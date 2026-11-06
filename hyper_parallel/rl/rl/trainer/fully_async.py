# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Complete-group asynchronous GRPO over the shared role and publication runtime."""
import logging
import sys
import time
from dataclasses import asdict, fields, replace
from functools import partial
from operator import methodcaller
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from rl.utils.async_audit import (
    EventTrace, record_async_state, record_async_rollout, record_async_checkpoint, install_async_audit,
    begin_async_update, finish_async_update, record_async_selection, record_publication_timing,
)
from rl.async_runtime.protocol import AsyncCheckpoint, ExperienceBuffer, decode_group, validate_async_config
from rl.async_runtime.producer import AsyncRuntime
from rl.data.experience import build_experience_batch
from rl.trainer.build import build_resources
from rl.trainer.distributed import run_synchronized
from rl.data.experience import pad_agent_call_batch_for_dp
from rl.utils.monitoring.async_metrics import report_async_update
from rl.weight_sync.sync import coordinator_call
from rl.utils.process_cleanup import temporary_environment


logger = logging.getLogger(__name__)


class FullyAsyncTrainer:
    """Keep optimizer updates and publication collectives on one training thread."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Build the learner runtime and bounded sampling coordinator."""
        self.settings = validate_async_config(config)
        self.config = dict(config)
        self.config["async_training"] = asdict(self.settings)
        role_config = {key: value for key, value in self.config.items() if key != "async_training"}
        resources = build_resources(role_config)
        self.state = resources.state
        self.actor = resources.actor
        self.targets = resources.targets
        self.residency = resources.residency
        self.publication = resources.publication
        self.lifecycle = resources.lifecycle
        self.device = resources.device
        self.device_handle = resources.device_handle
        self.parallel_dims = resources.parallel_dims
        self._dp_group_info = resources._dp_group_info
        self.rollout_engine = resources.rollout_engine
        self.rollout_manager = resources.rollout_manager
        self.evaluator = resources.evaluator
        self.checkpoints = resources.checkpoints
        self.tracker = resources._tracker
        self.codex_runtime = resources.codex_runtime
        self.run_synchronized = run_synchronized
        self.rank = dist.get_rank()
        self.trace = EventTrace(self.settings.audit_dir, "trainer", self.rank)
        self.loss_audit = install_async_audit(self, self.trace, self.settings.audit_dir is not None)
        self.buffer = ExperienceBuffer(self.settings, config["rollout"]["num_return_sequences"])
        self.sampler: Optional[AsyncRuntime] = None
        self.source_update = self.last_publication = self.optimizer_steps = 0
        self.version = 1

    def _root_command(self) -> tuple[str, Any]:
        """Select a complete global batch or request a fresh policy publication."""
        self.sampler.pump()
        self.buffer.evict_stale(self.state.global_step)
        required = self.parallel_dims.dp_size * self.settings.groups_per_dp_rank
        if self.state.global_step - self.source_update > self.settings.max_policy_lag_updates:
            return "publish", None
        deadline = time.monotonic() + self.settings.operation_timeout
        while len(self.buffer.ready) < required:
            self.sampler.fill(self.version, self.source_update)
            self.sampler.pump()
            self.buffer.evict_stale(self.state.global_step)
            if self.state.global_step > self.source_update and not self.buffer.reserved:
                self.sampler.drain()
                return "publish", None
            if time.monotonic() >= deadline:
                raise TimeoutError("No complete global batch available")
            time.sleep(0.01)
        self.sampler.fill(self.version, self.source_update)
        groups = self.buffer.select(required)
        record_async_selection(self.trace, groups, self.state.global_step + 1)
        return "train", tuple(self.buffer.ready[identity] for identity in self.buffer.selected)

    def _build_batch(self, payloads: tuple[bytes, ...]) -> Any:
        """Decode this data-parallel rank's complete groups onto its training device."""
        start = self.parallel_dims.dp_rank * self.settings.groups_per_dp_rank
        chosen = payloads[start:start + self.settings.groups_per_dp_rank]
        groups = [decode_group(data, self.buffer.candidates, self.settings.max_group_bytes) for data in chosen]
        settings = self.rollout_manager.agent_runner.settings
        batch = build_experience_batch(
            tuple(row for group in groups for row in group.trajectories),
            sum(group.generation_seconds for group in groups),
            settings, {"async_training": True}, mixed_policy_versions=True,
        )
        tensors = {item.name: getattr(batch, item.name).to(self.device)
                   for item in fields(batch) if isinstance(getattr(batch, item.name), torch.Tensor)}
        return replace(batch, **tensors)

    def _update(self, payloads: tuple[bytes, ...]) -> None:
        """Apply one Actor update before committing consumed sampling credits."""
        started = time.monotonic()
        step = self.state.global_step + 1
        self.trace.emit("prepare_start", update=step)
        rollout = self.run_synchronized("async batch validation", lambda: self._build_batch(payloads))
        rollout = pad_agent_call_batch_for_dp(rollout, self._dp_group_info)
        experience = self.targets.prepare(rollout, False, {})[0]
        begin_async_update(self, step, experience)
        update = self.actor.update(experience)
        finish_async_update(self)
        self.trace.emit("update_end", update=step, optimizer_steps=update.optimizer_steps,
                        gradient_norm=update.gradient_norm)

        record_async_rollout(self, experience, step)
        self.optimizer_steps += update.optimizer_steps
        self.state.global_step += 1
        self.state.consumed_samples += len(payloads) * self.buffer.candidates
        self.state.consumed_tokens += update.valid_tokens
        coordinator_call("commit async batch", self.buffer.commit)
        # Keep transport progress and grant new work before the next optimizer update.
        coordinator_call("async transport progress", partial(methodcaller("pump"), self.sampler))
        report_async_update(self, rollout, update, payloads, started)

    def _commit_policy(self) -> None:
        """Commit one Actor snapshot before exposing its version to sampling."""
        self.source_update = self.state.global_step
        self.version = self.source_update + 1
        self.trace.emit("publication_start", update=self.source_update, policy_version=self.version)
        self.publication.publish(self.version)
        self.trace.emit("publication_end", update=self.source_update, policy_version=self.version)
        self.last_publication = self.state.global_step

    def _publish(self, *, resume: bool = True) -> None:
        started = time.monotonic()
        coordinator_call("prepare async publication",
                         lambda: self.sampler.pause() if resume else self.sampler.drain())
        admitted_at = time.monotonic()
        self._commit_policy()
        record_publication_timing(self, started, admitted_at)
        if resume:
            coordinator_call("resume async sampling", lambda: self.sampler.resume(self.version))

    def _save(self, *, final: bool) -> None:
        """Persist quiescent source and buffer state after the model checkpoint."""
        coordinator_call("drain for async checkpoint", partial(methodcaller("drain"), self.sampler))
        snapshot = coordinator_call("snapshot async source", lambda: {
            "buffer": self.buffer.state_dict(), "source": self.sampler.snapshot(),
            "learner_update": self.state.global_step, "policy_version": self.version,
            "source_update": self.source_update, "optimizer_steps": self.optimizer_steps,
        })
        directory = self.checkpoints.directory(self.state.global_step)
        coordinator_call("invalidate async checkpoint", lambda: AsyncCheckpoint.invalidate(directory))
        record_async_state(self, "checkpoint_state")
        save = self.checkpoints.finalize if final else self.checkpoints.save
        save(self.state)
        coordinator_call("commit async checkpoint", lambda: AsyncCheckpoint.commit(directory, snapshot, self.config))
        record_async_checkpoint(self.trace, directory, snapshot)
        if not final:
            coordinator_call("resume after checkpoint", lambda: self.sampler.resume(self.version))

    def train(self) -> None:
        """Execute asynchronous collection, updates, publication and recoverable checkpoints."""
        completed = False
        try:
            load_path = coordinator_call("select async checkpoint", lambda: AsyncCheckpoint.resolve_path(self.config))
            self.config["train"]["checkpoint"]["load_path"] = load_path
            self.checkpoints.config["load_path"] = load_path
            restored = coordinator_call("validate async restore", lambda: AsyncCheckpoint.load(
                load_path, self.config,
            ))
            self.checkpoints.validate_resume()
            self.checkpoints.begin(self.state)
            if restored is not None:
                if restored["learner_update"] != self.state.global_step:
                    raise ValueError("Model and async snapshot update identities differ")
                self.optimizer_steps = restored["optimizer_steps"]
                if self.rank == 0:
                    self.buffer.load_state_dict(restored["buffer"])
            record_async_state(self, "restore_complete", checkpoint=load_path,
                               optimizer_steps=self.optimizer_steps)
            # Fresh vLLM workers start at version zero. Recovered Actor weights must always be republished.
            self.source_update = self.state.global_step
            self.version = self.source_update + 1
            service_env = {"HYPER_RL_ASYNC_TRAINING": "1",
                           "HYPER_RL_ASYNC_OPERATION_TIMEOUT": str(self.settings.operation_timeout),
                           "HYPER_RL_ASYNC_AUDIT_DIR": self.settings.audit_dir}
            with temporary_environment(service_env):
                self.rollout_engine.start_service()
            self._commit_policy()

            def start_sampler() -> None:
                """Start the CPU producer after committing inference weights."""
                self.sampler = AsyncRuntime(
                    self.config, self.buffer, None if restored is None else restored["source"],
                )
                self.sampler.resume(self.version)
            coordinator_call("start async sampler", start_sampler)
            while self.state.global_step < self.state.max_steps:
                kind, payload = coordinator_call("select async command", self._root_command)
                if kind == "publish":
                    self._publish()
                    continue
                self._update(payload)
                due = self.checkpoints.will_save(self.state.global_step)
                periodic = self.state.global_step - self.last_publication >= self.settings.publish_every_updates
                if due or periodic or self.state.global_step == self.state.max_steps:
                    self._publish(resume=not due)
                if due:
                    if self.evaluator is not None:
                        metrics, samples = self.evaluator.run(self.version)
                        self.tracker.log(metrics, step=self.state.global_step,
                                               sample_tables={"validation/samples": samples})
                    final = self.state.global_step == self.state.max_steps
                    self._save(final=final)
            coordinator_call("final drain", partial(methodcaller("drain"), self.sampler))
            completed = True
        finally:
            failed = sys.exc_info()[0] is not None
            cleanup_errors = []
            resources = [
                ("roles", self.lifecycle.close),
                ("trace", lambda: self.trace.emit("run_end", completed=completed, update=self.state.global_step)),
            ]
            if self.rank == 0 and self.sampler is not None:
                resources.insert(0, ("sampler", lambda: self.sampler.close(graceful=completed)))
            for name, close in resources:
                try:
                    close()
                except Exception as error:  # pylint: disable=broad-exception-caught
                    logger.warning("Async %s cleanup failed", name, exc_info=True)
                    cleanup_errors.append(error)
            if cleanup_errors and not failed:
                raise cleanup_errors[0]
