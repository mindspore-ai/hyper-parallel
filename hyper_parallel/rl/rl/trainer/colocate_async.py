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
"""Colocated GRPO: sample, abort, sleep, update, publish, then resume partial work."""
import logging
import sys
import time
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from operator import methodcaller
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from rl.algorithm.rollout_correction import correction_metrics, prepare_rollout_correction
from rl.async_runtime.protocol import AsyncCheckpoint, decode_group
from rl.async_runtime.colocated import ColocateAsyncRollouter
from rl.config.validate import validate_colocate_async_config
from rl.data.experience import build_experience_batch
from rl.rollout.client import AsyncGenerationClient
from rl.trainer.build import build_resources
from rl.trainer.distributed import run_synchronized
from rl.data.experience import pad_agent_call_batch_for_dp
from rl.utils.async_audit import EventTrace
from rl.utils.monitoring.colocate_async_metrics import report_colocate_update, summarize_colocate_payloads
from rl.utils.process_cleanup import temporary_environment
from rl.weight_sync.sync import coordinator_call

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColocateUpdateResult:
    """Carry one lease's update outcome and diagnostics without changing trainer state implicitly."""
    rollout: Any
    update: Any
    diagnostics: dict[str, float]
    rejected_groups: tuple[str, ...] = ()


class ColocateAsyncTrainer:
    """Own stage ordering; sampler and model runtimes retain their existing contracts."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Validate the independent mode and construct its shared model-role runtime."""
        self.settings = validate_colocate_async_config(config)
        self.config = deepcopy(dict(config))
        self.config["colocate_async"] = asdict(self.settings)
        if self.settings.off_policy.mode == "off":
            self.config["colocate_async"].pop("off_policy")
        resources = build_resources({key: value for key, value in self.config.items() if key != "colocate_async"})
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
        self.sampling = self.settings.sampling_config(self.parallel_dims.dp_size)
        self.batch_size = self.parallel_dims.dp_size * self.settings.groups_per_dp_rank
        self.candidates = self.config["rollout"]["num_return_sequences"]
        self.rollouter: Optional[ColocateAsyncRollouter] = None
        self.trace = EventTrace(self.settings.audit_dir, "colocate_trainer", self.rank)
        self.optimizer_steps = 0
        self._restored: Optional[dict] = None
        self._snapshot: Optional[dict] = None
        self.timings: dict[str, float] = {}

    @property
    def version(self) -> int:
        """Return the policy version published at the current successful update boundary."""
        return self.state.global_step + 1

    def _load_async_state(self, path: Optional[str]) -> tuple[Optional[int], int]:
        """Keep the potentially large restored buffer on its root owner only."""
        self._restored = AsyncCheckpoint.load(path, self.config)
        if self._restored is None:
            return None, 0
        if self._restored.get("execution_mode") != "colocate_async":
            raise ValueError("Checkpoint is not a colocated asynchronous snapshot")
        if self._restored["source_update"] != self._restored["learner_update"]:
            raise ValueError("Colocated checkpoint must contain the latest published Actor")
        return self._restored["learner_update"], self._restored["optimizer_steps"]

    def _pause_initial_service(self) -> None:
        """Close partial admission before any producer exists or any Actor is published."""
        with closing(AsyncGenerationClient.from_config(self.config, self.settings.operation_timeout)) as client:
            client.pause()

    def _start_rollouter(self, restored: Optional[dict]) -> None:
        """Start or rebuild the producer from its root-owned source and ready groups."""
        self.rollouter = ColocateAsyncRollouter(self.config, self.sampling, restored)
        self.rollouter.resume(self.version)

    def _restore_and_start(self) -> None:
        """Validate the checkpoint and publish its policy before starting sampling."""
        path = coordinator_call("select colocated checkpoint", partial(AsyncCheckpoint.resolve_path, self.config))
        self.config["train"]["checkpoint"]["load_path"] = path
        self.checkpoints.config["load_path"] = path
        restored_step, self.optimizer_steps = coordinator_call(
            "validate colocated snapshot", partial(self._load_async_state, path),
        )
        self.checkpoints.validate_resume()
        self.checkpoints.begin(self.state)
        if restored_step is not None and restored_step != self.state.global_step:
            raise ValueError("Model and colocated snapshot update identities differ")
        self.residency.release()
        environment = {"HYPER_RL_ASYNC_TRAINING": "1",
                       "HYPER_RL_ASYNC_OPERATION_TIMEOUT": str(self.settings.operation_timeout),
                       "HYPER_RL_ASYNC_AUDIT_DIR": self.settings.audit_dir}
        with temporary_environment(environment):
            self.rollout_engine.start_service()
        coordinator_call("close initial partial admission", self._pause_initial_service)
        self.rollout_engine.prepare_for_training()
        self._publish_policy()
        coordinator_call("start colocated sampling", lambda: self._start_rollouter(self._restored))
        self._restored = None
        self.trace.emit("restore_complete", update=self.state.global_step, checkpoint=path)

    def _pause_rollout_for_training(self) -> None:
        started = time.monotonic()
        coordinator_call("pause colocated sampling", partial(methodcaller("pause"), self.rollouter))
        self.timings["timing_s/pause"] = time.monotonic() - started
        started = time.monotonic()
        self.rollout_engine.prepare_for_training()
        self.timings["timing_s/sleep"] = time.monotonic() - started
        self.trace.emit("sleep_done", update=self.state.global_step)

    def _build_local_batch(self, payloads: tuple[bytes, ...]) -> Any:
        """Decode this rank's complete groups and move the batch to its training device."""
        if len(payloads) != self.batch_size:
            raise ValueError("Colocated payload must contain one complete global batch")
        start = self.parallel_dims.dp_rank * self.settings.groups_per_dp_rank
        chosen = payloads[start:start + self.settings.groups_per_dp_rank]
        groups = [decode_group(data, self.candidates, self.settings.max_group_bytes) for data in chosen]
        rollout = build_experience_batch(
            tuple(row for group in groups for row in group.trajectories),
            sum(group.generation_seconds for group in groups),
            self.rollout_manager.agent_runner.settings,
            {"async_training": True, "execution_mode": "colocate_async"}, mixed_policy_versions=True,
        )
        tensors = {item.name: getattr(rollout, item.name).to(self.device)
                   for item in fields(rollout) if isinstance(getattr(rollout, item.name), torch.Tensor)}
        self.trace.emit("device_batch_built", update=self.state.global_step + 1)
        return replace(rollout, **tensors)

    def _update_actor(self, payloads: tuple[bytes, ...]) -> ColocateUpdateResult:
        """Correct and update a leased batch before committing its sampling identities."""
        started = time.monotonic()
        rollout = self.run_synchronized("colocated batch validation", partial(self._build_local_batch, payloads))
        rollout = pad_agent_call_batch_for_dp(rollout, self._dp_group_info)
        experience = self.targets.prepare(rollout, False, {})[0]
        experience, diagnostics, rejected = self._correct_experience(experience)
        if rejected:
            coordinator_call("reject off-policy groups", partial(methodcaller("reject_batch_groups", rejected),
                                                                 self.rollouter))
            return ColocateUpdateResult(None, None, diagnostics, rejected)
        self.residency.release()
        step = self.state.global_step + 1
        self.trace.emit("update_start", update=step)
        update = self.run_synchronized("colocated Actor update", lambda: self.actor.update(experience))
        self.trace.emit("update_end", update=step, optimizer_steps=update.optimizer_steps,
                        gradient_norm=update.gradient_norm)
        self.optimizer_steps += update.optimizer_steps
        self.state.global_step = step
        self.state.consumed_samples += len(payloads) * self.candidates
        self.state.consumed_tokens += update.valid_tokens
        coordinator_call("commit colocated batch", partial(methodcaller("commit_batch"), self.rollouter))
        self.timings["timing_s/update"] = time.monotonic() - started
        return ColocateUpdateResult(rollout, update, diagnostics)

    def _correct_experience(self, experience: Any) -> tuple[Any, dict[str, float], tuple[str, ...]]:
        """Freeze correction weights and agree on rejected groups across training ranks."""
        config = self.settings.off_policy
        if config.mode == "off":
            return experience, {}, ()

        def prepare() -> Any:
            """Finish local validation before any diagnostic collective."""
            anchor = self.actor.compute_log_probs(experience)
            return prepare_rollout_correction(experience, anchor, config, self.version)

        experience, analysis = self.run_synchronized("fixed rollout correction anchor", prepare)
        world = self.parallel_dims.dp_size
        group = self.parallel_dims.dp_cp_mesh.get_group() if world > 1 else None
        diagnostics = correction_metrics(analysis, experience.loss_action_mask, config,
                                         group=group, world_size=world)
        rejected = [analysis.rejected_groups] * world
        if world > 1:
            dist.all_gather_object(rejected, analysis.rejected_groups, group=group)
        identities = tuple(sorted({identity for values in rejected for identity in values}))
        diagnostics["off_policy/rejected_groups"] = len(identities)
        self.trace.emit("correction_prepared", update=self.state.global_step,
                        anchor_version=self.version, rejected_groups=len(identities))
        return experience, diagnostics, identities

    def _resume_unchanged_policy(self) -> None:
        self.residency.release()
        self.rollout_engine.prepare_for_rollout()
        coordinator_call("resume for off-policy refill", partial(methodcaller("resume", self.version), self.rollouter))

    def _publish_policy(self) -> None:
        started = time.monotonic()
        self.trace.emit("publication_start", update=self.state.global_step, policy_version=self.version)
        self.publication.publish(self.version)
        self.trace.emit("publication_end", update=self.state.global_step, policy_version=self.version)
        self.timings["timing_s/publish"] = time.monotonic() - started

    def _publish_policy_and_resume(self) -> None:
        self._publish_policy()
        started = time.monotonic()
        coordinator_call("resume colocated sampling", partial(methodcaller("resume", self.version), self.rollouter))
        self.timings["timing_s/resume"] = time.monotonic() - started

    def _freeze_snapshot(self) -> None:
        self._snapshot = {
            **self.rollouter.drain_and_snapshot(), "execution_mode": "colocate_async",
            "learner_update": self.state.global_step, "source_update": self.state.global_step,
            "policy_version": self.version, "optimizer_steps": self.optimizer_steps,
        }

    def _evaluate_checkpoint(self) -> bool:
        """Hand the shared Codex gateway from a drained producer to evaluation."""
        if self.evaluator is None:
            return False
        handoff = (self.evaluator is not None and self.config.get("agentic", {}).get("runner") == "codex")
        if handoff:
            coordinator_call("release sampling gateway", partial(methodcaller("close", graceful=True), self.rollouter))
        self._evaluate_policy(self.version)
        return handoff

    def _save_checkpoint(self, *, final: bool) -> None:
        """Drain sampling and atomically commit matching model and producer snapshots."""
        started = time.monotonic()
        # Partial requests must be able to run while drain waits for their completion.
        coordinator_call("freeze colocated source", self._freeze_snapshot)
        self.timings["timing_s/checkpoint_drain"] = time.monotonic() - started
        restart_producer = self._evaluate_checkpoint()
        self._pause_rollout_for_training()
        directory = self.checkpoints.directory(self.state.global_step)
        coordinator_call("invalidate colocated snapshot", partial(AsyncCheckpoint.invalidate, directory))
        self.checkpoints.save(self.state)
        if final and self.config["train"]["checkpoint"].get("save_final", True):
            self.checkpoints.finalize(self.state)
        coordinator_call("commit colocated snapshot",
                         lambda: AsyncCheckpoint.commit(directory, self._snapshot, self.config))
        self.trace.emit("checkpoint_committed", update=self.state.global_step)
        if not final:
            self.residency.release()
            self.rollout_engine.prepare_for_rollout()
            if restart_producer:
                coordinator_call("restart sampling after evaluation", lambda: self._start_rollouter(self._snapshot))
            else:
                coordinator_call("resume after colocated checkpoint",
                                 partial(methodcaller("resume", self.version), self.rollouter))
        self._snapshot = None
        self.tracker.log({"timing_s/checkpoint": time.monotonic() - started,
                                "timing_s/checkpoint_drain": self.timings["timing_s/checkpoint_drain"]},
                               step=self.state.global_step)

    def _report_update(self, result: ColocateUpdateResult, payloads: tuple[bytes, ...], started: float) -> None:
        def diagnostics() -> dict[str, float]:
            """Read root-owned counts only after rollout reporting collectives finish."""
            counts = coordinator_call("colocated sampling metrics", partial(methodcaller("metrics"), self.rollouter))
            counts.update(summarize_colocate_payloads(payloads, self.candidates, self.settings.max_group_bytes,
                                                      self.state.global_step))
            counts.update({"colocate/optimizer_steps": self.optimizer_steps,
                           "timing_s/step": time.monotonic() - started, **self.timings, **result.diagnostics})
            return counts
        report_colocate_update(self, result.rollout, result.update, diagnostics)

    def _finish_sampling(self) -> None:
        if self.rollout_engine.phase == "rollout":
            coordinator_call("drain final colocated sampling", self._freeze_snapshot)
            self._snapshot = None
            self._pause_rollout_for_training()

    def _close(self, completed: bool, failed: bool) -> None:
        """Close every owned runtime and preserve the first failure."""
        errors = []
        if self.rank == 0 and self.rollouter is not None:
            try:
                self.rollouter.close(graceful=completed)
            # Continue closing roles even when sampler cleanup fails.
            except Exception as error:  # pylint: disable=broad-exception-caught
                logger.exception("Colocated sampler cleanup failed")
                errors.append(error)
        try:
            self.lifecycle.close()
        # Preserve an active training failure, or re-raise the first cleanup failure.
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.exception("Colocated role cleanup failed")
            errors.append(error)
        if errors and not failed:
            raise errors[0]

    def train(self) -> None:
        """Run the explicit sample/abort/sleep/update/publish/resume sequence."""
        completed = False
        refill_rounds = 0
        try:
            self._restore_and_start()
            while self.state.global_step < self.state.max_steps:
                started = time.monotonic()
                self.timings = {}
                payloads = coordinator_call("select colocated groups", lambda: self.rollouter.sample_batch(
                    self.version, self.state.global_step, self.batch_size,
                ))
                self.timings["timing_s/sample_wait"] = time.monotonic() - started
                self._pause_rollout_for_training()
                result = self._update_actor(payloads)
                if result.rejected_groups:
                    self.tracker.log(result.diagnostics, step=self.state.global_step)
                    if refill_rounds >= self.settings.off_policy.max_refill_rounds:
                        raise RuntimeError("Off-policy group rejection exhausted max_refill_rounds")
                    refill_rounds += 1
                    self._resume_unchanged_policy()
                    continue
                if self.settings.off_policy.mode != "off":
                    result.diagnostics["off_policy/refill_rounds"] = refill_rounds
                refill_rounds = 0
                self._publish_policy_and_resume()
                final = self.state.global_step == self.state.max_steps
                self._report_update(result, payloads, started)
                if self.checkpoints.will_save(self.state.global_step):
                    self._save_checkpoint(final=final)
            self._finish_sampling()
            completed = True
        finally:
            self._close(completed, sys.exc_info()[0] is not None)


    def _evaluate_policy(self, version: int) -> None:
        """Release the evaluation gateway before restarting colocated sampling."""
        try:
            metrics, samples = self.evaluator.run(version)
            self.tracker.log(metrics, step=self.state.global_step, sample_tables={"validation/samples": samples})
        finally:
            if self.config.get("agentic", {}).get("runner") == "codex":
                self.codex_runtime.close()
