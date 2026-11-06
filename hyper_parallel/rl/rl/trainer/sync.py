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
"""Synchronous generation, learning and policy publication."""
import time
from typing import Any, Mapping, Optional
import torch
import torch.distributed as dist
from rl.distillation.monitoring import validate_probability_gate
from rl.config.schema import required_mapping, uses_colocated_vllm
from rl.data.experience import pad_agent_call_batch_for_dp
from rl.data.contracts import ExperienceBatch
from rl.data.prompts import build_prompt_records
from rl.utils.failure_reporting import report_failure
from rl.weight_sync.sync import PolicySnapshot
from rl.utils.monitoring.metrics import build_training_metrics, enforce_learning_gate, summarize_rollout
from rl.trainer.distributed import run_synchronized, add_step_timing_metrics
import logging

logger = logging.getLogger(__name__)

class SyncTrainer:
    """Synchronous training, residency and preparation over the shared role runtime."""

    def __init__(self, resources: Any) -> None:
        """Retain explicit loop dependencies; construction state is not retained."""
        self.resolved_config = resources.resolved_config
        self.state = resources.state
        self.actor = resources.actor
        self.critic = resources.critic
        self.targets = resources.targets
        self.residency = resources.residency
        self.publication = resources.publication
        self.teacher_scorer = resources.teacher_scorer
        self.reward_scorer = resources.reward_scorer
        self.lifecycle = resources.lifecycle
        self.rollout_engine = resources.rollout_engine
        self.rollout_manager = resources.rollout_manager
        self.evaluator = resources.evaluator
        self.checkpoints = resources.checkpoints
        self._tracker = resources._tracker
        self.device = resources.device
        self.parallel_dims = resources.parallel_dims
        self._dp_group_info = resources._dp_group_info
        self.distillation_config = resources.distillation_config
        self.train_dataloader = resources.train_dataloader
        self.sampler = resources.sampler
        self._log_steps = resources._log_steps
        self._log_samples = resources._log_samples
        self.model_registration = resources.model_registration
        self._prefetch_iterator = None
        self._prefetched_batch = None

    _run_rank_synchronized = staticmethod(run_synchronized)
    _add_step_timing_metrics = staticmethod(add_step_timing_metrics)

    def train(self) -> None:
        """Run synchronous rollout, learning, publication, and checkpointing."""
        try:
            completed = self._train_and_checkpoint()
            if completed:
                dist.barrier()
        except BaseException as error:
            self.lifecycle.fail(error, "training")
            raise
        try:
            self.lifecycle.close()
        except BaseException as error:
            report_failure(error, stage="cleanup", step=self.state.global_step)
            raise

    def _train_and_checkpoint(self) -> bool:
        """Run training and final checkpointing, returning only after success."""
        self.checkpoints.validate_resume()
        self.checkpoints.begin(self.state)
        if self.state.global_step > self.rollout_engine.policy_version:
            self.residency.release()
            self.residency.move(offload=True)
            self.rollout_engine.prepare_for_training()
            self.residency.move(offload=False)
            self.rollout_engine.update_weights(
                PolicySnapshot(
                    version=self.state.global_step,
                    model_name=self.model_registration.name,
                    payload=self.actor.actor_model,
                )
            )
            self.residency.release()
            self.residency.move(offload=True)
            self.rollout_engine.prepare_for_rollout()
        else:
            self.residency.release()
            self.residency.move(offload=True)
        if self.evaluator is not None and self.resolved_config.get("evaluation", {}).get("before_train", False):
            validation_metrics, validation_samples = self.evaluator.run(self.state.global_step)
            self._tracker.log(
                validation_metrics,
                step=self.state.global_step,
                sample_tables={"validation/samples": validation_samples},
            )
        if hasattr(self, "sampler"):
            self.sampler.set_epoch(self.state.epoch)
        data_iterator = iter(self.train_dataloader)
        while self.state.global_step < self.state.max_steps:
            if getattr(self, "_prefetched_batch", None) is None:
                batch, data_iterator = self._next_batch(data_iterator)
                self._prepare_rollout_batch(batch, self.state.global_step)
            else:
                batch, self._prefetched_batch = self._prefetched_batch, None
            self._prefetch_iterator = data_iterator
            self._train_step(batch)
        self._prefetch_iterator = None
        save_final = bool(
            required_mapping(
                required_mapping(self.resolved_config, "train"),
                "checkpoint",
            ).get("save_final", True)
        )
        if (
            (save_final or self.resolved_config.get("evaluation", {}).get("at_end", False))
            and self.evaluator is not None
            and self.evaluator.last_step != self.state.global_step
        ):
            validation_metrics, validation_samples = self.evaluator.run(
                self.state.global_step
            )
            self._tracker.log(
                validation_metrics,
                step=self.state.global_step,
                sample_tables={"validation/samples": validation_samples},
            )
        if save_final and uses_colocated_vllm(self.resolved_config):
            self._close_rollout_for_final_checkpoint()
        self.checkpoints.finalize(self.state)
        return True

    def _prepare_rollout_batch(self, batch: Mapping[str, Any], policy_version: int) -> None:
        """Start future CPU sessions while leaving all model work on the main thread."""
        if getattr(self.rollout_manager, "preparation_enabled", False) is not True:
            return
        records = self._run_rank_synchronized(
            "CPU prompt preparation",
            lambda: build_prompt_records(batch, batch["input_ids"], batch["attention_mask"]),
        )
        self._run_rank_synchronized(
            "agent preparation launch", lambda: self.rollout_manager.prepare(records, policy_version),
        )

    def _prepare_next_rollout(self, next_step: int) -> None:
        """Prefetch within one epoch, excluding checkpoint and evaluation boundaries."""
        if (getattr(self.rollout_manager, "preparation_enabled", False) is not True
                or getattr(self, "_prefetch_iterator", None) is None
                or next_step >= self.state.max_steps
                or self._evaluation_due(next_step) or self.checkpoints.will_save(next_step)):
            return
        batch = self._run_rank_synchronized(
            "next prompt batch", lambda: next(self._prefetch_iterator, None),
        )
        if batch is not None:
            self._prefetched_batch = batch
            self._prepare_rollout_batch(batch, next_step)

    def _generate_training_rollout(self, batch: Mapping[str, Any],
                                   timings: dict[str, float]) -> tuple[Any, ExperienceBatch]:
        """Construct prompts and generate the current policy's training episodes."""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        stage_started = time.perf_counter()
        prompt_records = self._run_rank_synchronized(
            "prompt record construction",
            lambda: build_prompt_records(batch, input_ids, attention_mask),
        )
        if prompt_records is None:
            raise RuntimeError("Prompt record construction failed without a synchronized error")
        rollout = self.rollout_manager.generate(
            prompt_records=prompt_records,
            policy_version=self.state.global_step,
        )
        timings["gen"] = time.perf_counter() - stage_started
        return prompt_records, rollout

    def _train_step(self, batch: Mapping[str, Any]) -> None:
        """Run one explicit RL2-style role pipeline."""
        step_started = time.perf_counter()
        timings: dict[str, float] = {}
        next_step = self.state.global_step + 1
        logger.info(
            "rank=%d step=%d sample_index=%d",
            dist.get_rank(),
            next_step,
            int(batch["sample_indices"][0]),
        )
        prompt_records, rollout = self._generate_training_rollout(batch, timings)
        stage_started = time.perf_counter()
        if self.reward_scorer is None:
            self.rollout_engine.prepare_for_training()
        else:
            rollout = self.reward_scorer.score(prompt_records, rollout)
            timings["reward"] = float(rollout.metadata["reward_seconds"])
            timings["reward_wake"] = float(rollout.metadata["reward_wake_seconds"])
            timings["reward_sleep"] = float(rollout.metadata["reward_sleep_seconds"])
        timings["prepare_training"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        self.residency.move(offload=False)
        if self.residency.offload_manager is not None:
            timings["restore_training_state"] = time.perf_counter() - stage_started
        if rollout.old_log_probs is None and self.targets.old_log_prob_source != "actor":
            raise RuntimeError("Training rollout did not produce old_log_probs")
        rollout = pad_agent_call_batch_for_dp(rollout, self._dp_group_info)
        if self.teacher_scorer is not None:
            stage_started = time.perf_counter()
            rollout = self.teacher_scorer.score(prompt_records, rollout)
            timings['teacher'] = time.perf_counter() - stage_started
        collect_diagnostics = next_step % self._log_steps == 0 or self._evaluation_due(next_step)
        experience, diagnostic_metrics = self.targets.prepare(
            rollout,
            collect_diagnostics,
            timings,
        )
        if self.teacher_scorer is not None:
            diagnostic_metrics.update(self._run_rank_synchronized(
                "OPD pre-update probability gate",
                lambda: validate_probability_gate(experience, self.distillation_config,
                                                  group=self._dp_group_info.group,
                                                  size=self.parallel_dims.dp_size),
            ))
        stage_started = time.perf_counter()
        self._prepare_next_rollout(next_step)
        timings["prepare_next_rollout"] = time.perf_counter() - stage_started
        # Detached inference targets no longer need the full role parameters.
        self.residency.release()
        stage_started = time.perf_counter()
        actor_update = self.actor.update(experience)
        timings["update_actor"] = time.perf_counter() - stage_started
        if self.teacher_scorer is not None:
            diagnostic_metrics.update(self.actor.distillation_metrics())
            counts = torch.tensor([
                sum(key == teacher_key for key in experience.teacher_keys)
                for teacher_key, _ in self.distillation_config.teachers
            ], device=self.device, dtype=torch.int64)
            if self.parallel_dims.dp_size > 1:
                dist.all_reduce(counts, group=self._dp_group_info.group)
            diagnostic_metrics.update({
                f'distillation/teacher_samples/{key}': float(count)
                for (key, _), count in zip(self.distillation_config.teachers, counts.tolist())
            })
        critic_update = None
        if self.critic is not None:
            stage_started = time.perf_counter()
            critic_update = self.critic.update(experience)
            timings["update_critic"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        self.publication.publish(next_step)
        timings["weight_sync"] = time.perf_counter() - stage_started
        if self.residency.offload_manager is not None:
            diagnostic_metrics.update(self.residency.metrics)
        timings["step"] = time.perf_counter() - step_started
        if collect_diagnostics:
            self._add_step_timing_metrics(diagnostic_metrics, timings)
        self._complete_step(
            step=next_step,
            batch=batch,
            rollout=rollout,
            actor_update=actor_update,
            critic_update=critic_update,
            diagnostic_metrics=diagnostic_metrics,
        )

    def _complete_step(
        self,
        *,
        step: int,
        batch: Mapping[str, Any],
        rollout: ExperienceBatch,
        actor_update: Any,
        critic_update: Optional[Any],
        diagnostic_metrics: Mapping[str, float],
    ) -> None:
        """Record, evaluate, and checkpoint one successfully published policy."""
        self.state.global_step = step
        rollout_metrics, samples = summarize_rollout(
            rollout,
            batch,
            step=step,
            sample_limit=self._log_samples,
            is_request_owner=bool(getattr(self.rollout_engine, "is_request_owner", True)),
        )
        metrics = build_training_metrics(
            step=step,
            actor_update=actor_update,
            rollout_metrics=rollout_metrics,
            critic_update=critic_update,
            policy=self.rollout_engine,
            diagnostic_metrics=diagnostic_metrics,
        )
        train_config = required_mapping(self.resolved_config, "train")
        enforce_learning_gate(
            metrics,
            step=step,
            config=train_config.get("learning_gate", {}),
            run_synchronized=self._run_rank_synchronized,
        )
        checkpoint_will_save = self.checkpoints.will_save(step)
        validation_samples: list[dict[str, Any]] = []
        evaluation_due = self._evaluation_due(step)
        if evaluation_due:
            validation_metrics, validation_samples = self.evaluator.run(step)
            metrics.update(validation_metrics)
        if step % self._log_steps == 0 or evaluation_due:
            self._tracker.log(
                metrics,
                step=step,
                samples=samples,
                sample_tables={"validation/samples": validation_samples},
            )
        if checkpoint_will_save and uses_colocated_vllm(self.resolved_config):
            if step == self.state.max_steps:
                self._close_rollout_for_final_checkpoint()
            else:
                self.rollout_engine.prepare_for_training()
                self.residency.move(offload=False)
                self.residency.release()
        self.checkpoints.complete_step(
            self.state,
            loss=actor_update.total_loss,
            grad_norm=actor_update.gradient_norm,
        )
        if (
            checkpoint_will_save
            and uses_colocated_vllm(self.resolved_config)
            and step < self.state.max_steps
        ):
            self.residency.release()
            self.residency.move(offload=True)
            self.rollout_engine.prepare_for_rollout()

    def _close_rollout_for_final_checkpoint(self) -> None:
        """Release inference processes and IPC consumers before the final save."""
        if self.lifecycle._rollout_closed_for_final_checkpoint:
            return
        self.rollout_engine.close()
        self.lifecycle._rollout_closed_for_final_checkpoint = True
        self.residency.move(offload=False)
        self.residency.release()

    def _evaluation_due(self, step: int) -> bool:
        """Choose an independent evaluation interval or the legacy saving boundary."""
        if self.evaluator is None or step <= 0:
            return False
        evaluation = self.resolved_config.get("evaluation", {})
        checkpoint = self.resolved_config.get("train", {}).get("checkpoint", {})
        if step == self.state.max_steps and (
            checkpoint.get("save_final", True) or evaluation.get("at_end", False)
        ):
            # Final checkpointing closes rollout; evaluate the published policy first.
            return True
        interval = evaluation.get("every_n_steps")
        if interval is None:
            return self.checkpoints.will_save(step)
        return interval > 0 and step % interval == 0

    def _next_batch(self, data_iterator: Any) -> tuple[Mapping[str, Any], Any]:
        """Fetch one batch, advancing the distributed sampler epoch on exhaustion."""
        try:
            return next(data_iterator), data_iterator
        except StopIteration:
            self.state.epoch += 1
            self.sampler.set_epoch(self.state.epoch)
            data_iterator = iter(self.train_dataloader)
            return next(data_iterator), data_iterator
