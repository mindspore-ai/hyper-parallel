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
"""Actor model execution and algorithm-neutral policy optimization."""
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn import functional

from rl.algorithm.loss import RLAlgorithm
from rl.algorithm.distillation import compute_opd_loss
from rl.distillation.monitoring import DistillationMetricAccumulator
from rl.distillation.config import DistillationConfig
from rl.consistency import CONSISTENCY_PROFILE_OFF, consistency_runtime_state, trainer_sequence_log_probs
from rl.dataset.contracts import ExperienceBatch
from rl.router_replay import RouterReplay
from rl.weight_sync import synchronized_call
from rl.utils.monitoring.metrics import (
    ActorMetricAccumulator,
    ActorMicroBatchMetrics,
    ActorUpdateMetrics,
)
from hyper_parallel import HSDPModule, SkipDTensorDispatch, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.components.optim.mixed_precision_optimizer import MixedPrecisionOptimizer

# Role execution uses explicit compute/update APIs; forward remains the Module default.
class Actor(torch.nn.Module):  # pylint: disable=abstract-method
    """Own one policy model and, when trainable, its optimization runtime."""

    def __init__(
        self,
        actor_model: torch.nn.Module,
        algorithm: RLAlgorithm,
        micro_batch_size: int,
        *,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        device: Optional[Any] = None,
        dp_group_info: Optional[Any] = None,
        dp_size: int = 1,
        response_mini_batch_size: Optional[int] = None,
        update_epochs: int = 1,
        max_grad_norm: float = 1.0,
        distillation: Optional[DistillationConfig] = None,
        router_replay: Optional[RouterReplay] = None,
    ) -> None:
        """Initialize a trainable Actor or an inference-only reference Actor."""
        super().__init__()
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        if update_epochs <= 0:
            raise ValueError(f"update_epochs must be positive, got {update_epochs}")
        if optimizer is None and lr_scheduler is not None:
            raise ValueError("lr_scheduler requires a trainable Actor optimizer")
        if response_mini_batch_size is None:
            response_mini_batch_size = micro_batch_size
        if response_mini_batch_size <= 0:
            raise ValueError(
                "response_mini_batch_size must be positive, "
                f"got {response_mini_batch_size}"
            )
        if router_replay is not None and optimizer is None:
            raise ValueError("Router replay belongs only to the trainable Actor")
        self.router_replay = router_replay
        self.actor_model = actor_model
        self.algorithm = algorithm
        self.distillation = distillation
        self._opd_metrics = None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._device = device
        self._dp_group_info = dp_group_info
        self._dp_size = dp_size
        self._micro_batch_size = micro_batch_size
        self._response_mini_batch_size = response_mini_batch_size
        self._update_epochs = update_epochs
        self._max_grad_norm = max_grad_norm
        if optimizer is None:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            self.eval()
        elif device is None:
            raise ValueError("A trainable Actor requires a device")

    def sequence_log_probs(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute next-token probabilities, optionally selecting actions on capable models.

        Unselected positions are zero only when the model declares mask support.
        Consistency profiles and legacy models retain their complete-vector behavior.
        """
        if tuple(attention_mask.shape) != tuple(sequences.shape):
            raise ValueError("attention_mask must align with sequences")
        packed_log_probs = trainer_sequence_log_probs(
            self.actor_model,
            sequences,
            attention_mask,
        )
        if packed_log_probs is not None:
            return packed_log_probs
        if getattr(self.actor_model, "supports_token_log_probs", False):
            selected_kwargs = {}
            if action_mask is not None and getattr(self.actor_model, "supports_token_log_probs_mask", False):
                selected_kwargs["token_log_probs_mask"] = action_mask.bool()
            log_probs = self.actor_model(
                input_ids=sequences,
                attention_mask=attention_mask,
                use_cache=False,
                return_token_log_probs=True,
                **selected_kwargs,
            )
            if (not isinstance(log_probs, torch.Tensor) or log_probs.dtype != torch.float32
                    or tuple(log_probs.shape) != (sequences.shape[0], sequences.shape[1] - 1)):
                raise ValueError("Model token probabilities must be FP32 [batch, sequence_length - 1]")
            return log_probs
        outputs = self.actor_model(
            input_ids=sequences,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        next_token_ids = sequences[:, 1:]
        log_probs = logits[:, :-1, :].float().log_softmax(dim=-1)
        return log_probs.gather(
            dim=-1,
            index=next_token_ids.unsqueeze(-1),
        ).squeeze(-1)

    def _micro_batch_inputs(self, experience: ExperienceBatch, start: int, end: int) -> tuple:
        """Select identical token coordinates for model inputs and router replay."""
        sequences = experience.sequences[start:end]
        attention_mask = experience.attention_mask[start:end]
        padded_length = sequences.shape[1]
        kwargs = {}
        if getattr(self.actor_model, "supports_token_log_probs_mask", False):
            kwargs["action_mask"] = experience.loss_action_mask[start:end]
            if (experience.trajectories
                    and consistency_runtime_state()["trainer_recipe"] == CONSISTENCY_PROFILE_OFF):
                # Canonical trajectories retain the complete context before batch right-padding.
                length = max(row.token_ids.numel() for row in experience.trajectories[start:end])
                sequences = sequences[:, :length]
                attention_mask = attention_mask[:, :length]
                kwargs["action_mask"] = kwargs["action_mask"][:, :length - 1]
        return sequences, attention_mask, kwargs, padded_length

    def _experience_log_probs(self, experience: ExperienceBatch, start: int, end: int) -> torch.Tensor:
        """Trim selected-output micro-batches while retaining the batch's probability shape."""
        sequences, attention_mask, kwargs, padded_length = self._micro_batch_inputs(experience, start, end)
        log_probs = self.sequence_log_probs(sequences, attention_mask, **kwargs)
        padding = padded_length - sequences.shape[1]
        return functional.pad(log_probs, (0, padding)) if padding else log_probs

    def _validate_router_experience(self, experience: ExperienceBatch) -> None:
        """Reject invalid routing on all training ranks before entering model collectives."""
        if self.router_replay is not None:
            synchronized_call(
                "Actor router replay validation",
                lambda: self.router_replay.validate(
                    experience.router_traces, experience.attention_mask, experience.action_mask,
                ),
            )

    def _router_scope(self, experience: ExperienceBatch, start: int, end: int) -> Any:
        """Prepare exactly the trimmed input coordinates for one forward/backward scope."""
        if self.router_replay is None:
            return nullcontext()
        _, attention_mask, _, _ = self._micro_batch_inputs(experience, start, end)
        targets = self.router_replay.prepare(
            experience.router_traces[start:end],
            attention_mask,
            experience.action_mask[start:end, :attention_mask.shape[1]],
        )
        return self.router_replay.use(targets)

    def compute_log_probs(self, experience: ExperienceBatch) -> torch.Tensor:
        """Compute detached policy log-probabilities in response micro-batches."""
        self._validate_router_experience(experience)
        was_training = self.training
        self.eval()
        chunks = []
        try:
            with torch.no_grad():
                for start in range(
                    0,
                    experience.sequences.shape[0],
                    self._micro_batch_size,
                ):
                    end = min(
                        start + self._micro_batch_size,
                        experience.sequences.shape[0],
                    )
                    with self._router_scope(experience, start, end):
                        chunks.append(self._experience_log_probs(experience, start, end))
        finally:
            self.train(was_training)
        return torch.cat(chunks, dim=0).detach()

    def _loss_is_finite(
        self, scaled_loss: torch.Tensor, current_log_probs: torch.Tensor, action_mask: torch.Tensor,
    ) -> bool:
        """Synchronize the finite-loss decision before distillation or GSPO backward."""
        if self.algorithm.name == "gspo" or (self.distillation is not None and self.distillation.enabled):
            finite = scaled_loss.detach().isfinite() & current_log_probs.detach().masked_fill(
                ~action_mask.bool(), 0.0,
            ).isfinite().all()
            finite = finite.to(dtype=torch.int32)
            if self._dp_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=self._dp_group_info.group)
            return bool(finite.item())
        return bool(scaled_loss.isfinite().item())

    def forward_backward(
        self,
        experience: ExperienceBatch,
        start: int,
        end: int,
        *,
        global_tokens: int,
        global_sequences: Optional[int] = None,
    ) -> ActorMicroBatchMetrics:
        """Backpropagate a micro-batch; distributed callers validate replay via update first."""
        self._require_trainable()
        self._validate_experience(experience)
        response_count = int(experience.sequences.shape[0])
        if start < 0 or end <= start or end > response_count:
            raise ValueError(
                f"Invalid response slice [{start}:{end}] for {response_count} responses"
            )
        if global_tokens <= 0:
            raise ValueError(f"global_tokens must be positive, got {global_tokens}")
        normalizer = global_tokens
        if self.algorithm.requirements.loss_aggregation == "seq-mean-token-mean":
            if global_sequences is None or global_sequences <= 0 or global_sequences > global_tokens:
                raise ValueError("Sequence loss requires 0 < global_sequences <= global_tokens")
            normalizer = global_sequences
        with self._router_scope(experience, start, end):
            current_log_probs = self._experience_log_probs(experience, start, end)
            anchor_log_probs = experience.old_log_probs
            if self.distillation is not None and self.distillation.enabled:
                anchor_log_probs = experience.rollout_log_probs
            old_log_probs = anchor_log_probs[start:end].detach()
            action_mask = experience.loss_action_mask[start:end]
            reference_log_probs = experience.reference_log_probs
            output = self.algorithm.compute_actor_loss(
                current_log_probs=current_log_probs,
                old_log_probs=old_log_probs,
                reference_log_probs=(
                    None if reference_log_probs is None else reference_log_probs[start:end]
                ),
                advantages=experience.advantages[start:end],
                action_mask=action_mask,
            )
            total_loss_sum = output.total_loss_sum
            scaled_loss = total_loss_sum / normalizer * self._dp_size
            if self.distillation is not None and self.distillation.enabled:
                opd_loss = compute_opd_loss(current_log_probs, old_log_probs,
                                            experience.teacher_log_probs[start:end], action_mask, self.distillation)
                total_loss_sum = total_loss_sum + self.distillation.coefficient * opd_loss * normalizer / global_tokens
                scaled_loss = total_loss_sum / normalizer * self._dp_size
                if self._opd_metrics is None:
                    self._opd_metrics = DistillationMetricAccumulator(opd_loss, self.distillation)
                self._opd_metrics.add(current_log_probs, old_log_probs, experience.teacher_log_probs[start:end],
                                      action_mask, experience.teacher_keys[start:end])
            is_finite = self._loss_is_finite(scaled_loss, current_log_probs, action_mask)
            if not is_finite:
                raise RuntimeError(
                    f"Non-finite {self.algorithm.name} loss detected on response slice "
                    f"[{start}:{end}]"
                )
            scaled_loss.backward()
        return ActorMicroBatchMetrics(
            total_loss_sum=total_loss_sum.detach(),
            policy_loss_sum=output.policy_loss_sum.detach(),
            kl_loss_sum=output.regularization_loss_sum.detach(),
            old_policy_kl_sum=output.old_policy_kl_sum.detach(),
            log_ratio_abs_sum=(
                (current_log_probs.detach() - old_log_probs).abs().masked_fill(~action_mask.bool(), 0.0)
            ).flatten().sum(dim=0),
            clipped_token_count=output.clipped_token_count.detach(),
            clipped_sequence_count=output.clipped_sequence_count,
        )

    def update(self, experience: ExperienceBatch) -> ActorUpdateMetrics:
        """Run policy epochs, optimizer steps, and metric finalization."""
        self._require_trainable()
        self._opd_metrics = (DistillationMetricAccumulator(experience.rewards, self.distillation)
                             if self.distillation is not None and self.distillation.enabled else None)
        self._validate_router_experience(experience)
        response_count = (self._validate_update_experience(experience)
                          if self.algorithm.name == 'gspo' or self.distillation is not None
                          else self._validate_experience(experience))
        accumulator = ActorMetricAccumulator.create(
            experience.sequences.new_zeros(()),
            dp_group_info=self._dp_group_info,
            dp_size=self._dp_size,
            loss_aggregation=self.algorithm.requirements.loss_aggregation,
        )
        self.train()
        for _ in range(self._update_epochs):
            for mini_start in range(0, response_count, self._response_mini_batch_size):
                mini_end = min(
                    mini_start + self._response_mini_batch_size,
                    response_count,
                )
                local_mask = experience.loss_action_mask[mini_start:mini_end]
                if self.algorithm.name == "gspo":
                    global_tokens, global_sequences = self._global_batch_counts(local_mask)
                else:
                    global_tokens = self._global_token_count(local_mask)
                    global_sequences = None
                self.optimizer.zero_grad(set_to_none=True)
                for start in range(mini_start, mini_end, self._micro_batch_size):
                    end = min(start + self._micro_batch_size, mini_end)
                    self._set_gradient_sync(end == mini_end)
                    backward_kwargs = {"global_tokens": global_tokens}
                    if global_sequences is not None:
                        backward_kwargs["global_sequences"] = global_sequences
                    accumulator.add_micro_batch(
                        self.forward_backward(experience, start, end, **backward_kwargs)
                    )
                accumulator.add_optimizer_step(
                    global_tokens=global_tokens,
                    global_sequences=global_sequences,
                    gradient_norm=self._optimizer_step(),
                )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return accumulator.finalize(
            learning_rate=float(self.optimizer.param_groups[0]["lr"])
        )

    def distillation_metrics(self) -> dict[str, float]:
        """Reduce separate token-normalized OPD diagnostics without changing task metrics."""
        if self.distillation is None or not self.distillation.enabled:
            return {}
        if self._opd_metrics is None:
            raise RuntimeError('Distillation metrics require a completed Actor update')
        return self._opd_metrics.finalize(group=self._dp_group_info.group if self._dp_size > 1 else None,
                                          size=self._dp_size)

    def _require_trainable(self) -> None:
        """Reject optimization calls on an inference-only reference Actor."""
        if self.optimizer is None:
            raise RuntimeError("An inference-only reference Actor cannot be updated")

    def _validate_distillation_fields(self, experience: ExperienceBatch) -> None:
        """Require teacher supervision and the original sampling anchor."""
        if self.distillation is not None and self.distillation.enabled and experience.teacher_log_probs is None:
            raise ValueError('Enabled distillation requires teacher_log_probs before Actor.update')
        if self.distillation is not None and self.distillation.enabled and experience.rollout_log_probs is None:
            raise ValueError("Enabled distillation requires original rollout_log_probs before Actor.update")

    def _validate_experience(self, experience: ExperienceBatch) -> int:
        """Validate token-aligned fields required by policy optimization."""
        if experience.metadata.get("reward_status") == "pending":
            raise ValueError("Pending model rewards cannot be used for Actor optimization")
        self._validate_distillation_fields(experience)
        sequences = experience.sequences
        old_log_probs = experience.old_log_probs
        advantages = experience.advantages
        reference_log_probs = experience.reference_log_probs
        if old_log_probs is None:
            raise ValueError(
                f"Algorithm '{self.algorithm.name}' requires rollout log-probabilities"
            )
        if sequences.ndim != 2:
            raise ValueError(
                f"sequences must have shape [responses, tokens], got {sequences.shape}"
            )
        expected_shape = (sequences.shape[0], sequences.shape[1] - 1)
        for name, tensor in (
            ("action_mask", experience.loss_action_mask),
            ("old_log_probs", old_log_probs),
        ):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} shape mismatch: expected={expected_shape}, "
                    f"got={tuple(tensor.shape)}"
                )
        response_count = int(sequences.shape[0])
        segmented = bool(experience.trajectories) and all(
            "episode_id" in row.metadata or row.metadata.get("dp_padding", False)
            for row in experience.trajectories
        )
        if self._response_mini_batch_size > response_count and not segmented:
            raise ValueError(
                "response_mini_batch_size cannot exceed the local response count: "
                f"mini_batch={self._response_mini_batch_size}, responses={response_count}"
            )
        if advantages is None:
            raise RuntimeError("Training experience must include advantages before Actor.update")
        if tuple(advantages.shape) != expected_shape:
            raise ValueError(
                f"advantages shape mismatch: expected={expected_shape}, "
                f"got={tuple(advantages.shape)}"
            )
        if self.algorithm.requirements.data.reference_log_probs:
            if reference_log_probs is None:
                raise RuntimeError(
                    f"Algorithm '{self.algorithm.name}' requires reference log-probabilities"
                )
            if tuple(reference_log_probs.shape) != expected_shape:
                raise ValueError(
                    "reference_log_probs shape mismatch: "
                    f"expected={expected_shape}, got={tuple(reference_log_probs.shape)}"
                )
        return response_count

    def _validate_update_experience(self, experience: ExperienceBatch) -> int:
        """Make invalid fixed targets visible to every DP rank before backward."""
        local_error = None
        response_count = 0
        try:
            response_count = self._validate_experience(experience)
            mask = experience.loss_action_mask.bool()
            fields = [experience.old_log_probs, experience.advantages]
            if self.distillation is not None and self.distillation.enabled:
                fields.append(experience.teacher_log_probs)
            if self.algorithm.requirements.data.reference_log_probs:
                fields.append(experience.reference_log_probs)
            finite = torch.stack([
                value.detach().masked_fill(~mask, 0.0).isfinite().all() for value in fields
            ]).all() & experience.rewards.isfinite().all()
            if not bool(finite.item()):
                raise ValueError("Actor experience contains non-finite rewards or action targets")
        except (ValueError, RuntimeError) as error:
            local_error = str(error)
        if self._dp_size > 1:
            errors: list[Optional[str]] = [None] * self._dp_size
            dist.all_gather_object(errors, local_error, group=self._dp_group_info.group)
            if any(error is not None for error in errors):
                raise ValueError(f"Actor experience validation failed: {errors}")
        elif local_error is not None:
            raise ValueError(local_error)
        return response_count

    def _global_token_count(self, action_mask: torch.Tensor) -> int:
        """All-reduce the valid action-token count used for loss scaling."""
        count = torch.tensor(
            [float(action_mask.flatten().sum(dim=0).item())],
            dtype=torch.float32,
            device=self._device,
        )
        if self._dp_size > 1:
            dist.all_reduce(count, group=self._dp_group_info.group)
        result = int(count.item())
        if result <= 0:
            raise RuntimeError("Actor update has no valid action tokens on any rank")
        return result

    def _global_batch_counts(self, action_mask: torch.Tensor) -> tuple[int, int]:
        """Reduce token and nonempty-sequence counts over DP, excluding TP copies."""
        mask = action_mask.bool()
        counts = torch.stack((
            mask.flatten().sum(dim=0, dtype=torch.int64),
            mask.any(dim=-1).sum(dim=0, dtype=torch.int64),
        ))
        if self._dp_size > 1:
            dist.all_reduce(counts, group=self._dp_group_info.group)
        tokens, sequences = counts.tolist()
        if tokens <= 0:
            raise RuntimeError("Actor update has no valid action tokens on any rank")
        return tokens, sequences

    def _set_gradient_sync(self, is_last_micro_batch: bool) -> None:
        """Reduce each offloaded micro-batch, otherwise synchronize at the mini-batch boundary."""
        if isinstance(self.actor_model, HSDPModule):
            # No-sync retains full gradients on device even with CPU parameter offload.
            # Reduce each micro-batch so offloaded shards can accumulate on CPU instead.
            offloaded = isinstance(self.actor_model.hsdp_scheduler.offload_policy, CPUOffloadPolicy)
            self.actor_model.set_requires_gradient_sync(is_last_micro_batch or offloaded)
            self.actor_model.set_is_last_backward(is_last_micro_batch)

    def _optimizer_step(self) -> float:
        """Synchronize streams, clip gradients, and update Actor parameters."""
        hsdp_sync_stream()
        main_optimizer = isinstance(self.optimizer, MixedPrecisionOptimizer)
        if main_optimizer:
            self.optimizer.prepare_grads()
        parameters = (
            (parameter for group in self.optimizer.param_groups for parameter in group["params"])
            if main_optimizer else self.parameters()
        )
        norm_limit = self._max_grad_norm if self._max_grad_norm > 0 else float("inf")
        grad_norm = clip_grad_norm_(
            parameters,
            norm_limit,
            error_if_nonfinite=True,
        )
        with SkipDTensorDispatch():
            if main_optimizer:
                self.optimizer.step_with_ready_grads()
            else:
                self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.item())
