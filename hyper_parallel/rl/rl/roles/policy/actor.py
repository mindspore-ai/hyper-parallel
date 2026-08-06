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
"""Algorithm-neutral actor execution and optimization."""

from dataclasses import dataclass
from typing import Any, Optional

from rl.algorithm.base import RLAlgorithm
from rl.contracts import ExperienceBatch
from hyper_parallel import HSDPModule, SkipDTensorDispatch, get_platform, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.infer.mixin import GenerateMixin

platform = get_platform()


class ActorModel(GenerateMixin, platform.Module):
    """Expose ``model.generate()`` around a Hyper-Parallel causal LM.

    The wrapped object remains the exact FSDP-parallelized model used for
    training. ``GenerateMixin`` supplies only the public generation method.

    Args:
        module: Parallelized Hyper-Parallel causal language model.
    """

    def __init__(self, module: platform.Module) -> None:
        """Initialize the generation facade around one parallelized model."""
        platform.Module.__init__(self)
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward all arguments to the shared parallelized model."""
        return self.module(*args, **kwargs)

    def sequence_log_probs(
        self,
        sequences: platform.Tensor,
        attention_mask: platform.Tensor,
    ) -> platform.Tensor:
        """Compute chosen-token log-probabilities at every next-token position."""
        if tuple(attention_mask.shape) != tuple(sequences.shape):
            raise ValueError("attention_mask must align with sequences")
        outputs = self(input_ids=sequences, attention_mask=attention_mask, use_cache=False)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        next_token_ids = sequences[:, 1:]
        log_probs = logits[:, :-1, :].float().log_softmax(dim=-1)
        return log_probs.gather(
            dim=-1,
            index=next_token_ids.unsqueeze(-1),
        ).squeeze(-1)

    def response_log_probs(
        self,
        sequences: platform.Tensor,
        prompt_length: int,
        attention_mask: Optional[platform.Tensor] = None,
    ) -> platform.Tensor:
        """Return suffix log-probabilities for one generation engine call."""
        if attention_mask is None:
            attention_mask = sequences.new_ones(
                sequences.shape,
                dtype=platform.tensor_dtype.long,
            )
        return self.sequence_log_probs(sequences, attention_mask)[:, prompt_length - 1 :]

    def set_gradient_sync(self, is_last_micro_batch: bool) -> None:
        """Configure FSDP gradient synchronization for one response micro-batch.

        Args:
            is_last_micro_batch: Whether this is the final backward in the update.
        """
        if isinstance(self.module, HSDPModule):
            self.module.set_requires_gradient_sync(is_last_micro_batch)
            self.module.set_is_last_backward(is_last_micro_batch)


@dataclass(frozen=True)
class UpdateMetrics:
    """Globally reduced scalar diagnostics from one actor update."""

    total_loss: float
    policy_loss: float
    kl_loss: float
    old_policy_kl: float
    old_current_log_ratio_abs: float
    clip_fraction: float
    gradient_norm: float
    learning_rate: float
    valid_tokens: int
    optimizer_steps: int


@dataclass(frozen=True)
class _ActorBatch:
    """Validated tensor views consumed by one actor update."""

    sequences: platform.Tensor
    action_mask: platform.Tensor
    old_log_probs: platform.Tensor
    attention_mask: platform.Tensor
    advantages: platform.Tensor
    reference_log_probs: Optional[platform.Tensor]


@dataclass
class _UpdateAccumulator:
    """Mutable token sums and optimizer diagnostics for one actor update."""

    total_loss: platform.Tensor
    policy_loss: platform.Tensor
    kl_loss: platform.Tensor
    old_policy_kl: platform.Tensor
    log_ratio_abs: platform.Tensor
    clipped_tokens: platform.Tensor
    processed_global_tokens: int = 0
    gradient_norm_sum: float = 0.0
    optimizer_steps: int = 0

    @classmethod
    def create(cls, sequences: platform.Tensor) -> "_UpdateAccumulator":
        """Create zero-valued accumulators on the experience device."""
        zero = sequences.new_zeros((), dtype=platform.tensor_dtype.float32)
        return cls(
            total_loss=zero,
            policy_loss=zero.new_zeros(()),
            kl_loss=zero.new_zeros(()),
            old_policy_kl=zero.new_zeros(()),
            log_ratio_abs=zero.new_zeros(()),
            clipped_tokens=zero.new_zeros(()),
        )


class ActorManager:
    """Apply a math-only algorithm through Hyper-Parallel model capabilities.

    Args:
        actor: Trainable shared actor model.
        algorithm: Complete public algorithm Recipe responsible for actor loss.
        optimizer: Actor optimizer.
        lr_scheduler: Optional actor LR scheduler.
        device: Local accelerator device.
        dp_group_info: Hyper-Parallel data-parallel group descriptor.
        dp_size: Number of FSDP data-parallel ranks.
        micro_batch_size: Number of responses evaluated per model forward.
        response_mini_batch_size: Local responses consumed by one optimizer update.
        policy_update_epochs: Passes over the frozen rollout batch.
        max_grad_norm: Gradient clipping threshold; non-positive disables clipping.
    """

    def __init__(
        self,
        actor: ActorModel,
        algorithm: RLAlgorithm,
        optimizer: Any,
        lr_scheduler: Optional[Any],
        device: Any,
        dp_group_info: Any,
        dp_size: int,
        micro_batch_size: int,
        response_mini_batch_size: int,
        policy_update_epochs: int,
        max_grad_norm: float,
    ) -> None:
        """Initialize the actor update coordinator."""
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
        if response_mini_batch_size <= 0:
            raise ValueError(
                "response_mini_batch_size must be positive, "
                f"got {response_mini_batch_size}"
            )
        if policy_update_epochs <= 0:
            raise ValueError(f"policy_update_epochs must be positive, got {policy_update_epochs}")
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        self.actor = actor
        self.algorithm = algorithm
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._device = device
        self._dp_group_info = dp_group_info
        self._dp_size = dp_size
        self._micro_batch_size = micro_batch_size
        self._response_mini_batch_size = response_mini_batch_size
        self._policy_update_epochs = policy_update_epochs
        self._max_grad_norm = max_grad_norm

    def _global_token_count(self, response_mask: platform.Tensor) -> int:
        """All-reduce the number of valid response tokens."""
        local_count = float(response_mask.sum().item())
        token_tensor = platform.tensor(
            [local_count],
            dtype=platform.tensor_dtype.float32,
            device=self._device,
        )
        if self._dp_size > 1:
            platform.all_reduce(token_tensor, self._dp_group_info)
        global_count = int(token_tensor.item())
        if global_count <= 0:
            raise RuntimeError("Actor update has no valid action tokens on any rank")
        return global_count

    def _reduce_loss_metrics(
        self,
        local_metrics: platform.Tensor,
        global_tokens: int,
    ) -> tuple[float, ...]:
        """Reduce local token sums and convert them to global means."""
        if self._dp_size > 1:
            platform.all_reduce(local_metrics, self._dp_group_info)
        denominator = float(global_tokens)
        return tuple(float(value.item() / denominator) for value in local_metrics)

    def _validate_experience(self, experience: ExperienceBatch) -> _ActorBatch:
        """Validate actor-required tensor fields and return detached rollout data."""
        sequences = experience.sequences
        action_mask = experience.loss_action_mask
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
        for name, tensor in (("action_mask", action_mask), ("old_log_probs", old_log_probs)):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} shape mismatch: expected={expected_shape}, "
                    f"got={tuple(tensor.shape)}"
                )
        response_count = int(sequences.shape[0])
        if self._response_mini_batch_size > response_count:
            raise ValueError(
                "response_mini_batch_size cannot exceed the local response count: "
                f"mini_batch={self._response_mini_batch_size}, responses={response_count}"
            )
        if advantages is None:
            raise RuntimeError(
                "ExperienceBuilder must populate advantages before ActorManager.update"
            )
        if tuple(advantages.shape) != expected_shape:
            raise ValueError(
                f"advantages shape mismatch: expected={expected_shape}, "
                f"got={tuple(advantages.shape)}"
            )
        if self.algorithm.requirements.data.reference_log_probs and reference_log_probs is None:
            raise RuntimeError(
                f"Algorithm '{self.algorithm.name}' requires reference log-probabilities"
            )
        return _ActorBatch(
            sequences=sequences,
            action_mask=action_mask,
            old_log_probs=old_log_probs.detach(),
            attention_mask=experience.attention_mask,
            advantages=advantages,
            reference_log_probs=reference_log_probs,
        )

    def _backward_micro_batch(
        self,
        batch: _ActorBatch,
        start: int,
        end: int,
        global_mini_tokens: int,
        accumulator: _UpdateAccumulator,
    ) -> None:
        """Compute and backpropagate one actor micro-batch loss."""
        current_log_probs = self.actor.sequence_log_probs(
            batch.sequences[start:end],
            batch.attention_mask[start:end],
        )
        old_chunk = batch.old_log_probs[start:end]
        action_mask_chunk = batch.action_mask[start:end]
        output = self.algorithm.compute_actor_loss(
            current_log_probs=current_log_probs,
            old_log_probs=old_chunk,
            reference_log_probs=(
                None
                if batch.reference_log_probs is None
                else batch.reference_log_probs[start:end]
            ),
            advantages=batch.advantages[start:end],
            action_mask=action_mask_chunk,
        )
        scaled_loss = output.total_loss_sum / global_mini_tokens * self._dp_size
        if not bool(scaled_loss.isfinite().item()):
            raise RuntimeError(
                f"Non-finite {self.algorithm.name} loss detected on response slice "
                f"[{start}:{end}]"
            )
        scaled_loss.backward()
        numeric_mask = action_mask_chunk.to(dtype=current_log_probs.dtype)
        accumulator.total_loss = accumulator.total_loss + output.total_loss_sum.detach()
        accumulator.policy_loss = (
            accumulator.policy_loss + output.policy_loss_sum.detach()
        )
        accumulator.kl_loss = (
            accumulator.kl_loss + output.regularization_loss_sum.detach()
        )
        accumulator.old_policy_kl = accumulator.old_policy_kl + output.old_policy_kl_sum
        accumulator.log_ratio_abs = accumulator.log_ratio_abs + (
            (current_log_probs.detach() - old_chunk).abs() * numeric_mask
        ).sum()
        accumulator.clipped_tokens = (
            accumulator.clipped_tokens + output.clipped_token_count
        )

    def _optimizer_step(self, accumulator: _UpdateAccumulator) -> None:
        """Synchronize streams, clip gradients, and execute one optimizer step."""
        hsdp_sync_stream()
        norm_limit = self._max_grad_norm if self._max_grad_norm > 0 else float("inf")
        grad_norm = clip_grad_norm_(
            self.actor.parameters(),
            norm_limit,
            error_if_nonfinite=True,
        )
        accumulator.gradient_norm_sum += float(grad_norm.item())
        with SkipDTensorDispatch():
            self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        accumulator.optimizer_steps += 1

    def _run_update_epochs(
        self,
        batch: _ActorBatch,
        accumulator: _UpdateAccumulator,
    ) -> None:
        """Run configured response mini-batches and gradient micro-batches."""
        response_count = int(batch.sequences.shape[0])
        for _ in range(self._policy_update_epochs):
            for mini_start in range(0, response_count, self._response_mini_batch_size):
                mini_end = min(mini_start + self._response_mini_batch_size, response_count)
                global_tokens = self._global_token_count(
                    batch.action_mask[mini_start:mini_end]
                )
                accumulator.processed_global_tokens += global_tokens
                self._optimizer.zero_grad(set_to_none=True)
                for start in range(mini_start, mini_end, self._micro_batch_size):
                    end = min(start + self._micro_batch_size, mini_end)
                    self.actor.set_gradient_sync(end == mini_end)
                    self._backward_micro_batch(
                        batch,
                        start,
                        end,
                        global_tokens,
                        accumulator,
                    )
                self._optimizer_step(accumulator)

    def _build_update_metrics(self, accumulator: _UpdateAccumulator) -> UpdateMetrics:
        """Reduce token sums and construct the public actor update metrics."""
        reduced = self._reduce_loss_metrics(
            platform.cat(
                tuple(
                    metric.reshape(1)
                    for metric in (
                        accumulator.total_loss,
                        accumulator.policy_loss,
                        accumulator.kl_loss,
                        accumulator.old_policy_kl,
                        accumulator.log_ratio_abs,
                        accumulator.clipped_tokens,
                    )
                ),
                dim=0,
            ),
            accumulator.processed_global_tokens,
        )
        return UpdateMetrics(
            total_loss=reduced[0],
            policy_loss=reduced[1],
            kl_loss=reduced[2],
            old_policy_kl=reduced[3],
            old_current_log_ratio_abs=reduced[4],
            clip_fraction=reduced[5],
            gradient_norm=(
                accumulator.gradient_norm_sum / accumulator.optimizer_steps
            ),
            learning_rate=float(self._optimizer.param_groups[0]["lr"]),
            valid_tokens=accumulator.processed_global_tokens,
            optimizer_steps=accumulator.optimizer_steps,
        )

    def update(
        self,
        experience: ExperienceBatch,
    ) -> UpdateMetrics:
        """Execute old/reference/current log-probs, backward, and optimizer step.

        Args:
            experience: Token-aligned trajectory batch produced by rollout.

        Returns:
            Globally reduced actor-update metrics.
        """
        batch = self._validate_experience(experience)
        accumulator = _UpdateAccumulator.create(batch.sequences)
        self.actor.train()
        self._run_update_epochs(batch, accumulator)
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        return self._build_update_metrics(accumulator)
