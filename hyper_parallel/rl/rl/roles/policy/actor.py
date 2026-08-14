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
from typing import Any, Optional
from hyper_parallel import HSDPModule, SkipDTensorDispatch, get_platform, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_
from rl.algorithm.loss import RLAlgorithm
from rl.dataset.contracts import ExperienceBatch
from rl.utils.monitoring.metrics import (
    ActorMetricAccumulator,
    ActorMicroBatchMetrics,
    ActorUpdateMetrics,
)
platform = get_platform()
class Actor(platform.Module):
    """Own one policy model and, when trainable, its optimization runtime."""
    def __init__(
        self,
        actor_model: platform.Module,
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
    ) -> None:
        """Initialize a trainable Actor or an inference-only reference Actor."""
        platform.Module.__init__(self)
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
        self.actor_model = actor_model
        self.algorithm = algorithm
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
        sequences: platform.Tensor,
        attention_mask: platform.Tensor,
    ) -> platform.Tensor:
        """Compute chosen-token log-probabilities at each next-token position."""
        if tuple(attention_mask.shape) != tuple(sequences.shape):
            raise ValueError("attention_mask must align with sequences")
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
    def compute_log_probs(self, experience: ExperienceBatch) -> platform.Tensor:
        """Compute detached policy log-probabilities in response micro-batches."""
        was_training = self.training
        self.eval()
        chunks = []
        try:
            with platform.no_grad():
                for start in range(
                    0,
                    experience.sequences.shape[0],
                    self._micro_batch_size,
                ):
                    end = min(
                        start + self._micro_batch_size,
                        experience.sequences.shape[0],
                    )
                    chunks.append(
                        self.sequence_log_probs(
                            experience.sequences[start:end],
                            experience.attention_mask[start:end],
                        )
                    )
        finally:
            self.train(was_training)
        return platform.cat(chunks, dim=0).detach()
    def forward_backward(
        self,
        experience: ExperienceBatch,
        start: int,
        end: int,
        *,
        global_tokens: int,
    ) -> ActorMicroBatchMetrics:
        """Compute and backpropagate one policy-loss micro-batch."""
        self._require_trainable()
        self._validate_experience(experience)
        response_count = int(experience.sequences.shape[0])
        if start < 0 or end <= start or end > response_count:
            raise ValueError(
                f"Invalid response slice [{start}:{end}] for {response_count} responses"
            )
        if global_tokens <= 0:
            raise ValueError(f"global_tokens must be positive, got {global_tokens}")
        current_log_probs = self.sequence_log_probs(
            experience.sequences[start:end],
            experience.attention_mask[start:end],
        )
        old_log_probs = experience.old_log_probs[start:end].detach()
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
        scaled_loss = output.total_loss_sum / global_tokens * self._dp_size
        if not bool(scaled_loss.isfinite().item()):
            raise RuntimeError(
                f"Non-finite {self.algorithm.name} loss detected on response slice "
                f"[{start}:{end}]"
            )
        scaled_loss.backward()
        numeric_mask = action_mask.to(dtype=current_log_probs.dtype)
        return ActorMicroBatchMetrics(
            total_loss_sum=output.total_loss_sum.detach(),
            policy_loss_sum=output.policy_loss_sum.detach(),
            kl_loss_sum=output.regularization_loss_sum.detach(),
            old_policy_kl_sum=output.old_policy_kl_sum.detach(),
            log_ratio_abs_sum=(
                (current_log_probs.detach() - old_log_probs).abs() * numeric_mask
            ).sum(),
            clipped_token_count=output.clipped_token_count.detach(),
        )
    def update(self, experience: ExperienceBatch) -> ActorUpdateMetrics:
        """Run policy epochs, optimizer steps, and metric finalization."""
        self._require_trainable()
        response_count = self._validate_experience(experience)
        accumulator = ActorMetricAccumulator.create(
            experience.sequences.new_zeros(()),
            dp_group_info=self._dp_group_info,
            dp_size=self._dp_size,
        )
        self.train()
        for _ in range(self._update_epochs):
            for mini_start in range(0, response_count, self._response_mini_batch_size):
                mini_end = min(
                    mini_start + self._response_mini_batch_size,
                    response_count,
                )
                global_tokens = self._global_token_count(
                    experience.loss_action_mask[mini_start:mini_end]
                )
                self.optimizer.zero_grad(set_to_none=True)
                for start in range(mini_start, mini_end, self._micro_batch_size):
                    end = min(start + self._micro_batch_size, mini_end)
                    self._set_gradient_sync(end == mini_end)
                    accumulator.add_micro_batch(
                        self.forward_backward(
                            experience,
                            start,
                            end,
                            global_tokens=global_tokens,
                        )
                    )
                accumulator.add_optimizer_step(
                    global_tokens=global_tokens,
                    gradient_norm=self._optimizer_step(),
                )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return accumulator.finalize(
            learning_rate=float(self.optimizer.param_groups[0]["lr"])
        )
    def _require_trainable(self) -> None:
        """Reject optimization calls on an inference-only reference Actor."""
        if self.optimizer is None:
            raise RuntimeError("An inference-only reference Actor cannot be updated")
    def _validate_experience(self, experience: ExperienceBatch) -> int:
        """Validate token-aligned fields required by policy optimization."""
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
        if self._response_mini_batch_size > response_count:
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
    def _global_token_count(self, action_mask: platform.Tensor) -> int:
        """All-reduce the valid action-token count used for loss scaling."""
        count = platform.tensor(
            [float(action_mask.sum().item())],
            dtype=platform.tensor_dtype.float32,
            device=self._device,
        )
        if self._dp_size > 1:
            platform.all_reduce(count, self._dp_group_info)
        result = int(count.item())
        if result <= 0:
            raise RuntimeError("Actor update has no valid action tokens on any rank")
        return result
    def _set_gradient_sync(self, is_last_micro_batch: bool) -> None:
        """Enable HSDP gradient synchronization for the final micro-batch."""
        if isinstance(self.actor_model, HSDPModule):
            self.actor_model.set_requires_gradient_sync(is_last_micro_batch)
            self.actor_model.set_is_last_backward(is_last_micro_batch)
    def _optimizer_step(self) -> float:
        """Synchronize streams, clip gradients, and update Actor parameters."""
        hsdp_sync_stream()
        norm_limit = self._max_grad_norm if self._max_grad_norm > 0 else float("inf")
        grad_norm = clip_grad_norm_(
            self.parameters(),
            norm_limit,
            error_if_nonfinite=True,
        )
        with SkipDTensorDispatch():
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.item())
