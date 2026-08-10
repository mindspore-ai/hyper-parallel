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
"""Requirements-driven Critic optimization."""

from dataclasses import dataclass
from typing import Any, Optional

from rl.algorithm.base import RLAlgorithm
from rl.contracts import ExperienceBatch
from rl.roles.policy.value import CriticModel
from hyper_parallel import SkipDTensorDispatch, get_platform, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_

platform = get_platform()


@dataclass(frozen=True)
class CriticUpdateMetrics:
    value_loss: float
    gradient_norm: float
    learning_rate: float
    valid_tokens: int
    optimizer_steps: int


class CriticManager:
    """Optimize an independently allocated Critic through a public Recipe."""

    def __init__(
        self,
        critic: CriticModel,
        algorithm: RLAlgorithm,
        optimizer: Any,
        lr_scheduler: Optional[Any],
        device: Any,
        dp_group_info: Any,
        dp_size: int,
        micro_batch_size: int,
        response_mini_batch_size: int,
        update_epochs: int,
        max_grad_norm: float,
    ) -> None:
        """Initialize Critic optimization for a Recipe that requires it."""
        if not algorithm.requirements.roles.critic:
            raise ValueError(f"Algorithm '{algorithm.name}' does not require a Critic")
        self.critic = critic
        self.algorithm = algorithm
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._device = device
        self._dp_group_info = dp_group_info
        self._dp_size = dp_size
        self._micro_batch_size = micro_batch_size
        self._response_mini_batch_size = response_mini_batch_size
        self._update_epochs = update_epochs
        self._max_grad_norm = max_grad_norm

    def _global_token_count(self, mask: Any) -> int:
        """All-reduce and validate the number of trainable Critic tokens."""
        count = platform.tensor(
            [float(mask.sum().item())],
            dtype=platform.tensor_dtype.float32,
            device=self._device,
        )
        if self._dp_size > 1:
            platform.all_reduce(count, self._dp_group_info)
        result = int(count.item())
        if result <= 0:
            raise RuntimeError("Critic update has no valid action tokens")
        return result

    def update(self, experience: ExperienceBatch) -> CriticUpdateMetrics:
        """Run configured Critic mini-batch updates for one experience batch."""
        if experience.values is None or experience.returns is None:
            raise RuntimeError(
                "ExperienceBuilder must populate values and returns before Critic update"
            )
        sequences = experience.sequences
        attention_mask = experience.attention_mask
        action_mask = experience.loss_action_mask
        old_values = experience.values.detach()
        returns = experience.returns.detach()
        response_count = int(sequences.shape[0])
        local_loss = sequences.new_zeros((), dtype=platform.tensor_dtype.float32)
        processed_global_tokens = 0
        optimizer_steps = 0
        gradient_norm_sum = 0.0
        self.critic.train()
        for _ in range(self._update_epochs):
            for mini_start in range(0, response_count, self._response_mini_batch_size):
                mini_end = min(mini_start + self._response_mini_batch_size, response_count)
                global_tokens = self._global_token_count(action_mask[mini_start:mini_end])
                processed_global_tokens += global_tokens
                self._optimizer.zero_grad(set_to_none=True)
                for start in range(mini_start, mini_end, self._micro_batch_size):
                    end = min(start + self._micro_batch_size, mini_end)
                    self.critic.set_gradient_sync(end == mini_end)
                    current_values = self.critic.sequence_values(
                        sequences[start:end], attention_mask[start:end]
                    )
                    output = self.algorithm.compute_critic_loss(
                        current_values=current_values,
                        old_values=old_values[start:end],
                        returns=returns[start:end],
                        action_mask=action_mask[start:end],
                    )
                    scaled_loss = output.loss_sum / global_tokens * self._dp_size
                    if not bool(scaled_loss.isfinite().item()):
                        raise RuntimeError("Non-finite Critic loss detected")
                    scaled_loss.backward()
                    local_loss = local_loss + output.loss_sum.detach()
                hsdp_sync_stream()
                norm_limit = self._max_grad_norm if self._max_grad_norm > 0 else float("inf")
                grad_norm = clip_grad_norm_(
                    self.critic.parameters(), norm_limit, error_if_nonfinite=True
                )
                gradient_norm_sum += float(grad_norm.item())
                with SkipDTensorDispatch():
                    self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        if self._dp_size > 1:
            platform.all_reduce(local_loss, self._dp_group_info)
        return CriticUpdateMetrics(
            value_loss=float(local_loss.item()) / processed_global_tokens,
            gradient_norm=gradient_norm_sum / optimizer_steps,
            learning_rate=float(self._optimizer.param_groups[0]["lr"]),
            valid_tokens=processed_global_tokens,
            optimizer_steps=optimizer_steps,
        )
