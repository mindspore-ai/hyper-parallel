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
"""MOLT-style advantage registry and built-in target estimators."""
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import torch

from rl.registry import Registry

AdvantageEstimatorBuilder = Callable[..., "AdvantageEstimator"]


@dataclass(frozen=True)
class TargetOutput:
    """Token-aligned training targets prepared before optimizer execution."""
    advantages: Any
    returns: Optional[Any] = None


class AdvantageEstimator(Protocol):
    """Target-estimation component selected by a complete algorithm recipe."""

    def estimate(
        self,
        rewards: Any,
        action_mask: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
        values: Optional[Any] = None,
    ) -> TargetOutput:
        """Estimate token-aligned advantages and optional returns."""
ADVANTAGE_ESTIMATORS = Registry[AdvantageEstimatorBuilder]("advantage estimator")


def register_advantage_estimator(
    name: str,
) -> Callable[[AdvantageEstimatorBuilder], AdvantageEstimatorBuilder]:
    """Register an advantage estimator constructor under a stable name."""
    return ADVANTAGE_ESTIMATORS.register(name)


def get_advantage_estimator(name: str, **kwargs: Any) -> AdvantageEstimator:
    """Instantiate a registered advantage estimator."""
    return ADVANTAGE_ESTIMATORS.build(name, **kwargs)


def compute_group_advantages(
    rewards: Any,
    epsilon: float = 1.0e-6,
) -> Any:
    """Normalize rewards with the sample standard deviation of one group."""
    if rewards.shape[-1] < 2:
        raise ValueError("GRPO requires at least two responses in every reward group")
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=True)
    return (rewards - mean) / (std + epsilon)


@register_advantage_estimator("grpo")
@dataclass(frozen=True)
class GroupRelativeAdvantageEstimator:
    """Compute per-group reward normalization and broadcast it to action tokens."""
    epsilon: float = 1.0e-6

    def estimate(
        self,
        rewards: Any,
        action_mask: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
        values: Optional[Any] = None,
    ) -> TargetOutput:
        """Normalize rewards within groups and broadcast them to action tokens."""
        del values
        if group_ids is None:
            sequence_advantages = compute_group_advantages(rewards, self.epsilon)
        else:
            if len(group_ids) != rewards.shape[0]:
                raise ValueError("group_ids must contain one value per reward")
            grouped_indices: dict[Optional[str], list[int]] = {}
            for index, group_id in enumerate(group_ids):
                grouped_indices.setdefault(group_id, []).append(index)
            sequence_advantages = torch.full_like(rewards, 0)
            for indices in grouped_indices.values():
                index_tensor = torch.tensor(
                    indices,
                    dtype=torch.long,
                    device=rewards.device,
                )
                group_rewards = rewards.index_select(0, index_tensor)
                normalized = compute_group_advantages(group_rewards, self.epsilon)
                sequence_advantages.index_copy_(0, index_tensor, normalized)
        token_advantages = sequence_advantages.unsqueeze(-1).expand_as(action_mask)
        return TargetOutput(
            advantages=token_advantages * action_mask.to(token_advantages.dtype),
        )


@register_advantage_estimator("gae")
@dataclass(frozen=True)
class GAEAdvantageEstimator:
    """Generalized advantage estimation over trainable action positions only."""
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize: bool = True
    epsilon: float = 1.0e-6

    def estimate(
        self,
        rewards: Any,
        action_mask: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
        values: Optional[Any] = None,
        bootstrap_values: Optional[Any] = None,
    ) -> TargetOutput:
        """Compute generalized advantages and returns on action positions."""
        del group_ids
        if values is None:
            raise ValueError("GAE requires old critic values")
        if tuple(values.shape) != tuple(action_mask.shape):
            raise ValueError("GAE values must align with next-token action positions")
        value_fp = values.float()
        if bootstrap_values is not None and tuple(bootstrap_values.shape) != (values.shape[0],):
            raise ValueError("GAE bootstrap_values must contain one value per sequence")
        advantages = self._estimate_advantages(rewards, action_mask, value_fp, bootstrap_values)
        numeric_mask = action_mask.bool()
        returns = (advantages + value_fp) * numeric_mask.to(value_fp.dtype)
        if self.normalize:
            selected = advantages[numeric_mask]
            if selected.numel() > 1:
                advantages = advantages.clone()
                advantages[numeric_mask] = (
                    selected - selected.mean()
                ) / (selected.std(unbiased=False) + self.epsilon)
        return TargetOutput(advantages=advantages, returns=returns)

    def _estimate_advantages(
        self, rewards: Any, action_mask: Any, value_fp: Any, bootstrap_values: Optional[Any],
    ) -> Any:
        """Run the action-masked GAE recurrence in reverse token order."""
        advantages = value_fp.new_zeros(value_fp.shape)
        next_value = (value_fp.new_zeros(value_fp.shape[0]) if bootstrap_values is None
                      else bootstrap_values.detach().float())
        running = value_fp.new_zeros(value_fp.shape[0])
        last_action = action_mask.bool().any(dim=1)
        for index in range(value_fp.shape[1] - 1, -1, -1):
            active = action_mask[:, index].bool()
            token_reward = rewards.float() * last_action
            delta = token_reward + self.gamma * next_value - value_fp[:, index]
            candidate = delta + self.gamma * self.gae_lambda * running
            running = candidate.where(active, running)
            next_value = value_fp[:, index].where(active, next_value)
            advantages[:, index] = running * active
            last_action = last_action & ~active
        return advantages
