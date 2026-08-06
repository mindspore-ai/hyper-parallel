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
"""Public algorithm Recipe contracts.

Users register complete :class:`RLAlgorithm` recipes.  Reusable estimators,
objectives, and regularizers live under ``algorithms.components`` and remain
implementation details of those recipes.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class RoleRequirements:
    actor: bool = True
    rollout: bool = True
    reference: bool = False
    critic: bool = False
    reward_model: bool = False
    environment: bool = False


@dataclass(frozen=True)
class DataRequirements:
    rollout_log_probs: bool = True
    reference_log_probs: bool = False
    values: bool = False
    grouped_responses: bool = False
    action_mask: bool = True
    returns: bool = False


@dataclass(frozen=True)
class AlgorithmRequirements:
    roles: RoleRequirements
    data: DataRequirements


@dataclass(frozen=True)
class LossOutput:
    """Unreduced token sums returned to backend-owned optimization code."""

    total_loss_sum: Any
    policy_loss_sum: Any
    regularization_loss_sum: Any
    valid_token_count: Any
    old_policy_kl_sum: Any
    clipped_token_count: Any


@dataclass(frozen=True)
class TargetOutput:
    """Token-aligned training targets prepared before optimizer execution."""

    advantages: Any
    returns: Optional[Any] = None


@dataclass(frozen=True)
class CriticLossOutput:
    """Unreduced value-loss sum returned to backend-owned optimization code."""

    loss_sum: Any
    valid_token_count: Any


class RLAlgorithm(Protocol):
    """Complete public Recipe; it never owns models or steps optimizers."""

    name: str
    requirements: AlgorithmRequirements

    def compute_advantages(
        self,
        rewards: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
    ) -> Any:
        """Compute sequence-level advantages when the Recipe supports it."""

    def build_targets(
        self,
        rewards: Any,
        action_mask: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
        values: Optional[Any] = None,
    ) -> TargetOutput:
        """Build token-aligned advantages and optional returns."""

    def compute_actor_loss(
        self,
        current_log_probs: Any,
        old_log_probs: Any,
        reference_log_probs: Optional[Any],
        advantages: Any,
        action_mask: Any,
    ) -> LossOutput:
        """Compute unreduced actor loss terms for valid action tokens."""

    def compute_critic_loss(
        self,
        current_values: Any,
        old_values: Any,
        returns: Any,
        action_mask: Any,
    ) -> CriticLossOutput:
        """Compute an unreduced Critic loss when the Recipe requires it."""
