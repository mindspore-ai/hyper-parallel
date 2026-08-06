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
"""Complete Group Relative Policy Optimization public Recipe."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from rl.algorithm.base import (
    AlgorithmRequirements,
    CriticLossOutput,
    DataRequirements,
    LossOutput,
    RoleRequirements,
    TargetOutput,
)
from rl.algorithm.components.advantage import (
    GroupRelativeAdvantageEstimator,
    compute_group_advantages,
)
from rl.algorithm.components.objective import ClippedPolicyObjective
from rl.algorithm.components.regularizer import (
    LowVarianceKLRegularizer,
    low_variance_kl,
)
from rl.algorithm.registry import ALGORITHMS
from hyper_parallel import get_platform

platform = get_platform()

__all__ = [
    "GRPOAlgorithm",
    "GRPOConfig",
    "GRPO_REQUIREMENTS",
    "build_grpo",
    "compute_group_advantages",
    "masked_mean",
]


GRPO_REQUIREMENTS = AlgorithmRequirements(
    roles=RoleRequirements(reference=True, critic=False),
    data=DataRequirements(
        rollout_log_probs=True,
        reference_log_probs=True,
        grouped_responses=True,
    ),
)


def masked_mean(values: Any, mask: Any) -> Any:
    """Return the mean over elements selected by a non-empty mask."""
    numeric_mask = mask.to(dtype=values.dtype)
    count = numeric_mask.sum()
    if count.item() <= 0:
        raise ValueError("masked_mean requires at least one valid element")
    return (values * numeric_mask).sum() / count


@dataclass(frozen=True)
class GRPOConfig:
    """Validated hyperparameters for the complete GRPO Recipe."""

    advantage_epsilon: float = 1.0e-6
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_c: float = 3.0
    kl_coef: float = 0.001

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "GRPOConfig":
        """Validate and build a GRPO configuration from a mapping."""
        if config.get("loss_aggregation") != "token-mean":
            raise ValueError("GRPO requires algorithm.loss_aggregation=token-mean")
        instance = cls(
            advantage_epsilon=float(config.get("advantage_epsilon", 1.0e-6)),
            clip_ratio_low=float(config.get("clip_ratio_low", 0.2)),
            clip_ratio_high=float(config.get("clip_ratio_high", 0.2)),
            clip_ratio_c=float(config.get("clip_ratio_c", 3.0)),
            kl_coef=float(config.get("kl_coef", 0.001)),
        )
        if instance.clip_ratio_low < 0 or instance.clip_ratio_high < 0:
            raise ValueError("GRPO clip ratios must be non-negative")
        if instance.clip_ratio_c <= 1:
            raise ValueError("GRPO dual-clip constant must be greater than one")
        if instance.kl_coef < 0:
            raise ValueError("GRPO KL coefficient must be non-negative")
        return instance


class GRPOAlgorithm:
    """GRPO math with no optimizer, model, or distributed dependencies."""

    name = "grpo"
    requirements = GRPO_REQUIREMENTS

    def __init__(self, config: GRPOConfig) -> None:
        """Compose the complete GRPO Recipe from internal components."""
        self.config = config
        self._advantage_estimator = GroupRelativeAdvantageEstimator(
            epsilon=config.advantage_epsilon
        )
        self._policy_objective = ClippedPolicyObjective(
            clip_ratio_low=config.clip_ratio_low,
            clip_ratio_high=config.clip_ratio_high,
            dual_clip=config.clip_ratio_c,
        )
        self._regularizer = LowVarianceKLRegularizer(config.kl_coef)

    def compute_advantages(
        self,
        rewards: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
    ) -> Any:
        """Compute group-relative sequence advantages."""
        action_mask = rewards.new_ones(
            (rewards.shape[0], 1), dtype=platform.tensor_dtype.bool
        )
        return self._advantage_estimator.estimate(
            rewards, action_mask, group_ids
        ).advantages[:, 0]

    def build_targets(
        self,
        rewards: Any,
        action_mask: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
        values: Optional[Any] = None,
    ) -> TargetOutput:
        """Build token-aligned GRPO advantages through the internal estimator."""
        return self._advantage_estimator.estimate(
            rewards, action_mask, group_ids, values
        )

    def compute_actor_loss(
        self,
        current_log_probs: Any,
        old_log_probs: Any,
        reference_log_probs: Optional[Any],
        advantages: Any,
        action_mask: Any,
    ) -> LossOutput:
        """Compute clipped policy and reference-KL loss sums."""
        if reference_log_probs is None:
            raise ValueError("GRPO requires frozen-reference log-probabilities")
        objective = self._policy_objective.compute(
            current_log_probs, old_log_probs, advantages
        )
        regularization, reference_kl = self._regularizer.compute(
            current_log_probs, reference_log_probs
        )
        total_loss = objective.loss + regularization
        numeric_mask = action_mask.to(dtype=current_log_probs.dtype)
        old_policy_kl = low_variance_kl(current_log_probs, old_log_probs)
        return LossOutput(
            total_loss_sum=(total_loss * numeric_mask).sum(),
            policy_loss_sum=(objective.loss * numeric_mask).sum(),
            regularization_loss_sum=(reference_kl * numeric_mask).sum(),
            valid_token_count=numeric_mask.sum().detach(),
            old_policy_kl_sum=(old_policy_kl * numeric_mask).sum().detach(),
            clipped_token_count=(objective.clipped * numeric_mask).sum().detach(),
        )

    def compute_critic_loss(
        self,
        current_values: Any,
        old_values: Any,
        returns: Any,
        action_mask: Any,
    ) -> CriticLossOutput:
        """Reject Critic loss requests because GRPO has no Critic role."""
        del current_values, old_values, returns, action_mask
        raise RuntimeError("GRPO does not create or optimize a Critic")


@ALGORITHMS.register("grpo")
def build_grpo(config: Mapping[str, Any]) -> GRPOAlgorithm:
    """Build the registered GRPO Recipe from user configuration."""
    return GRPOAlgorithm(GRPOConfig.from_mapping(config))
