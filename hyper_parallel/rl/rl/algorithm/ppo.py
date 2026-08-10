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
"""Complete Proximal Policy Optimization public Recipe."""

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
from rl.algorithm.components.advantage import GAEAdvantageEstimator
from rl.algorithm.components.objective import ClippedPolicyObjective
from rl.algorithm.components.regularizer import (
    LowVarianceKLRegularizer,
    low_variance_kl,
)
from rl.algorithm.registry import ALGORITHMS


PPO_REQUIREMENTS = AlgorithmRequirements(
    roles=RoleRequirements(reference=True, critic=True),
    data=DataRequirements(
        rollout_log_probs=True,
        reference_log_probs=True,
        values=True,
        returns=True,
    ),
)


@dataclass(frozen=True)
class PPOConfig:
    """Validated hyperparameters for the complete PPO Recipe."""

    gamma: float = 1.0
    gae_lambda: float = 0.95
    advantage_epsilon: float = 1.0e-6
    normalize_advantages: bool = True
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    kl_coef: float = 0.001

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "PPOConfig":
        """Validate and build a PPO configuration from a mapping."""
        if config.get("loss_aggregation") != "token-mean":
            raise ValueError("PPO requires algorithm.loss_aggregation=token-mean")
        instance = cls(
            gamma=float(config.get("gamma", 1.0)),
            gae_lambda=float(config.get("gae_lambda", 0.95)),
            advantage_epsilon=float(config.get("advantage_epsilon", 1.0e-6)),
            normalize_advantages=bool(config.get("normalize_advantages", True)),
            clip_ratio=float(config.get("clip_ratio", 0.2)),
            value_clip_ratio=float(config.get("value_clip_ratio", 0.2)),
            kl_coef=float(config.get("kl_coef", 0.001)),
        )
        if not 0 <= instance.gamma <= 1 or not 0 <= instance.gae_lambda <= 1:
            raise ValueError("PPO gamma and gae_lambda must be in [0, 1]")
        if instance.clip_ratio < 0 or instance.value_clip_ratio < 0:
            raise ValueError("PPO clip ratios must be non-negative")
        if instance.kl_coef < 0:
            raise ValueError("PPO KL coefficient must be non-negative")
        return instance


class PPOAlgorithm:
    """PPO Recipe composed from internal GAE, clipped objective, and KL."""

    name = "ppo"
    requirements = PPO_REQUIREMENTS

    def __init__(self, config: PPOConfig) -> None:
        """Compose the complete PPO Recipe from internal components."""
        self.config = config
        self._advantage_estimator = GAEAdvantageEstimator(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize=config.normalize_advantages,
            epsilon=config.advantage_epsilon,
        )
        self._policy_objective = ClippedPolicyObjective(
            clip_ratio_low=config.clip_ratio,
            clip_ratio_high=config.clip_ratio,
        )
        self._regularizer = LowVarianceKLRegularizer(config.kl_coef)

    def compute_advantages(
        self,
        rewards: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
    ) -> Any:
        """Reject sequence-only estimation because PPO requires token values."""
        del rewards, group_ids
        raise ValueError("PPO advantages require token values; call build_targets")

    def build_targets(
        self,
        rewards: Any,
        action_mask: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
        values: Optional[Any] = None,
    ) -> TargetOutput:
        """Build PPO generalized advantages and value returns."""
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
            raise ValueError("PPO Recipe requires frozen-reference log-probabilities")
        objective = self._policy_objective.compute(
            current_log_probs, old_log_probs, advantages
        )
        regularization, reference_kl = self._regularizer.compute(
            current_log_probs, reference_log_probs
        )
        numeric_mask = action_mask.to(dtype=current_log_probs.dtype)
        old_policy_kl = low_variance_kl(current_log_probs, old_log_probs)
        return LossOutput(
            total_loss_sum=((objective.loss + regularization) * numeric_mask).sum(),
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
        """Compute a clipped token-value regression loss."""
        clipped_values = old_values + (current_values - old_values).clamp(
            min=-self.config.value_clip_ratio,
            max=self.config.value_clip_ratio,
        )
        current_error = (current_values - returns).square()
        clipped_error = (clipped_values - returns).square()
        loss = 0.5 * current_error.maximum(clipped_error)
        numeric_mask = action_mask.to(dtype=current_values.dtype)
        return CriticLossOutput(
            loss_sum=(loss * numeric_mask).sum(),
            valid_token_count=numeric_mask.sum().detach(),
        )


@ALGORITHMS.register("ppo")
def build_ppo(config: Mapping[str, Any]) -> PPOAlgorithm:
    """Build the registered PPO Recipe from user configuration."""
    return PPOAlgorithm(PPOConfig.from_mapping(config))
