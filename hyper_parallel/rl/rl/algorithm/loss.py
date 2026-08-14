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
"""MOLT-style loss and algorithm registries with built-in GRPO/PPO recipes."""
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol
from hyper_parallel import get_platform
from rl.algorithm.advantage import TargetOutput, get_advantage_estimator
from rl.registry import Registry
platform = get_platform()
@dataclass(frozen=True)
class RoleRequirements:
    reference: bool = False
    critic: bool = False
@dataclass(frozen=True)
class DataRequirements:
    rollout_log_probs: bool = True
    reference_log_probs: bool = False
    values: bool = False
    grouped_responses: bool = False
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
class CriticLossOutput:
    """Unreduced value-loss sum returned to backend-owned optimization code."""
    loss_sum: Any
    valid_token_count: Any
class RLAlgorithm(Protocol):
    """Complete public recipe; it never owns models or steps optimizers."""
    name: str
    requirements: AlgorithmRequirements
    def compute_advantages(
        self,
        rewards: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
    ) -> Any:
        """Compute sequence-level advantages when the recipe supports it."""
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
        """Compute an unreduced critic loss when the recipe requires it."""
@dataclass(frozen=True)
class PolicyObjectiveOutput:
    """Per-token objective values and clipping indicators."""
    loss: Any
    clipped: Any
class PolicyObjective(Protocol):
    """Policy-loss component selected by a complete algorithm recipe."""
    def compute(
        self,
        current_log_probs: Any,
        old_log_probs: Any,
        advantages: Any,
    ) -> PolicyObjectiveOutput:
        """Compute per-token policy loss and clipping indicators."""
PolicyLossBuilder = Callable[..., PolicyObjective]
POLICY_LOSSES = Registry[PolicyLossBuilder]("policy loss")
def register_policy_loss(name: str) -> Callable[[PolicyLossBuilder], PolicyLossBuilder]:
    """Register a policy-loss constructor under a stable name."""
    return POLICY_LOSSES.register(name)
def get_policy_loss(name: str, **kwargs: Any) -> PolicyObjective:
    """Instantiate a registered policy loss."""
    return POLICY_LOSSES.build(name, **kwargs)
@register_policy_loss("clipped")
@dataclass(frozen=True)
class ClippedPolicyObjective:
    """Clipped importance-ratio objective with optional dual clipping."""
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    dual_clip: Optional[float] = None
    def compute(
        self,
        current_log_probs: Any,
        old_log_probs: Any,
        advantages: Any,
    ) -> PolicyObjectiveOutput:
        """Compute the clipped importance-ratio policy objective."""
        log_ratio = current_log_probs - old_log_probs
        ratio = log_ratio.exp()
        unclipped_loss = -advantages * ratio
        clipped_ratio = ratio.clamp(
            min=1.0 - self.clip_ratio_low,
            max=1.0 + self.clip_ratio_high,
        )
        policy_loss = unclipped_loss.maximum(-advantages * clipped_ratio)
        if self.dual_clip is not None:
            dual_clip_loss = (-advantages * self.dual_clip).minimum(policy_loss)
            policy_loss = dual_clip_loss.where(advantages < 0, policy_loss)
        clipped = (
            (ratio < 1.0 - self.clip_ratio_low)
            | (ratio > 1.0 + self.clip_ratio_high)
        ).to(dtype=current_log_probs.dtype)
        return PolicyObjectiveOutput(loss=policy_loss, clipped=clipped)
def low_variance_kl(
    current_log_probs: Any,
    target_log_probs: Any,
) -> Any:
    """Non-negative k3 KL estimate used by GRPO and PPO recipes."""
    log_ratio = target_log_probs - current_log_probs
    return (log_ratio.exp() - log_ratio - 1.0).clamp(min=0.0, max=10.0)
@dataclass(frozen=True)
class LowVarianceKLRegularizer:
    """Apply a configurable coefficient to the k3 reference-policy KL."""
    coefficient: float
    def compute(
        self,
        current_log_probs: Any,
        reference_log_probs: Any,
    ) -> tuple[Any, Any]:
        """Compute weighted and raw low-variance KL estimates."""
        raw = low_variance_kl(current_log_probs, reference_log_probs)
        return self.coefficient * raw, raw
def _actor_loss(
    *,
    algorithm_name: str,
    objective: PolicyObjective,
    regularizer: LowVarianceKLRegularizer,
    current_log_probs: Any,
    old_log_probs: Any,
    reference_log_probs: Optional[Any],
    advantages: Any,
    action_mask: Any,
) -> LossOutput:
    """Assemble the shared clipped-policy and reference-KL Actor output."""
    if reference_log_probs is None:
        raise ValueError(
            f"{algorithm_name} requires frozen-reference log-probabilities"
        )
    policy = objective.compute(current_log_probs, old_log_probs, advantages)
    regularization, reference_kl = regularizer.compute(
        current_log_probs, reference_log_probs
    )
    numeric_mask = action_mask.to(dtype=current_log_probs.dtype)
    old_policy_kl = low_variance_kl(current_log_probs, old_log_probs)
    return LossOutput(
        total_loss_sum=((policy.loss + regularization) * numeric_mask).sum(),
        policy_loss_sum=(policy.loss * numeric_mask).sum(),
        regularization_loss_sum=(reference_kl * numeric_mask).sum(),
        valid_token_count=numeric_mask.sum().detach(),
        old_policy_kl_sum=(old_policy_kl * numeric_mask).sum().detach(),
        clipped_token_count=(policy.clipped * numeric_mask).sum().detach(),
    )
AlgorithmBuilder = Callable[[Mapping[str, Any]], RLAlgorithm]
ALGORITHMS = Registry[AlgorithmBuilder]("algorithm")
def register_algorithm(name: str) -> Callable[[AlgorithmBuilder], AlgorithmBuilder]:
    """Register a complete algorithm recipe builder under a stable name."""
    return ALGORITHMS.register(name)
def build_algorithm(config: Mapping[str, Any]) -> RLAlgorithm:
    """Build the complete recipe selected by ``algorithm.name``."""
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("algorithm.name must be a non-empty string")
    return ALGORITHMS.build(name, config)
GRPO_REQUIREMENTS = AlgorithmRequirements(
    roles=RoleRequirements(reference=True, critic=False),
    data=DataRequirements(
        rollout_log_probs=True,
        reference_log_probs=True,
        grouped_responses=True,
    ),
)
PPO_REQUIREMENTS = AlgorithmRequirements(
    roles=RoleRequirements(reference=True, critic=True),
    data=DataRequirements(
        rollout_log_probs=True,
        reference_log_probs=True,
        values=True,
        returns=True,
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
    """Validated hyperparameters for the complete GRPO recipe."""
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
        """Compose the complete GRPO recipe from registered components."""
        self.config = config
        self._advantage_estimator = get_advantage_estimator(
            "grpo",
            epsilon=config.advantage_epsilon,
        )
        self._policy_objective = get_policy_loss(
            "clipped",
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
        """Build token-aligned GRPO advantages through the registered estimator."""
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
        return _actor_loss(
            algorithm_name="GRPO",
            objective=self._policy_objective,
            regularizer=self._regularizer,
            current_log_probs=current_log_probs,
            old_log_probs=old_log_probs,
            reference_log_probs=reference_log_probs,
            advantages=advantages,
            action_mask=action_mask,
        )
    def compute_critic_loss(
        self,
        current_values: Any,
        old_values: Any,
        returns: Any,
        action_mask: Any,
    ) -> CriticLossOutput:
        """Reject critic loss requests because GRPO has no critic role."""
        del current_values, old_values, returns, action_mask
        raise RuntimeError("GRPO does not create or optimize a Critic")
@register_algorithm("grpo")
def build_grpo(config: Mapping[str, Any]) -> GRPOAlgorithm:
    """Build the registered GRPO recipe from user configuration."""
    return GRPOAlgorithm(GRPOConfig.from_mapping(config))
@dataclass(frozen=True)
class PPOConfig:
    """Validated hyperparameters for the complete PPO recipe."""
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
    """PPO recipe composed from registered GAE, clipped objective, and KL."""
    name = "ppo"
    requirements = PPO_REQUIREMENTS
    def __init__(self, config: PPOConfig) -> None:
        """Compose the complete PPO recipe from registered components."""
        self.config = config
        self._advantage_estimator = get_advantage_estimator(
            "gae",
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize=config.normalize_advantages,
            epsilon=config.advantage_epsilon,
        )
        self._policy_objective = get_policy_loss(
            "clipped",
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
        return _actor_loss(
            algorithm_name="PPO",
            objective=self._policy_objective,
            regularizer=self._regularizer,
            current_log_probs=current_log_probs,
            old_log_probs=old_log_probs,
            reference_log_probs=reference_log_probs,
            advantages=advantages,
            action_mask=action_mask,
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
@register_algorithm("ppo")
def build_ppo(config: Mapping[str, Any]) -> PPOAlgorithm:
    """Build the registered PPO recipe from user configuration."""
    return PPOAlgorithm(PPOConfig.from_mapping(config))
