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
"""Loss and algorithm registries with built-in GRPO, PPO, and GSPO recipes."""
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol

import torch

from rl.algorithm.advantage import TargetOutput, get_advantage_estimator
from rl.registry import Registry


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
    loss_aggregation: str = "token-mean"

    def __post_init__(self) -> None:
        """Reject aggregation modes the Actor cannot normalize."""
        if self.loss_aggregation not in ("token-mean", "seq-mean-token-mean"):
            raise ValueError(f"Unsupported loss aggregation: {self.loss_aggregation}")


@dataclass(frozen=True)
class LossOutput:
    """Local loss numerators and token diagnostics, normalized by the Actor."""
    total_loss_sum: Any
    policy_loss_sum: Any
    regularization_loss_sum: Any
    valid_token_count: Any
    old_policy_kl_sum: Any
    clipped_token_count: Any
    clipped_sequence_count: Optional[Any] = None


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
        *,
        action_mask: Optional[Any] = None,
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
        *,
        action_mask: Optional[Any] = None,
    ) -> PolicyObjectiveOutput:
        """Compute the clipped importance-ratio policy objective."""
        del action_mask
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


@register_policy_loss("gspo")
@dataclass(frozen=True)
class GSPOPolicyObjective:
    """Sequence importance ratios with token-local first-order gradients."""
    clip_ratio_low: float = 3.0e-4
    clip_ratio_high: float = 4.0e-4

    def compute(
        self,
        current_log_probs: Any,
        old_log_probs: Any,
        advantages: Any,
        *,
        action_mask: Optional[Any] = None,
    ) -> PolicyObjectiveOutput:
        """Compute masked sequence ratios and the asymmetric clipped objective."""
        if action_mask is None or action_mask.ndim != 2:
            raise ValueError("GSPO requires a rank-two action_mask")
        for name, tensor in (
            ("current_log_probs", current_log_probs),
            ("old_log_probs", old_log_probs),
            ("advantages", advantages),
        ):
            if tuple(tensor.shape) != tuple(action_mask.shape):
                raise ValueError(f"GSPO {name} must align with action_mask")
        mask = action_mask.bool()
        current = current_log_probs.float().masked_fill(~mask, 0.0)
        old = old_log_probs.detach().float().masked_fill(~mask, 0.0)
        advantage = advantages.detach().float().masked_fill(~mask, 0.0)
        lengths = mask.sum(dim=-1).clamp_min(1)
        sequence_log_ratio = (current - old).sum(dim=-1) / lengths
        # Sequence averaging in the loss supplies the length factor in the gradient.
        token_log_ratio = current - current.detach() + sequence_log_ratio.detach().unsqueeze(-1)
        ratio = token_log_ratio.clamp(max=10.0).exp()
        unclipped = -advantage * ratio
        clipped = -advantage * ratio.clamp(
            min=1.0 - self.clip_ratio_low,
            max=1.0 + self.clip_ratio_high,
        )
        return PolicyObjectiveOutput(
            loss=unclipped.maximum(clipped).masked_fill(~mask, 0.0),
            clipped=((clipped > unclipped) & mask).float(),
        )


def low_variance_kl(
    current_log_probs: Any,
    target_log_probs: Any,
) -> Any:
    """Non-negative k3 KL estimate used by GRPO and PPO recipes."""
    log_ratio = target_log_probs - current_log_probs
    return (log_ratio.exp() - log_ratio - 1.0).clamp(min=0.0, max=10.0)


def _masked_sum(values: Any, mask: Any) -> Any:
    """Sum values at valid action positions."""
    return values.masked_fill(~mask.bool(), 0.0).flatten().sum(dim=0)


def _loss_sum(values: Any, mask: Any, aggregation: str) -> Any:
    """Build a local loss numerator without dividing by the batch size."""
    if aggregation == "token-mean":
        return _masked_sum(values, mask)
    if aggregation == "seq-mean-token-mean":
        selected = values.masked_fill(~mask.bool(), 0.0)
        lengths = mask.sum(dim=-1).clamp_min(1)
        return (selected.sum(dim=-1) / lengths).sum(dim=0)
    raise ValueError(f"Unsupported loss aggregation: {aggregation}")


def _actor_loss(
    *,
    algorithm_name: str,
    objective: PolicyObjective,
    kl_coefficient: float,
    requirements: AlgorithmRequirements,
    current_log_probs: Any,
    old_log_probs: Any,
    reference_log_probs: Optional[Any],
    advantages: Any,
    action_mask: Any,
) -> LossOutput:
    """Assemble the shared clipped-policy and reference-KL Actor output."""
    if action_mask.ndim != 2:
        raise ValueError("Actor loss requires a rank-two action_mask")
    tensors = {
        "current_log_probs": current_log_probs,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    }
    if requirements.data.reference_log_probs and reference_log_probs is None:
        raise ValueError(
            f"{algorithm_name} requires frozen-reference log-probabilities"
        )
    if requirements.data.reference_log_probs:
        tensors["reference_log_probs"] = reference_log_probs
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != tuple(action_mask.shape):
            raise ValueError(f"{algorithm_name} {name} must align with action_mask")
    mask = action_mask.bool()
    current = current_log_probs.masked_fill(~mask, 0.0)
    old = old_log_probs.detach().masked_fill(~mask, 0.0)
    advantage = advantages.detach().masked_fill(~mask, 0.0)
    policy = objective.compute(current, old, advantage, action_mask=mask)
    reference_kl = torch.zeros_like(policy.loss)
    if requirements.data.reference_log_probs:
        reference = reference_log_probs.detach().masked_fill(~mask, 0.0)
        reference_kl = low_variance_kl(current.float(), reference.float())
    regularization = kl_coefficient * reference_kl
    old_policy_kl = low_variance_kl(current.detach().float(), old.float())
    aggregation = requirements.loss_aggregation
    clipped_sequences = None
    if aggregation == "seq-mean-token-mean":
        clipped_sequences = (policy.clipped.bool() & mask).any(dim=-1).float().sum(dim=0).detach()
    return LossOutput(
        total_loss_sum=_loss_sum(policy.loss + regularization, mask, aggregation),
        policy_loss_sum=_loss_sum(policy.loss, mask, aggregation),
        regularization_loss_sum=_loss_sum(reference_kl, mask, aggregation),
        valid_token_count=mask.float().flatten().sum(dim=0).detach(),
        old_policy_kl_sum=_masked_sum(old_policy_kl, mask).detach(),
        clipped_token_count=_masked_sum(policy.clipped, mask).detach(),
        clipped_sequence_count=clipped_sequences,
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


def _finite_float(value: Any, field: str) -> float:
    """Read a finite numeric setting without accepting boolean coefficients."""
    if isinstance(value, bool):
        raise ValueError(f"algorithm.{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"algorithm.{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"algorithm.{field} must be a finite number")
    return result


def _validate_kl_coefficient(value: Any) -> None:
    """Validate the coefficient controlling Reference ownership."""
    if _finite_float(value, "kl_coef") < 0:
        raise ValueError("algorithm.kl_coef must be non-negative")


@dataclass(frozen=True)
class GRPOConfig:
    """Validated hyperparameters for the complete GRPO recipe."""
    advantage_epsilon: float = 1.0e-6
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_c: float = 3.0
    kl_coef: float = 0.001

    def __post_init__(self) -> None:
        """Validate Reference ownership for direct and mapping construction."""
        _validate_kl_coefficient(self.kl_coef)

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
            kl_coef=_finite_float(config.get("kl_coef", 0.001), "kl_coef"),
        )
        if instance.clip_ratio_low < 0 or instance.clip_ratio_high < 0:
            raise ValueError("GRPO clip ratios must be non-negative")
        if instance.clip_ratio_c <= 1:
            raise ValueError("GRPO dual-clip constant must be greater than one")
        return instance


class GRPOAlgorithm:
    """GRPO math with no optimizer, model, or distributed dependencies."""
    name = "grpo"

    def __init__(self, config: GRPOConfig) -> None:
        """Compose the complete GRPO recipe from registered components."""
        self.config = config
        use_reference = config.kl_coef > 0.0
        self.requirements = AlgorithmRequirements(
            roles=RoleRequirements(reference=use_reference),
            data=DataRequirements(reference_log_probs=use_reference, grouped_responses=True),
        )
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
        self._kl_coefficient = config.kl_coef

    def compute_advantages(
        self,
        rewards: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
    ) -> Any:
        """Compute group-relative sequence advantages."""
        action_mask = rewards.new_ones(
            (rewards.shape[0], 1), dtype=torch.bool
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
            requirements=self.requirements,
            objective=self._policy_objective,
            kl_coefficient=self._kl_coefficient,
            current_log_probs=current_log_probs,
            old_log_probs=old_log_probs,
            reference_log_probs=reference_log_probs,
            advantages=advantages,
            action_mask=action_mask,
        )

    @staticmethod
    def compute_critic_loss(
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

    def __post_init__(self) -> None:
        """Validate Reference ownership independently of the Critic role."""
        _validate_kl_coefficient(self.kl_coef)

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
            kl_coef=_finite_float(config.get("kl_coef", 0.001), "kl_coef"),
        )
        if not 0 <= instance.gamma <= 1 or not 0 <= instance.gae_lambda <= 1:
            raise ValueError("PPO gamma and gae_lambda must be in [0, 1]")
        if instance.clip_ratio < 0 or instance.value_clip_ratio < 0:
            raise ValueError("PPO clip ratios must be non-negative")
        return instance


class PPOAlgorithm:
    """PPO recipe composed from registered GAE, clipped objective, and KL."""
    name = "ppo"

    def __init__(self, config: PPOConfig) -> None:
        """Compose the complete PPO recipe from registered components."""
        self.config = config
        use_reference = config.kl_coef > 0.0
        self.requirements = AlgorithmRequirements(
            roles=RoleRequirements(reference=use_reference, critic=True),
            data=DataRequirements(reference_log_probs=use_reference, values=True, returns=True),
        )
        self._advantage_estimator = get_advantage_estimator(
            "gae",
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize=False,  # The complete DP batch is normalized by ExperiencePreparer.
            epsilon=config.advantage_epsilon,
        )
        self._policy_objective = get_policy_loss(
            "clipped",
            clip_ratio_low=config.clip_ratio,
            clip_ratio_high=config.clip_ratio,
        )
        self._kl_coefficient = config.kl_coef

    @staticmethod
    def compute_advantages(
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
        bootstrap_values: Optional[Any] = None,
    ) -> TargetOutput:
        """Build PPO generalized advantages and value returns."""
        return self._advantage_estimator.estimate(
            rewards, action_mask, group_ids, values, bootstrap_values
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
            requirements=self.requirements,
            objective=self._policy_objective,
            kl_coefficient=self._kl_coefficient,
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
            loss_sum=_masked_sum(loss, numeric_mask),
            valid_token_count=numeric_mask.flatten().sum(dim=0).detach(),
        )


@register_algorithm("ppo")
def build_ppo(config: Mapping[str, Any]) -> PPOAlgorithm:
    """Build the registered PPO recipe from user configuration."""
    return PPOAlgorithm(PPOConfig.from_mapping(config))


@dataclass(frozen=True)
class GSPOConfig:
    """Validated sequence-policy hyperparameters without dual clipping."""
    advantage_epsilon: float = 1.0e-6
    clip_ratio_low: float = 3.0e-4
    clip_ratio_high: float = 4.0e-4
    kl_coef: float = 0.0

    def __post_init__(self) -> None:
        """Validate direct construction as well as parsed configuration."""
        for field in ("advantage_epsilon", "clip_ratio_low", "clip_ratio_high", "kl_coef"):
            _finite_float(getattr(self, field), field)
        if self.advantage_epsilon <= 0:
            raise ValueError("algorithm.advantage_epsilon must be positive")
        if not 0 <= self.clip_ratio_low < 1:
            raise ValueError("algorithm.clip_ratio_low must be in [0, 1)")
        if self.clip_ratio_high < 0:
            raise ValueError("algorithm.clip_ratio_high must be non-negative")
        _validate_kl_coefficient(self.kl_coef)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "GSPOConfig":
        """Reject incompatible recipe fields before allocating any models."""
        allowed = {
            "name", "advantage_epsilon", "clip_ratio_low", "clip_ratio_high",
            "kl_coef", "kl_type", "loss_aggregation",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unsupported GSPO algorithm fields: {sorted(unknown)}")
        if config.get("loss_aggregation") != "seq-mean-token-mean":
            raise ValueError("GSPO requires algorithm.loss_aggregation=seq-mean-token-mean")
        if config.get("kl_type", "low_var_kl") != "low_var_kl":
            raise ValueError("GSPO requires algorithm.kl_type=low_var_kl")
        defaults = cls()
        return cls(**{
            field: _finite_float(config.get(field, getattr(defaults, field)), field)
            for field in ("advantage_epsilon", "clip_ratio_low", "clip_ratio_high", "kl_coef")
        })


class GSPOAlgorithm:
    """Group-relative targets and sequence-level clipped policy optimization."""
    name = "gspo"

    def __init__(self, config: GSPOConfig) -> None:
        """Compose GSPO and declare optional Reference ownership per instance."""
        self.config = config
        use_reference = config.kl_coef > 0.0
        self.requirements = AlgorithmRequirements(
            roles=RoleRequirements(reference=use_reference),
            data=DataRequirements(reference_log_probs=use_reference, grouped_responses=True),
            loss_aggregation="seq-mean-token-mean",
        )
        self._advantage_estimator = get_advantage_estimator("grpo", epsilon=config.advantage_epsilon)
        self._policy_objective = get_policy_loss(
            "gspo", clip_ratio_low=config.clip_ratio_low, clip_ratio_high=config.clip_ratio_high,
        )

    def compute_advantages(
        self, rewards: Any, group_ids: Optional[tuple[Optional[str], ...]] = None,
    ) -> Any:
        """Return one group-relative advantage per trajectory."""
        mask = rewards.new_ones((rewards.shape[0], 1), dtype=torch.bool)
        return self._advantage_estimator.estimate(rewards, mask, group_ids).advantages[:, 0]

    def build_targets(
        self,
        rewards: Any,
        action_mask: Any,
        group_ids: Optional[tuple[Optional[str], ...]] = None,
        values: Optional[Any] = None,
    ) -> TargetOutput:
        """Build fixed group advantages before optimization splits the batch."""
        return self._advantage_estimator.estimate(rewards, action_mask, group_ids, values)

    def compute_actor_loss(
        self,
        current_log_probs: Any,
        old_log_probs: Any,
        reference_log_probs: Optional[Any],
        advantages: Any,
        action_mask: Any,
    ) -> LossOutput:
        """Return sequence-averaged loss sums and token-level diagnostics."""
        return _actor_loss(
            algorithm_name="GSPO",
            objective=self._policy_objective,
            kl_coefficient=self.config.kl_coef,
            requirements=self.requirements,
            current_log_probs=current_log_probs,
            old_log_probs=old_log_probs,
            reference_log_probs=reference_log_probs,
            advantages=advantages,
            action_mask=action_mask,
        )

    def compute_critic_loss(
        self, current_values: Any, old_values: Any, returns: Any, action_mask: Any,
    ) -> CriticLossOutput:
        """Reject Critic requests because GSPO only optimizes the Actor."""
        del current_values, old_values, returns, action_mask
        raise RuntimeError("GSPO does not create or optimize a Critic")


@register_algorithm("gspo")
def build_gspo(config: Mapping[str, Any]) -> GSPOAlgorithm:
    """Build the complete GSPO recipe through the public algorithm registry."""
    return GSPOAlgorithm(GSPOConfig.from_mapping(config))
