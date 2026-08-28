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
"""Training metric accumulation and public metric-name mapping."""
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from rl.dataset.contracts import ExperienceBatch
from hyper_parallel import get_platform
platform = get_platform()
_GIB = 1024 ** 3


@dataclass(frozen=True)
class ActorMicroBatchMetrics:
    """Detached token sums produced by one Actor forward/backward call."""
    total_loss_sum: Any
    policy_loss_sum: Any
    kl_loss_sum: Any
    old_policy_kl_sum: Any
    log_ratio_abs_sum: Any
    clipped_token_count: Any


@dataclass(frozen=True)
class ActorUpdateMetrics:
    """Globally reduced diagnostics from one Actor update."""
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
class CriticUpdateMetrics:
    """Globally reduced diagnostics from one Critic update."""
    value_loss: float
    gradient_norm: float
    learning_rate: float
    valid_tokens: int
    optimizer_steps: int


class ActorMetricAccumulator:
    """Accumulate detached Actor statistics without controlling optimization."""

    def __init__(self, totals: Any, dp_group_info: Any, dp_size: int) -> None:
        """Initialize detached metric totals and data-parallel reduction metadata."""
        self._totals = totals
        self._dp_group_info = dp_group_info
        self._dp_size = dp_size
        self._global_tokens = 0
        self._gradient_norm_sum = 0.0
        self._optimizer_steps = 0

    @classmethod
    def create(
        cls,
        zero: Any,
        *,
        dp_group_info: Any,
        dp_size: int,
    ) -> "ActorMetricAccumulator":
        """Create six scalar accumulators on the supplied tensor device."""
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        totals = zero.new_zeros((6,), dtype=platform.tensor_dtype.float32)
        return cls(totals, dp_group_info, dp_size)

    def add_micro_batch(self, metrics: ActorMicroBatchMetrics) -> None:
        """Add detached loss and diagnostic sums from one micro-batch."""
        values = (
            metrics.total_loss_sum,
            metrics.policy_loss_sum,
            metrics.kl_loss_sum,
            metrics.old_policy_kl_sum,
            metrics.log_ratio_abs_sum,
            metrics.clipped_token_count,
        )
        self._totals.add_(
            platform.cat(tuple(value.detach().reshape(1) for value in values), dim=0)
        )

    def add_optimizer_step(self, *, global_tokens: int, gradient_norm: float) -> None:
        """Record one optimizer step and its global token denominator."""
        if global_tokens <= 0:
            raise ValueError(f"global_tokens must be positive, got {global_tokens}")
        self._global_tokens += global_tokens
        self._gradient_norm_sum += gradient_norm
        self._optimizer_steps += 1

    def finalize(self, *, learning_rate: float) -> ActorUpdateMetrics:
        """Reduce token sums and return public Actor update metrics."""
        if self._optimizer_steps <= 0 or self._global_tokens <= 0:
            raise RuntimeError("Actor metrics require at least one optimizer step")
        totals = self._totals.clone()
        if self._dp_size > 1:
            platform.all_reduce(totals, self._dp_group_info)
        means = totals / float(self._global_tokens)
        return ActorUpdateMetrics(
            total_loss=float(means[0].item()),
            policy_loss=float(means[1].item()),
            kl_loss=float(means[2].item()),
            old_policy_kl=float(means[3].item()),
            old_current_log_ratio_abs=float(means[4].item()),
            clip_fraction=float(means[5].item()),
            gradient_norm=self._gradient_norm_sum / self._optimizer_steps,
            learning_rate=learning_rate,
            valid_tokens=self._global_tokens,
            optimizer_steps=self._optimizer_steps,
        )


def _system_memory_metrics() -> dict[str, float]:
    """Read peak device memory metrics when the backend exposes them."""
    handle = platform.get_device_handle(platform.device_type())
    allocated_fn = getattr(handle, "max_memory_allocated", None)
    reserved_fn = getattr(handle, "max_memory_reserved", None)
    allocated = float(allocated_fn()) / _GIB if allocated_fn is not None else 0.0
    reserved = float(reserved_fn()) / _GIB if reserved_fn is not None else 0.0
    return {
        "system/world_size": float(platform.get_world_size()),
        "system/max_memory_allocated_gb": allocated,
        "system/max_memory_reserved_gb": reserved,
    }


def build_training_metrics(
    *,
    step: int,
    actor_update: ActorUpdateMetrics,
    rollout_metrics: Mapping[str, float],
    critic_update: Optional[CriticUpdateMetrics] = None,
    policy: Optional[Any] = None,
    diagnostic_metrics: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Combine role, rollout, system, and policy diagnostics for tracking."""
    metrics = {
        "train/global_step": float(step),
        "train/learning_rate": actor_update.learning_rate,
        "train/gradient_norm": actor_update.gradient_norm,
        "train/total_loss": actor_update.total_loss,
        "train/policy_loss": actor_update.policy_loss,
        "train/kl_loss": actor_update.kl_loss,
        "train/old_policy_kl": actor_update.old_policy_kl,
        "train/old_current_log_ratio_abs": actor_update.old_current_log_ratio_abs,
        "train/clip_fraction": actor_update.clip_fraction,
        "train/valid_tokens": float(actor_update.valid_tokens),
        "train/optimizer_steps": float(actor_update.optimizer_steps),
    }
    metrics.update(rollout_metrics)
    if critic_update is not None:
        metrics.update(
            {
                "critic/value_loss": critic_update.value_loss,
                "critic/gradient_norm": critic_update.gradient_norm,
                "critic/learning_rate": critic_update.learning_rate,
                "critic/valid_tokens": float(critic_update.valid_tokens),
                "critic/optimizer_steps": float(critic_update.optimizer_steps),
            }
        )
    if diagnostic_metrics is not None:
        metrics.update(diagnostic_metrics)
    metrics.update(_system_memory_metrics())
    if policy is not None and getattr(policy, "policy_fingerprint", None) is not None:
        metrics["policy/version"] = float(policy.policy_version)
        metrics["policy/fingerprint_changed"] = float(
            bool(policy.policy_fingerprint_changed)
        )
        configured_strategy = getattr(
            policy,
            "weight_sync_configured_strategy",
            None,
        )
        last_strategy = getattr(policy, "weight_sync_last_strategy", None)
        metrics.update(
            {
                "weight_sync/configured_direct_reshard": float(
                    configured_strategy == "direct_reshard"
                ),
                "weight_sync/configured_full_gather": float(
                    configured_strategy == "full_gather"
                ),
                "weight_sync/last_direct_reshard": float(
                    last_strategy == "direct_reshard"
                ),
                "weight_sync/last_full_gather": float(
                    last_strategy == "full_gather"
                ),
                "weight_sync/fallback_count": float(
                    getattr(policy, "weight_sync_fallback_count", 0)
                ),
                "weight_sync/direct_success_count": float(
                    getattr(policy, "weight_sync_direct_success_count", 0)
                ),
            }
        )
    return metrics


def _local_statistics(values: Any) -> tuple[int, float, float, float, float]:
    """Return mergeable count, sum, squared sum, minimum, and maximum."""
    values = values.detach().float()
    count = int(values.numel())
    if count == 0:
        return 0, 0.0, 0.0, math.inf, -math.inf
    return (
        count,
        float(values.sum(dim=0).item()),
        float(values.square().sum(dim=0).item()),
        float(values.min().item()),
        float(values.max().item()),
    )


def _masked_statistics(values: Optional[Any], mask: Any) -> tuple[int, float, float, float, float]:
    """Build local statistics from valid action positions."""
    if values is None:
        return 0, 0.0, 0.0, math.inf, -math.inf
    return _local_statistics(values.detach().float().masked_select(mask))


def _merge_statistics(
    records: list[dict[str, Any]],
    key: str,
) -> Optional[dict[str, float]]:
    """Merge scalar moments gathered from every data-parallel rank."""
    count = sum(int(record[key][0]) for record in records)
    if count == 0:
        return None
    total = sum(float(record[key][1]) for record in records)
    square_total = sum(float(record[key][2]) for record in records)
    variance_numerator = max(square_total - total * total / count, 0.0)
    return {
        "count": float(count),
        "sum": total,
        "square_sum": square_total,
        "mean": total / count,
        "std": math.sqrt(variance_numerator / (count - 1)) if count > 1 else 0.0,
        "min": min(float(record[key][3]) for record in records if record[key][0]),
        "max": max(float(record[key][4]) for record in records if record[key][0]),
    }


def _pearson_correlation(
    left: dict[str, float],
    right: dict[str, float],
    cross_sum: float,
) -> float:
    """Compute Pearson correlation from globally merged scalar moments."""
    count = left["count"]
    covariance = count * cross_sum - left["sum"] * right["sum"]
    left_scale = count * left["square_sum"] - left["sum"] ** 2
    right_scale = count * right["square_sum"] - right["sum"] ** 2
    denominator = math.sqrt(max(left_scale * right_scale, 0.0))
    return covariance / denominator if denominator > 0.0 else float("nan")


def _add_distribution_metrics(
    metrics: dict[str, float],
    prefix: str,
    statistics: Optional[dict[str, float]],
) -> None:
    """Expose a compact mean/std/min/max group when data is available."""
    if statistics is None:
        return
    for name in ("mean", "std", "min", "max"):
        metrics[f"{prefix}_{name}"] = statistics[name]


def _local_training_diagnostics(
    experience: ExperienceBatch,
    actor_log_probs: Any,
) -> dict[str, Any]:
    """Build mergeable pre-update probability and target statistics."""
    rollout_log_probs = experience.old_log_probs
    if rollout_log_probs is None:
        raise ValueError("Log-probability diagnostics require rollout log-probabilities")
    expected_shape = tuple(rollout_log_probs.shape)
    if tuple(actor_log_probs.shape) != expected_shape:
        raise ValueError(
            "Actor and rollout log-probabilities must align: "
            f"actor={tuple(actor_log_probs.shape)}, rollout={expected_shape}"
        )
    mask = experience.loss_action_mask.bool()
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"Action mask must align with log-probabilities: mask={tuple(mask.shape)}, "
            f"log_probs={expected_shape}"
        )
    actor = actor_log_probs.detach().float().masked_select(mask)
    rollout = rollout_log_probs.detach().float().masked_select(mask)
    actor_probabilities = actor.exp()
    rollout_probabilities = rollout.exp()
    return {
        "actor_log": _local_statistics(actor),
        "rollout_log": _local_statistics(rollout),
        "absolute_log_diff": _local_statistics((actor - rollout).abs()),
        "absolute_probability_diff": _local_statistics(
            (actor_probabilities - rollout_probabilities).abs()
        ),
        "actor_probability": _local_statistics(actor_probabilities),
        "rollout_probability": _local_statistics(rollout_probabilities),
        "log_cross_sum": float((actor * rollout).sum(dim=0).item()),
        "probability_cross_sum": float(
            (actor_probabilities * rollout_probabilities).sum(dim=0).item()
        ),
        "advantages": _masked_statistics(experience.advantages, mask),
        "returns": _masked_statistics(experience.returns, mask),
        "values": _masked_statistics(experience.values, mask),
        "return_errors": _masked_statistics(
            None
            if experience.returns is None or experience.values is None
            else experience.returns - experience.values,
            mask,
        ),
        "action_tokens": int(mask.flatten().sum(dim=0).item()),
        "total_tokens": int(experience.attention_mask.flatten().sum(dim=0).item()),
    }


def _gather_diagnostic_records(local: dict[str, Any]) -> list[dict[str, Any]]:
    """Gather one diagnostics record from every data-parallel rank."""
    world_size = platform.get_world_size()
    gathered: list[Optional[dict[str, Any]]] = [None] * world_size
    if world_size == 1:
        gathered[0] = local
    else:
        platform.all_gather_object(gathered, local)
    return [record for record in gathered if record is not None]


def _probability_diagnostic_metrics(
    records: list[dict[str, Any]],
) -> Optional[dict[str, float]]:
    """Merge rollout/Actor probability diagnostics across ranks."""
    actor_log = _merge_statistics(records, "actor_log")
    rollout_log = _merge_statistics(records, "rollout_log")
    log_diff = _merge_statistics(records, "absolute_log_diff")
    probability_diff = _merge_statistics(records, "absolute_probability_diff")
    actor_probability = _merge_statistics(records, "actor_probability")
    rollout_probability = _merge_statistics(records, "rollout_probability")
    statistics = (
        actor_log,
        rollout_log,
        log_diff,
        probability_diff,
        actor_probability,
        rollout_probability,
    )
    if any(item is None for item in statistics):
        return None
    actor_log, rollout_log, log_diff, probability_diff = statistics[:4]
    actor_probability, rollout_probability = statistics[4:]
    log_cross_sum = sum(float(record["log_cross_sum"]) for record in records)
    probability_cross_sum = sum(
        float(record["probability_cross_sum"]) for record in records
    )
    action_tokens = sum(int(record["action_tokens"]) for record in records)
    total_tokens = sum(int(record["total_tokens"]) for record in records)
    return {
        "training/rollout_probs_diff_valid": 1.0,
        "training/rollout_probs_diff_max": probability_diff["max"],
        "training/rollout_probs_diff_mean": probability_diff["mean"],
        "training/rollout_probs_diff_std": probability_diff["std"],
        "training/rollout_actor_probs_pearson_corr": _pearson_correlation(
            actor_probability, rollout_probability, probability_cross_sum
        ),
        "training/rollout_log_probs_diff_max": log_diff["max"],
        "training/rollout_log_probs_diff_mean": log_diff["mean"],
        "training/rollout_log_probs_diff_std": log_diff["std"],
        "training/rollout_actor_log_probs_pearson_corr": _pearson_correlation(
            actor_log, rollout_log, log_cross_sum
        ),
        "training/rollout_log_probs_mean": rollout_log["mean"],
        "training/actor_log_probs_mean": actor_log["mean"],
        "training/rollout_actor_kl": rollout_log["mean"] - actor_log["mean"],
        "training/log_probs_valid_tokens": float(action_tokens),
        "training/total_tokens": float(total_tokens),
        "training/action_token_fraction": action_tokens / max(total_tokens, 1),
    }


def _add_target_diagnostics(
    metrics: dict[str, float],
    records: list[dict[str, Any]],
) -> None:
    """Add advantage, return, value, and explained-variance metrics."""
    advantages = _merge_statistics(records, "advantages")
    returns = _merge_statistics(records, "returns")
    values = _merge_statistics(records, "values")
    return_errors = _merge_statistics(records, "return_errors")
    _add_distribution_metrics(metrics, "train/advantage", advantages)
    _add_distribution_metrics(metrics, "critic/return", returns)
    _add_distribution_metrics(metrics, "critic/value", values)
    if returns is None or return_errors is None:
        return
    return_variance = max(
        returns["square_sum"] / returns["count"] - returns["mean"] ** 2,
        0.0,
    )
    error_variance = max(
        return_errors["square_sum"] / return_errors["count"]
        - return_errors["mean"] ** 2,
        0.0,
    )
    metrics["critic/explained_variance"] = 1.0 - error_variance / (
        return_variance + 1.0e-5
    )


def summarize_training_diagnostics(
    experience: ExperienceBatch,
    actor_log_probs: Any,
) -> dict[str, float]:
    """Compare rollout and pre-update Actor probabilities across DP ranks.

    The supplied Actor log-probabilities must be computed before optimization.
    This function only reads detached tensors and never changes training inputs.
    """
    records = _gather_diagnostic_records(
        _local_training_diagnostics(experience, actor_log_probs)
    )
    if platform.get_rank() != 0:
        return {}
    metrics = _probability_diagnostic_metrics(records)
    if metrics is None:
        return {"training/rollout_probs_diff_valid": 0.0}
    _add_target_diagnostics(metrics, records)
    return metrics


def select_round_robin_samples(
    records: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select globally bounded samples while representing every rank."""
    selected = []
    sample_index = 0
    while len(selected) < limit:
        added = False
        for record in records:
            samples = record["samples"]
            if sample_index < len(samples):
                selected.append(samples[sample_index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        sample_index += 1
    return selected


def _local_rollout_record(
    rollout: ExperienceBatch,
    batch: Mapping[str, Any],
    *,
    step: int,
    sample_limit: int,
) -> dict[str, Any]:
    """Build mergeable rollout statistics and bounded local samples."""
    response_lengths = rollout.action_mask.sum(dim=-1).detach().cpu().tolist()
    rewards = rollout.rewards.detach().cpu().tolist()
    rank = platform.get_rank()
    batch_rows = {
        str(int(sample_index)): row
        for row, sample_index in enumerate(batch["sample_indices"])
    }
    samples = []
    for index, response in enumerate(rollout.responses[:sample_limit]):
        trajectory = rollout.trajectories[index]
        batch_row = batch_rows[trajectory.prompt_id]
        samples.append(
            {
                "step": step,
                "rank": rank,
                "prompt": batch["prompts"][batch_row],
                "response": response,
                "ground_truth": batch["ground_truths"][batch_row],
                "extracted_answer": trajectory.metadata.get("extracted_answer"),
                "reward": float(rewards[index]),
            }
        )
    group_rewards: dict[str, list[float]] = {}
    for trajectory, reward in zip(rollout.trajectories, rewards):
        group_id = trajectory.group_id or trajectory.prompt_id
        group_rewards.setdefault(group_id, []).append(float(reward))
    return {
        "reward_sum": float(sum(rewards)),
        "reward_square_sum": float(sum(reward * reward for reward in rewards)),
        "reward_count": len(rewards),
        "reward_min": float(min(rewards)),
        "reward_max": float(max(rewards)),
        "zero_std_groups": sum(
            int(max(group) == min(group)) for group in group_rewards.values()
        ),
        "length_sum": int(sum(response_lengths)),
        "length_square_sum": int(sum(length * length for length in response_lengths)),
        "length_min": int(min(response_lengths, default=0)),
        "length_max": int(max(response_lengths, default=0)),
        "truncated_count": sum(
            int(trajectory.truncated) for trajectory in rollout.trajectories
        ),
        "generated_tokens": int(rollout.action_mask.flatten().sum(dim=0).item()),
        "generation_seconds": float(rollout.generation_seconds),
        "samples": samples,
    }


def _rollout_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Merge rollout reward, response-length, and throughput statistics."""
    reward_sum = sum(record["reward_sum"] for record in records)
    reward_count = sum(record["reward_count"] for record in records)
    generated_tokens = sum(record["generated_tokens"] for record in records)
    generation_seconds = max(record["generation_seconds"] for record in records)
    length_sum = sum(record["length_sum"] for record in records)
    reward_mean = reward_sum / max(reward_count, 1)
    reward_variance = max(
        sum(record["reward_square_sum"] for record in records)
        / max(reward_count, 1)
        - reward_mean ** 2,
        0.0,
    )
    length_mean = length_sum / max(reward_count, 1)
    length_variance = max(
        sum(record["length_square_sum"] for record in records)
        / max(reward_count, 1)
        - length_mean ** 2,
        0.0,
    )
    return {
        "reward/mean": reward_mean,
        "reward/std": math.sqrt(reward_variance),
        "reward/min": min(record["reward_min"] for record in records),
        "reward/max": max(record["reward_max"] for record in records),
        "reward/accuracy": reward_mean,
        "reward/zero_std_groups": float(
            sum(record["zero_std_groups"] for record in records)
        ),
        "rollout/response_length_mean": length_mean,
        "rollout/response_length_std": math.sqrt(length_variance),
        "rollout/response_length_min": float(
            min(record["length_min"] for record in records)
        ),
        "rollout/response_length_max": float(
            max(record["length_max"] for record in records)
        ),
        "rollout/truncated_ratio": sum(
            record["truncated_count"] for record in records
        )
        / max(reward_count, 1),
        "rollout/sequence_count": float(reward_count),
        "rollout/generated_tokens": float(generated_tokens),
        "rollout/generation_seconds": generation_seconds,
        "rollout/tokens_per_second": generated_tokens / max(generation_seconds, 1.0e-9),
    }


def summarize_rollout(
    rollout: ExperienceBatch,
    batch: Mapping[str, Any],
    *,
    step: int,
    sample_limit: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Gather rollout metrics and bounded samples on rank zero."""
    rank = platform.get_rank()
    local = _local_rollout_record(
        rollout,
        batch,
        step=step,
        sample_limit=sample_limit,
    )
    gathered: list[Optional[dict[str, Any]]] = [None] * platform.get_world_size()
    platform.all_gather_object(gathered, local)
    if rank != 0:
        return {}, []
    records = [record for record in gathered if record is not None]
    return _rollout_metrics(records), select_round_robin_samples(records, sample_limit)


def enforce_learning_gate(
    metrics: Mapping[str, float],
    *,
    step: int,
    config: Mapping[str, Any],
    run_synchronized: Callable[[str, Callable[[], None]], None],
) -> None:
    """Fail numerical acceptance runs when configured invariants are absent."""
    if not bool(config.get("enabled", False)):
        return

    def validate() -> None:
        """Validate rank-zero learning evidence before synchronized publication."""
        if platform.get_rank() != 0:
            return
        failures = []
        gradient_norm = float(metrics["train/gradient_norm"])
        minimum_gradient = float(config.get("min_gradient_norm", 0.0))
        if gradient_norm <= minimum_gradient:
            failures.append(
                f"Learning gate requires gradient_norm > {minimum_gradient}, "
                f"got {gradient_norm}"
            )
        reward_minimum = float(metrics.get("reward/min", 0.0))
        reward_maximum = float(metrics.get("reward/max", 0.0))
        if bool(config.get("require_mixed_rewards", False)) and (
            reward_maximum <= reward_minimum
        ):
            failures.append(
                "Learning gate requires nonzero global reward variance, "
                f"got reward/min={reward_minimum} and reward/max={reward_maximum}"
            )
        if bool(config.get("require_fingerprint_change", False)) and not bool(
            metrics.get("policy/fingerprint_changed", 0.0)
        ):
            failures.append("Learning gate requires a changed replicated norm probe")
        if int(metrics.get("policy/version", -1)) != step:
            failures.append(
                f"Learning gate expected policy version {step}, "
                f"got {metrics.get('policy/version')}"
            )
        if failures:
            raise RuntimeError("; ".join(failures))
    run_synchronized(f"learning gate step {step}", validate)
