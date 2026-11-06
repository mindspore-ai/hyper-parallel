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
"""Conditional-action correction; model execution and sampling remain with their owners."""
import math
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from rl.config.schema import RolloutCorrectionConfig


@dataclass(frozen=True)
class CorrectionAnalysis:
    """Frozen batch analysis before any optimizer step or distributed statistics."""
    log_ratios: Any
    weights: Any
    rejected_groups: tuple[str, ...]


def prepare_rollout_correction(
    experience: Any, anchor: Any, config: RolloutCorrectionConfig, version: int,
) -> tuple[Any, CorrectionAnalysis]:
    """Preserve behavior scores while freezing the current Actor as the PPO anchor.

    The ratio corrects actions conditional on their recorded contexts. It is not
    an unbiased correction of multi-turn state visitation or grouped rewards.
    """
    mask = experience.loss_action_mask.bool()
    behavior = experience.rollout_log_probs
    if behavior is None or tuple(anchor.shape) != tuple(mask.shape):
        raise ValueError("Correction requires sampled probabilities and token-aligned anchor scores")
    behavior = behavior.detach().float().masked_fill(~mask, 0.)
    anchor = anchor.detach().float().masked_fill(~mask, 0.)
    for scores in (behavior, anchor):
        active = scores.masked_select(mask)
        if not bool(active.isfinite().all()) or bool((active > 1.e-5).any()):
            raise ValueError("Correction action probabilities must be finite and nonpositive")
    delta = anchor - behavior
    if not bool(delta.isfinite().all()):
        raise ValueError("Non-finite rollout-to-anchor log ratio")
    # Clip in log space before exp so a valid tiny behavior probability cannot overflow IS weights.
    weights = delta.clamp(max=math.log(config.max_importance_weight)).exp().masked_fill(~mask, 0.)
    rejected = _rejected_groups(experience, delta, mask, config)
    analysis = CorrectionAnalysis(delta, weights, rejected)
    metadata = {**experience.metadata, "off_policy_anchor_version": version}
    if config.mode != "decoupled":
        return replace(experience, metadata=metadata), analysis
    corrected = replace(experience, old_log_probs=anchor, rollout_is_weights=weights,
                        metadata=metadata)
    return corrected, analysis


def _rejected_groups(experience: Any, delta: Any, mask: Any, config: RolloutCorrectionConfig) -> tuple[str, ...]:
    """Collect complete group identities that exceed the configured log-ratio bound."""
    if config.rejection == "disabled":
        return ()
    if len(experience.trajectories) != mask.shape[0]:
        raise ValueError("Group rejection requires complete trajectory identities")
    rejected = set()
    for index, row in enumerate(experience.trajectories):
        if row.metadata.get("dp_padding", False):
            continue
        if row.group_id is None:
            raise ValueError("Group rejection requires GRPO group identities")
        active = delta[index].masked_select(mask[index]).abs()
        if active.numel() and bool((active > config.max_abs_log_ratio).any()):
            rejected.add(str(row.group_id))
    return tuple(sorted(rejected))


def correction_metrics(
    analysis: CorrectionAnalysis, mask: Any, config: RolloutCorrectionConfig,
    *, group: Any = None, world_size: int = 1,
) -> dict[str, float]:
    """Reduce sufficient statistics; histogram quantiles are approximate upper bin edges.

    Call only after all ranks have validated their local analysis. No token arrays
    or private contexts are gathered. ESS counts correlated action tokens.
    """
    delta = analysis.log_ratios.masked_select(mask.bool())
    weights = analysis.weights.masked_select(mask.bool())
    cap = config.max_importance_weight
    stats = weights.new_tensor([weights.numel(), weights.sum(), weights.square().sum(), delta.sum(),
                               (delta > math.log(cap)).sum(), (delta.abs() > 20).sum(),
                               (torch.expm1(delta.clamp(-20, 20)) - delta.clamp(-20, 20)).sum()])
    maxima = weights.new_tensor([weights.max() if weights.numel() else 0.,
                                delta.abs().max() if delta.numel() else 0.])
    histogram = torch.histc(weights.detach().cpu(), bins=128, min=0., max=cap).to(weights.device)
    if world_size > 1:
        dist.all_reduce(stats, group=group)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(histogram, group=group)
    if not bool(stats.isfinite().all()):
        raise ValueError("Non-finite correction statistics")
    count, total, squares, delta_sum, clipped, guarded, divergence = stats.tolist()
    if count <= 0:
        raise ValueError("Correction batch has no global action tokens")
    ess = total * total / squares if squares > 0 else 0.
    metrics = {"action_tokens": count, "log_ratio_mean": delta_sum / count,
               "abs_log_ratio_max": maxima[1].item(), "is_weight_mean": total / count,
               "is_weight_max": maxima[0].item(), "is_weight_clip_fraction": clipped / count,
               "is_ess": ess, "is_ess_fraction": ess / count,
               "k3_clamped_mean": divergence / count, "diagnostic_clamp_fraction": guarded / count}
    cumulative = histogram.cumsum(0).cpu()
    for percentile in (50, 95, 99):
        index = int(torch.searchsorted(cumulative, count * percentile / 100.).clamp(max=127).item())
        metrics[f"is_weight_p{percentile}_approx"] = (index + 1) * cap / 128
    return {f"off_policy/{key}": value for key, value in metrics.items()}
