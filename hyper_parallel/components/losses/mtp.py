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
"""Multi-Token-Prediction auxiliary loss objective."""

__all__ = ["calculate_mtp_loss", "deepseek_v3_mtp_loss"]

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def calculate_mtp_loss(  # pylint: disable=unused-argument
    mtp_per_depth_logits: list[torch.Tensor],
    mtp_per_depth_h: list[torch.Tensor],
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """Multi-Token-Prediction auxiliary loss.

    Computes CE per depth and sums them.

    Args:
        mtp_per_depth_logits: Per-depth logits from the MTP heads.
        mtp_per_depth_h: Per-depth hidden states. Reserved for future MTP
            variants that condition the loss on hidden states; currently
            unused.
        labels: Target token indices.
        loss_fn: Loss module applied per depth.

    Returns:
        Summed MTP loss over all depths.
    """
    total_mtp_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    for logits in mtp_per_depth_logits:
        logits_shifted = logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()
        depth_loss = loss_fn(
            logits_shifted.view(-1, logits_shifted.size(-1)),
            labels_shifted.view(-1),
        )
        total_mtp_loss = total_mtp_loss + depth_loss
    return total_mtp_loss


def deepseek_v3_mtp_loss(
    per_depth_logits: Sequence[torch.Tensor],
    next_token_labels: torch.Tensor,
    *,
    loss_factor: float = 0.3,
    normalization: Literal["sequence", "valid_tokens"] = "sequence",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute loss_factor / D times the sum of per-depth cross-entropies.

    Args:
        per_depth_logits: Full-vocabulary logits, depth k shaped [B, S-k, V].
        next_token_labels: ALREADY shifted main-LM targets [B, S]. If input_ids
            are [t0, t1, t2, t3], these labels are [t1, t2, t3, t4]; if t4 is
            unavailable, use ignore_index. Depth 1 scores [t2, t3, t4].
        loss_factor: Nonnegative finite weight for the MTP objective.
        normalization: 'sequence' divides each depth's loss sum by B*S, matching
            the original-T denominator in DeepSeek-V3 paper Eq. 24. 'valid_tokens'
            divides by each depth's unmasked target count. The choice changes
            gradient scaling and is explicit; neither is a claim about unpublished
            DeepSeek training code. Empty valid-target sets contribute zero.
        ignore_index: Label excluded from cross-entropy.

    Returns:
        Scalar FP32 weighted MTP loss, connected to every depth's logits.

    Note:
        No DP/CP/global accumulation normalization or vocabulary-sharded CE is
        performed here. Distributed adapters own those reductions. The existing
        calculate_mtp_loss API is intentionally unchanged.
    """
    if not per_depth_logits:
        raise ValueError("per_depth_logits must contain at least one prediction depth")
    if next_token_labels.ndim != 2 or next_token_labels.numel() == 0:
        raise ValueError("next_token_labels must be nonempty [B, S]")
    if next_token_labels.dtype != torch.long:
        raise ValueError("next_token_labels must have dtype torch.long")
    if normalization not in ("sequence", "valid_tokens"):
        raise ValueError("normalization must be 'sequence' or 'valid_tokens'")
    if not math.isfinite(loss_factor) or loss_factor < 0:
        raise ValueError("loss_factor must be finite and nonnegative")
    total = torch.zeros((), device=next_token_labels.device, dtype=torch.float32)
    for depth, logits in enumerate(per_depth_logits, start=1):
        targets = next_token_labels[:, depth:]
        if targets.shape[1] == 0 or logits.ndim != 3 or logits.shape[:2] != targets.shape:
            raise ValueError("Depth k logits must have shape [B, S-k, V] for next_token_labels [B, S]")
        loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                   ignore_index=ignore_index, reduction="sum")
        denominator = (next_token_labels.numel() if normalization == "sequence"
                       else (targets != ignore_index).sum().clamp_min(1))
        total = total + loss_sum / denominator
    return total * (loss_factor / len(per_depth_logits))
