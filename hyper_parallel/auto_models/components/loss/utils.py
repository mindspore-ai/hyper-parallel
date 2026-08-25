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
"""Loss calculation utilities — calculate_loss dispatcher following design doc §10."""

from typing import Any, Optional

import torch
from torch import nn

# FusedLinearCrossEntropy is a stub placeholder — actual implementation
# depends on cut_cross_entropy or similar fused kernel. The dispatcher
# handles it via type check; if missing, all paths fall through to logit-based CE.
try:
    from hyper_parallel.auto_models.components.loss.linear_ce import FusedLinearCrossEntropy
except ImportError:
    FusedLinearCrossEntropy = None


def _get_lm_head_weight(model: nn.Module) -> Optional[torch.Tensor]:
    """Return the model's LM-head weight, materializing DTensor when needed."""
    if hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
        if lm_head is not None and hasattr(lm_head, "weight"):
            weight = lm_head.weight
            return weight.full_tensor() if hasattr(weight, "full_tensor") else weight
    for name, param in model.named_parameters(remove_duplicate=False):
        if "lm_head" in name and name.endswith(".weight"):
            return param.full_tensor() if hasattr(param, "full_tensor") else param
    return None


def calculate_loss(loss_fn: nn.Module, **kwargs: Any) -> torch.Tensor:
    """Unified loss calculation dispatcher.

    Follows design doc §10:
    - Path A: FusedLinearCrossEntropy — uses hidden_states + lm_weight
    - Path B: Standard logit-based CE — uses logits + labels

    Returns raw ce_sum (not divided by N) for token_weighted aggregation.
    Token-mean normalization is done by scale_grads_and_clip_grad_norm.

    Args:
        loss_fn: Loss module (MaskedCrossEntropy, FusedLinearCrossEntropy, etc.)
        **kwargs: Must contain logits+labels (Path B) or hidden_states+lm_weight (Path A).

    Returns:
        Raw loss tensor (reduction="sum", not divided by num_label_tokens).
    """
    num_label_tokens = kwargs.pop("num_label_tokens", None)
    loss_aggregation = kwargs.pop("loss_aggregation", "token_weighted")

    if FusedLinearCrossEntropy is not None and isinstance(loss_fn, FusedLinearCrossEntropy):
        # ── Path A: FusedLinearCrossEntropy ──
        hidden_states = kwargs.get("hidden_states")
        lm_weight = kwargs.get("lm_weight")
        labels = kwargs.get("labels")
        if hidden_states is not None and lm_weight is not None:
            return loss_fn(
                hidden_states=hidden_states,
                labels=labels,
                lm_weight=lm_weight,
            )
        # Fallback: use logits
        logits = kwargs.get("logits")
        if logits is None:
            raise ValueError(
                "FusedLinearCrossEntropy requires hidden_states+lm_weight or logits"
            )
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        return loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

    # ── Path B: Standard logit-based loss ──
    logits = kwargs["logits"]
    labels = kwargs["labels"]

    # Shift: causal LM always needs shift
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    if loss_aggregation == "token_weighted":
        # Return raw ce_sum; token-mean normalization by scale_grads
        return loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
    # rank_average: mean-scale loss
    return loss_fn(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )


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
    for depth_idx, logits in enumerate(mtp_per_depth_logits):
        logits_shifted = logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()
        depth_loss = loss_fn(
            logits_shifted.view(-1, logits_shifted.size(-1)),
            labels_shifted.view(-1),
        )
        total_mtp_loss = total_mtp_loss + depth_loss
    return total_mtp_loss
