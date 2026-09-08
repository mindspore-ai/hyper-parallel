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

import torch
from torch import nn


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


__all__ = ["calculate_mtp_loss"]
