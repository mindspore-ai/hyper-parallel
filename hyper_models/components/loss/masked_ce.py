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
"""MaskedCrossEntropy — fp32 recast masked CE loss, following design doc §10.2."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropy(nn.Module):
    """Cross-entropy loss with masked (ignored) target positions and optional fp32 upcast.

    Designed to match the design doc §10.2: recasts bf16 logits to fp32 before
    CE computation to avoid precision loss from large-vocabulary accumulations.
    """

    def __init__(self, fp32_upcast: bool = True, ignore_index: int = -100, reduction: str = "sum"):
        super().__init__()
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss.

        Args:
            logits: [batch_size, seq_len, vocab_size] or [N, V] pre-flattened.
            labels: [batch_size, seq_len] target indices.
            mask: Optional mask tensor (1 = keep, 0 = ignore).
            num_label_tokens: If provided, divide loss by this (for token-mean normalization).

        Returns:
            Scalar loss tensor.
        """
        if labels.device != logits.device:
            labels = labels.to(logits.device)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)
                labels.masked_fill_(mask.view(-1) == 0, self.ignore_index)
                del mask

        if self.fp32_upcast:
            logits = logits.float()

        loss = F.cross_entropy(logits, labels, reduction=self.reduction, ignore_index=self.ignore_index)

        if num_label_tokens is not None:
            if self.reduction != "sum":
                raise ValueError("num_label_tokens is only supported when reduction is 'sum'")
            if num_label_tokens == 0:
                return loss * 0.0
            loss = loss / num_label_tokens

        return loss