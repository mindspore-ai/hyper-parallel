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
"""Directly configurable masked cross-entropy callable."""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


@dataclass(kw_only=True, slots=True)
class MaskedCrossEntropy:
    """YAML-targeted masked cross-entropy callable."""

    fp32_upcast: bool = True
    ignore_index: int = -100
    reduction: Literal["none", "mean", "sum"] = "sum"

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the configured loss.

        Args:
            logits: Model logits whose final dimension is the vocabulary.
            labels: Target token IDs.
            mask: Optional mask where zero marks ignored target positions.
            num_label_tokens: Optional normalization denominator.

        Returns:
            Masked cross-entropy loss.

        Raises:
            ValueError: If token normalization is requested for a non-sum loss.
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

        if self.fp32_upcast:
            logits = logits.float()

        loss = F.cross_entropy(
            logits,
            labels,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        if num_label_tokens is not None:
            if self.reduction != "sum":
                raise ValueError(
                    "num_label_tokens is only supported when reduction is 'sum'"
                )
            if num_label_tokens == 0:
                return loss * 0.0
            loss = loss / num_label_tokens

        return loss


__all__ = ["MaskedCrossEntropy"]
