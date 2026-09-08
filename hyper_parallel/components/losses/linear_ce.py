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
"""FusedLinearCrossEntropy — stub for fused lm_head + CE loss.

Reference: nemo_automodel.components.loss.linear_ce
Full implementation requires cut_cross_entropy package.
"""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class FusedLinearCrossEntropy(nn.Module):
    """Fused linear + cross-entropy loss.

    Stub implementation: falls back to separate linear + CE.
    The production version fuses lm_head matmul with CE computation
    for memory efficiency (avoids materializing the full logits tensor).
    """

    def __init__(self, fp32_upcast: bool = True, ignore_index: int = -100) -> None:
        """Initialize the fused linear + cross-entropy loss module.

        Args:
            fp32_upcast: Whether to upcast inputs to float32 before the
                matmul / CE computation for numerical stability.
            ignore_index: Label index ignored by the cross-entropy loss.
        """
        super().__init__()
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        lm_weight: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute fused linear + cross-entropy loss.

        Args:
            hidden_states: [B, S, H] final hidden states (when ``lm_weight``
                is provided), or pre-computed logits [N, V] (when
                ``lm_weight=None``, i.e. the logits fallback path of the
                ``calculate_loss`` dispatcher).
            labels: [B, S] or [N] target indices.
            lm_weight: [V, H] LM head weight; when None, ``hidden_states`` is
                treated as pre-computed logits.
            num_label_tokens: Optional token count for normalization.

        Returns:
            Scalar loss (reduction="sum").
        """
        if lm_weight is not None:
            logits = torch.matmul(hidden_states.float() if self.fp32_upcast else hidden_states, lm_weight.t())
        else:
            # dispatcher fallback path: hidden_states are pre-computed logits
            logits = hidden_states.float() if self.fp32_upcast else hidden_states
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = F.cross_entropy(logits, labels, reduction="sum", ignore_index=self.ignore_index)
        if num_label_tokens is not None:
            if num_label_tokens > 0:
                loss = loss / num_label_tokens
        return loss
