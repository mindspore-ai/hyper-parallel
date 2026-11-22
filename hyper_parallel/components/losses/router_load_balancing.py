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
"""Shared token-choice router load-balancing objective."""
from __future__ import annotations

import torch
from torch.nn import functional as F



def token_choice_load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...], num_experts: int, top_k: int, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute the expert-frequency/probability objective for softmax token-choice routing.

    Probabilities retain the source logit dtype. Counts and probability sums
    accumulate in FP32, in layer order, without a layers-by-tokens temporary.
    """
    if not isinstance(router_logits, tuple) or not router_logits:
        raise ValueError("router_logits must contain at least one layer")
    if not 0 < top_k <= num_experts:
        raise ValueError("top_k must be positive and no larger than num_experts")
    device = router_logits[0].device
    assignments = torch.zeros(num_experts, dtype=torch.float32, device=device)
    probability_sum = torch.zeros_like(assignments)
    rows = 0.0
    valid = None
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError("padding mask must have shape [batch, sequence]")
        valid = attention_mask.reshape(-1).to(device=device, dtype=torch.float32)
    for logits in router_logits:
        if logits.ndim != 2 or logits.shape[1] != num_experts:
            raise ValueError("router logits must have shape [tokens, num_experts]")
        if valid is not None and valid.numel() != logits.shape[0]:
            raise ValueError("padding mask must cover every token in each router layer")
        probabilities = F.softmax(logits.to(device), dim=-1)
        selected = probabilities.topk(top_k, dim=-1).indices
        if valid is None:
            counts = torch.bincount(selected.reshape(-1), minlength=num_experts).float()
            probability_sum = probability_sum + probabilities.float().sum(dim=0)
            rows = rows + logits.shape[0]
        else:
            counts = torch.zeros_like(assignments).scatter_add_(0, selected.reshape(-1), valid.repeat_interleave(top_k))
            probability_sum = probability_sum + (probabilities.float() * valid[:, None]).sum(dim=0)
            rows = rows + valid.sum()
        assignments = assignments + counts
    rows = torch.as_tensor(rows, dtype=torch.float32, device=device).clamp_min(1.0)
    return ((assignments / rows) * (probability_sum / rows)[None, :]).sum() * num_experts
