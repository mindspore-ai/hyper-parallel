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
"""Bounded full-vocabulary probabilities for RL-selected token positions."""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint


def chunked_token_log_probs(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    projection: Callable[[torch.Tensor], torch.Tensor],
    *,
    position_mask: torch.Tensor | None = None,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return differentiable FP32 target log-probabilities without retaining full logits.

    Args:
        hidden_states: Complete decoder output aligned to targets, shaped [B, S, H].
        targets: Next-token int64 labels shaped [B, S].
        projection: Live, full-vocabulary output head; must not introduce collectives per chunk.
        position_mask: Optional boolean [B, S] mask. Unselected positions return zero.
        chunk_size: Maximum projected positions per call, between one and 512.

    Returns:
        FP32 [B, S] probabilities. Context gradients remain connected through hidden states.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 0 < chunk_size <= 512:
        raise ValueError("token probability chunk_size must be an integer between 1 and 512")
    if (hidden_states.ndim != 3 or not hidden_states.is_floating_point() or min(hidden_states.shape) < 1
            or targets.dtype != torch.long or tuple(targets.shape) != tuple(hidden_states.shape[:2])
            or targets.device != hidden_states.device):
        raise ValueError("Expected floating [B,S,H] hidden states and aligned int64 targets on the same device")
    if position_mask is not None and (
            position_mask.dtype != torch.bool or position_mask.shape != targets.shape
            or position_mask.device != targets.device):
        raise ValueError("position_mask must be boolean and aligned with targets")
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    labels = targets.reshape(-1)
    indices = None
    if position_mask is not None:
        indices = position_mask.reshape(-1).nonzero(as_tuple=True)[0]
        # Zero-action DP padding still needs a head/decoder path with zero gradients.
        if indices.numel() == 0:
            indices = torch.zeros(1, dtype=torch.long, device=targets.device)
        hidden = hidden.index_select(0, indices)
        labels = labels.index_select(0, indices)

    def project(chunk: torch.Tensor, chunk_targets: torch.Tensor) -> torch.Tensor:
        """Resolve live head weights during forward and checkpoint recomputation."""
        logits = projection(chunk)
        return logits.float().log_softmax(dim=-1).gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)

    chunks = []
    for start in range(0, labels.numel(), chunk_size):
        hidden_chunk = hidden[start:start + chunk_size]
        target_chunk = labels[start:start + chunk_size]
        # Plain chunking retains every vocabulary matrix until backward; checkpoint each head call.
        if torch.is_grad_enabled():
            values = checkpoint(project, hidden_chunk, target_chunk, use_reentrant=False, preserve_rng_state=True)
        else:
            values = project(hidden_chunk, target_chunk)
        chunks.append(values)
    values = torch.cat(chunks)
    if indices is None:
        return values.reshape(targets.shape)
    values = values * position_mask.reshape(-1).index_select(0, indices)
    return values.new_zeros(targets.numel()).scatter(0, indices, values).reshape(targets.shape)
