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


def _validate_probability_inputs(hidden_states: torch.Tensor, targets: torch.Tensor,
                                 position_mask: torch.Tensor | None, chunk_size: int) -> None:
    """Validate layout and selection contracts before allocating probability buffers."""
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
    _validate_probability_inputs(hidden_states, targets, position_mask, chunk_size)
    labels = targets.reshape(-1)
    if position_mask is None:
        indices = torch.arange(labels.numel(), device=targets.device)
    else:
        indices = position_mask.reshape(-1).nonzero(as_tuple=True)[0]
        # Zero-action DP padding still needs a head/decoder path with zero gradients.
        if indices.numel() == 0:
            indices = torch.zeros(1, dtype=torch.long, device=targets.device)

    def project(states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Gather only this chunk, including during checkpoint recomputation."""
        rows = positions // targets.shape[1]
        columns = positions % targets.shape[1]
        logits = projection(states[rows, columns])
        return logits.float().log_softmax(dim=-1).gather(-1, labels[positions, None]).squeeze(-1)

    chunks = []
    for start in range(0, indices.numel(), chunk_size):
        positions = indices[start:start + chunk_size]
        if torch.is_grad_enabled():
            values = checkpoint(project, hidden_states, positions, use_reentrant=False, preserve_rng_state=True)
        else:
            values = project(hidden_states, positions)
        chunks.append(values)
    values = torch.cat(chunks)
    if position_mask is None:
        return values.reshape(targets.shape)
    values = values * position_mask.reshape(-1)[indices]
    return values.new_zeros(targets.numel()).scatter(0, indices, values).reshape(targets.shape)


class _SelectedLogProbs(torch.autograd.Function):
    """Recompute normalization in backward without saving FP32 vocabulary matrices."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(logits, targets)
        return logits.float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits, targets = ctx.saved_tensors
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            values = logits.float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            gradient, = torch.autograd.grad(values, logits, grad_output, create_graph=create_graph)
        return gradient, None


def chunked_logits_log_probs(logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """Bound fallback FP32 normalization without copying the full vocabulary tensor."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 0 < chunk_size <= 512:
        raise ValueError("token probability chunk_size must be an integer between 1 and 512")
    if targets.numel() == 0:
        return logits[..., 0].float()

    rows = []
    for batch in range(targets.shape[0]):
        chunks = []
        for start in range(0, targets.shape[1], chunk_size):
            chunk = logits[batch, start:start + chunk_size]
            labels = targets[batch, start:start + chunk_size]
            values = _SelectedLogProbs.apply(chunk, labels)
            chunks.append(values)
        rows.append(torch.cat(chunks))
    return torch.stack(rows)
