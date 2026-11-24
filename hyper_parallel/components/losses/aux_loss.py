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
"""MoE load balancing and shared auxiliary-loss gradient scaling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F

from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch
from hyper_parallel.core.utils.communication import differentiable_all_reduce


def sequence_load_balancing_loss(
        scores: torch.Tensor,
        selected_experts: torch.Tensor,
        sequence_ids: torch.Tensor,
        num_sequences: int,
        token_mask: torch.Tensor | None = None,
        sequence_partition_groups: tuple[Any, ...] = (),
) -> torch.Tensor:
    """Average ``E * sum_i(fraction_i * mean_probability_i)`` over sequences.

    Args:
        scores: Nonnegative, unbiased scores for all experts, shaped ``[T, E]``.
        selected_experts: Actual dispatch expert IDs after selection/fix-router, shaped ``[T, K]``.
        sequence_ids: Logical sample index of each token, shaped ``[T]``.
        num_sequences: Global number of samples shared by the token-partition ranks.
        token_mask: Optional boolean ``[T]`` mask; true tokens contribute.
        sequence_partition_groups: Independent axes partitioning the same
            sequences' tokens. Never include a data-parallel/expert-dispatch
            group. Probability sums use a differentiable SUM all-reduce;
            callers must account for replicated losses in gradient reduction.

    Returns:
        Unweighted mean over nonempty sequences. Empty or fully masked inputs
        return a differentiable zero, including when a local shard is empty.
    """
    if scores.ndim != 2 or selected_experts.ndim != 2 or scores.shape[0] != selected_experts.shape[0]:
        raise ValueError("scores and selected_experts must have shapes [T, E] and [T, K]")
    num_tokens, num_experts = scores.shape
    if num_sequences <= 0 or sequence_ids.shape != (num_tokens,) or sequence_ids.dtype != torch.long:
        raise ValueError("sequence_ids must be int64 [T] and num_sequences must be positive")
    top_k = selected_experts.shape[1]
    if not 0 < top_k <= num_experts:
        raise ValueError("top_k must be positive and no greater than num_experts")
    if token_mask is not None and (token_mask.shape != (num_tokens,) or token_mask.dtype != torch.bool):
        raise ValueError("token_mask must be a boolean tensor with shape [T]")
    scores = scores.float()
    probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    valid = torch.ones(num_tokens, device=scores.device, dtype=torch.float32)
    if token_mask is not None:
        valid = token_mask.to(dtype=torch.float32)
    indices = sequence_ids.unsqueeze(-1) * num_experts + selected_experts
    counts = scores.new_zeros(num_sequences * num_experts).scatter_add(
        0, indices.reshape(-1), valid.unsqueeze(-1).expand(-1, top_k).reshape(-1),
    ).view(num_sequences, num_experts)
    probability_sum = scores.new_zeros(num_sequences, num_experts).index_add(
        0, sequence_ids, probabilities * valid.unsqueeze(-1),
    )
    for group in sequence_partition_groups:
        dist.all_reduce(counts, group=group)
        probability_sum = differentiable_all_reduce(probability_sum, "sum", group)
    assignments = counts.sum(dim=-1, keepdim=True)
    token_count = (assignments / top_k).clamp_min(1.0)
    losses = num_experts * ((counts / assignments.clamp_min(1.0)) * (probability_sum / token_count)).sum(-1)
    return losses.sum() / (assignments > 0).sum().clamp_min(1)


def token_choice_load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...], num_experts: int, top_k: int, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute the token-choice objective used by Qwen3-MoE chunked loss in PR #1652.

    Pool statistics across router layers and valid tokens. Probabilities retain
    the source logit dtype, and counts and probability sums accumulate in FP32.

    Args:
        router_logits: Per-layer router logits of shape [tokens, num_experts].
        num_experts: Number of routed experts per layer.
        top_k: Number of experts selected per token.
        attention_mask: Optional [batch, sequence] mask with 1 for valid tokens.

    Returns:
        Scalar auxiliary loss before applying the model's coefficient.
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


@dataclass
class _AuxLossScaleState:
    """Per-forward scale, shared with its backward and recomputation."""

    scale: torch.Tensor | float = 1.0
    active: bool = False
    handles: list[Any] = field(default_factory=list)


_AUX_LOSS_STATE: ContextVar[_AuxLossScaleState | None] = ContextVar("aux_loss_state", default=None)


@contextmanager
def aux_loss_scale_context() -> Iterator[None]:
    """Isolate auxiliary gradient scaling for one forward/backward microstep.

    Call :func:`bind_aux_loss_scale` on the local scalar task loss before
    weighting it. Keep this context active through backward so checkpoint
    recomputation observes the same scale. Nested contexts are independent.
    """
    state = _AuxLossScaleState()
    token = _AUX_LOSS_STATE.set(state)
    try:
        yield
    finally:
        for handle in state.handles:
            handle.remove()
        _AUX_LOSS_STATE.reset(token)


def bind_aux_loss_scale(loss: torch.Tensor) -> None:
    """Make injected losses follow the scalar task loss's backward multiplier.

    This is a no-op outside :func:`aux_loss_scale_context` or when no auxiliary
    loss was attached. The hook runs before upstream model backward, including
    checkpoint recomputation, and never changes the task-loss gradient.

    Args:
        loss: Unweighted scalar foundation loss connected to the model output.
    """
    state = _AUX_LOSS_STATE.get()
    if state is None or not state.active or not loss.requires_grad:
        return
    if loss.numel() != 1:
        raise ValueError("Auxiliary gradient scaling requires a scalar task loss")

    def save_scale(gradient: torch.Tensor) -> None:
        """Capture the actual multiplier after token and parallel weighting.

        Args:
            gradient: Upstream scalar loss multiplier, possibly a DTensor.
        """
        with SkipDTensorDispatch():
            local_gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
            state.scale = local_gradient.detach()

    state.handles.append(loss.register_hook(save_scale))


class _AuxLossAutoScaler(torch.autograd.Function):
    """Inject auxiliary-loss gradients without changing the forward loss value."""

    main_loss_backward_scale = torch.tensor(1.0)

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        """Save the auxiliary loss and return the output unchanged.

        Args:
            ctx: Autograd context for the auxiliary loss and scale state.
            output: Forward tensor carrying the injected gradient.
            aux_loss: Scalar objective whose gradient is injected in backward.
        """
        ctx.save_for_backward(aux_loss)
        ctx.scale_state = _AUX_LOSS_STATE.get()
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass through the output gradient and inject the auxiliary-loss gradient.

        Args:
            ctx: Context saved by forward.
            grad_output: Unchanged gradient returned to the forward tensor.
        """
        (aux_loss,) = ctx.saved_tensors
        scale = (
            ctx.scale_state.scale if ctx.scale_state is not None
            else _AuxLossAutoScaler.main_loss_backward_scale
        )
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=aux_loss.device, dtype=aux_loss.dtype)
        aux_loss_grad = torch.ones_like(aux_loss) * scale
        return grad_output, aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """Set the auxiliary-loss gradient scale."""
        _AuxLossAutoScaler.main_loss_backward_scale = scale


def aux_loss_auto_scale(output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
    """Attach an auxiliary-loss gradient to an unchanged forward tensor.

    Args:
        output: Main forward output.
        aux_loss: Scalar auxiliary loss whose gradient should be injected.

    Returns:
        ``output`` unchanged in value, with an autograd edge to ``aux_loss``.
    """
    state = _AUX_LOSS_STATE.get()
    if state is not None:
        state.active = True
    return _AuxLossAutoScaler.apply(output, aux_loss)


def set_aux_loss_scale(scale: torch.Tensor) -> None:
    """Set the gradient multiplier used by :func:`aux_loss_auto_scale`.

    Args:
        scale: Tensor used to scale the injected auxiliary-loss gradient.
    """
    _AuxLossAutoScaler.set_loss_scale(scale)
