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
"""Selected-block KL with an externally supplied, detached main-attention teacher."""
from __future__ import annotations

from typing import Any, Callable, Protocol

import torch
from torch.autograd.function import once_differentiable

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class PooledIndexerKlFn(Protocol):
    """Scalar KL sum; only index Q/K/weights receive gradients.

    Q [B,S,H,D], pooled K [B,K,D], weights [B,S,H], selected block IDs and
    normalized teacher [B,S,C]. Invalid IDs are -1; teacher rows may be zero.
    Caller owns loss coefficient and valid-query normalization. The CANN path
    requires min(causal complete blocks, capacity) valid IDs per row, followed by
    -1 padding. Official master accepts capacities from 1 to 8192. Keep the
    actual capacity: padding can change the kernel's inferred sparse budget.
    """

    def __call__(self, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
                 indices: torch.Tensor, teacher: torch.Tensor, *, pool_size: int,
                 query_chunk_size: int = 64) -> torch.Tensor:
        """Use score sum_h(weights * relu(QK / sqrt(D))) on selected blocks."""


POOLED_INDEXER_KL = OpSpec[PooledIndexerKlFn](config_key="pooled_indexer_kl")


def validate_inputs(
    query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
    teacher: torch.Tensor, pool_size: int, query_chunk_size: int,
) -> None:
    """Validate the shared external-teacher contract."""
    if query.ndim != 4 or key.ndim != 3 or key.shape[0] != query.shape[0] or key.shape[-1] != query.shape[-1]:
        raise ValueError("pooled KL requires BSHD query and BKD pooled key")
    if min(query.shape) < 1 or key.shape[1] < 1 or weights.shape != query.shape[:3]:
        raise ValueError("pooled KL requires positive dimensions and BSH head weights")
    if indices.ndim != 3 or indices.shape[:2] != query.shape[:2] or teacher.shape != indices.shape:
        raise ValueError("selected blocks and teacher must have matching [B,S,C] shapes")
    if indices.dtype != torch.int32 or indices.shape[-1] < 1:
        raise ValueError("selected block IDs must have nonempty INT32 capacity")
    if any(t.device != query.device for t in (key, weights, indices, teacher)) or key.dtype != query.dtype:
        raise ValueError("pooled KL tensors must share a device and Q/K dtype")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (pool_size, query_chunk_size)):
        raise ValueError("pool and query chunk sizes must be positive integers")


def native_pooled_indexer_kl(
    query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
    teacher: torch.Tensor, *, pool_size: int, query_chunk_size: int = 64,
) -> torch.Tensor:
    """Differentiate only selected scores; unselected blocks have no direct gradient."""
    del pool_size
    batch, length, _, dim = query.shape
    rows = torch.arange(batch, device=query.device)[:, None, None]
    # Accumulate repeated selections in FP32 before casting a pooled-key gradient.
    keys = key.float()
    total = query[..., :0].float().sum() + key[..., :0].float().sum() + weights[..., :0].float().sum()
    for start in range(0, length, query_chunk_size):
        end = min(start + query_chunk_size, length)
        ids = indices[:, start:end]
        valid = ids >= 0
        selected = keys[rows, ids.clamp_min(0).long()]
        dots = torch.einsum("bqhd,bqkd->bqhk", query[:, start:end].float(), selected) * dim ** -0.5
        scores = (dots.relu() * weights[:, start:end].float()[..., None]).sum(2)
        target = teacher[:, start:end].detach().float().masked_fill(~valid, 0)
        active = valid.any(-1) & (target.sum(-1) > 0)
        scores = scores.masked_fill(~valid, float("-inf"))
        scores = torch.where(active[..., None], scores, torch.zeros_like(scores))
        log_prob = scores.log_softmax(-1).masked_fill(~valid, 0)
        total = total + (target * (target.clamp_min(1e-30).log() - log_prob)).sum()
    return total


class _CANNPooledKL(torch.autograd.Function):
    """Retain official KL gradients for the scalar sum's eventual upstream scale."""

    @staticmethod
    def forward(
        ctx: Any, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
        teacher: torch.Tensor, pool_size: int, kernels: tuple[Callable, ...],
    ) -> torch.Tensor:
        """Evaluate the selected-block KL sum and retain unscaled student gradients."""
        grad_op, metadata_op = kernels
        batch, length, heads, dim = query.shape
        opts = {"layout_q": "BSND", "layout_k": "BSND", "mask_mode": 3, "cmp_ratio": pool_size,
                "cmp_residual_k": torch.full((batch,), length % pool_size, device=query.device, dtype=torch.int32)}
        metadata = metadata_op(heads, 1, dim, batch_size=batch, max_seqlen_q=length,
                               max_seqlen_k=key.shape[1], topk=indices.shape[-1], **opts)
        dq, dk, dw, prob = grad_op(query, key[:, :, None], weights, indices[:, :, None],
                                   teacher[:, :, None], metadata=metadata, **opts)
        ctx.save_for_backward(dq, dk.squeeze(2), dw)
        target = teacher[:, :, None]
        # Invalid slots do not contribute. Mask before log/multiply so the scalar
        # does not depend on the probability buffer's padding contents.
        prob = prob.masked_fill(indices[:, :, None] < 0, 1.0)
        return (target * (target.clamp_min(1e-30).log() - prob.clamp_min(1e-30).log())).sum()

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Apply the caller's loss coefficient and reduction to student gradients."""
        dq, dk, dw = ctx.saved_tensors
        return dq * grad_loss, dk * grad_loss, dw * grad_loss, None, None, None, None


def cann_rejection(
    query: torch.Tensor, key: torch.Tensor, indices: torch.Tensor, pool_size: int,
) -> str | None:
    """Select only the official A2/A3 indexer ABI, padding four QSA heads to eight."""
    if query.dtype not in (torch.float16, torch.bfloat16) or query.shape[2] not in (4, 8, 16, 32, 64):
        return "CANN pooled KL requires FP16/BF16 and 4/8/16/32/64 index heads"
    if query.shape[-1] != 128 or key.shape[1] != query.shape[1] // pool_size or pool_size > 128:
        return "CANN pooled KL requires D128 and the complete unpadded pooled key bank"
    if indices.shape[-1] > 8192:
        return "CANN pooled KL supports at most 8192 selected blocks"
    return None


def _load_native():
    return native_pooled_indexer_kl


def _load_cann():
    package = require_package("cann_ops_transformer.ops")
    names = ("sparse_lightning_indexer_kl_loss_grad", "sparse_lightning_indexer_kl_loss_grad_metadata")
    for name in names:
        require_symbol(package, name)
    kernels = tuple(getattr(package, name) for name in names)

    def _operation(query, key, weights, indices, teacher, *, pool_size, query_chunk_size=64):
        del query_chunk_size
        target = teacher.detach().float().masked_fill(indices < 0, 0)
        active = target.sum(-1) > 0
        # The kernel infers valid count from causal visibility and slot capacity,
        # not from -1 values. Keep the route and disable zero-teacher rows through
        # zero Q/weights so they cannot contribute to the shared key gradient.
        ids = indices
        query = torch.where(active[..., None, None], query, 0)
        weights = torch.where(active[..., None], weights, 0)
        scaled = weights.float() * query.shape[-1] ** -0.5
        if query.shape[2] == 4:
            query = torch.cat((query, torch.zeros_like(query)), dim=2)
            scaled = torch.cat((scaled, torch.zeros_like(scaled)), dim=2)
        return _CANNPooledKL.apply(query, key, scaled, ids, target, pool_size, kernels)

    return _operation


def register_pooled_indexer_kl(registry: OpRegistry) -> None:
    """Register a separate teacher contract instead of reusing compressed-main-KV KL."""
    registry.register_family("pooled_indexer_kl", specs=(POOLED_INDEXER_KL,))
    registry.register_impl(POOLED_INDEXER_KL, name="native", devices=("cpu", "cuda", "npu"), load=_load_native)
    registry.register_impl(POOLED_INDEXER_KL, name="cann", devices=("npu",), load=_load_cann)
