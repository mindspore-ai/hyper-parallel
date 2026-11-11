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
"""Native Ascend 910 V4 causal BF16 attention, shared by training and rollout."""

from __future__ import annotations

from functools import lru_cache
import importlib
from importlib.metadata import version
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def _provider() -> ModuleType:
    """Load the optional native operator only after the worker selects its device."""
    installed = version("flash-attn-npu")
    if installed.split("+")[0] != "0.5.0":
        raise ValueError(f"Unified attention requires flash-attn-npu 0.5.0, got {installed}")
    provider = importlib.import_module("flash_attn_npu_4")
    native = importlib.import_module("flash_attn_npu_4.flash_attn_npu_4")
    if not str(native.__file__).endswith(".so"):
        raise ValueError("Unified attention requires the native V4 extension")
    if provider.flash_attn_func.__module__ != "flash_attn_npu_4.flash_attn_npu_interface":
        raise ValueError("Unified attention requires the Ascend 910 V4 public interface")
    return provider


def require_attention_provider() -> None:
    """Fail at initialization when the explicitly selected provider is unavailable."""
    _provider()


def _validate_device(query: torch.Tensor) -> None:
    """Keep accidental CPU/CUDA dispatch out of the numerical contract."""
    if query.device.type not in {"npu", "privateuseone"}:
        raise ValueError("Unified V4 attention requires Ascend NPU tensors")


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    """Check shape metadata without transferring any device values to the host."""
    if any(tensor.ndim not in (3, 4) for tensor in (query, key, value)):
        raise ValueError("Unified attention requires TND or BSND tensors")
    if any(tensor.dtype != torch.bfloat16 or tensor.device != query.device for tensor in (query, key, value)):
        raise ValueError("Unified attention requires BF16 Q/K/V on the same device")
    if key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise ValueError("Unified attention requires matching K/V shapes and equal Q/K/V head dimensions")
    if not 0 < query.shape[-1] <= 256 or key.shape[-2] <= 0 or query.shape[-2] % key.shape[-2]:
        raise ValueError("Unified attention requires head dimension 1..256 and query heads divisible by KV heads")
    _validate_device(query)


def dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    *, scale: float | None = None) -> torch.Tensor:
    """Compute causal BSND attention with native deterministic autograd.

    Args:
        query: BF16 query in BSND layout.
        key: BF16 key in BSND layout, with matching batch and head dimension.
        value: BF16 value with the same shape as key.
        scale: Attention score scale; None uses inverse square root of head dimension.

    Returns:
        BSND attention output with gradients for Q, K and V.
    """
    _validate_qkv(query, key, value)
    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != key.shape[0]:
        raise ValueError("Dense unified attention requires matching BSND batches")
    return _provider().flash_attn_func(
        query, key, value, softmax_scale=scale, causal=True, deterministic=True, num_splits=1,
    )


def _validate_lengths(tensor: torch.Tensor | None, query: torch.Tensor, name: str,
                      *, rank: int = 1) -> None:
    """Validate graph-resident integer metadata without reading its contents."""
    if tensor is not None and (tensor.dtype != torch.int32 or tensor.device != query.device or tensor.ndim != rank):
        raise ValueError(f"{name} must be a rank-{rank} int32 tensor on the query device")


def prepare_scheduler_metadata(
    *, max_seqlen_q: int, max_seqlen_k: int, num_heads: int, num_kv_heads: int, head_dim: int,
    cu_seqlens_q: torch.Tensor, seqused_k: torch.Tensor,
) -> torch.Tensor:
    """Prepare native paged-attention scheduling metadata outside graph capture.

    Args:
        max_seqlen_q: Query length upper bound used by the numerical call.
        max_seqlen_k: KV length upper bound used by the numerical call.
        num_heads: Local query head count.
        num_kv_heads: Local key/value head count.
        head_dim: Common Q/K/V head dimension.
        cu_seqlens_q: Device int32 query offsets for packed queries.
        seqused_k: Device int32 logical KV lengths, one per request.

    Returns:
        Native opaque metadata to copy into the graph's persistent buffer.

    Note:
        Native V4 0.5.0 creates ordinary events while preparing this metadata.
        Call this before capture and before each replay whose lengths change;
        copy the result into a stable buffer passed to varlen_attention.
    """
    _validate_device(cu_seqlens_q)
    _validate_lengths(cu_seqlens_q, cu_seqlens_q, "cu_seqlens_q")
    _validate_lengths(seqused_k, cu_seqlens_q, "seqused_k")
    if cu_seqlens_q.numel() != seqused_k.numel() + 1:
        raise ValueError("Query offsets and KV lengths must describe the same request count")
    return _provider().get_scheduler_metadata(
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        nheads=num_heads, nheads_kv=num_kv_heads, headdim=head_dim, headdim_v=head_dim,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
        causal=True, num_splits=1, window_size_left=-1, window_size_right=-1,
    )


def _validate_kv_layout(query, key, cu_seqlens_q, cu_seqlens_k, seqused_k, page_table) -> None:
    """Validate packed and paged request geometry without reading device lengths."""
    if (query.ndim == 3) != (cu_seqlens_q is not None):
        raise ValueError("TND query requires cu_seqlens_q; BSND query must omit it")
    batch_size = cu_seqlens_q.numel() - 1 if cu_seqlens_q is not None else query.shape[0]
    if seqused_k is not None and seqused_k.numel() != batch_size:
        raise ValueError("seqused_k must contain one length per query request")
    if page_table is not None:
        if key.ndim != 4 or page_table.shape[0] != batch_size or seqused_k is None or cu_seqlens_k is not None:
            raise ValueError("Paged attention requires BSND pages, page table and seqused_k, without cu_seqlens_k")
    elif key.ndim != query.ndim or (key.ndim == 3 and cu_seqlens_k is None):
        raise ValueError("Packed attention requires TND Q/K/V and both cumulative length tensors")
    if cu_seqlens_k is not None and cu_seqlens_k.numel() != batch_size + 1:
        raise ValueError("Query and key cumulative lengths must describe the same request count")


def varlen_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *,
    max_seqlen_q: int, max_seqlen_k: int,
    cu_seqlens_q: torch.Tensor | None = None, cu_seqlens_k: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None, page_table: torch.Tensor | None = None,
    scheduler_metadata: torch.Tensor | None = None, scale: float | None = None,
) -> torch.Tensor:
    """Compute packed or paged causal attention without host length queries.

    Args:
        query: TND packed query when cu_seqlens_q is supplied, otherwise BSND.
        key: TND packed keys, or paged (pages, page_size, KV heads, head dimension).
        value: Same layout and shape as key.
        max_seqlen_q: Explicit query length upper bound, including during graph replay.
        max_seqlen_k: Explicit KV length upper bound, including during graph replay.
        cu_seqlens_q: Device int32 query offsets for packed queries.
        cu_seqlens_k: Device int32 key offsets for packed keys.
        seqused_k: Device int32 logical cache lengths, required for paged KV.
        page_table: Device int32 logical-to-physical page table.
        scheduler_metadata: Optional native scheduler metadata; otherwise generated on device.
        scale: Attention score scale.

    Returns:
        Attention output in the query layout. Paged KV is inference-only;
        packed training uses native deterministic backward without seqused_k.

    Note:
        Callers own metadata contents and buffer updates during graph replay.
        Bounds are Python shape metadata; this adapter never calls item or cpu.
        Graph capture requires explicit scheduler_metadata prepared outside
        capture, since the native V4 metadata builder creates ordinary events.
    """
    _validate_qkv(query, key, value)
    if any(not isinstance(bound, int) or isinstance(bound, bool) or bound < 0
           for bound in (max_seqlen_q, max_seqlen_k)):
        raise ValueError("Unified attention requires explicit nonnegative integer maximum lengths")
    for name, tensor in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k),
                         ("seqused_k", seqused_k)):
        _validate_lengths(tensor, query, name)
    _validate_lengths(page_table, query, "page_table", rank=2)
    _validate_kv_layout(query, key, cu_seqlens_q, cu_seqlens_k, seqused_k, page_table)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value)):
        if page_table is not None or seqused_k is not None:
            raise ValueError("Deterministic V4 backward requires dense or packed attention without paged/seqused KV")
    return _provider().flash_attn_varlen_func(
        query, key, value, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k, page_table=page_table, scheduler_metadata=scheduler_metadata,
        softmax_scale=scale, causal=True, deterministic=True, num_splits=1,
    )
