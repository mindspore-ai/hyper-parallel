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
"""CANN-specific Torch adapters for DeepSeek DSA.

These local-tensor boundaries support BSND/TND causal training. They do not
infer distributed layouts; CP hooks must finish communication before calling
them. Kernel outputs are raw sums/gradients; normalization belongs to the caller.
"""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from cann_ops_transformer import (
    dense_lightning_indexer_kl_loss_grad,
    dense_lightning_indexer_kl_loss_grad_metadata,
    dense_lightning_indexer_softmax_lse,
    dense_lightning_indexer_softmax_lse_metadata,
    lightning_indexer,
    sparse_flash_attention,
    sparse_flash_attention_grad,
    sparse_lightning_indexer_kl_loss_grad,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)

def _cumulative_lengths(lengths: torch.Tensor | None, total: int, device: torch.device) -> torch.Tensor:
    """Convert the existing TND end offsets to the package's cu-seqlens form."""
    if lengths is None:
        return torch.tensor((0, total), dtype=torch.int32, device=device)
    lengths = lengths.to(device=device, dtype=torch.int32).contiguous()
    if lengths.ndim != 1 or lengths.numel() == 0:
        raise ValueError("DSA sequence lengths must be a non-empty one-dimensional tensor")
    return torch.cat((torch.zeros(1, dtype=torch.int32, device=device), lengths))


def _dense_metadata(
    query: torch.Tensor,
    key: torch.Tensor,
    actual_seq_qlen: torch.Tensor | None,
    actual_seq_klen: torch.Tensor | None,
    layout: str,
    *,
    kl: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build dense DSA operator metadata and the matching sequence arguments."""
    metadata_op = dense_lightning_indexer_kl_loss_grad_metadata if kl else dense_lightning_indexer_softmax_lse_metadata
    if layout == "TND":
        cu_q = _cumulative_lengths(actual_seq_qlen, query.shape[0], query.device)
        cu_k = _cumulative_lengths(actual_seq_klen, key.shape[0], key.device)
        call_kwargs = {"cu_seqlens_q": cu_q, "cu_seqlens_k": cu_k}
        batch_size = cu_q.numel() - 1
        max_seqlen_q = int((cu_q[1:] - cu_q[:-1]).max().item())
        max_seqlen_k = int((cu_k[1:] - cu_k[:-1]).max().item())
    else:
        seqused_q = None if actual_seq_qlen is None else actual_seq_qlen.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        seqused_k = None if actual_seq_klen is None else actual_seq_klen.to(
            device=key.device, dtype=torch.int32
        ).contiguous()
        call_kwargs = {"seqused_q": seqused_q, "seqused_k": seqused_k}
        batch_size, max_seqlen_q, max_seqlen_k = query.shape[:2] + (key.shape[1],)
    attributes = {
        **call_kwargs,
        "batch_size": batch_size,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "layout_q": layout,
        "layout_k": layout,
        "mask_mode": 3,
        "cmp_ratio": 1,
    }
    metadata = metadata_op(query.shape[-2], key.shape[-2], query.shape[-1], **attributes)
    return metadata, {**call_kwargs, "layout_q": layout, "layout_k": layout, "mask_mode": 3, "cmp_ratio": 1}


def _dense_target_mask(
    target: torch.Tensor,
    actual_seq_qlen: torch.Tensor | None,
    actual_seq_klen: torch.Tensor | None,
    layout: str,
) -> torch.Tensor:
    """Mark valid causal keys in a dense DSA target distribution."""
    max_seqlen_k = target.shape[-1]
    key_positions = torch.arange(max_seqlen_k, device=target.device)
    mask = torch.zeros_like(target, dtype=torch.bool)
    if layout == "TND":
        query_ends = [0, *actual_seq_qlen.tolist()]
        key_ends = [0, *actual_seq_klen.tolist()]
        for q_start, q_end, k_start, k_end in zip(
            query_ends[:-1], query_ends[1:], key_ends[:-1], key_ends[1:]
        ):
            q_len, k_len = q_end - q_start, k_end - k_start
            if q_len:
                causal_lengths = torch.arange(k_len - q_len + 1, k_len + 1, device=target.device)
                mask[q_start:q_end] = key_positions < causal_lengths[:, None, None]
        return mask
    for batch in range(target.shape[0]):
        q_len = target.shape[1] if actual_seq_qlen is None else int(actual_seq_qlen[batch].item())
        k_len = max_seqlen_k if actual_seq_klen is None else int(actual_seq_klen[batch].item())
        causal_lengths = torch.arange(k_len - q_len + 1, k_len + 1, device=target.device)
        mask[batch, :q_len] = key_positions < causal_lengths[:, None, None]
    return mask


def _package_indexer(
    query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, *,
    actual_seq_qlen: torch.Tensor | None, actual_seq_klen: torch.Tensor | None,
    layout: str, sparse_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call LightningIndexer V2 with the package's argument contract."""
    kwargs = {
        "layout_q": layout,
        "layout_k": layout,
        "mask_mode": 3,
        "cmp_ratio": 1,
        "return_value": 1,
    }
    if layout == "TND":
        kwargs.update({
            "cu_seqlens_q": _cumulative_lengths(actual_seq_qlen, query.shape[0], query.device),
            "cu_seqlens_k": _cumulative_lengths(actual_seq_klen, key.shape[0], query.device),
        })
    else:
        # BSND uses per-batch valid lengths rather than cumulative TND ends.
        if actual_seq_qlen is not None:
            kwargs["seqused_q"] = actual_seq_qlen.to(device=query.device, dtype=torch.int32).contiguous()
        if actual_seq_klen is not None:
            kwargs["seqused_k"] = actual_seq_klen.to(device=query.device, dtype=torch.int32).contiguous()
    indices, values = lightning_indexer(
        query.contiguous(), key.contiguous(), weights.float().contiguous(), sparse_count, **kwargs,
    )
    return indices, values.to(dtype=query.dtype)


def _package_sparse_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, indices: torch.Tensor, scale: float,
    *, query_rope: torch.Tensor | None, key_rope: torch.Tensor | None,
    actual_seq_qlen: torch.Tensor | None, actual_seq_klen: torch.Tensor | None, layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call SparseFlashAttention using the existing V3.2 TND contract."""
    kwargs = {
        "query_rope": None if query_rope is None else query_rope.contiguous(),
        "key_rope": None if key_rope is None else key_rope.contiguous(),
        "sparse_block_size": 1,
        "layout_query": layout,
        "layout_kv": layout,
        "sparse_mode": 3,
        "attention_mode": 2,
        "return_softmax_lse": True,
    }
    if layout == "TND":
        kwargs.update({
            "actual_seq_lengths_query": actual_seq_qlen,
            "actual_seq_lengths_kv": actual_seq_klen,
        })
    shared_value = value.data_ptr() == key.data_ptr() and value.shape == key.shape
    key = key.contiguous()
    if shared_value:
        value = key
    else:
        value = value.contiguous()
    return sparse_flash_attention(
        query.contiguous(), key, value, indices.contiguous(), scale, **kwargs,
    )


def _package_sparse_attention_grad(
    query: torch.Tensor, key: torch.Tensor, indices: torch.Tensor, grad_output: torch.Tensor,
    output: torch.Tensor, maximum: torch.Tensor, total: torch.Tensor, scale: float, *,
    query_rope: torch.Tensor | None, key_rope: torch.Tensor | None,
    actual_seq_qlen: torch.Tensor | None, actual_seq_klen: torch.Tensor | None, layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call SparseFlashAttentionGrad in KV-merge mode.

    The package returns the merged shared-KV gradient as ``dk`` and an empty
    ``dv``. The autograd adapter therefore returns ``dk`` for the key input and
    a zero tensor for the aliased value input, preventing double accumulation.
    """
    result = sparse_flash_attention_grad(
        query.contiguous(), key.contiguous(), None, indices.contiguous(), grad_output.contiguous(),
        output.contiguous(), maximum.contiguous(), total.contiguous(), None, scale, 1,
        query_rope=None if query_rope is None else query_rope.contiguous(),
        key_rope=None if key_rope is None else key_rope.contiguous(),
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_klen,
        layout=layout, sparse_mode=3, attention_mode=2,
    )
    dq, dk, _dv, dq_rope, dk_rope, _d_sinks = result
    return (
        dq,
        dk,
        torch.zeros_like(key),
        None if query_rope is None else dq_rope,
        None if key_rope is None else dk_rope,
    )

def _package_sparse_kl_target(
    query: torch.Tensor, key: torch.Tensor, sparse_indices: torch.Tensor,
    softmax_max: torch.Tensor, softmax_sum: torch.Tensor, scale: float,
    query_rope: torch.Tensor | None, key_rope: torch.Tensor | None,
    actual_seq_qlen: torch.Tensor | None, actual_seq_klen: torch.Tensor | None,
) -> torch.Tensor:
    """Reconstruct package KL's per-candidate teacher probability tensor.

    The package-facing adapter exposes sparse ``max``/``sum`` statistics, while
    ops-transformer consumes ``attn_softmax_l1_norm``. Reconstructing the
    selected-candidate probabilities keeps the teacher detached and preserves
    the old attention statistics exactly.
    """
    query_f = query.float()
    key_f = key.float()
    batch = query.shape[0] if query.ndim == 4 else 1
    if query.ndim == 3:
        q_count, kv_heads, topk = query.shape[0], key.shape[1], sparse_indices.shape[-1]
        q_heads = query.shape[1]
    else:
        _, q_count, q_heads, _ = query.shape
        kv_heads, topk = key.shape[2], sparse_indices.shape[-1]
    if q_heads % kv_heads:
        raise ValueError("DSA package KL requires query heads divisible by key heads")
    groups = q_heads // kv_heads
    if query_rope is not None:
        query_f = torch.cat((query_f, query_rope.float()), dim=-1)
        key_f = torch.cat((key_f, key_rope.float()), dim=-1)
    lengths = actual_seq_qlen.to(device=query.device, dtype=torch.int64) if actual_seq_qlen is not None else None
    key_lengths = actual_seq_klen.to(device=query.device, dtype=torch.int64) if actual_seq_klen is not None else None
    if query.ndim == 4:
        grouped_query = query_f.reshape(batch, q_count, kv_heads, groups, query_f.shape[-1])
        safe = sparse_indices.to(torch.int64).clamp_min(0)
        key_by_head = key_f.permute(0, 2, 1, 3)
        gather_index = safe.permute(0, 2, 1, 3).unsqueeze(-1).expand(
            batch, kv_heads, q_count, topk, key_f.shape[-1]
        )
        gathered = torch.gather(
            key_by_head.unsqueeze(2).expand(-1, -1, q_count, -1, -1), 3, gather_index
        ).permute(0, 2, 1, 3, 4)
        scores = torch.einsum("bskgd,bskpd->bskgp", grouped_query, gathered).mul(scale)
        stats_max = softmax_max.permute(0, 2, 1, 3).unsqueeze(-1)
        stats_sum = softmax_sum.permute(0, 2, 1, 3).unsqueeze(-1)
        probabilities = (scores - stats_max).exp()
        probabilities = probabilities / stats_sum.clamp_min(torch.finfo(torch.float32).tiny)
        return probabilities.masked_fill(~(sparse_indices >= 0).unsqueeze(3), 0).mean(dim=3).to(torch.float32)
    positions = torch.arange(q_count, device=query.device, dtype=torch.int64)
    if lengths is None:
        doc = torch.zeros_like(positions)
        key_offsets = torch.zeros(1, device=query.device, dtype=torch.int64)
    else:
        # End offsets are inclusive of the preceding document. A token at an
        # exact boundary belongs to the next document.
        doc = torch.bucketize(positions, lengths, right=True)
        key_offsets = torch.cat((torch.zeros(1, device=query.device, dtype=torch.int64), key_lengths[:-1]))
    local_indices = sparse_indices.to(torch.int64)
    valid = local_indices >= 0
    safe = local_indices.clamp_min(0) + key_offsets[doc].view(-1, 1, 1)
    gathered = key_f.index_select(0, safe.reshape(-1)).reshape(q_count, kv_heads, topk, key_f.shape[-1])
    grouped_query = query_f.reshape(q_count, kv_heads, groups, query_f.shape[-1])
    scores = torch.einsum("tkgd,tkpd->tkgp", grouped_query, gathered).mul(scale)
    # Sparse-attention stats are [Nkv,T,G]. Invalid candidates contribute zero.
    probabilities = (scores - softmax_max.permute(1, 0, 2).unsqueeze(-1)).exp()
    denominator = softmax_sum.permute(1, 0, 2).unsqueeze(-1)
    probabilities = probabilities / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    return probabilities.masked_fill(~valid.unsqueeze(2), 0).mean(dim=2).to(torch.float32)


def _package_sparse_kl(
    query: torch.Tensor, key: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor,
    weights: torch.Tensor, sparse_indices: torch.Tensor, softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor, scale: float, *, query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None, actual_seq_qlen: torch.Tensor | None,
    actual_seq_klen: torch.Tensor | None, layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SparseLightningIndexerKLLossGrad and recover the scalar KL loss."""
    target = _package_sparse_kl_target(
        query, key, sparse_indices, softmax_max, softmax_sum, scale,
        query_rope, key_rope, actual_seq_qlen, actual_seq_klen,
    )
    if layout == "TND":
        cu_q = _cumulative_lengths(actual_seq_qlen, query_index.shape[0], query_index.device)
        cu_k = _cumulative_lengths(actual_seq_klen, key_index.shape[0], key_index.device)
        metadata_kwargs = {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "batch_size": cu_q.numel() - 1,
            "max_seqlen_q": int((cu_q[1:] - cu_q[:-1]).max().item()),
            "max_seqlen_k": int((cu_k[1:] - cu_k[:-1]).max().item()),
            "layout_q": "TND",
            "layout_k": "TND",
        }
        call_kwargs = {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "layout_q": "TND",
            "layout_k": "TND",
        }
    elif layout == "BSND":
        seqused_q = None if actual_seq_qlen is None else actual_seq_qlen.to(
            device=query_index.device, dtype=torch.int32
        ).contiguous()
        seqused_k = None if actual_seq_klen is None else actual_seq_klen.to(
            device=key_index.device, dtype=torch.int32
        ).contiguous()
        metadata_kwargs = {
            "seqused_q": seqused_q,
            "seqused_k": seqused_k,
            "batch_size": query_index.shape[0],
            "max_seqlen_q": query_index.shape[1],
            "max_seqlen_k": key_index.shape[1],
            "layout_q": "BSND",
            "layout_k": "BSND",
        }
        call_kwargs = {
            "seqused_q": seqused_q,
            "seqused_k": seqused_k,
            "layout_q": "BSND",
            "layout_k": "BSND",
        }
    else:
        raise ValueError(f"Unsupported package KL layout: {layout!r}")
    metadata = sparse_lightning_indexer_kl_loss_grad_metadata(
        query_index.shape[-2], key_index.shape[-2], query_index.shape[-1],
        topk=sparse_indices.shape[-1], mask_mode=3, cmp_ratio=1, **metadata_kwargs,
    )
    dq, dk, dw, prediction = sparse_lightning_indexer_kl_loss_grad(
        query_index.contiguous(), key_index.contiguous(), weights.float().contiguous(),
        sparse_indices.contiguous(), target.contiguous(), metadata=metadata,
        mask_mode=3, cmp_ratio=1, **call_kwargs,
    )
    valid = sparse_indices >= 0
    log_prediction = prediction.float().masked_fill(~valid, 1).clamp_min(torch.finfo(torch.float32).tiny).log()
    loss = (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - log_prediction)).sum()
    return dq.to(query_index.dtype), dk.to(key_index.dtype), dw.to(weights.dtype), loss.reshape(1)


def _validate(query: torch.Tensor, layout: str) -> None:
    if layout not in ("BSND", "TND"):
        raise ValueError(f"DSA layout must be BSND or TND, got {layout!r}")
    if query.ndim != (4 if layout == "BSND" else 3):
        raise ValueError(f"DSA {layout} query has incompatible shape {tuple(query.shape)}")
    if query.device.type != "npu":
        raise ValueError("DSA operators require local Ascend NPU tensors")
    if query.dtype not in (torch.float16, torch.bfloat16) or query.numel() == 0:
        raise ValueError("DSA query must be a nonempty float16/bfloat16 tensor")


def _check_key(query, key, weights=None):
    if (key.ndim != query.ndim or key.shape[-1] != query.shape[-1] or key.shape[-2] == 0
            or query.shape[-2] % key.shape[-2] or key.numel() == 0
            or (query.ndim == 4 and query.shape[0] != key.shape[0])):
        raise ValueError(f"Incompatible DSA query/key shapes: {tuple(query.shape)}, {tuple(key.shape)}")
    if key.device != query.device or key.dtype != query.dtype:
        raise ValueError("DSA query and key must share a device and dtype")
    if weights is not None and (weights.shape != query.shape[:-1] or weights.device != query.device):
        raise ValueError("DSA weights must match the query's token/head dimensions and device")


def _check_rope(query, key, query_rope, key_rope):
    if (query_rope is None) != (key_rope is None):
        raise ValueError("query_rope and key_rope must be supplied together")
    if query_rope is None:
        return
    for tensor, rope in ((query, query_rope), (key, key_rope)):
        if rope.shape[:-1] != tensor.shape[:-1] or rope.device != tensor.device or rope.dtype != tensor.dtype:
            raise ValueError("DSA rotary tensors must match the corresponding query/key dimensions, dtype and device")
    if query_rope.shape[-1] != key_rope.shape[-1]:
        raise ValueError("DSA query/key rotary dimensions must match")


def _check_kl_inputs(query, key, query_index, key_index, weights, query_rope, key_rope, layout):
    _validate(query, layout)
    _validate(query_index, layout)
    _check_key(query, key)
    _check_key(query_index, key_index, weights)
    _check_rope(query, key, query_rope, key_rope)
    if (query.shape[:-2] != query_index.shape[:-2] or key.shape[:-2] != key_index.shape[:-2]
            or query_index.device != query.device or query_index.dtype != query.dtype):
        raise ValueError("DSA teacher and indexer must match token dimensions, dtype and device")


def _check_indices(query, key, indices):
    if (indices.ndim != query.ndim or indices.shape[:-2] != query.shape[:-2]
            or indices.shape[-2] != key.shape[-2] or indices.shape[-1] == 0
            or indices.dtype != torch.int32 or indices.device != query.device):
        raise ValueError("DSA sparse indices must be int32 [B,S,Nkv,TopK]/[T,Nkv,TopK] tensors on the query device")


def _lengths(query, actual_seq_qlen, actual_seq_klen, layout, dtype=torch.int64):
    if layout == "TND" and (actual_seq_qlen is None or actual_seq_klen is None):
        raise ValueError("TND DSA requires cumulative query and key sequence lengths without a leading zero")
    if (actual_seq_qlen is None) != (actual_seq_klen is None):
        raise ValueError("DSA query and key sequence lengths must be supplied together")
    for lengths in (actual_seq_qlen, actual_seq_klen):
        if lengths is not None and (lengths.ndim != 1 or lengths.numel() == 0
                                    or lengths.dtype not in (torch.int32, torch.int64)):
            raise ValueError("DSA sequence lengths must be one-dimensional int32/int64 tensors")
    if actual_seq_qlen is not None and actual_seq_qlen.numel() != actual_seq_klen.numel():
        raise ValueError("DSA query and key sequence lengths must describe the same number of documents")
    # Dense warm-up metadata is kept on host; sparse package calls use device lengths.
    device = query.device if dtype == torch.int32 else "cpu"
    return tuple(value.to(device=device, dtype=dtype).contiguous() if value is not None else None
                 for value in (actual_seq_qlen, actual_seq_klen))


def _lightning_indexer(
    query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, *,
    actual_seq_qlen: torch.Tensor | None = None, actual_seq_klen: torch.Tensor | None = None,
    layout: str = "BSND", sparse_count: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal TopK indices (int32) and values (query dtype).

    The value buffer is always fully allocated: kernels may write padding even
    when their return-values flag is false. Selection is not differentiable.

    Args:
        query: Local float16/bfloat16 NPU query in BSND or TND layout.
        key: Indexer key with matching dtype and head dimension, normally one head.
        weights: Per-query-head weights with shape ``query.shape[:-1]``. The
            tested vendor indexer requires the query dtype.
        actual_seq_qlen: Sequence lengths for BSND, or cumulative ends for TND
            without a leading zero. Converted to NPU int32.
        actual_seq_klen: Corresponding key lengths; required together with query lengths.
        layout: ``"BSND"`` or ``"TND"``.
        sparse_count: TopK width; 2048 is covered by the precision tests.

    Returns:
        Indices and values shaped ``[B,S,Nkv,TopK]`` or ``[T,Nkv,TopK]``.
        Rows with fewer causal candidates use -1 index padding; padded values
        are unspecified.
    """
    _validate(query, layout)
    _check_key(query, key, weights)
    if sparse_count <= 0:
        raise ValueError("sparse_count must be positive")
    lengths = _lengths(query, actual_seq_qlen, actual_seq_klen, layout, torch.int32)
    return _package_indexer(
        query, key, weights, actual_seq_qlen=lengths[0],
        actual_seq_klen=lengths[1], layout=layout, sparse_count=sparse_count,
    )


def _dense_indexer_lse_v2(
    query_index: torch.Tensor, key_index: torch.Tensor, weights: torch.Tensor, *,
    actual_seq_qlen: torch.Tensor | None = None, actual_seq_klen: torch.Tensor | None = None,
    layout: str = "BSND",
) -> torch.Tensor:
    """Compute dense indexer LSE for the dense DSA warm-up path.

    Args:
        query_index: Local float16/bfloat16 NPU indexer query in BSND or TND layout.
        key_index: Indexer key, normally with one head.
        weights: Per-query-head weights with shape ``query_index.shape[:-1]``.
        actual_seq_qlen: BSND lengths or TND cumulative ends without a leading
            zero. CPU int64 metadata avoids device-to-host synchronization.
        actual_seq_klen: Corresponding key lengths; required together with query lengths.
        layout: ``"BSND"`` or ``"TND"``.

    Returns:
        Float32 LSE in ``[B,Nkv,S]`` or ``[Nkv,T]``.
    """
    _validate(query_index, layout)
    _check_key(query_index, key_index, weights)
    lengths = _lengths(query_index, actual_seq_qlen, actual_seq_klen, layout, torch.int32)
    metadata, options = _dense_metadata(
        query_index, key_index, *lengths, layout, kl=False,
    )
    return dense_lightning_indexer_softmax_lse(
        query_index.contiguous(), key_index.contiguous(), weights.float().contiguous(),
        metadata=metadata, **options,
    )


def _dense_indexer_kl_loss_grad(
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    attention_target: torch.Tensor,
    index_softmax_lse: torch.Tensor,
    *,
    actual_seq_qlen: torch.Tensor | None = None, actual_seq_klen: torch.Tensor | None = None,
    layout: str = "TND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run dense indexer KL backward and recover the raw summed loss.

    ``attention_target`` is the detached dense attention distribution; the
    indexer LSE comes from ``dense_lightning_indexer_softmax_lse``.
    """
    _validate(query_index, layout)
    _check_key(query_index, key_index, weights)
    lengths = _lengths(query_index, actual_seq_qlen, actual_seq_klen, layout, torch.int32)
    metadata, options = _dense_metadata(
        query_index, key_index, *lengths, layout, kl=True,
    )
    dq, dk, dw, prediction = dense_lightning_indexer_kl_loss_grad(
        query_index.contiguous(), key_index.contiguous(), weights.float().contiguous(),
        attention_target.contiguous(), index_softmax_lse.contiguous(),
        metadata=metadata, **options,
    )
    mask = _dense_target_mask(attention_target, *lengths, layout)
    target = attention_target.float().masked_fill(~mask, 0)
    prediction = prediction.float().masked_fill(~mask, 1).clamp_min(torch.finfo(torch.float32).tiny)
    target_log = target.clamp_min(torch.finfo(torch.float32).tiny).log()
    loss = (target * (target_log - prediction.log())).sum()
    return dq.to(query_index.dtype), dk.to(key_index.dtype), dw.to(weights.dtype), loss.reshape(1)


def _sparse_indexer_kl_loss_grad(
    query: torch.Tensor, key: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor,
    weights: torch.Tensor, sparse_indices: torch.Tensor, softmax_max: torch.Tensor, softmax_sum: torch.Tensor,
    scale_value: float, *, query_rope: torch.Tensor | None = None, key_rope: torch.Tensor | None = None,
    actual_seq_qlen: torch.Tensor | None = None, actual_seq_klen: torch.Tensor | None = None,
    layout: str = "BSND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute sparse-stage indexer gradients and summed KL loss.

    Args:
        query: Local float16/bfloat16 NPU teacher query after MLA absorption.
        key: Single-head absorbed MLA key in the same BSND or TND layout.
        query_index: Indexer query for the same query tokens.
        key_index: Single-head indexer key for the same key tokens.
        weights: Per-indexer-query-head weights.
        sparse_indices: Int32 candidates from :func:`_lightning_indexer`.
        softmax_max: Detached float32 teacher maximum from sparse attention.
        softmax_sum: Detached float32 teacher exponential sum from sparse attention.
        scale_value: Teacher attention score scaling factor.
        query_rope: Optional teacher query rotary states.
        key_rope: Optional teacher key rotary states; paired with query_rope.
        actual_seq_qlen: BSND lengths or TND cumulative ends without a leading
            zero. Prefer CPU int64 metadata.
        actual_seq_klen: Corresponding key lengths; required together with query lengths.
        layout: ``"BSND"`` or ``"TND"``.

    Returns:
        Explicit gradients matching query_index, key_index and weights, followed
        by a float32 one-element summed loss. These raw outputs have no autograd
        graph; use :func:`dsa_kl_loss` for normalized training integration.
    """
    _check_kl_inputs(query, key, query_index, key_index, weights, query_rope, key_rope, layout)
    _check_indices(query, key, sparse_indices)
    _lengths(query, actual_seq_qlen, actual_seq_klen, layout)
    return _package_sparse_kl(
        query, key, query_index, key_index, weights, sparse_indices,
        softmax_max, softmax_sum, scale_value, query_rope=query_rope,
        key_rope=key_rope, actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen, layout=layout,
    )


class _SparseAttention(torch.autograd.Function):
    """Attach the cann_ops_transformer sparse-attention backward."""

    @staticmethod
    def forward(
        ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, indices: torch.Tensor,
        query_rope: torch.Tensor | None, key_rope: torch.Tensor | None,
        qlen: torch.Tensor | None, klen: torch.Tensor | None, scale: float, layout: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute local attention and save inputs and statistics for backward."""
        outputs = _package_sparse_attention(
            query, key, value, indices, scale, query_rope=query_rope, key_rope=key_rope,
            actual_seq_qlen=qlen, actual_seq_klen=klen, layout=layout,
        )
        ctx.save_for_backward(query, key, value, indices, query_rope, key_rope, qlen, klen, *outputs)
        ctx.scale, ctx.layout = scale, layout
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad_output: torch.Tensor, _grad_max: torch.Tensor | None, _grad_sum: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Return first-order gradients for Q, shared K/V and rotary inputs."""
        query, key, value, indices, query_rope, key_rope, qlen, klen, output, maximum, total = ctx.saved_tensors
        gradients = _package_sparse_attention_grad(
            query, key, indices, grad_output, output, maximum, total, ctx.scale,
            query_rope=query_rope, key_rope=key_rope, actual_seq_qlen=qlen,
            actual_seq_klen=klen, layout=ctx.layout,
        )
        return (*gradients[:3], None, *gradients[3:], None, None, None, None)


def _sparse_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, sparse_indices: torch.Tensor, scale_value: float, *,
    query_rope: torch.Tensor | None = None, key_rope: torch.Tensor | None = None,
    actual_seq_qlen: torch.Tensor | None = None, actual_seq_klen: torch.Tensor | None = None,
    layout: str = "BSND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute causal MLA sparse attention with a shared key/value tensor.

    ``value`` must alias ``key`` with identical shape and strides, as required
    by the CANN MLA-absorb kernel. Autograd accumulates both derivatives into
    that shared tensor. Statistics are detached [B,Nkv,S,G]/[Nkv,T,G] tensors.
    Each row must contain min(causal length, TopK) distinct valid indices from
    :func:`_lightning_indexer`, followed by -1 padding. CANN backward relies
    on this count; arbitrary shorter candidate lists are unsupported.

    Args:
        query: Local float16/bfloat16 NPU query after MLA absorption.
        key: Absorbed MLA key, normally with one head.
        value: Shared key/value tensor with the same storage, shape and strides as key.
        sparse_indices: Int32 ``[B,S,Nkv,TopK]`` or ``[T,Nkv,TopK]`` candidates.
        scale_value: Attention score scaling factor.
        query_rope: Optional query rotary states with matching token/head dimensions.
        key_rope: Optional key rotary states; paired with query_rope.
        actual_seq_qlen: BSND lengths or TND cumulative ends without a leading
            zero. Converted to NPU int32.
        actual_seq_klen: Corresponding key lengths; required together with query lengths.
        layout: ``"BSND"`` or ``"TND"``.

    Returns:
        Attention output with the query token/head dimensions and value width,
        followed by float32 softmax maximum and exponential sum. First-order
        gradients cover Q, shared K/V and rotary states; statistics are detached.
    """
    _validate(query, layout)
    _check_key(query, key)
    _check_indices(query, key, sparse_indices)
    _check_rope(query, key, query_rope, key_rope)
    if value is not key:
        raise ValueError("The ops-transformer DSA attention backend requires value to alias key")
    lengths = _lengths(query, actual_seq_qlen, actual_seq_klen, layout, torch.int32)
    return _SparseAttention.apply(query, key, value, sparse_indices, query_rope, key_rope,
                                  *lengths, scale_value, layout)


def _cann_dsa_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_qlen: torch.Tensor | None = None,
    actual_seq_klen: torch.Tensor | None = None,
    layout: str = "BSND",
    sparse_count: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select causal DSA candidates with CANN LightningIndexer."""
    return _lightning_indexer(
        query,
        key,
        weights,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        layout=layout,
        sparse_count=sparse_count,
    )


def _cann_dsa_dense_indexer_lse(
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_qlen: torch.Tensor | None = None,
    actual_seq_klen: torch.Tensor | None = None,
    layout: str = "BSND",
) -> torch.Tensor:
    """Compute dense indexer log-sum-exp with the CANN V2 operator."""
    return _dense_indexer_lse_v2(
        query_index,
        key_index,
        weights,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        layout=layout,
    )


def _cann_dsa_dense_indexer_kl_loss_grad(
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    attention_target: torch.Tensor,
    index_softmax_lse: torch.Tensor,
    *,
    actual_seq_qlen: torch.Tensor | None = None,
    actual_seq_klen: torch.Tensor | None = None,
    layout: str = "TND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return explicit dense indexer gradients and summed KL loss from CANN."""
    return _dense_indexer_kl_loss_grad(
        query_index,
        key_index,
        weights,
        attention_target,
        index_softmax_lse,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        layout=layout,
    )


def _cann_dsa_sparse_indexer_kl_loss_grad(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    sparse_indices: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    scale_value: float,
    *,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    actual_seq_qlen: torch.Tensor | None = None,
    actual_seq_klen: torch.Tensor | None = None,
    layout: str = "BSND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return explicit sparse indexer gradients and summed KL loss from CANN."""
    return _sparse_indexer_kl_loss_grad(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        query_rope=query_rope,
        key_rope=key_rope,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        layout=layout,
    )


def _cann_dsa_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    *,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    actual_seq_qlen: torch.Tensor | None = None,
    actual_seq_klen: torch.Tensor | None = None,
    layout: str = "BSND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run CANN sparse attention with its CANN backward and return softmax stats."""
    return _sparse_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        query_rope=query_rope,
        key_rope=key_rope,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        layout=layout,
    )
