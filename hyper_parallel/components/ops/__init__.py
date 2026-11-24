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
"""OpSelect operator entry points and backend configuration."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo
from hyper_parallel.components.functional.npu_fusion_attention import resolve_packed_sequence_lengths

from ._dsa_layout import DsaQueryLayout, ReduceSum
from .factory import _dispatch, _select_for_call, ops
from .impl.attention import (
    GROUPED_QUERY_ATTENTION, MULTI_LATENT_ATTENTION, MULTI_LATENT_ATTENTION_WITH_STATS, VISION_ATTENTION,
    DenseAttentionFn, MultiLatentAttentionStatsFn, VisionAttentionFn,
)
from .impl.causal_conv1d import CAUSAL_CONV1D
from .impl.dsa_indexer import (
    DSA_INDEXER, DSA_PACKED_INDEXER, PackedDsaIndexerFn, indexer_native_only, native_dsa_indexer,
)
from .impl.dsa_indexer_kl import (
    DSA_DENSE_KL, DSA_INDEXER_KL, DSA_PACKED_KL, DenseDsaKlFn, PackedDsaKlFn, indexer_kl_native_only,
    native_dsa_indexer_kl,
)
from .impl.deepseek_sparse_attention import (
    DSA_ABSORBED, DSA_ABSORBED_SINK, DSA_COMPRESSED_BANKS, PackedDeepseekSparseAttentionFn, attention_native_only,
)
from .impl.gdn import GDN
from .impl.grouped_matmul import GROUPED_MATMUL, GroupedMatmulFn
from .impl.grouped_swiglu import GROUPED_SWIGLU
from .impl.indexer_topk_restore import (
    INDEXER_TOPK_RESTORE, IndexerTopKRestoreFn, native_indexer_topk_restore,
)
from .impl.kda import KDA
from .impl.mhc_post import MHC_POST, MhcPostFn
from .impl.moe_router import MOE_ROUTER, router_scores
from .impl.moe_token_reorder import MOE_TOKEN_REORDER, MOE_TOKEN_RESTORE, PermutedTokens
from .impl.mome import MOME, MomeFn
from .impl.rms_norm import RMS_NORM, RMS_NORM_FP32_AFFINE, RMSNormFn
from .impl.rope import ROPE, ROPE_INTERLEAVED, InterleavedRopeFn, RopeFn
from .impl.sinkhorn import SINKHORN, SinkhornFn
from .impl.swiglu import SWIGLU, SWIGLU_CLAMPED, ClampedSwiGLUFn, SwiGLUFn
from .spec import OpSpec

__all__ = [
    "DsaQueryLayout", "causal_conv1d", "dsa_indexer", "dsa_indexer_kl", "dsa_indexer_kl_dense",
    "dsa_indexer_kl_packed", "dsa_indexer_packed",
    "deepseek_sparse_attention_absorbed", "deepseek_sparse_attention_compressed_banks",
    "deepseek_sparse_attention_with_sink",
    "indexer_topk_restore", "prepare_indexer_topk_restore",
    "gdn", "grouped_query_attention", "grouped_matmul", "grouped_swiglu", "kda", "mhc_post", "multi_latent_attention",
    "multi_latent_attention_with_stats",
    "moe_router", "moe_token_reorder", "moe_token_restore", "mome", "ops", "rms_norm", "rms_norm_fp32_affine",
    "router_scores",
    "rope", "rope_interleaved", "rope_qk",
    "sinkhorn", "swiglu", "swiglu_clamped", "vision_attention",
]

indexer_topk_restore: IndexerTopKRestoreFn = _dispatch(INDEXER_TOPK_RESTORE)

grouped_matmul: GroupedMatmulFn = _dispatch(GROUPED_MATMUL)
dsa_indexer_packed: PackedDsaIndexerFn = _dispatch(DSA_PACKED_INDEXER)
dsa_indexer_kl_dense: DenseDsaKlFn = _dispatch(DSA_DENSE_KL)
dsa_indexer_kl_packed: PackedDsaKlFn = _dispatch(DSA_PACKED_KL)
deepseek_sparse_attention_absorbed: PackedDeepseekSparseAttentionFn = _dispatch(DSA_ABSORBED)
deepseek_sparse_attention_with_sink: PackedDeepseekSparseAttentionFn = _dispatch(DSA_ABSORBED_SINK)
mhc_post: MhcPostFn = _dispatch(MHC_POST)
multi_latent_attention_with_stats: MultiLatentAttentionStatsFn = _dispatch(MULTI_LATENT_ATTENTION_WITH_STATS)
mome: MomeFn = _dispatch(MOME)
rms_norm: RMSNormFn = _dispatch(RMS_NORM)
rms_norm_fp32_affine: RMSNormFn = _dispatch(RMS_NORM_FP32_AFFINE)
rope: RopeFn = _dispatch(ROPE)
rope_interleaved: InterleavedRopeFn = _dispatch(ROPE_INTERLEAVED)
sinkhorn: SinkhornFn = _dispatch(SINKHORN)
swiglu: SwiGLUFn = _dispatch(SWIGLU)
swiglu_clamped: ClampedSwiGLUFn = _dispatch(SWIGLU_CLAMPED)
vision_attention: VisionAttentionFn = _dispatch(VISION_ATTENTION)


def _dense_mask(mask: torch.Tensor | None, key_length: int, packed: bool) -> torch.Tensor | None:
    """Normalize key padding and pairwise mask dimensions before dense attention."""
    if mask is not None and not packed:
        mask = mask[..., :key_length]
    if mask is not None and mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask


def _dense_window(
    module: torch.nn.Module, query_length: int, key_length: int, causal: bool, kwargs: dict[str, Any],
) -> int | None:
    """Resolve an explicit sliding window or the module's causal token limit."""
    window = kwargs.get("sliding_window")
    if window is None and causal and query_length == key_length:
        pre_tokens = kwargs.get("pre_tokens", getattr(module, "pre_tockens", None))
        if pre_tokens is not None and pre_tokens < query_length:
            window = pre_tokens + 1
    return window


def _dense_attention(
    spec: OpSpec[DenseAttentionFn], module: torch.nn.Module, query: torch.Tensor,
    key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None,
    dropout: float, scaling: float | None, kwargs: dict[str, Any],
) -> tuple[torch.Tensor, None]:
    """Normalize dense attention arguments and dispatch the selected contract."""
    if kwargs.get("indices") is not None:
        raise ValueError("Dense attention does not consume sparse attention indices")
    q_len, k_len = query.shape[2], key.shape[2]
    q_ends, k_ends = resolve_packed_sequence_lengths(kwargs, query.shape[0] * q_len, key.shape[0] * k_len)
    lengths = (q_ends, k_ends) if q_ends is not None else None
    if lengths is not None and len(q_ends) != len(k_ends):
        raise ValueError("Packed query and key lengths must contain the same number of segments")
    attention_mask = _dense_mask(attention_mask, k_len, lengths is not None)
    causal = kwargs.get("is_causal", getattr(module, "is_causal", True))
    if q_len != k_len and attention_mask is not None and attention_mask.ndim >= 3 and lengths is None:
        # CP masks already encode the query's global offset.
        causal = False
    window = _dense_window(module, q_len, k_len, causal, kwargs)
    query_offset = kwargs.get("query_offset")
    blocked = ()
    if lengths is not None and attention_mask is not None:
        blocked = (("torch_npu", "packed attention requires segment boundaries without an external mask"),)
    operation = _select_for_call(spec, query, blocked_backends=blocked)
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    return operation(
        query, key, value, attention_mask, causal=causal, scale=scale, dropout=dropout,
        window=window, lengths=lengths, query_offset=query_offset,
    ), None


def grouped_query_attention(
    module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None, **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run the dense grouped-query contract through a Transformers-compatible entry point."""
    return _dense_attention(
        GROUPED_QUERY_ATTENTION, module, query, key, value, attention_mask, dropout, scaling, kwargs,
    )


def multi_latent_attention(
    module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None, **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attend over projected MLA states, allowing the value dimension to differ from Q and K."""
    return _dense_attention(MULTI_LATENT_ATTENTION, module, query, key, value, attention_mask, dropout, scaling, kwargs)


def rope_qk(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply half-layout RoPE on Q and K with Transformers position embeddings."""
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.shape[0] == 1 and query.shape[0] > 1:
        cos = cos.expand(query.shape[0], -1, -1)
        sin = sin.expand(query.shape[0], -1, -1)
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return rope(query, cos, sin), rope(key, cos, sin)


_LINEAR_ATTENTION_BACKENDS = {"eager": "native", "triton": "triton_ascend"}


def grouped_swiglu(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, counts: torch.Tensor,
    scores: torch.Tensor | None = None, *, backend: str = "auto",
) -> torch.Tensor:
    """Apply grouped expert SwiGLU after resolving the family backend."""
    operation = _select_for_call(GROUPED_SWIGLU, x, default_backend=backend)
    return operation(x, w1, w2, w3, counts, scores)


def causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, info: PackedSequenceInfo | None = None,
    *, dilation: int = 1, backend: str = "auto",
) -> torch.Tensor:
    """Apply local depthwise causal convolution and SiLU to dense or packed tokens."""
    reason = "fused convolution requires nonempty batches and dilation one" if dilation != 1 or not x.numel() else None
    blocked = ()
    if (
        weight.shape[-1] not in (2, 3, 4) or x.shape[-1] % 16 or x.dtype not in (torch.bfloat16, torch.float16)
        or weight.dtype != x.dtype or (bias is not None and bias.dtype != x.dtype)
    ):
        blocked = (("ascendc", "requires width 2/3/4, 16-aligned channels and matching BF16/FP16 dtypes"),)
    operation = _select_for_call(
        CAUSAL_CONV1D, x, native_only=reason, blocked_backends=blocked,
        default_backend=_LINEAR_ATTENTION_BACKENDS.get(backend, backend),
    )
    return operation(x, weight, bias, info, dilation=dilation)


def gdn(
    x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
    *, chunk_size: int = 64, normalize_qk: bool = True, info: PackedSequenceInfo | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute zero-state GDN with effective gate and beta inputs."""
    reason = None
    if (
        chunk_size != 64 or x.shape[-1] != 128 or value.shape[-1] != 128
        or x.dtype != torch.bfloat16 or key.dtype != x.dtype or value.dtype != x.dtype
        or beta.dtype != x.dtype or gate.dtype != torch.float32
    ):
        reason = "fused GDN requires chunk size 64, head dimension 128, BF16 Q/K/V/beta and FP32 gate"
    operation = _select_for_call(
        GDN, x, native_only=reason, default_backend=_LINEAR_ATTENTION_BACKENDS.get(backend, backend),
    )
    return operation(x, key, value, gate, beta, chunk_size=chunk_size, normalize_qk=normalize_qk, info=info)


def kda(
    x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor,
    *, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None, chunk_size: int = 64,
    scale: float | None = None, info: PackedSequenceInfo | None = None, backend: str = "auto",
) -> torch.Tensor:
    """Compute zero-state KDA with raw gate and beta logits."""
    reason = None
    if (
        lower_bound is None or not -5.0 <= lower_bound < 0 or chunk_size != 64
        or x.shape[-1] != 128 or value.shape[-1] != 128
        or x.dtype != torch.bfloat16 or key.dtype != x.dtype or value.dtype != x.dtype
    ):
        reason = "fused KDA requires a safe gate, chunk size 64, head dimension 128 and BF16 Q/K/V"
    blocked = []
    if gate.dtype != x.dtype or beta.dtype != x.dtype:
        blocked.append(("triton_ascend", "requires BF16 gate and beta"))
    if x.shape[2] != value.shape[2]:
        blocked.append(("ascendc", "requires matching Q/K/V head counts"))
    operation = _select_for_call(
        KDA, x, native_only=reason, blocked_backends=tuple(blocked),
        default_backend=_LINEAR_ATTENTION_BACKENDS.get(backend, backend),
    )
    return operation(x, key, value, gate, beta, a_log=a_log, dt_bias=dt_bias, lower_bound=lower_bound,
                     chunk_size=chunk_size, scale=scale, info=info)


def dsa_indexer(
    x: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, *,
    layout: DsaQueryLayout, sparse_count: int, query_chunk_size: int = 256,
    reduce_sum: ReduceSum | None = None,
) -> torch.Tensor:
    """Select causal compressed keys and return sorted int32 IDs."""
    if x.ndim != 4 or key.ndim != 3 or min(sparse_count, key.shape[1]) <= 0 or x.shape[1] == 0:
        implementation = native_dsa_indexer
    else:
        implementation = _select_for_call(DSA_INDEXER, x, native_only=indexer_native_only(x, layout, reduce_sum))
    return implementation(
        x, key, merge_weight, layout=layout, sparse_count=sparse_count,
        query_chunk_size=query_chunk_size, reduce_sum=reduce_sum,
    )


def dsa_indexer_kl(
    x: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
    attention_query: torch.Tensor, compressed_key: torch.Tensor, topk_indices: torch.Tensor,
    sinks: torch.Tensor, *, attention_scale: float, loss_coeff: float, query_chunk_size: int = 256,
    reduce_sum: ReduceSum | None = None, layout: DsaQueryLayout | None = None,
) -> torch.Tensor:
    """Compute the scalar Indexer KL loss, with gradients restricted to the student."""
    if not loss_coeff or index_key.shape[1] == 0 or x.shape[1] == 0:
        implementation = native_dsa_indexer_kl
    else:
        reason = indexer_kl_native_only(x, index_key, topk_indices, layout, reduce_sum)
        implementation = _select_for_call(DSA_INDEXER_KL, x, native_only=reason)
    return implementation(
        x, index_key, merge_weight, attention_query, compressed_key, topk_indices, sinks,
        attention_scale=attention_scale, loss_coeff=loss_coeff, query_chunk_size=query_chunk_size,
        reduce_sum=reduce_sum, layout=layout,
    )


def deepseek_sparse_attention_compressed_banks(
    x: torch.Tensor, raw_key: torch.Tensor, compressed_key: torch.Tensor | None,
    topk_indices: torch.Tensor | None, sinks: torch.Tensor, *, scale: float, window: int,
    layout: DsaQueryLayout, global_sequence_length: int, raw_key_offset: int = 0,
    main_key_offset: int = 0, global_key_length: int | None = None,
) -> torch.Tensor:
    """Attend a sliding window and optional compressed keys in logical document coordinates."""
    reason = attention_native_only(x, layout, window, topk_indices, global_sequence_length, global_key_length)
    implementation = _select_for_call(DSA_COMPRESSED_BANKS, x, native_only=reason)
    return implementation(
        x, raw_key, compressed_key, topk_indices, sinks, scale=scale, window=window,
        layout=layout, global_sequence_length=global_sequence_length,
        raw_key_offset=raw_key_offset, main_key_offset=main_key_offset, global_key_length=global_key_length,
    )


def moe_token_reorder(x: torch.Tensor, indices: torch.Tensor) -> PermutedTokens:
    """Arrange routed slots for expert execution with the selected permutation pair."""
    reason = "fused permutation requires at least one routed slot" if indices.numel() == 0 else None
    return _select_for_call(MOE_TOKEN_REORDER, x, native_only=reason)(x, indices)


def moe_token_restore(x: torch.Tensor, slot_positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reduce weighted expert outputs in the original token order."""
    reason = "fused restoration requires at least one routed slot" if slot_positions.numel() == 0 else None
    return _select_for_call(MOE_TOKEN_RESTORE, x, native_only=reason)(x, slot_positions, weights)


def _check_native_ep_reorder(x: torch.Tensor) -> None:
    """Check the configured backend before using EP's chunked or compact dispatch."""
    _select_for_call(
        MOE_TOKEN_REORDER, x, native_only="fused permutation requires full, unchunked routing without a capacity limit",
    )


def moe_router(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, vision_bias: torch.Tensor | None,
    image_mask: torch.Tensor | None, top_k: int, scoring_func: str, scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute logits, routing weights and expert IDs for DeepSeek's top-k router."""
    native_only = None
    if scoring_func != "sqrtsoftplus" or x.ndim != 3 or x.numel() == 0:
        native_only = "MegaGate requires nonempty batched tokens and sqrtsoftplus scoring"
    route = _select_for_call(MOE_ROUTER, x, native_only=native_only)
    return route(x, weight, bias, vision_bias, image_mask, top_k, scoring_func, scale)


def prepare_indexer_topk_restore(selected: torch.Tensor) -> IndexerTopKRestoreFn | None:
    """Resolve a deferred restore callable before launching the return collective.

    None preserves Torch's early local-copy and post-wait remote-copy schedule.
    A fused callable instead places both partitions after the receive wait. It
    must be retained by the pending handle, along with the source buffers, so
    changing process configuration cannot alter an in-flight return operation.
    """
    operation = _select_for_call(INDEXER_TOPK_RESTORE, selected)
    return None if operation is native_indexer_topk_restore else operation
