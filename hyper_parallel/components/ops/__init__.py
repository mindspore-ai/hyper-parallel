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
"""Direct HyperOps functions and the model configuration entry point."""

from __future__ import annotations

import torch

from ._dsa_layout import DsaQueryLayout, ReduceSum
from .factory import _dispatch, _select_for_call, ops
from .families.attention import VISION_ATTENTION, VisionAttentionFn
from .families.dsa_indexer import DSA_INDEXER, indexer_native_only, native_dsa_indexer
from .families.dsa_indexer_kl import DSA_INDEXER_KL, indexer_kl_native_only, native_dsa_indexer_kl
from .families.dsa_sparse_attention import DSA_COMPRESSED_BANKS, attention_native_only
from .families.grouped_matmul import GROUPED_MATMUL, GroupedMatmulFn
from .families.mhc_post import MHC_POST, MhcPostFn
from .families.moe_router import MOE_ROUTER, router_scores
from .families.moe_token_reorder import MOE_TOKEN_REORDER, MOE_TOKEN_RESTORE, PermutedTokens
from .families.rms_norm import RMS_NORM, RMS_NORM_FP32_AFFINE, RMSNormFn
from .families.sinkhorn import SINKHORN, SinkhornFn
from .families.swiglu import SWIGLU, SWIGLU_CLAMPED, ClampedSwiGLUFn, SwiGLUFn

__all__ = [
    "DsaQueryLayout", "dsa_indexer", "dsa_indexer_kl", "dsa_sparse_attention_compressed_banks", "grouped_matmul",
    "mhc_post",
    "moe_router", "moe_token_reorder", "moe_token_restore", "ops", "rms_norm", "rms_norm_fp32_affine", "router_scores",
    "sinkhorn", "swiglu", "swiglu_clamped", "vision_attention",
]

grouped_matmul: GroupedMatmulFn = _dispatch(GROUPED_MATMUL)
mhc_post: MhcPostFn = _dispatch(MHC_POST)
rms_norm: RMSNormFn = _dispatch(RMS_NORM)
rms_norm_fp32_affine: RMSNormFn = _dispatch(RMS_NORM_FP32_AFFINE)
sinkhorn: SinkhornFn = _dispatch(SINKHORN)
swiglu: SwiGLUFn = _dispatch(SWIGLU)
swiglu_clamped: ClampedSwiGLUFn = _dispatch(SWIGLU_CLAMPED)
vision_attention: VisionAttentionFn = _dispatch(VISION_ATTENTION)


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


def dsa_sparse_attention_compressed_banks(
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
