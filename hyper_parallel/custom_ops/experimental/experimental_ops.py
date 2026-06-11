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
"""Experimental custom operator implementations.

Each function is a thin delegation wrapper around ``_platform.custom_ops``,
which routes to the platform-specific Ascend NPU custom C++ kernel.
"""
from typing import Optional, Tuple

from mindspore import Tensor

from hyper_parallel.platform import get_platform

_platform = get_platform()

_MAX_INT64 = 9223372036854775807


def npu_dense_lightning_indexer_softmax_lse(
        query_index,
        key_index,
        weights,
        *,
        actual_seq_qlen: Optional[Tensor] = None,
        actual_seq_klen: Optional[Tensor] = None,
        layout: str = 'BSND',
        sparse_mode: int = 3,
        pre_tokens: int = _MAX_INT64,
        next_tokens: int = _MAX_INT64,
) -> Tuple:
    """Compute softmax max/sum indices for Lightning Indexer attention.

    .. warning::
        This is an experimental API that subject to change or deletion.

    Pre-computes the Softmax max and sum values to reduce memory usage.

    The call is routed through the platform ``custom_ops`` layer, which
    delegates to a ``DFunction`` wrapping the Ascend custom C++ kernel.
    DTensor inputs are transparently handled by distributed dispatch.

    Args:
        query_index: Lightning Indexer query input (Q̃). dtype bfloat16/float16.
        key_index: Lightning Indexer key input (K̃). Same dtype as query_index.
        weights: Weight coefficient (W). dtype bfloat16/float16/float32.
        actual_seq_qlen: Cumulative query sequence lengths (int32 Tensor).
        actual_seq_klen: Cumulative key sequence lengths (int32 Tensor).
        layout: Data layout format — 'BSND' (default) or 'TND'.
        sparse_mode: Sparse computation mode; only mode 3 is supported.
        pre_tokens: Preceding token window size for sparse attention (int64).
        next_tokens: Following token window size for sparse attention (int64).

    Returns:
        tuple[Tensor, Tensor]: ``(softmax_max_index, softmax_sum_index)``.
    """
    return _platform.custom_ops.npu_dense_lightning_indexer_softmax_lse(
        query_index, key_index, weights,
        actual_seq_qlen, actual_seq_klen,
        layout, sparse_mode, pre_tokens, next_tokens,
    )


def npu_dense_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        softmax_max,
        softmax_sum,
        softmax_max_index,
        softmax_sum_index,
        scale_value,
        *,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen: Optional[Tensor] = None,
        actual_seq_klen: Optional[Tensor] = None,
        layout: str = 'BSND',
        sparse_mode: int = 3,
        pre_tokens: int = _MAX_INT64,
        next_tokens: int = _MAX_INT64,
) -> Tuple:
    """Compute backward gradients and KL-divergence loss for dense Lightning Indexer.

    .. warning::
        This is an experimental API that subject to change or deletion.

    The call is routed through the platform ``custom_ops`` layer.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            ``(d_query_index, d_key_index, d_weights, loss)``.
    """
    return _platform.custom_ops.npu_dense_lightning_indexer_grad_kl_loss(
        query, key, query_index, key_index, weights,
        softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
        scale_value,
        query_rope, key_rope,
        actual_seq_qlen, actual_seq_klen,
        layout, sparse_mode,
        pre_tokens, next_tokens,
    )


def npu_sparse_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        *,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen: Optional[Tensor] = None,
        actual_seq_klen: Optional[Tensor] = None,
        layout: str = 'BSND',
        sparse_mode: int = 3,
        pre_tokens: int = _MAX_INT64,
        next_tokens: int = _MAX_INT64,
) -> Tuple:
    """Compute backward gradients and KL-divergence loss for sparse Lightning Indexer.

    .. warning::
        This is an experimental API that subject to change or deletion.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            ``(d_query_index, d_key_index, d_weights, loss)``.
    """
    return _platform.custom_ops.npu_sparse_lightning_indexer_grad_kl_loss(
        query, key, query_index, key_index, weights,
        sparse_indices, softmax_max, softmax_sum, scale_value,
        query_rope, key_rope,
        actual_seq_qlen, actual_seq_klen,
        layout, sparse_mode,
        pre_tokens, next_tokens,
    )


def npu_mhc_post(x, h_res, h_out, h_post) -> Tuple:
    """MHC post-processing with residual connection.

    .. warning::
        This is an experimental API that subject to change or deletion.

    Returns:
        Tensor: Output tensor with same shape and dtype as x.
    """
    return _platform.custom_ops.npu_mhc_post(x, h_res, h_out, h_post)


def npu_mhc_pre_sinkhorn(
        x,
        phi,
        alpha,
        bias,
        *,
        hc_mult: int = 4,
        num_iters: int = 20,
        hc_eps: float = 1e-6,
        norm_eps: float = 1e-6,
        out_flag: bool = True,
) -> Tuple:
    """MHC pre-processing with Sinkhorn normalization.

    .. warning::
        This is an experimental API that subject to change or deletion.

    Returns:
        tuple: 8 output tensors.
    """
    return _platform.custom_ops.npu_mhc_pre_sinkhorn(
        x, phi, alpha, bias,
        hc_mult, num_iters,
        hc_eps, norm_eps, out_flag,
    )


def npu_mhc_pre_clamp_sinkhorn(
        x,
        phi,
        alpha,
        bias,
        *,
        hc_mult: int = 4,
        num_iters: int = 20,
        hc_eps: float = 1e-6,
        norm_eps: float = 1e-6,
        out_flag: bool = True,
        clamp_min: float = 0.0,
        clamp_max: float = 0.0,
) -> Tuple:
    """MHC pre-processing with clamp and Sinkhorn normalization.

    .. warning::
        This is an experimental API that subject to change or deletion.

    Returns:
        tuple: 9 output tensors.
    """
    return _platform.custom_ops.npu_mhc_pre_clamp_sinkhorn(
        x, phi, alpha, bias,
        hc_mult, num_iters,
        hc_eps, norm_eps, out_flag,
        clamp_min, clamp_max,
    )
