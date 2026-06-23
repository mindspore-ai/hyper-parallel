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


def npu_lightning_indexer(
        query,
        key,
        weights,
        sparse_count: int,
        *,
        cu_seq_lens_q: Optional[Tensor] = None,
        cu_seq_lens_k: Optional[Tensor] = None,
        cmp_residual_k: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        layout: str = 'BSND',
        sparse_mode: int = 0,
        cmp_ratio: int = 1,
        return_value: bool = False,
) -> Tuple:
    """Sparse attention preprocessing — select top-K key tokens per query token.

    .. warning::
        This is an experimental API that subject to change or deletion.

    Aligned with the ``lightning_indexer`` benchmark signature: positional
    ``(query, key, weights, sparse_count)`` (``sparse_count`` is benchmark
    ``topk``); the two ``layout_q`` / ``layout_k`` parameters are merged into a
    single ``layout``. The underlying kernel handles all ``cmp_ratio`` values
    (1 / 4 / 128) directly.

    The remaining benchmark kwargs (``seqused_q`` / ``seqused_k`` /
    ``output_idx_offset`` / ``metadata`` / ``max_seqlen_q``) are not yet exposed
    by this external API and are pinned to ``None`` / ``-1`` inside the DFunction.

    Args:
        query: Lightning Indexer query input (Q_index). Must be contiguous.
            layout='BSND': shape ``(B, S1, N1, D)``; layout='TND': ``(T1, N1, D)``.
            dtype bfloat16/float16.
        key: Lightning Indexer key input (K_index). Must be contiguous.
            Same dtype as query.
        weights: Weight coefficient (W). shape ``(B, S1, N1)``. Same dtype as query.
        sparse_count: Number of top-K key tokens to retain (benchmark ``topk``).
        cu_seq_lens_q: Cumulative query sequence lengths (int32); None for BSND.
        cu_seq_lens_k: Cumulative key sequence lengths (int32); None for BSND.
        cmp_residual_k: Per-batch compression residual (original_k_len % cmp_ratio),
            int32. Affects the valid compressed-key range when ``cmp_ratio != 1``.
        block_table: Block table for PageAttention (optional).
        layout: Data layout — 'BSND' (default) or 'TND'. Used for both Q and K.
        sparse_mode: Sparse mask mode (benchmark ``mask_mode``); 0 = defaultMask.
        cmp_ratio: Key compression ratio (1 / 4 / 128).
        return_value: Whether to also output sparse_values (benchmark ``return_value``).

    Returns:
        tuple[Tensor, Tensor]: ``(sparse_indices, sparse_values)``.
    """
    return _platform.custom_ops.npu_lightning_indexer(
        query, key, weights, sparse_count,
        cu_seq_lens_q, cu_seq_lens_k, cmp_residual_k, block_table,
        layout, sparse_mode, cmp_ratio, return_value,
    )


def npu_sparse_flash_mla(
        query,
        *,
        ori_kv: Optional[Tensor] = None,
        cmp_kv: Optional[Tensor] = None,
        cmp_sparse_indices: Optional[Tensor] = None,
        cu_seq_lens_q: Optional[Tensor] = None,
        cu_seq_lens_ori_kv: Optional[Tensor] = None,
        cu_seq_lens_cmp_kv: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_ori_kv: Optional[Tensor] = None,
        seqused_cmp_kv: Optional[Tensor] = None,
        cmp_residual_kv: Optional[Tensor] = None,
        sinks: Optional[Tensor] = None,
        softmax_scale: float = 1.0,
        cmp_ratio: int = 1,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout: str = 'BSND',
        return_softmax_lse: bool = False,
):
    """MLA sparse attention (SparseFlashMla).

    .. warning::
        This is an experimental API that subject to change or deletion.

    Computes:  O = softmax(Q @ K̃^T · scale) @ Ṽ  where K̃ = Ṽ is derived from
    ``ori_kv``, ``cmp_kv`` and associated sparse indices.

    The single ``layout`` argument applies to both Q and KV (the DFunction
    splits it into ``layout_q`` / ``layout_kv`` when calling the kernel).  Only
    ``query`` is positional.

    Args:
        query: Query tensor.  shape ``(B, S1, N1, D)``, layout BSND.
            Must be contiguous.  dtype bfloat16/float16.
        ori_kv: Original KV tensor.  shape ``(B, S2, 1, D)``.  None when absent (band mode).
        cmp_kv: Compressed KV tensor.  shape ``(B, S_cmp, 1, D)``,
            where ``S_cmp = ceil(S2 / cmp_ratio)``.  None when absent.
        cmp_sparse_indices: Sparse indices for cmp_kv.
            shape ``(B, S1, 1, K)``, dtype int32.  None when cmp_ratio != 4.
        cu_seq_lens_q: Cumulative query sequence lengths (int32); required for TND.
        cu_seq_lens_ori_kv: Cumulative ori_kv sequence lengths (int32); for TND.
        cu_seq_lens_cmp_kv: Cumulative cmp_kv sequence lengths (int32); for TND.
        seqused_q: Used query sequence lengths (int32); None when absent. For BSND it
            marks the valid query rows per batch (truncation participates); for TND the
            query range is governed by cu_seq_lens_q so this is inert.
        seqused_ori_kv: Used ori_kv sequence lengths (int32); None when absent.
        seqused_cmp_kv: Used cmp_kv sequence lengths (int32); None when absent.
        cmp_residual_kv: Per-batch ori_kv-vs-cmp_ratio residual
            (``ori_len % cmp_ratio``), int32.
        sinks: Attention-sink tensor.  shape ``(N1,)``, dtype float32.
        softmax_scale: Softmax scaling factor (benchmark default 1.0).
        cmp_ratio: KV compression ratio (benchmark default 1).
        ori_mask_mode: Mask mode for q×ori_kv (benchmark default 4 = band).
        cmp_mask_mode: Mask mode for q×cmp_kv (benchmark default 3 = rightDownCausal).
        ori_win_left: Band-mask left window (benchmark default 127).
        ori_win_right: Band-mask right window (benchmark default 0).
        layout: Data layout for Q and KV — 'BSND' (default) or 'TND'.
        return_softmax_lse: Whether to also return softmax LSE.

    Returns:
        If ``return_softmax_lse=False``: Tensor ``attention_out``,
            shape ``(B, S1, N1, D)``.
        If ``return_softmax_lse=True``:
            tuple[Tensor, Tensor] ``(attention_out, softmax_lse)``.
    """
    result = _platform.custom_ops.npu_sparse_flash_mla(
        query, ori_kv, cmp_kv,
        cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
        None, cmp_sparse_indices, sinks,
        softmax_scale, cmp_ratio,
        ori_mask_mode, cmp_mask_mode,
        ori_win_left, ori_win_right,
        layout, layout,
        cmp_residual_kv, seqused_ori_kv, seqused_cmp_kv, seqused_q,
    )
    return result if return_softmax_lse else result[0]


def npu_sparse_flash_mla_grad(
        query,
        dout,
        attn_out,
        softmax_lse,
        *,
        ori_kv: Optional[Tensor] = None,
        cmp_kv: Optional[Tensor] = None,
        ori_sparse_indices: Optional[Tensor] = None,
        cmp_sparse_indices: Optional[Tensor] = None,
        cu_seq_lens_q: Optional[Tensor] = None,
        cu_seq_lens_ori_kv: Optional[Tensor] = None,
        cu_seq_lens_cmp_kv: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_ori_kv: Optional[Tensor] = None,
        seqused_cmp_kv: Optional[Tensor] = None,
        cmp_residual_kv: Optional[Tensor] = None,
        ori_topk_length: Optional[Tensor] = None,
        cmp_topk_length: Optional[Tensor] = None,
        sinks: Optional[Tensor] = None,
        softmax_scale: float = 1.0,
        cmp_ratio: int = 1,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout: str = 'BSND',
) -> Tuple:
    """MLA sparse-attention backward (SparseFlashMlaGrad), full 6-output form.

    .. warning::
        This is an experimental API that subject to change or deletion.

    Exposes the raw grad kernel so a network-defined custom backward can obtain
    the ``softmax_l1_norm`` outputs (the main-attention target distribution
    ``p``) alongside the input gradients, then feed ``p`` straight into
    :func:`npu_sparse_lightning_indexer_kl_loss_grad` within the same backward.
    Call this from inside your own autograd function's ``backward`` — it does not
    build an autograd graph itself.

    ``metadata`` is not exposed: the grad kernel asserts it is nullptr and
    re-derives its own tiling internally.

    Args:
        query: Query tensor used in the forward.  dtype bfloat16/float16.
        dout: Gradient of the attention output (``grad_attention_out``).
        attn_out: Attention output from the forward.
        softmax_lse: Softmax log-sum-exp from the forward.
        ori_kv: Original KV tensor; None when absent (band mode).
        cmp_kv: Compressed KV tensor; None when absent.
        ori_sparse_indices: Sparse indices for ori_kv; None = band mode.  Its
            shape drives ``ori_softmax_l1_norm``.
        cmp_sparse_indices: Sparse indices for cmp_kv (int32); None when
            cmp_ratio != 4.  Its shape drives ``cmp_softmax_l1_norm``.
        cu_seq_lens_q: Cumulative query seq lengths (int32); None for BSND.
        cu_seq_lens_ori_kv: Cumulative ori_kv seq lengths (int32); for TND.
        cu_seq_lens_cmp_kv: Cumulative cmp_kv seq lengths (int32); for TND.
        seqused_q: Used query seq lengths (int32); None when absent.
        seqused_ori_kv: Used ori_kv seq lengths (int32); None when absent.
        seqused_cmp_kv: Used cmp_kv seq lengths (int32); None when absent.
        cmp_residual_kv: Per-batch ori_kv-vs-cmp_ratio residual (int32);
            required for CFA/SCFA (cmp_ratio != 1) with cmp_mask_mode=3.
        ori_topk_length: Optional ori top-k length; None when absent.
        cmp_topk_length: Optional cmp top-k length; None when absent.
        sinks: Attention-sink tensor (float32); None when absent.
        softmax_scale: Softmax scaling factor (must match the forward).
        cmp_ratio: KV compression ratio (must match the forward).
        ori_mask_mode: Mask mode for q×ori_kv (default 4 = band).
        cmp_mask_mode: Mask mode for q×cmp_kv (default 3 = rightDownCausal).
        ori_win_left: Band-mask left window (default 127).
        ori_win_right: Band-mask right window (default 0).
        layout: Data layout for Q and KV — 'BSND' (default) or 'TND'.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            ``(d_query, d_ori_kv, d_cmp_kv, d_sinks, ori_softmax_l1_norm,
            cmp_softmax_l1_norm)``.  ``ori/cmp_softmax_l1_norm`` are float32 and
            share the shape of ``ori/cmp_sparse_indices``; each is the
            ``reduceG(softmax)/G`` main-attention distribution over the selected
            tokens.  Absent branches yield empty tensors.
    """
    return _platform.custom_ops.npu_sparse_flash_mla_grad(
        query, dout, attn_out, softmax_lse,
        ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
        seqused_q, seqused_ori_kv, seqused_cmp_kv,
        cmp_residual_kv, ori_topk_length, cmp_topk_length,
        sinks, None,  # metadata=None → grad kernel self-derives its own tiling
        softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
        ori_win_left, ori_win_right, layout, layout,
    )


def npu_sparse_lightning_indexer_kl_loss_grad(
        query,
        key,
        weights,
        sparse_indices,
        attn_softmax_l1_norm,
        *,
        cu_seq_lens_q: Optional[Tensor] = None,
        cu_seq_lens_k: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_k: Optional[Tensor] = None,
        cmp_residual_k: Optional[Tensor] = None,
        layout: str = 'BSND',
        mask_mode: int = 3,
        cmp_ratio: int = 1,
) -> Tuple:
    """Compute backward gradients for sparse Lightning Indexer KL loss.

    .. warning::
        This is an experimental API that subject to change or deletion.

    The main-attention target distribution is supplied directly via
    ``attn_softmax_l1_norm`` (e.g. the ``softmax_l1_norm`` output of
    :func:`npu_sparse_flash_mla_grad`); the kernel neither recomputes the main
    attention nor outputs a loss, and returns the indexer-branch softmax as
    ``softmax_out``.

    Args:
        query: Lightning Indexer query input (q̃). Must be contiguous.
            layout='BSND': shape ``(B, S1, N_qi, D_qi)``;
            layout='TND': shape ``(T1, N_qi, D_qi)``. dtype bfloat16/float16.
        key: Lightning Indexer key input (k̃). Must be contiguous.
            layout='BSND': shape ``(B, S2, N_ki, D_ki)``;
            layout='TND': shape ``(T2, N_ki, D_ki)``. dtype bfloat16/float16.
        weights: Weight coefficient (W). Same dtype as query.
        sparse_indices: Sorted token indices. shape ``(B, S1, 1, K)``, dtype int32.
        attn_softmax_l1_norm: Main-attention target distribution p (float32),
            pre-computed by the main-attention branch (e.g. the
            ``softmax_l1_norm`` output of ``npu_sparse_flash_mla`` backward).
        cu_seq_lens_q: Cumulative query sequence lengths. shape ``(B+1,)``,
            dtype int32; None for BSND layout.
        cu_seq_lens_k: Cumulative key sequence lengths. shape ``(B+1,)``,
            dtype int32; None for BSND layout.
        seqused_q: Used query sequence lengths; None when absent.
        seqused_k: Used key sequence lengths; None when absent.
        cmp_residual_k: Optional compressed-KV residual.
        layout: Data layout format — 'TND' (default) or 'BSND'.
        mask_mode: Sparse mask mode (only 3 supported).
        cmp_ratio: KV compression ratio.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            ``(d_query, d_key, d_weights, softmax_out)``.
    """
    return _platform.custom_ops.npu_sparse_lightning_indexer_kl_loss_grad(
        query, key, weights, sparse_indices, attn_softmax_l1_norm,
        cu_seq_lens_q, cu_seq_lens_k, seqused_q, seqused_k, cmp_residual_k,
        layout, mask_mode, cmp_ratio,
    )
