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
"""MindSpore custom kernel implementations and DFunction wrappers."""
import importlib
import os
import sys
from typing import NamedTuple, Optional

import mindspore as ms # pylint: disable=C0415

from hyper_parallel.core.shard.dfunction import DFunction


_CC_DIR = os.path.dirname(os.path.abspath(__file__))
_MS_EXTENSION_NAME = "hyper_parallel_custom_ops_ms"
_BUILD_LIB = os.path.join(_CC_DIR, "build", "lib")

if _BUILD_LIB not in sys.path:
    sys.path.insert(0, _BUILD_LIB)

_CUSTOM_OP_SOURCES = [
    os.path.join(_CC_DIR, "module.cc"),
    os.path.join(_CC_DIR, "dense_lightning_indexer_grad_kl_loss.cc"),
    os.path.join(_CC_DIR, "dense_lightning_indexer_softmax_lse.cc"),
    os.path.join(_CC_DIR, "sparse_lightning_indexer_grad_kl_loss.cc"),
    os.path.join(_CC_DIR, "mhc_post.cc"),
    os.path.join(_CC_DIR, "mhc_post_backward.cc"),
    os.path.join(_CC_DIR, "mhc_pre_sinkhorn.cc"),
    os.path.join(_CC_DIR, "mhc_pre_sinkhorn_backward.cc"),
    os.path.join(_CC_DIR, "mhc_pre_clamp_sinkhorn.cc"),
    os.path.join(_CC_DIR, "mhc_pre_clamp_sinkhorn_backward.cc"),
    os.path.join(_CC_DIR, "lightning_indexer_v2.cc"),
    os.path.join(_CC_DIR, "flash_attention_varlen_v4.cc"),
    os.path.join(_CC_DIR, "sparse_flash_mla.cc"),
    os.path.join(_CC_DIR, "sparse_flash_mla_grad.cc"),
    os.path.join(_CC_DIR, "sparse_lightning_indexer_kl_loss_grad.cc"),
    os.path.join(_CC_DIR, "chunk_kda_fwd.cc"),
    os.path.join(_CC_DIR, "chunk_kda_bwd.cc"),
]


def _build_custom_ops():
    return ms.ops.CustomOpBuilder(
        _MS_EXTENSION_NAME,
        _CUSTOM_OP_SOURCES,
        backend="Ascend",
    ).load()


try:
    _custom_ops = importlib.import_module(_MS_EXTENSION_NAME)
except ImportError:
    # Source-tree development: .so not pre-built; JIT-compile from local .cc files.
    _custom_ops = _build_custom_ops()
else:
    # Rebuild stale source-tree extensions that predate newly added symbols.
    if not hasattr(_custom_ops, "npu_mhc_pre_clamp_sinkhorn") or not hasattr(_custom_ops, "npu_chunk_kda_bwd"):
        _custom_ops = _build_custom_ops()


def _ensure_contiguous(*tensors):
    """Ensure all tensors are contiguous (no-op if already contiguous)."""
    return tuple(t.contiguous() if not t.is_contiguous() else t for t in tensors)


def _to_list_int64(val):
    """Convert Tensor(int32) to List[int64] for aclnn kernel consumption."""
    if isinstance(val, ms.Tensor):
        return val.asnumpy().astype("int64").tolist()
    return val


class _SparseFlashMlaSavedTensors(NamedTuple):
    """Tensors retained by sparse flash MLA for its backward pass."""

    query: ms.Tensor
    ori_kv: Optional[ms.Tensor]
    cmp_kv: Optional[ms.Tensor]
    sinks: Optional[ms.Tensor]
    ori_sparse_indices: Optional[ms.Tensor]
    cmp_sparse_indices: Optional[ms.Tensor]
    cu_seq_lens_q: Optional[ms.Tensor]
    cu_seq_lens_ori_kv: Optional[ms.Tensor]
    cu_seq_lens_cmp_kv: Optional[ms.Tensor]
    attention_out: ms.Tensor
    softmax_lse: ms.Tensor
    cmp_residual_kv: Optional[ms.Tensor]


def _restore_sparse_flash_mla_saved_tensors(ctx):
    """Restore optional tensors from the compact autograd saved-tensor sequence."""
    saved_tensors = iter(ctx.saved_tensors)
    presence = (
        True,
        ctx.has_ori_kv,
        ctx.has_cmp_kv,
        ctx.has_sinks,
        ctx.has_ori_sparse,
        ctx.has_cmp_sparse,
        ctx.has_cu_q,
        ctx.has_cu_ori_kv,
        ctx.has_cu_cmp_kv,
        True,
        True,
        ctx.has_cmp_residual,
    )
    values = (next(saved_tensors) if is_present else None for is_present in presence)
    return _SparseFlashMlaSavedTensors(*values)


class NpuDenseLightningIndexerSoftmaxLseDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for npu_dense_lightning_indexer_softmax_lse on MindSpore.

    Routes plain-tensor calls directly to the MindSpore custom kernel, and
    DTensor calls through the distributed dispatch framework using the
    registered DistributedOp with the same op_name.

    All 11 forward arguments after ``ctx`` are positional to stay compatible
    with both MindSpore autograd function conventions.

    No backward is defined because the operator does not require gradients.
    """

    _op_name = "npu_dense_lightning_indexer_softmax_lse"

    @staticmethod
    def forward(ctx, query_index, key_index, weights,
                actual_seq_qlen, actual_seq_klen,
                layout, sparse_mode, pre_tokens, next_tokens):
        """Forward pass: delegates to the MindSpore Ascend custom kernel.

        Args:
            ctx: Autograd context.
            query_index: Lightning Indexer query input (Q̃).
            key_index: Lightning Indexer key input (K̃).
            weights: Lightning Indexer weight coefficient (W).
            actual_seq_qlen: Cumulative query sequence lengths; None for BSND.
            actual_seq_klen: Cumulative key sequence lengths; None for BSND.
            layout: Data layout format, 'BSND' or 'TND'.
            sparse_mode: Sparse computation mode (only mode 3 supported).
            pre_tokens: Number of preceding tokens for sparse attention.
            next_tokens: Number of following tokens for sparse attention.

        Returns:
            tuple[Tensor, Tensor]: (softmax_max_index, softmax_sum_index), both float32.
        """
        return _custom_ops.npu_dense_lightning_indexer_softmax_lse(
            query_index, key_index, weights,
            _to_list_int64(actual_seq_qlen), _to_list_int64(actual_seq_klen),
            layout, sparse_mode, pre_tokens, next_tokens,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        """No-op backward — this operator does not require gradients."""
        return (None,) * 9


class NpuFlashAttentionVarLenV4DFunction(DFunction):  # pylint: disable=W0221
    """TND varlen FlashAttention through aclnn V4, differentiable.

    Unlike the indexer kernels around it this one **does** need a backward: the dense stage's
    teacher attention sits in the graph. The backward calls
    ``aclnnFlashAttentionUnpaddingScoreGradV4`` with the *same* softmax layout the forward
    produced -- ``softmaxInLayout`` and ``softmaxOutLayout`` must agree, and disagreeing is
    silent (the shapes match either way, only the numbers come out wrong).

    ``tnd_softmax_out=True`` asks the kernel for TND-ordered statistics, which is the whole
    point: the consumer then only transposes and needs no knowledge of where this chunk sits
    in the global sequence, so CP fold / head-tail balance wrappers can wrap this freely.
    """

    _op_name = "npu_flash_attention_varlen_v4"

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, actual_seq_qlen, actual_seq_kvlen,
                scale_value, head_num, sparse_mode, pre_tokens, next_tokens, inner_precise,
                tnd_softmax_out):
        """Forward: returns ``(softmax_max, softmax_sum, attention_out)``."""
        q_len = _to_list_int64(actual_seq_qlen)
        kv_len = _to_list_int64(actual_seq_kvlen)
        outs = _custom_ops.npu_flash_attention_varlen_v4(
            query, key, value, atten_mask, q_len, kv_len, scale_value, head_num,
            sparse_mode, pre_tokens, next_tokens, inner_precise, tnd_softmax_out,
        )
        softmax_max, softmax_sum, attention_out = outs
        ctx.save_for_backward(query, key, value, atten_mask, softmax_max, softmax_sum, attention_out)
        ctx.fa_args = (q_len, kv_len, scale_value, head_num, sparse_mode, pre_tokens, next_tokens,
                       inner_precise, tnd_softmax_out)
        return softmax_max, softmax_sum, attention_out

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward through the attention output only.

        The softmax statistics are consumed by the indexer loss, which produces its own
        gradients for the indexer parameters and routes nothing back here -- so their
        incoming gradients are ignored, mirroring what the stock FlashAttentionScore bprop
        does with them.
        """
        query, key, value, atten_mask, softmax_max, softmax_sum, attention_out = ctx.saved_tensors
        (q_len, kv_len, scale_value, head_num, sparse_mode, pre_tokens, next_tokens,
         inner_precise, tnd_softmax) = ctx.fa_args
        dy = grad_outputs[2]
        if dy is None:
            return (None,) * 13
        dq, dk, dv = _custom_ops.npu_flash_attention_varlen_grad_v4(
            query, key, value, dy, atten_mask, softmax_max, softmax_sum, attention_out,
            q_len, kv_len, scale_value, head_num, sparse_mode, pre_tokens, next_tokens,
            inner_precise, tnd_softmax,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


class NpuDenseLightningIndexerGradKlLossDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for npu_dense_lightning_indexer_grad_kl_loss on MindSpore.

    Routes plain-tensor calls directly to the MindSpore custom kernel, and
    DTensor calls through the distributed dispatch framework using the
    registered DistributedOp with the same op_name.

    All 18 forward arguments after ``ctx`` are positional to stay compatible
    with both MindSpore autograd function conventions.
    """

    _op_name = "npu_dense_lightning_indexer_grad_kl_loss"

    @staticmethod
    def forward(ctx, query, key, query_index, key_index, weights,
                softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
                scale_value, query_rope, key_rope,
                actual_seq_qlen, actual_seq_klen,
                layout, sparse_mode, pre_tokens, next_tokens):
        """Forward pass: delegates to the MindSpore Ascend custom kernel.

        Args:
            ctx: Autograd context.
            query: Main attention query (Q). dtype bfloat16/float16.
            key: Main attention key (K). dtype bfloat16/float16.
            query_index: Lightning Indexer query input (Q̃). dtype bfloat16/float16.
            key_index: Lightning Indexer key input (K̃). dtype bfloat16/float16.
            weights: Lightning Indexer weight coefficient (W).
            softmax_max: Attention softmax max values. dtype float32.
            softmax_sum: Attention softmax sum values. dtype float32.
            softmax_max_index: Index attention softmax max (from softmax_lse). dtype float32.
            softmax_sum_index: Index attention softmax sum (from softmax_lse). dtype float32.
            scale_value: Scaling factor. dtype float32.
            query_rope: Optional MLA query rope tensor.
            key_rope: Optional MLA key rope tensor.
            actual_seq_qlen: Cumulative query sequence lengths; None for BSND.
            actual_seq_klen: Cumulative key sequence lengths; None for BSND.
            layout: Data layout format, 'BSND' or 'TND'.
            sparse_mode: Sparse computation mode (only mode 3 supported).
            pre_tokens: Number of preceding tokens for sparse attention.
            next_tokens: Number of following tokens for sparse attention.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                (d_query_index, d_key_index, d_weights, loss).
        """
        result = _custom_ops.npu_dense_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights,
            softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
            scale_value, query_rope, key_rope,
            _to_list_int64(actual_seq_qlen), _to_list_int64(actual_seq_klen),
            layout, sparse_mode, pre_tokens, next_tokens,
        )
        ctx.save_for_backward(result[0], result[1], result[2])
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        d_query_index, d_key_index, d_weights = _ensure_contiguous(*ctx.saved_tensors)
        return (None, None, d_query_index, d_key_index, d_weights,
                None, None, None, None, None, None, None, None, None, None, None, None, None)


class NpuSparseLightningIndexerGradKlLossDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for npu_sparse_lightning_indexer_grad_kl_loss on MindSpore.

    Routes plain-tensor calls directly to the MindSpore custom kernel, and
    DTensor calls through the distributed dispatch framework using the
    registered DistributedOp with the same op_name.

    All 17 forward arguments after ``ctx`` are positional to stay compatible
    with both MindSpore autograd function conventions.
    """

    _op_name = "npu_sparse_lightning_indexer_grad_kl_loss"

    @staticmethod
    def forward(ctx, query, key, query_index, key_index, weights,
                sparse_indices, softmax_max, softmax_sum, scale_value,
                query_rope, key_rope,
                actual_seq_qlen, actual_seq_klen,
                layout, sparse_mode, pre_tokens, next_tokens):
        """Forward pass: delegates to the MindSpore Ascend custom kernel.

        Args:
            ctx: Autograd context.
            query: Main attention query (q_t). dtype bfloat16/float16.
            key: Main attention key (K_t). dtype bfloat16/float16.
            query_index: Lightning Indexer query input (q̃_t). dtype bfloat16/float16.
            key_index: Lightning Indexer key input (K̃_t). dtype bfloat16/float16.
            weights: Lightning Indexer weight coefficient (W_t).
            sparse_indices: Sorted token indices for key/key_index. dtype bfloat16/float16.
            softmax_max: Attention softmax max values.
            softmax_sum: Attention softmax sum values.
            scale_value: Scaling factor. dtype float.
            query_rope: Optional MLA query rope tensor.
            key_rope: Optional MLA key rope tensor.
            actual_seq_qlen: Cumulative query sequence lengths; None for BSND.
            actual_seq_klen: Cumulative key sequence lengths; None for BSND.
            layout: Data layout format, 'BSND' or 'TND'.
            sparse_mode: Sparse computation mode (only mode 3 supported).
            pre_tokens: Number of preceding tokens for sparse attention.
            next_tokens: Number of following tokens for sparse attention.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                (d_query_index, d_key_index, d_weights, loss).
        """
        result = _custom_ops.npu_sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights,
            sparse_indices, softmax_max, softmax_sum, scale_value,
            query_rope, key_rope,
            _to_list_int64(actual_seq_qlen), _to_list_int64(actual_seq_klen),
            layout, sparse_mode, pre_tokens, next_tokens,
        )
        ctx.save_for_backward(result[0], result[1], result[2])
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        d_query_index, d_key_index, d_weights = _ensure_contiguous(*ctx.saved_tensors)
        return (None, None, d_query_index, d_key_index, d_weights,
                None, None, None, None, None, None, None, None, None, None, None, None)


class NpuMhcPostDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for npu_mhc_post on MindSpore.

    Routes plain-tensor calls directly to the MindSpore custom kernel, and
    DTensor calls through the distributed dispatch framework using the
    registered DistributedOp with the same op_name.

    All 4 forward arguments after ``ctx`` are positional to stay compatible
    with both MindSpore autograd function conventions.
    """

    _op_name = "npu_mhc_post"

    @staticmethod
    def forward(ctx, x, h_res, h_out, h_post):
        """Forward pass: delegates to the MindSpore Ascend custom kernel.

        Args:
            ctx: Autograd context.
            x: Input tensor of shape [B,S,N,D] or [T,N,D]. dtype bfloat16/float16.
            h_res: mHC h_res transformation matrix. dtype float32.
            h_out: Attention/MLP layer output. dtype bfloat16/float16.
            h_post: mHC h_post transformation matrix. dtype float32.

        Returns:
            Tensor: Output tensor with same shape and dtype as x.
        """
        ctx.save_for_backward(x, h_res, h_out, h_post)
        return _custom_ops.npu_mhc_post(x, h_res, h_out, h_post)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass: calls npu_mhc_post_backward kernel.

        Args:
            ctx: Autograd context.
            grad_outputs: Upstream gradients; grad_outputs[0] is grad_y.

        Returns:
            tuple: (grad_x, grad_h_res, grad_h_out, grad_h_post).
        """
        x, h_res, h_out, h_post = ctx.saved_tensors
        grad_y, x, h_res, h_out, h_post = _ensure_contiguous(
            grad_outputs[0], x, h_res, h_out, h_post)
        grads = _custom_ops.npu_mhc_post_backward(
            grad_y, x, h_res, h_out, h_post)
        return grads[0], grads[1], grads[2], grads[3]


class NpuMhcPreSinkhornDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for npu_mhc_pre_sinkhorn on MindSpore.

    Routes plain-tensor calls directly to the MindSpore custom kernel, and
    DTensor calls through the distributed dispatch framework using the
    registered DistributedOp with the same op_name.

    All 9 forward arguments after ``ctx`` are positional to stay compatible
    with both MindSpore autograd function conventions.
    """

    _op_name = "npu_mhc_pre_sinkhorn"

    @staticmethod
    def forward(ctx, x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag):
        """Forward pass: delegates to the MindSpore Ascend custom kernel.

        Args:
            ctx: Autograd context.
            x: Input tensor. dtype bfloat16/float16.
            phi: mHC parameter matrix. dtype float32.
            alpha: mHC scaling parameters. dtype float32.
            bias: mHC bias parameters. dtype float32.
            hc_mult: HC dimension size (currently only 4 supported).
            num_iters: Sinkhorn iteration count.
            hc_eps: H_pre sigmoid eps parameter.
            norm_eps: RmsNorm eps parameter.
            out_flag: Whether to output intermediate gradients.

        Returns:
            tuple[Tensor, ...]: 8 output tensors
                (h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out).
        """
        result = _custom_ops.npu_mhc_pre_sinkhorn(
            x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag
        )
        _, _, _, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = result
        ctx.save_for_backward(x, phi, alpha, bias,
                              h_pre, hc_before_norm, inv_rms, sum_out, norm_out)
        ctx.hc_eps = hc_eps
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass: calls npu_mhc_pre_sinkhorn_backward kernel.

        Args:
            ctx: Autograd context.
            grad_outputs: Upstream gradients for the 8 forward outputs.
                grad_outputs[0]=grad_h_in, [1]=grad_h_post, [2]=grad_h_res;
                [3..7] correspond to saved intermediates and are None.

        Returns:
            tuple: (grad_x, grad_phi, grad_alpha, grad_bias, None×5) —
                gradients for the 9 forward inputs.
        """
        x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = ctx.saved_tensors
        (grad_h_in, grad_h_post, grad_h_res,
         x, phi, alpha, bias,
         h_pre, hc_before_norm, inv_rms, sum_out, norm_out) = _ensure_contiguous(
            grad_outputs[0], grad_outputs[1], grad_outputs[2],
            x, phi, alpha, bias,
            h_pre, hc_before_norm, inv_rms, sum_out, norm_out)
        b, s, n = grad_h_post.shape
        grad_h_res = grad_h_res.reshape(b, s, n, n)
        grads = _custom_ops.npu_mhc_pre_sinkhorn_backward(
            grad_h_in, grad_h_post, grad_h_res,
            x, phi, alpha, bias,
            h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
            ctx.hc_eps)
        return grads[0], grads[1], grads[2], grads[3], None, None, None, None, None


_MHC_PRE_CLAMP_NONE_GRADS = (None,) * 7


class NpuMhcPreClampSinkhornDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for npu_mhc_pre_clamp_sinkhorn on MindSpore.

    This matches the static-graph aclnnMhcPreClampSinkhorn integration:
    forward has 11 arguments and returns 9 tensors, and backward consumes
    h_res_logits plus clamp_min/clamp_max.
    """

    _op_name = "npu_mhc_pre_clamp_sinkhorn"

    @staticmethod
    def forward(ctx, x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps,
                out_flag, clamp_min, clamp_max):
        """Forward pass: delegates to the clamp-enabled Ascend custom kernel."""
        result = _custom_ops.npu_mhc_pre_clamp_sinkhorn(
            x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps,
            out_flag, clamp_min, clamp_max
        )
        _, _, _, h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits = result
        ctx.save_for_backward(x, phi, alpha, bias,
                              h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits)
        ctx.hc_eps = hc_eps
        ctx.clamp_min = clamp_min
        ctx.clamp_max = clamp_max
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass: calls npu_mhc_pre_clamp_sinkhorn_backward kernel."""
        x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits = ctx.saved_tensors
        (grad_h_in, grad_h_post, grad_h_res,
         x, phi, alpha, bias,
         h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits) = _ensure_contiguous(
            grad_outputs[0], grad_outputs[1], grad_outputs[2],
            x, phi, alpha, bias,
            h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits)
        n = grad_h_post.shape[-1]
        grad_h_res = ms.ops.reshape(grad_h_res, tuple(grad_h_res.shape[:-1]) + (n, n))

        grads = _custom_ops.npu_mhc_pre_clamp_sinkhorn_backward(
            grad_h_in, grad_h_post, grad_h_res,
            x, phi, alpha, bias,
            h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits,
            ctx.hc_eps, ctx.clamp_min, ctx.clamp_max)
        return tuple(grads[:4]) + _MHC_PRE_CLAMP_NONE_GRADS


class NpuLightningIndexerDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for npu_lightning_indexer.

    The underlying kernel handles all cmp_ratio values (1 / 4 / 128) directly.
    Forward-only: indexer gradients are produced by the network's explicit
    ``sparse_lightning_indexer_kl_loss_grad`` call.

    Signature mirrors the torch-extension ``lightning_indexer`` benchmark:
    positional ``(query, key, weights, sparse_count)`` (``sparse_count`` is
    benchmark ``topk``); the two layouts are merged into a single ``layout``
    (the kernel is fed identical ``layout_q`` / ``layout_k``).
    """

    _op_name = "npu_lightning_indexer"

    @staticmethod
    def forward(ctx, query, key, weights, sparse_count,
                cu_seq_lens_q=None, cu_seq_lens_k=None, cmp_residual_k=None,
                block_table=None, layout="BSND",
                sparse_mode=0, cmp_ratio=1, return_value=False):
        """Forward pass: call the custom kernel for all cmp_ratios.

        Remaining benchmark kwargs (seqused_q/k, output_idx_offset, metadata,
        max_seqlen_q) are presently unused by the external API and pinned to
        ``None`` / ``-1``.

        Returns:
            tuple[Tensor, Tensor]: (sparse_indices, sparse_values).
        """
        return _custom_ops.npu_lightning_indexer_v2(
            query, key, weights, sparse_count,
            cu_seq_lens_q, cu_seq_lens_k,
            None, None, cmp_residual_k, block_table, None, None, -1,
            layout, layout, sparse_mode, cmp_ratio, return_value)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """No-op backward — indexer gradients come from kl_loss_grad."""
        return (None,) * 12


class NpuSparseFlashMlaDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for the MLA sparse-attention kernel.

    Forward runs ``npu_sparse_flash_mla`` (the kernel derives its metadata from
    the tensor shapes internally); backward runs ``npu_sparse_flash_mla_grad``.
    """

    _op_name = "npu_sparse_flash_mla"

    @staticmethod
    def forward(ctx,  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
                query, ori_kv, cmp_kv,
                cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
                ori_sparse_indices, cmp_sparse_indices, sinks,
                softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
                ori_win_left, ori_win_right,
                layout_q, layout_kv,
                cmp_residual_kv=None, seqused_ori_kv=None, seqused_cmp_kv=None,
                seqused_q=None):
        """Forward pass: runs MLA sparse attention (metadata computed in-kernel).

        Args:
            ctx: Autograd context.
            query: Query tensor.  dtype bfloat16/float16.
            ori_kv: Original KV tensor; None when absent.
            cmp_kv: Compressed KV tensor; None when absent.
            cu_seq_lens_q: Cumulative query seq lengths (TND); None for BSND.
            cu_seq_lens_ori_kv: Cumulative ori_kv seq lengths; None for PA_ND.
            cu_seq_lens_cmp_kv: Cumulative cmp_kv seq lengths; None for PA_ND.
            ori_sparse_indices: Sparse indices for ori_kv; None = band mode.
            cmp_sparse_indices: Sparse indices for cmp_kv (int32 Tensor).
            sinks: Attention-sink tensor (float32); None when absent.
            softmax_scale: Softmax scaling factor (float).
            cmp_ratio: KV compression ratio (int).
            ori_mask_mode: Mask mode for q×ori_kv (default 4=band).
            cmp_mask_mode: Mask mode for q×cmp_kv (default 3=rightDownCausal).
            ori_win_left: Band-mask left window (default 127).
            ori_win_right: Band-mask right window (default 0).
            layout_q: Q data layout — 'BSND' or 'TND'.
            layout_kv: KV data layout — 'PA_ND' or 'BSND'.

        Returns:
            tuple[Tensor, Tensor]: (attention_out, softmax_lse).
        """
        if cmp_ratio != 4:
            cmp_sparse_indices = None

        # The kernel computes its metadata internally.  topk_value_mode=1;
        # return_softmax_lse is forced True internally so the backward always
        # receives a valid LSE (a stale/zero LSE makes the grad kernel explode);
        # the external return value is gated separately by the wrapper's own
        # return_softmax_lse flag, independent of this.
        result = _custom_ops.npu_sparse_flash_mla(
            query, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
            None, None,                       # ori_block_table, cmp_block_table
            cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
            seqused_q, seqused_ori_kv, seqused_cmp_kv,   # seq_used_q, seq_used_ori_kv, seq_used_cmp_kv
            cmp_residual_kv, None, None,            # cmp_residual_kv, ori_topk_length, cmp_topk_length
            sinks,
            softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
            ori_win_left, ori_win_right, layout_q, layout_kv, 1, True,
        )
        attention_out, softmax_lse = result[0], result[1]

        ctx.has_ori_kv = ori_kv is not None
        ctx.has_cmp_kv = cmp_kv is not None
        ctx.has_sinks = sinks is not None
        ctx.has_ori_sparse = ori_sparse_indices is not None
        ctx.has_cmp_sparse = cmp_sparse_indices is not None
        ctx.has_cu_q = cu_seq_lens_q is not None
        ctx.has_cu_ori_kv = cu_seq_lens_ori_kv is not None
        ctx.has_cu_cmp_kv = cu_seq_lens_cmp_kv is not None
        ctx.has_cmp_residual = cmp_residual_kv is not None
        # metadata is NOT saved for backward: the grad kernel asserts metadata
        # must be nullptr and re-derives its own tiling internally.  cmp_residual_kv
        # IS saved — the grad kernel requires it for CFA/SCFA with cmp_mask_mode=3.
        saved_tensors = [
            query, ori_kv, cmp_kv, sinks, ori_sparse_indices, cmp_sparse_indices,
            cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
            attention_out, softmax_lse, cmp_residual_kv,
        ]
        ctx.save_for_backward(*[tensor for tensor in saved_tensors if tensor is not None])
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        ctx.layout_kv = layout_kv
        return attention_out, softmax_lse

    @staticmethod
    def backward(ctx, grad_attention_out, grad_softmax_lse):  # pylint: disable=unused-argument
        """Backward pass: calls npu_sparse_flash_mla_grad kernel."""
        state = _restore_sparse_flash_mla_saved_tensors(ctx)
        # metadata MUST be None: the grad kernel asserts it is nullptr and
        # re-derives tiling internally.  cmp_residual_kv is passed through —
        # required for CFA/SCFA (cmp_ratio!=1) with cmp_mask_mode=3.
        grads = _custom_ops.npu_sparse_flash_mla_grad(
            state.query, grad_attention_out, state.attention_out, state.softmax_lse,
            state.ori_kv, state.cmp_kv, state.ori_sparse_indices, state.cmp_sparse_indices,
            state.cu_seq_lens_q, state.cu_seq_lens_ori_kv, state.cu_seq_lens_cmp_kv,
            None, None, None,                 # seq_used_q, seq_used_ori_kv, seq_used_cmp_kv
            state.cmp_residual_kv, None, None,  # cmp_residual_kv, ori_topk_length, cmp_topk_length
            state.sinks, None,                 # sinks, metadata(None → grad kernel self-derives)
            ctx.softmax_scale, ctx.cmp_ratio, ctx.ori_mask_mode, ctx.cmp_mask_mode,
            ctx.ori_win_left, ctx.ori_win_right, ctx.layout_q, ctx.layout_kv,
        )
        d_query = grads[0]
        d_ori_kv = grads[1] if state.ori_kv is not None else None
        d_cmp_kv = grads[2] if state.cmp_kv is not None else None
        d_sinks = grads[3] if state.sinks is not None else None
        # grads[4], grads[5] = ori/cmp_softmax_l1_norm — discarded here.
        # 21 positional forward args (ctx excluded):
        # query, ori_kv, cmp_kv, cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
        # ori_sparse_indices, cmp_sparse_indices, sinks,
        # softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
        # layout_q, layout_kv, cmp_residual_kv, seqused_ori_kv, seqused_cmp_kv, seqused_q
        return (d_query, d_ori_kv, d_cmp_kv,
                None, None, None,
                None, None, d_sinks,
                None, None, None, None, None, None,
                None, None, None, None, None, None)


def npu_sparse_flash_mla_grad(*args, **kwargs):
    """Raw ``sparse_flash_mla_grad`` kernel passthrough (stateless, no autograd).

    Runs the same backward kernel as ``NpuSparseFlashMlaDFunction.backward``, but
    returns its full 6-tuple so a network-defined custom backward can also
    consume ``ori/cmp_softmax_l1_norm`` (the main-attention target distribution
    ``p`` for the Lightning-Indexer KL loss).  Intended to be called from inside
    another custom function's ``backward`` (autograd already off); it builds no
    graph.  ``metadata`` must be ``None`` — the grad kernel re-derives its own
    tiling internally.

    Returns:
        tuple[Tensor, ...]: ``(d_query, d_ori_kv, d_cmp_kv, d_sinks,
        ori_softmax_l1_norm, cmp_softmax_l1_norm)``.
    """
    return _custom_ops.npu_sparse_flash_mla_grad(*args, **kwargs)


class NpuSparseLightningIndexerKlLossGradDFunction(DFunction):  # pylint: disable=W0221
    """DFunction wrapper for ``npu_sparse_lightning_indexer_kl_loss_grad``.

    The kernel takes the pre-computed main-attention target distribution
    ``attn_softmax_l1_norm`` and produces ``(dq, dk, dw, softmax_out)`` — the
    gradients w.r.t. ``query``/``key``/``weights`` plus the indexer-branch
    softmax; it neither recomputes the main attention nor outputs a loss.
    Metadata is computed inside the kernel from the tensor shapes.  Backward
    propagates ``(dq, dk, dw)`` to those inputs.
    """

    _op_name = "npu_sparse_lightning_indexer_kl_loss_grad"

    @staticmethod
    def forward(ctx, query, key, weights, sparse_indices, attn_softmax_l1_norm,
                cu_seq_lens_q, cu_seq_lens_k, seqused_q, seqused_k, cmp_residual_k,
                layout, mask_mode, cmp_ratio):
        """Forward pass: runs the KL-loss grad kernel (metadata computed in-kernel).

        Args:
            ctx: Autograd context.
            query: Lightning Indexer query (q̃). dtype bfloat16/float16.
            key: Lightning Indexer key (k̃). dtype bfloat16/float16.
            weights: Lightning Indexer weight coefficient (w).
            sparse_indices: Sorted token indices (int32).
            attn_softmax_l1_norm: Main-attention target distribution p (float32),
                pre-computed by the main-attention branch.
            cu_seq_lens_q: Cumulative query sequence lengths; None for BSND.
            cu_seq_lens_k: Cumulative key sequence lengths; None for BSND.
            seqused_q: Used query sequence lengths; None when absent.
            seqused_k: Used key sequence lengths; None when absent.
            cmp_residual_k: Optional compressed-KV residual.
            layout: Data layout format — 'BSND' or 'TND'.
            mask_mode: Sparse mask mode (only 3 supported).
            cmp_ratio: KV compression ratio.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                (d_query, d_key, d_weights, softmax_out).
        """
        # The kernel computes its metadata internally.
        result = _custom_ops.npu_sparse_lightning_indexer_kl_loss_grad(
            query, key, weights, sparse_indices, attn_softmax_l1_norm,
            cu_seq_lens_q, cu_seq_lens_k, seqused_q, seqused_k, cmp_residual_k,
            layout, layout, mask_mode, cmp_ratio,
        )
        ctx.save_for_backward(result[0], result[1], result[2])
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward: propagate the fused gradients to query/key/weights inputs."""
        d_query, d_key, d_weights = _ensure_contiguous(*ctx.saved_tensors)
        # 13 positional forward args: query, key, weights, sparse_indices,
        # attn_softmax_l1_norm, cu_seq_lens_q, cu_seq_lens_k, seqused_q, seqused_k,
        # cmp_residual_k, layout, mask_mode, cmp_ratio.
        return (d_query, d_key, d_weights,
                None, None, None, None, None, None, None, None, None, None)


def _canonical_chunk_indices(cu_seqlens, chunk_size):
    """Build flattened ``(sequence, local_chunk)`` metadata."""
    if cu_seqlens is None:
        return None
    indices = []
    for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for chunk in range((int(end) - int(begin) + chunk_size - 1) // chunk_size):
            indices.extend((sequence, chunk))
    return indices


def _canonicalize_chunk_kda_inputs(q, k, v, g, beta, layout):
    """Convert public inputs to dense BNSD or packed NTD."""
    if layout in ("BNSD", "NTD"):
        return q, k, v, g, beta
    if layout == "BSND":
        return (
            ms.ops.transpose(q, (0, 2, 1, 3)).contiguous(),
            ms.ops.transpose(k, (0, 2, 1, 3)).contiguous(),
            ms.ops.transpose(v, (0, 2, 1, 3)).contiguous(),
            ms.ops.transpose(g, (0, 2, 1, 3)).contiguous(),
            ms.ops.transpose(beta, (0, 2, 1)).contiguous(),
        )
    if layout == "TND":
        return (
            ms.ops.transpose(q, (1, 0, 2)).contiguous(),
            ms.ops.transpose(k, (1, 0, 2)).contiguous(),
            ms.ops.transpose(v, (1, 0, 2)).contiguous(),
            ms.ops.transpose(g, (1, 0, 2)).contiguous(),
            ms.ops.transpose(beta, (1, 0)).contiguous(),
        )
    raise ValueError("Chunk KDA layout must be BSND, BNSD, TND, or NTD.")


def _restore_chunk_kda_gradients(dq, dk, dv, dg, db, layout):
    """Restore gradients from canonical layout to the public input layout."""
    if layout == "BSND":
        return (
            ms.ops.transpose(dq, (0, 2, 1, 3)).contiguous(),
            ms.ops.transpose(dk, (0, 2, 1, 3)).contiguous(),
            ms.ops.transpose(dv, (0, 2, 1, 3)).contiguous(),
            ms.ops.transpose(dg, (0, 2, 1, 3)).contiguous(),
            ms.ops.transpose(db, (0, 2, 1)).contiguous(),
        )
    if layout == "TND":
        return (
            ms.ops.transpose(dq, (1, 0, 2)).contiguous(),
            ms.ops.transpose(dk, (1, 0, 2)).contiguous(),
            ms.ops.transpose(dv, (1, 0, 2)).contiguous(),
            ms.ops.transpose(dg, (1, 0, 2)).contiguous(),
            ms.ops.transpose(db, (1, 0)).contiguous(),
        )
    return dq, dk, dv, dg, db


def _validate_chunk_kda_fwd_options(chunk_size, output_final_state, safe_gate,
                                    lower_bound, use_gate_in_kernel, a_log,
                                    dt_bias, state_v_first):
    """Validate forward attributes independent of tensor layout."""
    if chunk_size not in (64, 128):
        raise ValueError("Chunk KDA forward supports chunk_size=64 or 128.")
    if output_final_state not in (True, False):
        raise ValueError("output_final_state must be a bool.")
    if state_v_first not in (True, False):
        raise ValueError("state_v_first must be a bool.")
    if use_gate_in_kernel and a_log is None:
        raise ValueError("a_log is required when use_gate_in_kernel=True.")
    if not use_gate_in_kernel and (a_log is not None or dt_bias is not None):
        raise ValueError("a_log and dt_bias require use_gate_in_kernel=True.")
    if safe_gate and use_gate_in_kernel and not -5.0 <= lower_bound < 0.0:
        raise ValueError("lower_bound must be in [-5, 0) when safe_gate=True.")


def _validate_chunk_kda_fwd_layout(q, layout, cu_seqlens, chunk_indices, chunk_size):
    """Validate layout metadata and return canonical chunk indices."""
    if layout not in ("BSND", "BNSD", "TND", "NTD"):
        raise ValueError("Chunk KDA layout must be BSND, BNSD, TND, or NTD.")
    rank = len(q.shape)
    is_packed = layout in ("TND", "NTD")
    if is_packed and (rank != 3 or cu_seqlens is None):
        raise ValueError("TND/NTD require rank-3 inputs and cu_seqlens.")
    if not is_packed and rank != 4:
        raise ValueError("BSND/BNSD require rank-4 inputs.")
    if not is_packed and cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("Rank-4 variable-length Chunk KDA requires B=1.")
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    if (cu_seqlens is None) != (chunk_indices is None):
        raise ValueError("cu_seqlens and chunk_indices must be supplied together.")
    return chunk_indices


def _validate_chunk_kda_fwd(q, chunk_size, layout, output_final_state,
                            safe_gate, lower_bound, use_gate_in_kernel,
                            a_log, dt_bias, cu_seqlens, chunk_indices,
                            state_v_first):
    """Validate the full forward-only capability exposed by FLA-NPU."""
    _validate_chunk_kda_fwd_options(
        chunk_size, output_final_state, safe_gate, lower_bound,
        use_gate_in_kernel, a_log, dt_bias, state_v_first
    )
    return _validate_chunk_kda_fwd_layout(q, layout, cu_seqlens, chunk_indices, chunk_size)


def _validate_chunk_kda_autograd(q, v, chunk_size, layout, cu_seqlens,
                                 initial_state, disable_recompute,
                                 state_v_first):
    """Validate the narrower capability supported by fused backward."""
    if initial_state is not None:
        raise ValueError("initial_state is not supported until backward provides dht/dh0 semantics.")
    if not disable_recompute:
        raise ValueError("disable_recompute=False is not supported by the fused backward kernel.")
    if state_v_first:
        raise ValueError("state_v_first=True is not supported by the fused backward kernel.")
    if chunk_size != 64:
        raise ValueError("Differentiable Chunk KDA currently supports only chunk_size=64.")
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("Differentiable Chunk KDA currently requires K=V=128.")
    if layout in ("BSND", "BNSD") and cu_seqlens is not None:
        raise ValueError("Differentiable Chunk KDA does not support rank-4 variable-length inputs.")


def _run_chunk_kda_fwd(q, k, v, g, beta, scale, chunk_size, layout,
                       output_final_state, safe_gate, lower_bound,
                       use_gate_in_kernel, a_log, dt_bias, cu_seqlens,
                       chunk_indices, initial_state=None,
                       state_v_first=False):
    """Run the shared Chunk KDA forward implementation."""
    chunk_indices = _validate_chunk_kda_fwd(
        q, chunk_size, layout, output_final_state, safe_gate, lower_bound,
        use_gate_in_kernel, a_log, dt_bias, cu_seqlens, chunk_indices,
        state_v_first
    )
    q_head, k_head, v_head, g_head, beta_head = _canonicalize_chunk_kda_inputs(
        q, k, v, g, beta, layout
    )
    canonical_layout = "NTD" if len(q_head.shape) == 3 else "BNSD"
    result = _custom_ops.npu_chunk_kda_fwd(
        q_head, k_head, v_head, g_head, beta_head, scale, chunk_size,
        canonical_layout, True, safe_gate, lower_bound,
        use_gate_in_kernel, a_log, dt_bias, initial_state, cu_seqlens,
        chunk_indices, state_v_first
    )
    canonical_inputs = (q_head, k_head, v_head, g_head, beta_head)
    return result, canonical_inputs, chunk_indices


def _run_chunk_kda_bwd(q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h,
                       d_o, scale, chunk_size, layout="BNSD", raw_g=None,
                       a_log=None, dt_bias=None, cu_seqlens=None,
                       chunk_indices=None, safe_gate=False,
                       use_gate_in_kernel=False, lower_bound=-5.0):
    """Run backward after canonicalizing layout and adapting GQA heads."""
    zero_gate = ms.ops.zeros_like(gk)
    q, k, v, raw_head, beta = _canonicalize_chunk_kda_inputs(
        q, k, v, raw_g if raw_g is not None else zero_gate, beta, layout
    )
    canonical_layout = "NTD" if len(q.shape) == 3 else "BNSD"
    if layout == "BSND":
        d_o = ms.ops.transpose(d_o, (0, 2, 1, 3)).contiguous()
    elif layout == "TND":
        d_o = ms.ops.transpose(d_o, (1, 0, 2)).contiguous()
    head_axis = 0 if canonical_layout == "NTD" else 1
    heads = q.shape[head_axis]
    value_heads = v.shape[head_axis]
    if value_heads < heads or value_heads % heads != 0:
        raise ValueError("Chunk KDA requires HV >= H and HV % H == 0.")
    ratio = value_heads // heads
    q_kernel = q if ratio == 1 else ms.ops.repeat_interleave(q, ratio, axis=head_axis).contiguous()
    k_kernel = k if ratio == 1 else ms.ops.repeat_interleave(k, ratio, axis=head_axis).contiguous()
    bias_head = None if dt_bias is None else dt_bias.reshape((value_heads, 128))
    outputs = _custom_ops.npu_chunk_kda_bwd(
        q_kernel, k_kernel, v, beta, gk, aqk, akk, w, qg, kg, v_new, h,
        d_o, raw_head if use_gate_in_kernel else None,
        a_log if use_gate_in_kernel else None, bias_head,
        cu_seqlens, chunk_indices, scale, chunk_size, safe_gate,
        use_gate_in_kernel, lower_bound, canonical_layout
    )
    dq, dk, dv, db, dg, d_a, d_bias = outputs
    if ratio != 1:
        if canonical_layout == "BNSD":
            batch, _, tokens, dim = dq.shape
            dq = dq.reshape((batch, heads, ratio, tokens, dim)).sum(axis=2)
            dk = dk.reshape((batch, heads, ratio, tokens, dim)).sum(axis=2)
        else:
            _, tokens, dim = dq.shape
            dq = dq.reshape((heads, ratio, tokens, dim)).sum(axis=1)
            dk = dk.reshape((heads, ratio, tokens, dim)).sum(axis=1)
    dq, dk, dv, dg, db = _restore_chunk_kda_gradients(dq, dk, dv, dg, db, layout)
    if dt_bias is not None:
        d_bias = d_bias.reshape(dt_bias.shape)
    return dq, dk, dv, db, dg, d_a, d_bias


class NpuChunkKdaDFunction(DFunction):
    """Differentiable Chunk KDA wrapper for dense and packed layouts."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, chunk_size, layout,
                output_final_state, safe_gate, lower_bound,
                use_gate_in_kernel, a_log, dt_bias, cu_seqlens,
                chunk_indices):
        """Run forward and retain the intermediates required by backward."""
        _validate_chunk_kda_autograd(
            q, v, chunk_size, layout, cu_seqlens, None, True, False
        )
        result, canonical_inputs, chunk_indices = _run_chunk_kda_fwd(
            q, k, v, g, beta, scale, chunk_size, layout,
            output_final_state, safe_gate, lower_bound, use_gate_in_kernel,
            a_log, dt_bias, cu_seqlens, chunk_indices
        )
        q_head, k_head, v_head, g_head, beta_head = canonical_inputs
        saved = [q_head, k_head, v_head, beta_head, result[2], result[3], result[4],
                 result[5], result[7], result[8], result[9], result[10], g_head]
        if a_log is not None:
            saved.append(a_log)
        if dt_bias is not None:
            saved.append(dt_bias)
        ctx.save_for_backward(*saved)
        ctx.has_a_log = a_log is not None
        ctx.has_dt_bias = dt_bias is not None
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.layout = layout
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        return result[0], result[1] if output_final_state else None

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Run fused backward and reject unsupported final-state gradients."""
        grad_attn = grad_outputs[0]
        grad_state = grad_outputs[1] if len(grad_outputs) > 1 else None
        if grad_state is not None:
            raise RuntimeError(
                "Chunk KDA final_state is not differentiable in the current integration. "
                "Do not include final_state in the training loss."
            )
        if grad_attn is None:
            return (None,) * 16
        saved = ctx.saved_tensors
        q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, raw_g = saved[:13]
        index = 13
        a_log = saved[index] if ctx.has_a_log else None
        index += int(ctx.has_a_log)
        dt_bias = saved[index] if ctx.has_dt_bias else None
        grad_attn = (ms.ops.transpose(grad_attn, (0, 2, 1, 3)).contiguous()
                     if len(q.shape) == 4
                     else ms.ops.transpose(grad_attn, (1, 0, 2)).contiguous())
        outputs = _run_chunk_kda_bwd(
            q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h,
            grad_attn, ctx.scale, ctx.chunk_size,
            "NTD" if len(q.shape) == 3 else "BNSD",
            raw_g=raw_g, a_log=a_log, dt_bias=dt_bias,
            cu_seqlens=ctx.cu_seqlens, chunk_indices=ctx.chunk_indices,
            safe_gate=ctx.safe_gate, use_gate_in_kernel=ctx.use_gate_in_kernel,
            lower_bound=ctx.lower_bound,
        )
        dq, dk, dv, db, dg, d_a, d_bias = outputs
        dq, dk, dv, dg, db = _restore_chunk_kda_gradients(
            dq, dk, dv, dg, db, ctx.layout
        )
        return (dq, dk, dv, dg, db, None, None, None, None, None,
                None, None, d_a if ctx.use_gate_in_kernel else None,
                d_bias if ctx.has_dt_bias else None, None, None)


def npu_chunk_kda_fwd(q, k, v, g, beta, scale, chunk_size, layout="BNSD",
                      output_final_state=False, safe_gate=False, lower_bound=-5.0,
                      use_gate_in_kernel=False, a_log=None, dt_bias=None,
                      cu_seqlens=None, chunk_indices=None,
                      initial_state=None, disable_recompute=True,
                      state_v_first=False,
                      return_intermediates=False):
    """Run forward-only KDA, restricting backward intermediates when requested."""
    if return_intermediates not in (True, False):
        raise ValueError("return_intermediates must be a bool.")
    if return_intermediates:
        _validate_chunk_kda_autograd(
            q, v, chunk_size, layout, cu_seqlens, initial_state,
            disable_recompute, state_v_first
        )
    result, _, _ = _run_chunk_kda_fwd(
        q, k, v, g, beta, scale, chunk_size, layout, output_final_state,
        safe_gate, lower_bound, use_gate_in_kernel, a_log, dt_bias,
        cu_seqlens, chunk_indices, initial_state, state_v_first
    )
    public_outputs = (result[0], result[1] if output_final_state else None)
    if not return_intermediates:
        return public_outputs
    intermediates = (result[2], result[3], result[4], result[5],
                     result[7], result[8], result[9], result[10])
    return public_outputs + (intermediates,)


def npu_chunk_kda_bwd(q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h,
                      d_o, scale, chunk_size, layout="BNSD", raw_g=None,
                      a_log=None, dt_bias=None, cu_seqlens=None,
                      chunk_indices=None, safe_gate=False,
                      use_gate_in_kernel=False, lower_bound=-5.0):
    """Stateless explicit Chunk KDA backward operator."""
    outputs = _run_chunk_kda_bwd(
        q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h,
        d_o, scale, chunk_size, layout, raw_g, a_log, dt_bias,
        cu_seqlens, chunk_indices, safe_gate, use_gate_in_kernel, lower_bound
    )
    return outputs[:5] + (outputs[5] if use_gate_in_kernel else None,
                          outputs[6] if dt_bias is not None else None)
