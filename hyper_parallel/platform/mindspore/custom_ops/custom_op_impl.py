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
import os
import sys

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
]


def _build_custom_ops():
    return ms.ops.CustomOpBuilder(
        _MS_EXTENSION_NAME,
        _CUSTOM_OP_SOURCES,
        backend="Ascend",
    ).load()


try:
    _custom_ops = __import__(_MS_EXTENSION_NAME)
except ImportError:
    # Source-tree development: .so not pre-built; JIT-compile from local .cc files.
    _custom_ops = _build_custom_ops()
else:
    # Rebuild stale source-tree extensions that predate newly added symbols.
    if not hasattr(_custom_ops, "npu_mhc_pre_clamp_sinkhorn"):
        _custom_ops = _build_custom_ops()


def _ensure_contiguous(*tensors):
    """Ensure all tensors are contiguous (no-op if already contiguous)."""
    return tuple(t.contiguous() if not t.is_contiguous() else t for t in tensors)


def _to_list_int64(val):
    """Convert Tensor(int32) to List[int64] for aclnn kernel consumption."""
    if isinstance(val, ms.Tensor):
        return val.asnumpy().astype("int64").tolist()
    return val


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
        grads = _custom_ops.npu_mhc_pre_sinkhorn_backward(
            grad_h_in, grad_h_post, grad_h_res,
            x, phi, alpha, bias,
            h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
            ctx.hc_eps)
        return grads[0], grads[1], grads[2], grads[3], None, None, None, None, None


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
        return grads[0], grads[1], grads[2], grads[3], None, None, None, None, None, None, None
