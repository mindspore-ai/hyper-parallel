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
from dataclasses import dataclass
from importlib import machinery
from importlib import util
import os
from pathlib import Path
import sys

import mindspore as ms  # pylint: disable=C0415

from hyper_parallel.core.shard.dfunction import DFunction


_CC_DIR = os.path.dirname(os.path.abspath(__file__))
_MS_EXTENSION_NAME = "hyper_parallel_custom_ops_ms"
_BUILD_LIB = os.path.join(_CC_DIR, "lib")

_MHC_PRE_CLAMP_NONE_GRADS = (None,) * 7


def _extension_search_paths():
    """Return source-build and installed extension directories in priority order."""
    module_path = Path(__file__).resolve()
    repository_root = module_path.parents[4]
    search_paths = []
    if (repository_root / "setup.py").is_file():
        search_paths.append(str(
            repository_root / "build" / "native" / "payload" / "hyper_parallel"
            / "platform" / "mindspore" / "custom_ops" / "lib"
        ))
    search_paths.append(_BUILD_LIB)
    return list(dict.fromkeys(search_paths))


def _load_prebuilt_extension():
    """Load the exact prebuilt extension without mutating process-global ``sys.path``."""
    loaded_module = sys.modules.get(_MS_EXTENSION_NAME)
    if loaded_module is not None:
        return loaded_module
    load_errors = []
    for extension_path in _extension_search_paths():
        root = Path(extension_path)
        for suffix in machinery.EXTENSION_SUFFIXES:
            library_path = root / f"{_MS_EXTENSION_NAME}{suffix}"
            if not library_path.is_file():
                continue
            module = None
            try:
                spec = util.spec_from_file_location(_MS_EXTENSION_NAME, library_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot create an extension loader for {library_path}")
                module = util.module_from_spec(spec)
                sys.modules[_MS_EXTENSION_NAME] = module
                spec.loader.exec_module(module)
                return module
            except (ImportError, OSError, RuntimeError) as error:
                if module is not None and sys.modules.get(_MS_EXTENSION_NAME) is module:
                    sys.modules.pop(_MS_EXTENSION_NAME, None)
                load_errors.append(f"{library_path}: {error}")
    details = "; ".join(load_errors) if load_errors else "no prebuilt extension file was found"
    raise ImportError(details)


@dataclass(frozen=True)
class _MhcPreClampArgs:
    """Bound arguments for npu_mhc_pre_clamp_sinkhorn."""

    x: ms.Tensor
    phi: ms.Tensor
    alpha: ms.Tensor
    bias: ms.Tensor
    hc_mult: int
    num_iters: int
    hc_eps: float
    norm_eps: float
    out_flag: bool
    clamp_min: float
    clamp_max: float


def _bind_mhc_pre_clamp_args(args, kwargs):
    """Bind npu_mhc_pre_clamp_sinkhorn arguments with Python defaults."""
    names = (
        "x", "phi", "alpha", "bias", "hc_mult", "num_iters",
        "hc_eps", "norm_eps", "out_flag", "clamp_min", "clamp_max",
    )
    values = {
        "hc_mult": 4,
        "num_iters": 20,
        "hc_eps": 1e-6,
        "norm_eps": 1e-6,
        "out_flag": True,
        "clamp_min": 0.0,
        "clamp_max": 0.0,
    }
    if len(args) > len(names):
        raise TypeError(f"npu_mhc_pre_clamp_sinkhorn expected at most {len(names)} arguments")
    for name, value in zip(names, args):
        values[name] = value
    for name, value in kwargs.items():
        if name in values and name in names[:len(args)]:
            raise TypeError(f"npu_mhc_pre_clamp_sinkhorn got multiple values for argument '{name}'")
        if name not in names:
            raise TypeError(f"npu_mhc_pre_clamp_sinkhorn got an unexpected keyword argument '{name}'")
        values[name] = value
    missing = [name for name in names[:4] if name not in values]
    if missing:
        raise TypeError(f"npu_mhc_pre_clamp_sinkhorn missing required arguments: {missing}")
    return _MhcPreClampArgs(*(values[name] for name in names))


try:
    _custom_ops = _load_prebuilt_extension()
except ImportError as error:
    raise ImportError(
        "[HP-NATIVE-LOAD-FAILED] component=custom_ops framework=mindspore. "
        "No compatible prebuilt extension could be loaded. For a source/PYTHONPATH checkout, run "
        "`./build.sh --multicore off --shmem off --custom-ops on`."
    ) from error
if not hasattr(_custom_ops, "npu_mhc_pre_clamp_sinkhorn"):
    raise ImportError(
        "[HP-NATIVE-INCOMPATIBLE] component=custom_ops framework=mindspore. "
        "The prebuilt extension is stale or was built for a different HyperParallel source revision."
    )


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
        b, s, n = grad_h_post.shape
        grad_h_res = grad_h_res.reshape(b, s, n, n)
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
    def forward(ctx, *args, **kwargs):
        """Forward pass: delegates to the clamp-enabled Ascend custom kernel."""
        bound = _bind_mhc_pre_clamp_args(args, kwargs)
        result = _custom_ops.npu_mhc_pre_clamp_sinkhorn(
            bound.x, bound.phi, bound.alpha, bound.bias,
            bound.hc_mult, bound.num_iters, bound.hc_eps, bound.norm_eps,
            bound.out_flag, bound.clamp_min, bound.clamp_max
        )
        _, _, _, h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits = result
        ctx.save_for_backward(bound.x, bound.phi, bound.alpha, bound.bias,
                              h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits)
        ctx.hc_eps = bound.hc_eps
        ctx.clamp_min = bound.clamp_min
        ctx.clamp_max = bound.clamp_max
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass: calls npu_mhc_pre_clamp_sinkhorn_backward kernel."""
        tensors = _ensure_contiguous(
            grad_outputs[0], grad_outputs[1], grad_outputs[2],
            *ctx.saved_tensors
        )
        n = tensors[1].shape[-1]
        grad_h_res = ms.ops.reshape(tensors[2], tuple(tensors[2].shape[:-1]) + (n, n))

        grads = _custom_ops.npu_mhc_pre_clamp_sinkhorn_backward(
            tensors[0], tensors[1], grad_h_res,
            tensors[3], tensors[4], tensors[5], tensors[6],
            tensors[7], tensors[8], tensors[9], tensors[10], tensors[11], tensors[12],
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
        ctx.save_for_backward(*[t for t in [
            query, ori_kv, cmp_kv, sinks, ori_sparse_indices, cmp_sparse_indices,
            cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
            attention_out, softmax_lse, cmp_residual_kv,
        ] if t is not None])
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
        it = iter(ctx.saved_tensors)
        q = next(it)
        ori_kv = next(it) if ctx.has_ori_kv else None
        cmp_kv = next(it) if ctx.has_cmp_kv else None
        sinks = next(it) if ctx.has_sinks else None
        ori_sparse_indices = next(it) if ctx.has_ori_sparse else None
        cmp_sparse_indices = next(it) if ctx.has_cmp_sparse else None
        cu_seq_lens_q = next(it) if ctx.has_cu_q else None
        cu_seq_lens_ori_kv = next(it) if ctx.has_cu_ori_kv else None
        cu_seq_lens_cmp_kv = next(it) if ctx.has_cu_cmp_kv else None
        attention_out = next(it)
        softmax_lse = next(it)
        cmp_residual_kv = next(it) if ctx.has_cmp_residual else None
        # metadata MUST be None: the grad kernel asserts it is nullptr and
        # re-derives tiling internally.  cmp_residual_kv is passed through —
        # required for CFA/SCFA (cmp_ratio!=1) with cmp_mask_mode=3.
        grads = _custom_ops.npu_sparse_flash_mla_grad(
            q, grad_attention_out, attention_out, softmax_lse,
            ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
            cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv,
            None, None, None,                 # seq_used_q, seq_used_ori_kv, seq_used_cmp_kv
            cmp_residual_kv, None, None,       # cmp_residual_kv, ori_topk_length, cmp_topk_length
            sinks, None,                      # sinks, metadata(None → grad kernel self-derives)
            ctx.softmax_scale, ctx.cmp_ratio, ctx.ori_mask_mode, ctx.cmp_mask_mode,
            ctx.ori_win_left, ctx.ori_win_right, ctx.layout_q, ctx.layout_kv,
        )
        d_query = grads[0]
        d_ori_kv = grads[1] if ori_kv is not None else None
        d_cmp_kv = grads[2] if cmp_kv is not None else None
        d_sinks = grads[3] if sinks is not None else None
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
