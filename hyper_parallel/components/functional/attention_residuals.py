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
"""Block attention residuals with eager math and the ops-transformer fused API."""

import importlib
from collections.abc import Callable

# Torch is the native framework; the legacy platform-agnostic lint rule is obsolete here.
# pylint: disable=forbidden-backend-import
import torch


def _block_attention_residuals_eager(prefix: torch.Tensor, history: torch.Tensor,
                                    projection: torch.Tensor, norm_weight: torch.Tensor,
                                    norm_eps: float) -> torch.Tensor:
    values = torch.cat((history, prefix.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    normalized = values_fp32 * torch.rsqrt(values_fp32.square().mean(-1, keepdim=True) + norm_eps)
    query = projection.float().reshape(-1) * norm_weight.float()
    probabilities = (normalized * query).sum(-1).softmax(-1)
    return (probabilities.unsqueeze(-1) * values_fp32).sum(1).to(values.dtype)


def _load_block_attention_residuals() -> Callable:
    """Resolve the differentiable ops-transformer API and its backward dependency."""
    try:
        # CANN is optional for CPU and eager-only callers.
        ops = importlib.import_module("cann_ops_transformer")
    except (ImportError, OSError) as error:
        raise ImportError(
            "AttnRes backend='ascendc' could not load cann_ops_transformer. Install the standalone "
            "ops-transformer Python package with compatible PyTorch, torch-npu and CANN libraries. "
            f"See hyper_parallel/components/functional/README.md. Original error: {error}"
        ) from error
    block_attention_residuals_op = getattr(ops, "block_attention_residuals", None)
    block_attention_residuals_backward = getattr(ops, "block_attention_residuals_backward", None)
    if not callable(block_attention_residuals_op) or not callable(block_attention_residuals_backward):
        raise ImportError(
            "AttnRes backend='ascendc' requires cann_ops_transformer.block_attention_residuals and "
            "block_attention_residuals_backward. Install a package containing both training APIs "
            "and build block_attention_residuals plus block_attention_residuals_grad. "
            "See hyper_parallel/components/functional/README.md."
        )
    return block_attention_residuals_op


def _block_attention_residuals_fused(prefix: torch.Tensor, history: torch.Tensor,
                                    projection: torch.Tensor, norm_weight: torch.Tensor,
                                    norm_eps: float) -> torch.Tensor:
    block_attention_residuals_op = _load_block_attention_residuals()
    # Native forward accepts [H], but its training backward requires [1,H].
    if projection.ndim == 1:
        projection = projection.unsqueeze(0)
    inputs = (prefix, history, projection, norm_weight)
    common_dtype = prefix.dtype
    for tensor in inputs[1:]:
        common_dtype = torch.promote_types(common_dtype, tensor.dtype)
    # The native API requires one dtype; casts must preserve all four gradient edges.
    prepared = tuple(tensor.to(dtype=common_dtype).contiguous() for tensor in inputs)
    result = block_attention_residuals_op(*prepared, valid_block_num=-1, norm_eps=norm_eps)
    return result.to(torch.promote_types(prefix.dtype, history.dtype))


def block_attention_residuals(prefix: torch.Tensor, history: torch.Tensor,
                             projection: torch.Tensor, norm_weight: torch.Tensor, *, norm_eps: float,
                             backend: str = "eager") -> torch.Tensor:
    """Mix previous blocks and the current residual through depth attention.

    Args:
        prefix: Current residual of shape [T,H], with complete hidden H.
        history: Previous blocks of shape [T,N,H].
        projection: Learned pseudo-query of shape [H] or [1,H].
        norm_weight: Learned RMS normalization scale of shape [H].
        norm_eps: RMSNorm epsilon from the calling model layer.
        backend: ``eager`` uses FP32 intermediates. ``ascendc`` calls the
            differentiable ops-transformer API; native errors propagate unchanged.

    Returns:
        Tensor of shape [T,H] with the promoted prefix/history dtype. Mixed
        inputs are differentiably promoted to one dtype for the fused operator.
        Empty T or N retains zero gradients without a native launch.

    Raises:
        ImportError: The optional CANN package cannot load or lacks a training API.

    Note:
        Models own block boundaries and learned parameters. The fused operator
        supports first-order gradients. Its dispatcher and native entry validate
        device and shape limits; kernel execution errors propagate unchanged.
    """
    if backend == "eager":
        return _block_attention_residuals_eager(prefix, history, projection, norm_weight, norm_eps)
    if backend != "ascendc":
        raise ValueError("AttnRes backend must be 'eager' or 'ascendc'")
    if prefix.shape[0] == 0 or history.shape[1] == 0:
        # Empty views keep zero gradients without inspecting unused values.
        zero = sum(tensor.reshape(-1)[:0].sum() for tensor in (history, projection, norm_weight))
        dtype = torch.promote_types(prefix.dtype, history.dtype)
        return prefix.to(dtype) + zero.to(dtype)
    return _block_attention_residuals_fused(prefix, history, projection, norm_weight, norm_eps)
