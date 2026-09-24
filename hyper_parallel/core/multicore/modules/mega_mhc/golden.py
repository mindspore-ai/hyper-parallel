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
"""Independent PyTorch and CANN references for shifted Single-Pass mHC."""

from __future__ import annotations

import torch
import torch_npu  # pylint: disable=unused-import  # Registers CANN NPU operators.

from hyper_parallel.core.multicore.torch import ops as multicore_ops


MHC_STREAMS = 4
MHC_MAPPINGS = MHC_STREAMS * MHC_STREAMS + 2 * MHC_STREAMS
MHC_SINKHORN_ITERS = 20

MegaMhcOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _sinkhorn(logits: torch.Tensor, num_iters: int, eps: float) -> torch.Tensor:
    """Apply the mHC row/column Sinkhorn sequence in FP32."""
    current = torch.softmax(logits.float(), dim=-1) + eps
    current = current / (current.sum(dim=-2, keepdim=True) + eps)
    for _ in range(1, num_iters):
        current = current / (current.sum(dim=-1, keepdim=True) + eps)
        current = current / (current.sum(dim=-2, keepdim=True) + eps)
    return current


def _predict_mappings(
    residual: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    *,
    hc_eps: float,
    norm_eps: float,
    num_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict current A/C/B mappings from the current residual streams."""
    leading_shape = residual.shape[:-2]
    residual_fp32 = residual.float()
    normalized = residual_fp32 * torch.rsqrt(
        residual_fp32.square().mean(dim=(-2, -1), keepdim=True) + norm_eps
    )
    logits = torch.matmul(normalized.flatten(-2), phi.float().transpose(-1, -2))
    pre_logits, post_logits, residual_logits = torch.split(
        logits,
        (MHC_STREAMS, MHC_STREAMS, MHC_STREAMS * MHC_STREAMS),
        dim=-1,
    )
    pre_bias, post_bias, residual_bias = torch.split(
        bias.float(),
        (MHC_STREAMS, MHC_STREAMS, MHC_STREAMS * MHC_STREAMS),
    )
    pre_mix = torch.sigmoid(pre_logits * alpha[0].float() + pre_bias) + hc_eps
    post_mix = 2.0 * torch.sigmoid(post_logits * alpha[1].float() + post_bias)
    residual_mix = _sinkhorn(
        (residual_logits * alpha[2].float() + residual_bias).reshape(
            *leading_shape,
            MHC_STREAMS,
            MHC_STREAMS,
        ),
        num_iters,
        hc_eps,
    )
    return pre_mix, post_mix, residual_mix


def _validate_common(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    num_iters: int,
) -> None:
    """Validate the fixed first-version mHC contract shared by all paths."""
    if residual.ndim < 3 or residual.shape[-2] != MHC_STREAMS:
        raise ValueError(f"residual must end in [{MHC_STREAMS}, H], got {tuple(residual.shape)}.")
    leading_shape = tuple(residual.shape[:-2])
    hidden_size = residual.shape[-1]
    expected_shapes = {
        "previous_output": (*leading_shape, hidden_size),
        "previous_pre_mix": (*leading_shape, MHC_STREAMS),
        "previous_post_mix": (*leading_shape, MHC_STREAMS),
        "previous_residual_mix": (*leading_shape, MHC_STREAMS, MHC_STREAMS),
        "phi": (MHC_MAPPINGS, MHC_STREAMS * hidden_size),
        "alpha": (3,),
        "bias": (MHC_MAPPINGS,),
        "norm_weight": (hidden_size,),
    }
    values = {
        "previous_output": previous_output,
        "previous_pre_mix": previous_pre_mix,
        "previous_post_mix": previous_post_mix,
        "previous_residual_mix": previous_residual_mix,
        "phi": phi,
        "alpha": alpha,
        "bias": bias,
        "norm_weight": norm_weight,
    }
    for name, expected in expected_shapes.items():
        actual = tuple(values[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}.")
    if num_iters != MHC_SINKHORN_ITERS:
        raise ValueError(f"initial HyperMegaMhc requires num_iters={MHC_SINKHORN_ITERS}, got {num_iters}.")


def torch_mega_mhc(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = MHC_SINKHORN_ITERS,
) -> MegaMhcOutputs:
    """Evaluate shifted Single-Pass mHC with ordinary tensor operations.

    This implementation is the semantic oracle. It intentionally does not call
    either the HyperMegaMhc custom op or a CANN mHC large operator.

    Args:
        previous_output: Previous block output.
        residual: Previous residual streams.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weights.
        alpha: Mapping scale parameters.
        bias: Mapping bias parameters.
        norm_weight: Shifted RMSNorm weight.
        hc_eps: Sinkhorn numerical-stability epsilon.
        norm_eps: RMS normalization epsilon.
        num_iters: Sinkhorn iteration count.
    """
    _validate_common(
        previous_output,
        residual,
        previous_pre_mix,
        previous_post_mix,
        previous_residual_mix,
        phi,
        alpha,
        bias,
        norm_weight,
        num_iters,
    )
    residual_fp32 = residual.float()
    new_residual = (
        torch.einsum("...ji,...jh->...ih", previous_residual_mix.float(), residual_fp32)
        + previous_post_mix.float().unsqueeze(-1) * previous_output.float().unsqueeze(-2)
    ).to(residual.dtype)
    next_pre_mix, next_post_mix, next_residual_mix = _predict_mappings(
        new_residual,
        phi,
        alpha,
        bias,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        num_iters=num_iters,
    )
    mixed_input = (new_residual.float() * previous_pre_mix.float().unsqueeze(-1)).sum(dim=-2)
    block_input = mixed_input * torch.rsqrt(mixed_input.square().mean(dim=-1, keepdim=True) + norm_eps)
    block_input = (block_input * norm_weight.float()).to(residual.dtype)
    return new_residual, next_pre_mix, next_post_mix, next_residual_mix, block_input


def cann_mega_mhc(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = MHC_SINKHORN_ITERS,
) -> MegaMhcOutputs:
    """Evaluate the five-kernel CANN compatibility graph on an NPU.

    Args:
        previous_output: Previous block output.
        residual: Previous residual streams.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weights.
        alpha: Mapping scale parameters.
        bias: Mapping bias parameters.
        norm_weight: Shifted RMSNorm weight.
        hc_eps: Sinkhorn numerical-stability epsilon.
        norm_eps: RMS normalization epsilon.
        num_iters: Sinkhorn iteration count.
    """
    _validate_common(
        previous_output,
        residual,
        previous_pre_mix,
        previous_post_mix,
        previous_residual_mix,
        phi,
        alpha,
        bias,
        norm_weight,
        num_iters,
    )
    if not residual.is_npu or residual.dtype != torch.bfloat16:
        raise TypeError("cann_mega_mhc requires BF16 NPU activations.")
    leading_shape = residual.shape[:-2]
    hidden_size = residual.shape[-1]
    token_count = residual.numel() // (MHC_STREAMS * hidden_size)
    residual_bsnd = residual.reshape(1, token_count, MHC_STREAMS, hidden_size).contiguous()
    new_residual_bsnd = torch.ops.npu.npu_mhc_post(
        residual_bsnd,
        previous_residual_mix.reshape(1, token_count, MHC_STREAMS, MHC_STREAMS).contiguous(),
        previous_output.reshape(1, token_count, hidden_size).contiguous(),
        previous_post_mix.reshape(1, token_count, MHC_STREAMS).contiguous(),
    )
    new_residual_flat = new_residual_bsnd.reshape(token_count, MHC_STREAMS, hidden_size)
    _, next_post, next_res, next_pre = multicore_ops.cann_mhc_pre_sinkhorn(
        new_residual_flat,
        phi.contiguous(),
        alpha.contiguous(),
        bias.contiguous(),
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        num_iters=num_iters,
    )
    previous_pre_bsn = previous_pre_mix.reshape(1, token_count, MHC_STREAMS).contiguous()
    mixed_input = torch.mul(new_residual_bsnd, previous_pre_bsn.unsqueeze(-1)).sum(
        dim=2, dtype=torch.bfloat16
    )
    block_input, _ = torch.ops.npu.npu_rms_norm(mixed_input, norm_weight, norm_eps)
    return (
        new_residual_flat.reshape(*leading_shape, MHC_STREAMS, hidden_size),
        next_pre.reshape(*leading_shape, MHC_STREAMS),
        next_post.reshape(*leading_shape, MHC_STREAMS),
        next_res.reshape(*leading_shape, MHC_STREAMS, MHC_STREAMS),
        block_input.reshape(*leading_shape, hidden_size),
    )
