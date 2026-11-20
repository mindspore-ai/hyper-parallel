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
"""Independent Torch mHC oracle retained for public component acceptance."""
from __future__ import annotations
import torch
from hyper_multicore.modules.mega_mhc_common import (
    MHC_SINKHORN_ITERS,
    MHC_STREAMS,
    MegaMhcOutputs,
    validate_mega_mhc_inputs,
)


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
    validate_mega_mhc_inputs(
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
