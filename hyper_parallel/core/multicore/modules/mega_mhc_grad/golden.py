# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Independent PyTorch semantic reference and native forward cache for mHC backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_npu  # pylint: disable=unused-import  # Registers CANN NPU operators.

from hyper_parallel.core.multicore.modules.mega_mhc.golden import (
    MHC_SINKHORN_ITERS,
    _validate_common,
    torch_mega_mhc,
)
from hyper_parallel.core.multicore.torch import ops as multicore_ops


MegaMhcGradOutputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True)
class MegaMhcGradCache:
    """Forward tensors required by the native backward kernels."""

    new_residual: torch.Tensor
    hc_before_norm: torch.Tensor
    inv_rms: torch.Tensor
    sum_out: torch.Tensor
    norm_out: torch.Tensor
    mixed_input: torch.Tensor
    rms_rstd: torch.Tensor


def make_cann_grad_cache(
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
) -> MegaMhcGradCache:
    """Run the forward compatibility graph once and retain backward caches.

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
    hidden_size = residual.shape[-1]
    token_count = residual.numel() // (4 * hidden_size)
    residual_bsnd = residual.reshape(1, token_count, 4, hidden_size).contiguous()
    new_residual = torch.ops.npu.npu_mhc_post(
        residual_bsnd,
        previous_residual_mix.reshape(1, token_count, 4, 4).contiguous(),
        previous_output.reshape(1, token_count, hidden_size).contiguous(),
        previous_post_mix.reshape(1, token_count, 4).contiguous(),
    )
    (
        _unused_hin,
        _next_post,
        _next_res,
        _next_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
    ) = multicore_ops.cann_mhc_pre_sinkhorn_with_cache(
        new_residual.reshape(token_count, 4, hidden_size),
        phi.contiguous(),
        alpha.contiguous(),
        bias.contiguous(),
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        num_iters=num_iters,
    )
    mixed_input = torch.mul(
        new_residual, previous_pre_mix.reshape(1, token_count, 4, 1)
    ).sum(dim=2, dtype=torch.bfloat16)
    _block_input, rms_rstd = torch.ops.npu.npu_rms_norm(
        mixed_input, norm_weight, norm_eps
    )
    return MegaMhcGradCache(
        new_residual=new_residual,
        hc_before_norm=hc_before_norm,
        inv_rms=inv_rms,
        sum_out=sum_out,
        norm_out=norm_out,
        mixed_input=mixed_input,
        rms_rstd=rms_rstd,
    )


def torch_mega_mhc_grad(
    grad_new_residual: torch.Tensor,
    grad_next_pre_mix: torch.Tensor,
    grad_next_post_mix: torch.Tensor,
    grad_next_residual_mix: torch.Tensor,
    grad_block_input: torch.Tensor,
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    cache: MegaMhcGradCache | None = None,
    *,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_iters: int = MHC_SINKHORN_ITERS,
) -> MegaMhcGradOutputs:
    """Differentiate the independent shifted forward semantic oracle.

    Args:
        grad_new_residual: Gradient of the updated residual streams.
        grad_next_pre_mix: Gradient of the next pre-mix coefficients.
        grad_next_post_mix: Gradient of the next post-mix coefficients.
        grad_next_residual_mix: Gradient of the next residual-mix coefficients.
        grad_block_input: Gradient of the shifted block input.
        previous_output: Previous block output.
        residual: Previous residual streams.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weights.
        alpha: Mapping scale parameters.
        bias: Mapping bias parameters.
        norm_weight: Shifted RMSNorm weight.
        cache: Optional native cache, unused by the independent oracle.
        hc_eps: Sinkhorn numerical-stability epsilon.
        norm_eps: RMS normalization epsilon.
        num_iters: Sinkhorn iteration count.
    """
    del cache
    with torch.enable_grad():
        differentiable = tuple(
            value.detach().clone().requires_grad_(True)
            for value in (
                previous_output,
                residual,
                previous_pre_mix,
                previous_post_mix,
                previous_residual_mix,
                phi,
                alpha,
                bias,
                norm_weight,
            )
        )
        outputs = torch_mega_mhc(
            *differentiable,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            num_iters=num_iters,
        )
        gradients = torch.autograd.grad(
            outputs,
            differentiable,
            grad_outputs=(
                grad_new_residual,
                grad_next_pre_mix,
                grad_next_post_mix,
                grad_next_residual_mix,
                grad_block_input,
            ),
        )
    return (*gradients[:-1], gradients[-1].float())
