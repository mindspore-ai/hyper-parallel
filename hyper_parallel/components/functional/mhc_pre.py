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
"""Manifold hyper-connection pre-processing functions."""

from typing import Any, Optional, Tuple

import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

try:
    import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import
except ImportError:
    omni_training_custom_ops = None

from hyper_parallel.components.functional.sinkhorn import sinkhorn, sinkhorn_knopps


class _MhcPre(torch.autograd.Function):
    """Autograd bridge for the NPU MHC pre custom operator."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        out_flag: bool = True,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the NPU MHC pre forward operator."""
        if omni_training_custom_ops is None:
            raise ImportError("MHC AscendC requires omni_training_custom_ops")
        hin, h_post, h_comb_before, inv_rms, h_mix, h_pre = (
            torch.ops.custom.npu_manifold_constrained_hyper_connection_pre(
                x,
                phi,
                alpha,
                bias,
                gamma=gamma,
                out_flag=out_flag,
                norm_eps=norm_eps,
                hc_eps=hc_eps,
            )
        )
        ctx.save_for_backward(x, phi, alpha, gamma, h_post, inv_rms, h_mix, h_pre)
        ctx.hc_eps = hc_eps
        ctx.has_gamma = gamma is not None
        return hin, h_pre, h_post, h_comb_before, x

    @staticmethod
    def backward(
        ctx: Any,
        dh_in: torch.Tensor,
        dh_pre: torch.Tensor,
        dh_post: torch.Tensor,
        dh_res: torch.Tensor,
        dh_x: torch.Tensor,
    ) -> tuple:
        """Run the NPU MHC pre backward operator."""
        del dh_pre
        x, phi, alpha, gamma, h_post, inv_rms, h_mix, h_pre = ctx.saved_tensors
        hc_eps = ctx.hc_eps
        dx, dphi, dalpha, dbias, dgamma = (
            torch.ops.custom.npu_manifold_constrained_hyper_connection_pre_grad(
                x,
                phi,
                alpha,
                dh_in,
                dh_post,
                dh_res,
                inv_rms,
                h_mix,
                h_pre,
                h_post,
                gamma=gamma,
                hc_eps=hc_eps,
                grad_x_post=dh_x,
            )
        )
        if not ctx.has_gamma:
            dgamma = None
        grads = [dx, dphi, dalpha, dbias, dgamma, None, None, None]
        return tuple(grads)


def hc_split_sinkhorn_torch(
    weight: torch.Tensor,
    branch_alpha: torch.Tensor,
    branch_beta: torch.Tensor,
    num_stream: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive MHC pre, post, and residual coefficients from projection weights.

    Args:
        weight: Projected MHC coefficient logits.
        branch_alpha: Learned scale for each coefficient group.
        branch_beta: Learned bias for each coefficient group.
        num_stream: Number of residual streams.
        sinkhorn_iters: Number of Sinkhorn iterations.
        eps: Numerical stability epsilon.

    Returns:
        Pre-connection, post-connection, and residual mixing coefficients.
    """
    h_pre, h_post, h_res = weight.split([num_stream, num_stream, num_stream * num_stream], dim=-1)
    h_res = h_res.unflatten(-1, (num_stream, num_stream))

    hpre_input_alpha = h_pre * branch_alpha[0]
    hpre_input_beta = branch_beta[:num_stream].unsqueeze(0).unsqueeze(0)
    h_pre = torch.sigmoid(hpre_input_alpha + hpre_input_beta) + eps

    hpost_input_alpha = h_post * branch_alpha[1]
    hpost_input_beta = branch_beta[num_stream:2 * num_stream].unsqueeze(0).unsqueeze(0)
    h_post = 2 * torch.sigmoid(hpost_input_alpha + hpost_input_beta)

    hres_input_alpha = h_res * branch_alpha[2]
    hres_input_beta = branch_beta[2 * num_stream:].view(num_stream, num_stream).unsqueeze(0).unsqueeze(0)
    h_res = hres_input_alpha + hres_input_beta
    h_res = sinkhorn_knopps(h_res, sinkhorn_iters, eps)

    return h_pre, h_post, h_res


def mhc_pre(
    x: torch.Tensor,
    phi: torch.Tensor,
    branch_alpha: torch.Tensor,
    branch_beta: torch.Tensor,
    num_stream: int,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    gamma: Optional[torch.Tensor] = None,
    hpre_renorm: bool = False,
    use_ascendc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare an input and its mixing coefficients for an MHC-wrapped block.

    Args:
        x: Flattened residual streams.
        phi: MHC projection weight.
        branch_alpha: Learned scale for each coefficient group.
        branch_beta: Learned bias for each coefficient group.
        num_stream: Number of residual streams.
        sinkhorn_iters: Number of Sinkhorn iterations.
        norm_eps: RMS normalization epsilon.
        hc_eps: Hyper-connection numerical stability epsilon.
        gamma: Optional RMS normalization scale.
        hpre_renorm: Whether to normalize pre-connection coefficients.
        use_ascendc: Whether to use the fused NPU custom operators.

    Returns:
        Block input, post coefficients, residual coefficients, and optional residual buffer.
    """
    shape, dtype = x.size(), x.dtype
    if use_ascendc:
        x = x.reshape(shape[0], shape[1], num_stream, -1)
        if gamma is not None:
            gamma = gamma.reshape(num_stream, -1).float()
        y, h_pre, h_post, h_comb_before, residual = _MhcPre.apply(
            x,
            phi.float(),
            branch_alpha.float(),
            branch_beta.float(),
            gamma,
            True,
            norm_eps,
            hc_eps,
        )
        h_res = sinkhorn(h_comb_before, sinkhorn_iters, hc_eps)
        residual = residual.reshape(shape[0], shape[1], -1)
        return y, h_post, h_res, residual

    x = x.float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    if gamma is not None:
        weight = F.linear(x * rsqrt * gamma, phi)  # pylint: disable=not-callable
    else:
        weight = F.linear(x, phi) * rsqrt  # pylint: disable=not-callable
    h_pre, h_post, h_res = hc_split_sinkhorn_torch(
        weight, branch_alpha, branch_beta, num_stream, sinkhorn_iters, hc_eps
    )
    if hpre_renorm:
        eps_cache = torch.full((), 1e-30, dtype=h_pre.dtype, device=h_pre.device)
        h_pre = h_pre / h_pre.sum(dim=-1, keepdim=True).maximum(eps_cache)
    y = torch.sum(h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(num_stream, -1)), dim=2).to(dtype)
    return y, h_post, h_res, None
