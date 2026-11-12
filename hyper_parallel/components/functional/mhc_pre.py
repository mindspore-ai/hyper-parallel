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

from hyper_parallel.components.functional.sinkhorn import sinkhorn
from hyper_parallel.tools.logging import get_logger

logger = get_logger()


def _mhc_pre_backward(ctx, dh_in, dh_post, dh_res, dh_x):
    """Invoke the legacy omni MHC gradient operator with the saved forward state."""
    x, phi, alpha, gamma, h_post, inv_rms, h_mix, h_pre = ctx.saved_tensors
    return torch.ops.custom.npu_manifold_constrained_hyper_connection_pre_grad(
        x, phi, alpha, dh_in, dh_post, dh_res, inv_rms, h_mix, h_pre, h_post,
        gamma=gamma,
        hc_eps=ctx.hc_eps,
        grad_x_post=dh_x,
    )


class _MhcPre(torch.autograd.Function):
    """Autograd bridge for the legacy omni NPU MHC pre operator."""

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
        """Run the legacy omni NPU MHC pre forward operator."""
        # The package is optional and is needed only for the legacy omni backend.
        # pylint: disable=import-outside-toplevel
        try:
            import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import
        except ImportError as error:
            raise ImportError("MHC backend='omni' requires omni_training_custom_ops") from error
        # pylint: enable=import-outside-toplevel
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
        """Run the legacy omni NPU MHC pre backward operator."""
        del dh_pre
        dx, dphi, dalpha, dbias, dgamma = _mhc_pre_backward(ctx, dh_in, dh_post, dh_res, dh_x)
        if not ctx.has_gamma:
            dgamma = None
        grads = [dx, dphi, dalpha, dbias, dgamma, None, None, None]
        return tuple(grads)


def _cann_mhc_pre(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    num_stream: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    gamma: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused ops-transformer MHC pre and Sinkhorn operator.

    The CANN operator has no RMS-scale input, so an optional gamma is folded
    into the projection as a differentiable ``phi * gamma`` product; autograd
    then keeps the phi and gamma gradients exact.
    """
    # The package is optional and is needed only for this selected CANN path.
    from cann_ops_transformer.ops import (  # pylint: disable=import-outside-toplevel
        mhc_pre_sinkhorn,
    )

    if gamma is not None:
        phi = phi * gamma.reshape(1, -1).to(phi.dtype)
    hin, h_post, h_res = mhc_pre_sinkhorn(
        x, phi, alpha, bias, num_stream, sinkhorn_iters, hc_eps, norm_eps
    )
    # The operator returns the flattened [.., N*N] residual matrix while the
    # shared post path consumes the square [.., N, N] layout.
    return hin, h_post, h_res.reshape(*h_res.shape[:-1], num_stream, num_stream)


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
    *,
    backend: str = "cann",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        backend: ``"cann"`` (default) calls the CANN ops-transformer
            ``mhc_pre_sinkhorn`` operator; the legacy ``"omni"`` backend is
            deprecated. CANN constraints: fp32 phi/alpha/beta, fp16/bf16
            ``x`` with a 128-aligned tail axis, and fixed N=4 streams with 20
            Sinkhorn iterations on Atlas A2/A3 series.

    Returns:
        Block input, post coefficients, residual coefficients, and residual buffer.
    """
    if backend not in ("cann", "omni"):
        raise ValueError(f"backend must be 'cann' or 'omni', got {backend!r}")
    shape = x.size()
    x = x.reshape(shape[0], shape[1], num_stream, -1)
    if gamma is not None:
        gamma = gamma.reshape(num_stream, -1).float()
    if backend == "cann":
        hin, h_post, h_res = _cann_mhc_pre(
            x,
            phi.float(),
            branch_alpha.float(),
            branch_beta.float(),
            num_stream,
            sinkhorn_iters,
            norm_eps,
            hc_eps,
            gamma,
        )
        return hin, h_post, h_res, x.reshape(shape[0], shape[1], -1)
    logger.warning(
        "mhc_pre backend='omni' is deprecated and will be removed in a future release; "
        "use backend='cann'."
    )
    y, _, h_post, h_comb_before, residual = _MhcPre.apply(
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
