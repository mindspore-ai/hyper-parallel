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
"""Manifold hyper-connection post-processing functions."""

from typing import Any, Optional, Tuple

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as functional  # pylint: disable=forbidden-backend-import
import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import


class _MhcPost(torch.autograd.Function):
    """Autograd bridge for the NPU MHC post custom operator."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        """Run the NPU MHC post forward operator."""
        ctx.save_for_backward(x, h_res, h_out, h_post)
        return torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
            x,
            h_res,
            h_out,
            h_post,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the NPU MHC post backward operator."""
        x, h_res, h_out, h_post = ctx.saved_tensors
        grad_x, grad_h_res, grad_h_out, grad_h_post = torch.ops.custom.npu_ai_infra_mhc_post_grad(
            grad_output.contiguous(),
            x,
            h_res,
            h_out,
            h_post,
        )
        return grad_x, grad_h_res, grad_h_out, grad_h_post


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    num_stream: int,
    use_custom_op: bool = False,
) -> torch.Tensor:
    """Combine transformed and residual streams after an MHC-wrapped block.

    Args:
        x: Output of the wrapped block.
        residual: Flattened residual streams.
        h_post: Per-stream output mixing coefficients.
        h_res: Residual stream mixing matrix.
        num_stream: Number of residual streams.
        use_custom_op: Whether to use the fused NPU custom operator.

    Returns:
        Flattened mixed residual streams.
    """
    if use_custom_op:
        x_shape = x.size()
        residual_reshape = residual.reshape(x_shape[0], x_shape[1], num_stream, -1)
        y_flat = _MhcPost.apply(residual_reshape, h_res, x, h_post)
        y_flat = y_flat.flatten(2)
        return y_flat
    y = (
        h_post.unsqueeze(-1) * x.unsqueeze(-2)
        + torch.sum(
            h_res.unsqueeze(-1)
            * residual.unflatten(dim=-1, sizes=(num_stream, -1)).unsqueeze(-2),
            dim=2,
        )
    ).flatten(2)
    return y.type_as(x)


def mhc_post_process(
    x: torch.Tensor,
    phi: torch.Tensor,
    branch_alpha: torch.Tensor,
    branch_beta: torch.Tensor,
    num_stream: int,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    gamma: Optional[torch.Tensor] = None,
    hpre_renorm: bool = False,
) -> torch.Tensor:
    """Merge all residual streams at the end of an MHC stack.

    Args:
        x: Flattened residual streams.
        phi: Final MHC projection weight.
        branch_alpha: Learned coefficient scale.
        branch_beta: Learned coefficient bias.
        num_stream: Number of residual streams.
        norm_eps: RMS normalization epsilon.
        hc_eps: Hyper-connection numerical stability epsilon.
        gamma: Optional RMS normalization scale.
        hpre_renorm: Whether to normalize merge coefficients.

    Returns:
        The merged hidden states.
    """
    dtype = x.dtype
    x = x.float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    if gamma is not None:
        weight = functional.linear(x * rsqrt * gamma, phi)
    else:
        weight = functional.linear(x, phi) * rsqrt
    h_pre = functional.sigmoid(
        weight * branch_alpha + branch_beta.unsqueeze(0).unsqueeze(0)
    ) + hc_eps
    if hpre_renorm:
        eps_cache = torch.full((), 1e-30, dtype=h_pre.dtype, device=h_pre.device)
        h_pre = h_pre / h_pre.sum(dim=-1, keepdim=True).maximum(eps_cache)
    y = torch.sum(h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(num_stream, -1)), dim=2).to(dtype)
    return y
