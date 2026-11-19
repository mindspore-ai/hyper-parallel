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
"""Memory-efficient MHC coefficients with independent pre-coefficient gradients."""

import math
from typing import Any

import torch
from torch.autograd.function import once_differentiable


def _native_pre(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
    """Load the optional Ascend backend only when the kernel is selected."""
    import torch_npu  # pylint: disable=import-outside-toplevel

    return torch_npu.npu_mhc_pre(*args, **kwargs)


def _native_rms_backward(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the FP32 RMS derivative without materializing broadcast products."""
    import torch_npu  # pylint: disable=import-outside-toplevel

    return torch_npu.npu_rms_norm_backward(*args)


class _MhcCoefficients(torch.autograd.Function):
    """Differentiate pre, post and residual logits independently of stream collapse."""

    @staticmethod
    def forward(
            ctx: Any, x: torch.Tensor, phi: torch.Tensor, alpha: torch.Tensor, bias: torch.Tensor,
            norm_eps: float, hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep the source BF16 streams and small FP32 statistics for backward."""
        _, post, residual_logits, inv_rms, raw_mix, pre = _native_pre(
            x, phi, alpha, bias, norm_eps=norm_eps, hc_eps=hc_eps, out_flag=1,
        )
        ctx.save_for_backward(x, phi, alpha, inv_rms, raw_mix, pre, post)
        ctx.hc_eps = hc_eps
        return pre, post, residual_logits

    @staticmethod
    @once_differentiable
    def backward(
            ctx: Any, grad_pre: torch.Tensor, grad_post: torch.Tensor, grad_residual: torch.Tensor,
    ) -> tuple:
        """Propagate a pre gradient even when it was consumed by another sublayer."""
        x, phi, alpha, inv_rms, raw_mix, pre, post = ctx.saved_tensors
        streams, hidden = x.shape[-2:]
        width = streams * hidden
        inverse = inv_rms.reshape(*x.shape[:-2], 1)
        pre_sigmoid = pre - ctx.hc_eps
        grad_logits = torch.cat((
            grad_pre * pre_sigmoid * (1 - pre_sigmoid),
            grad_post * post * (1 - post / 2),
            grad_residual.flatten(-2),
        ), dim=-1)
        normalized_mix = raw_mix * inverse
        slices = ((0, streams), (streams, 2 * streams), (2 * streams, (streams + 2) * streams))
        grad_alpha = torch.stack([
            (grad_logits[..., start:end] * normalized_mix[..., start:end]).sum()
            for start, end in slices
        ])
        grad_bias = grad_logits.reshape(-1, phi.shape[0]).sum(0)
        scales = torch.cat((alpha[0].expand(streams), alpha[1].expand(streams),
                            alpha[2].expand(streams * streams)))
        grad_mix = grad_logits * scales
        flat = x.reshape(-1, width).float()
        # Scale the small adjoint instead of storing the full normalized FP32 input.
        grad_phi = (grad_mix * inverse).reshape(-1, phi.shape[0]).T @ flat
        grad_normalized = (grad_mix.reshape(-1, phi.shape[0]) @ phi).contiguous()
        grad_x, _ = _native_rms_backward(
            grad_normalized, flat, torch.ones(width, device=x.device, dtype=torch.float32), inverse.reshape(-1, 1),
        )
        return grad_x.reshape_as(x).to(x.dtype), grad_phi, grad_alpha, grad_bias, None, None


def mhc_coefficients(
        x: torch.Tensor, phi: torch.Tensor, alpha: torch.Tensor, bias: torch.Tensor,
        *, norm_eps: float = 1e-6, hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute unweighted-RMS MHC coefficients before residual Sinkhorn projection.

    Args:
        x: BF16/FP16 streams with shape ``[T, streams, hidden]`` or ``[B, S, streams, hidden]``.
        phi: FP32 mapping weight with shape ``[streams * (streams + 2), streams * hidden]``.
        alpha: Three FP32 group scales.
        bias: FP32 mapping bias matching the first dimension of ``phi``.
        norm_eps: Positive unweighted RMS epsilon.
        hc_eps: Positive offset added to the pre sigmoid.

    Returns:
        FP32 pre and post coefficients and unnormalized residual mixing logits.
        Stream collapse is deliberately separate, so pre may act on another sublayer's streams.

    Raises:
        ValueError: If tensor shapes, dtypes, device or epsilons violate the native contract.

    Note:
        Requires Ascend 950 native MHC Pre and RMS backward kernels. Supports first-order gradients.
    """
    if x.device.type != "npu" or x.dtype not in (torch.bfloat16, torch.float16) or x.ndim not in (3, 4):
        raise ValueError("MHC coefficients require 3-D/4-D BF16 or FP16 streams on NPU")
    streams, hidden = x.shape[-2:]
    expected = (streams * (streams + 2), streams * hidden)
    if tuple(phi.shape) != expected or tuple(alpha.shape) != (3,) or tuple(bias.shape) != (expected[0],):
        raise ValueError("MHC coefficient weights, scales or bias have incompatible shapes")
    if any(tensor.dtype != torch.float32 or tensor.device != x.device for tensor in (phi, alpha, bias)):
        raise ValueError("MHC coefficient weights, scales and bias must be FP32 on the streams' device")
    if not all(math.isfinite(v) and v > 0 for v in (norm_eps, hc_eps)):
        raise ValueError("MHC coefficient epsilons must be positive")
    return _MhcCoefficients.apply(x, phi, alpha, bias, norm_eps, hc_eps)
