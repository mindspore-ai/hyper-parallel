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
"""Partial interleaved RoPE with fused forward and reference-compatible gradients."""

from functools import lru_cache
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import


@lru_cache(maxsize=1)
def _load_partial_rotary_mul() -> Callable[..., Any]:
    """Load the optional forward entry only when partial RoPE is requested."""
    # Keep the optional NPU extension out of CPU and reference-only imports.
    try:
        from cann_ops_transformer import inplace_partial_rotary_mul  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise RuntimeError(
            "use_fused_rope=True requires cann_ops_transformer.inplace_partial_rotary_mul. "
            "See hyper_parallel/components/functional/README.md for installation and supported inputs."
        ) from err
    return inplace_partial_rotary_mul


class _PartialRotaryMul(torch.autograd.Function):
    """Keep caller storage intact and preserve the reference's branch-wise casts."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate a private contiguous output, never a saved projection tensor."""
        ctx.save_for_backward(cos, sin)
        ctx.plain_dim = tensor.shape[-1] - cos.shape[-1]
        output = tensor.clone(memory_format=torch.contiguous_format)
        _load_partial_rotary_mul()(
            output, cos, sin, rotary_mode="interleave", partial_slice=[ctx.plain_dim, tensor.shape[-1]],
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Match the two low-precision casts of the eager rotation's adjoint."""
        cos, sin = ctx.saved_tensors
        plain, rotary = grad_output.split((ctx.plain_dim, cos.shape[-1]), dim=-1)
        floating = rotary.float()
        direct = (floating * cos).to(grad_output.dtype)
        rotated = (floating * sin).to(grad_output.dtype)
        del floating
        even, odd = rotated.unflatten(-1, (-1, 2)).unbind(-1)
        adjoint = torch.stack((odd, -even), dim=-1).flatten(-2)
        del rotated, even, odd
        tail = direct + adjoint
        del direct, adjoint
        return (torch.cat((plain, tail), dim=-1) if ctx.plain_dim else tail), None, None


def _validate_rotary_coefficients(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Check positional constants before preparing the native broadcast layout."""
    if cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("Partial RoPE requires matching cos/sin shapes [B, S, R/2]")
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("Partial RoPE requires constant cos/sin without gradients")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("Partial RoPE requires FP32 cos/sin to preserve V4.1 rounding")
    if cos.device != tensor.device or sin.device != tensor.device:
        raise ValueError("Partial RoPE input and cos/sin must share a device")


def partial_rotary_mul(
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Rotate the trailing channels without changing input or gradient storage.

    Args:
        tensor: Four-dimensional BSND or BNSD input on NPU.
        cos: FP32 coefficients [B, S, R/2], one frequency per adjacent pair.
            Batch and sequence dimensions may be broadcast from one.
        sin: Constant FP32 sine coefficients with the same shape as cos.
        unsqueeze_dim: 1 for BNSD, 2 for BSND.

    Returns:
        A tensor with the input's shape and dtype. Empty inputs keep connected
        gradients without a kernel launch. Backward uses Torch operations to
        retain the eager V4.1 rounding order; it does not call the native grad op.

    Raises:
        ValueError: Shapes, positional gradients or coefficient dtypes are unsupported.
        RuntimeError: Inputs are not on NPU or the required extension is missing.
    """
    if tensor.ndim != 4 or unsqueeze_dim not in (1, 2):
        raise ValueError("Partial RoPE requires a 4D tensor and unsqueeze_dim 1 or 2")
    _validate_rotary_coefficients(tensor, cos, sin)
    if tensor.device.type != "npu":
        raise RuntimeError(f"Fused partial RoPE requires NPU inputs; got device={tensor.device}")
    value = tensor if unsqueeze_dim == 2 else tensor.transpose(1, 2)
    if cos.shape[0] not in (1, value.shape[0]) or cos.shape[1] not in (1, value.shape[1]):
        raise ValueError("Partial RoPE cos/sin batch and sequence dimensions must broadcast to the input")
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > tensor.shape[-1]:
        raise ValueError("Partial RoPE frequencies cannot exceed the input head dimension")
    if not tensor.numel() or not rotary_dim:
        return tensor.clone()
    cosine = cos.expand(value.shape[0], -1, -1).repeat_interleave(2, dim=-1).unsqueeze(2).contiguous()
    sine = sin.expand(value.shape[0], -1, -1).repeat_interleave(2, dim=-1).unsqueeze(2).contiguous()
    output = _PartialRotaryMul.apply(value, cosine, sine)
    return output if unsqueeze_dim == 2 else output.transpose(1, 2)
