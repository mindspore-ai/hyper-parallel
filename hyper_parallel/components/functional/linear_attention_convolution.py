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
"""Optional NPU kernels for dense and packed depthwise causal convolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import (
        PackedSequenceInfo,
    )


def validate_causal_convolution(convolution: nn.Conv1d, *, require_activation: bool = False) -> None:
    """Reject source convolutions whose semantics differ from the supported causal kernel.

    Args:
        convolution: Source depthwise short convolution.
        require_activation: KDA source ShortConv declares its SiLU activation.
    """
    if convolution.stride != (1,):
        raise ValueError("Linear-attention ShortConv requires stride=1")
    if convolution.groups != convolution.in_channels or convolution.out_channels != convolution.in_channels:
        raise ValueError("Linear-attention ShortConv requires depthwise equal input/output channels")
    padding = (convolution.kernel_size[0] - 1) * convolution.dilation[0]
    if convolution.padding != (padding,) or convolution.padding_mode != "zeros":
        raise ValueError("Linear-attention ShortConv requires causal zero padding")
    if require_activation and getattr(convolution, "activation", None) not in ("silu", "swish"):
        raise ValueError("KDA ShortConv requires a declared SiLU/Swish activation")


def validate_conv_backend(backend: str) -> str:
    """Validate an explicit convolution backend without loading kernels.

    Args:
        backend: One of ``eager``, ``triton``, or ``ascendc``.

    Returns:
        The validated backend, without changing its spelling or default.

    Raises:
        ValueError: If the backend is unsupported.
    """
    if backend not in ("eager", "triton", "ascendc"):
        raise ValueError(f"Unsupported convolution backend {backend!r}; expected eager, triton, or ascendc.")
    return backend


class _AscendCConvolution(torch.autograd.Function):
    """Shared dense/packed convolution with joint SiLU and convolution backward."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        offsets: tuple[int, ...] | None,
    ) -> torch.Tensor:
        """Save preactivation and per-call metadata for joint backward.

        Args:
            ctx: Autograd context owning this call's saved tensors.
            inputs: Dense or packed token activations, including any valid history.
            weight: Depthwise convolution parameter with shape ``[C,1,W]``.
            bias: Optional channel bias.
            offsets: Packed document boundaries, or None for independent dense samples.
        """
        # Optional native operators must not be loaded by ordinary framework imports.
        from fla_npu.ops.ascendc import causal_conv1d_fn  # pylint: disable=import-outside-toplevel

        op_inputs = inputs if offsets is None else inputs.reshape(-1, inputs.shape[-1])
        op_inputs = op_inputs.contiguous()
        op_weight = weight[:, 0].transpose(0, 1).contiguous()
        cpu_offsets = None if offsets is None else torch.tensor(offsets, dtype=torch.int64, device="cpu")
        preactivation = causal_conv1d_fn(
            op_inputs, op_weight, bias, conv_states=None, activation=None,
            query_start_loc_cpu=cpu_offsets, head_num=0,
        )
        ctx.save_for_backward(op_inputs, op_weight, preactivation, cpu_offsets)
        ctx.input_shape = tuple(inputs.shape)
        ctx.has_bias = bias is not None
        ctx.input_layout = "BSND" if offsets is None else "TND"
        return F.silu(preactivation).reshape_as(inputs)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        """Differentiate SiLU once inside the native convolution backward.

        Args:
            ctx: Autograd context holding this call's forward tensors and layout.
            grad_output: Gradient of the activated convolution output.
        """
        # Keep the optional native dependency lazy in backward as well as forward.
        from fla_npu.ops.ascendc import causal_conv1d_bwd  # pylint: disable=import-outside-toplevel

        inputs, weight, preactivation, cpu_offsets = ctx.saved_tensors
        grad_input, grad_weight, grad_bias, _ = causal_conv1d_bwd(
            x=inputs, y=preactivation, weight=weight,
            dy=grad_output.reshape_as(inputs).contiguous(),
            query_start_loc=cpu_offsets, activation=1, input_layout=ctx.input_layout,
        )
        return (
            grad_input.reshape(ctx.input_shape),
            grad_weight.transpose(0, 1).unsqueeze(1).contiguous(),
            grad_bias if ctx.has_bias else None,
            None,
        )


def _with_history(
    inputs: torch.Tensor,
    halo: torch.Tensor | None,
    info: PackedSequenceInfo | None,
    width: int,
) -> tuple[torch.Tensor, tuple[int, ...] | None, int]:
    """Prepend valid history while preserving dense samples and packed resets."""
    if info is None:
        if halo is None:
            return inputs, None, 0
        # Dense batch samples are independent; a CP cut is not a document reset.
        return torch.cat((halo, inputs), dim=1), None, halo.shape[1]
    if halo is None:
        return inputs, info.local_offsets, 0
    prefix = min(width - 1, info.preceding_sample_tokens)
    # The empty slice preserves a zero halo gradient when a real document resets.
    extended = torch.cat((halo[:, width - 1 - prefix:], inputs), dim=1)
    offsets = (0, *(offset + prefix for offset in info.local_offsets[1:]))
    return extended, offsets, prefix


def _triton_convolution(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    info: PackedSequenceInfo | None,
    offsets: tuple[int, ...] | None,
    use_plan: bool,
) -> torch.Tensor:
    """Use one FLA entry for dense batches and immutable packed document plans."""
    # FLA is optional and must not be imported for the default eager backend.
    from fla.modules.conv import causal_conv1d  # pylint: disable=import-outside-toplevel

    if info is None:
        cu_seqlens, cu_seqlens_cpu, chunk_indices = None, None, None
    elif use_plan:
        plan = info.materialize(inputs.device, 64)
        cu_seqlens, cu_seqlens_cpu = plan.cu_seqlens, plan.cu_seqlens_cpu
        chunk_indices = plan.chunk_indices
    else:
        cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.int64, device="cpu")
        cu_seqlens = cu_seqlens_cpu.to(device=inputs.device, dtype=torch.int32)
        chunk_indices = None
    # Contiguous BTD enables the NPU kernel without its strided fallback workspace.
    output, _ = causal_conv1d(
        inputs.contiguous(), weight=weight[:, 0], bias=bias, activation=None, backend="triton",
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, chunk_indices=chunk_indices,
    )
    return F.silu(output)


def _validate_fused_inputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    info: PackedSequenceInfo | None,
    halo: torch.Tensor | None,
) -> None:
    """Check layouts before a selected backend is allowed to launch kernels."""
    if inputs.ndim != 3 or any(size < 1 for size in inputs.shape):
        raise ValueError("Fused convolution expects nonempty [B,S,C] or packed [1,T,C] inputs.")
    if weight.ndim != 3 or weight.shape[:2] != (inputs.shape[-1], 1):
        raise ValueError("Fused convolution weight must have shape [C,1,W] matching input channels.")
    if bias is not None and bias.shape != (inputs.shape[-1],):
        raise ValueError("Fused convolution bias must have shape [C] matching input channels.")
    if weight.device != inputs.device or (bias is not None and bias.device != inputs.device):
        raise ValueError("Fused convolution input, weight and bias must be on the same device.")
    if info is not None and inputs.shape[:2] != (1, info.length):
        raise ValueError("Packed convolution expects [1,T,C] matching its token interval.")
    if halo is not None:
        _validate_fused_history(inputs, halo, weight.shape[-1])


def _validate_fused_history(
    inputs: torch.Tensor,
    halo: torch.Tensor,
    width: int,
) -> None:
    """Check the received history layout for dense and packed inputs."""
    if halo.shape != (inputs.shape[0], width - 1, inputs.shape[-1]):
        raise ValueError("Fused convolution halo has an incompatible shape.")
    if halo.dtype != inputs.dtype or halo.device != inputs.device:
        raise ValueError("Fused convolution halo must match the input dtype and device.")


def fused_causal_conv1d(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    info: PackedSequenceInfo | None = None,
    halo: torch.Tensor | None = None,
    *,
    backend: str,
    dilation: int = 1,
) -> torch.Tensor:
    """Apply fused depthwise causal convolution and SiLU to dense or packed tokens.

    Args:
        inputs: Dense ``[B,S,C]`` or packed ``[1,T,C]`` projected tokens.
        weight: The original parameter's local view with shape ``[C,1,W]``.
        bias: Optional local bias with shape ``[C]``.
        info: Packed document boundaries, or None for independent dense batch samples.
        halo: Already exchanged ``[B,W-1,C]`` history for dense batches, or
            ``[1,W-1,C]`` history whose valid prefix is selected by packed metadata.
            This helper adds no communication.
        backend: Explicit ``triton`` or ``ascendc`` selection, with no fallback.
        dilation: Source convolution dilation; fused kernels require one.

    Returns:
        Local SiLU output with the input shape and dtype.

    Raises:
        ValueError: If the backend, device, dilation, or native dtype/shape is unsupported.
        ImportError: If the selected optional kernel package is missing.

    Note:
        AscendC jointly differentiates SiLU and convolution using a saved low-precision
        preactivation. Unlike eager or Triton, its backward does not round the SiLU
        gradient to the input dtype before convolution backward. Dense batch samples
        remain separate; only packed inputs with explicit metadata are flattened.
    """
    if backend not in ("triton", "ascendc"):
        raise ValueError(f"Fused convolution requires triton or ascendc, got {backend!r}.")
    if dilation != 1:
        raise ValueError("Fused convolution requires dilation=1.")
    if inputs.device.type != "npu":
        raise ValueError("Fused convolution currently supports NPU tensors only.")
    _validate_fused_inputs(inputs, weight, bias, info, halo)
    width = weight.shape[-1]
    if backend == "ascendc":
        if width not in (2, 3, 4) or inputs.shape[-1] % 16:
            raise ValueError("AscendC convolution requires width 2/3/4 and channels divisible by 16.")
        if inputs.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("AscendC convolution requires BF16 or FP16 activations.")
        if weight.dtype != inputs.dtype or (bias is not None and bias.dtype != inputs.dtype):
            raise ValueError("AscendC convolution requires matching input/weight/bias dtypes; no implicit cast.")
    extended, offsets, prefix = _with_history(inputs, halo, info, width)
    if backend == "triton":
        output = _triton_convolution(extended, weight, bias, info, offsets, halo is None)
    else:
        output = _AscendCConvolution.apply(extended, weight, bias, offsets)
    return output[:, prefix:]


class _PackedConvTap(torch.autograd.Function):
    """Recreate a masked tap in backward instead of retaining its FP32 copy."""

    @staticmethod
    def forward(
        ctx: Any, extended: torch.Tensor, weight: torch.Tensor, valid: torch.Tensor, offset: int, length: int,
    ) -> torch.Tensor:
        """Save shared input storage while preserving the eager product precision.

        Args:
            ctx: Autograd context owning tensors and communication metadata for this invocation.
            extended: Input tokens extended by the predecessor halo.
            weight: Depthwise convolution weights.
            valid: Mask excluding taps that cross document boundaries.
            offset: Start of the local output within the extended input.
            length: Number of local output tokens.
        """
        ctx.save_for_backward(extended, weight, valid)
        ctx.offset, ctx.length = offset, length
        values = extended[:, offset : offset + length].to(weight.dtype)
        values = torch.where(valid[None, :, None], values, 0.0)
        return values * weight[None, None, :]

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        """Rebuild only this tap; differentiable autograd operations retain higher derivatives.

        Args:
            ctx: Autograd context owning tensors and communication metadata for this invocation.
            grad_output: Gradient of the local operator output.
        """
        extended, weight, valid = ctx.saved_tensors
        grad_extended = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_values = grad_output * weight[None, None, :]
            grad_values = torch.where(valid[None, :, None], grad_values, 0.0).to(extended.dtype)
            grad_extended = F.pad(grad_values, (0, 0, ctx.offset, extended.shape[1] - ctx.offset - ctx.length))
        if ctx.needs_input_grad[1]:
            values = extended[:, ctx.offset : ctx.offset + ctx.length].to(weight.dtype)
            values = torch.where(valid[None, :, None], values, 0.0)
            grad_weight = (grad_output * values).sum(dim=(0, 1))
        return grad_extended, grad_weight, None, None, None


def packed_causal_conv1d(
    inputs: torch.Tensor,
    convolution: nn.Conv1d,
    info: PackedSequenceInfo,
    halo: torch.Tensor | None = None,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    backend: str = "eager",
) -> torch.Tensor:
    """Apply depthwise causal SiLU convolution with exact sample isolation.

    Vectorize over tokens rather than loop over samples. Accumulate low-precision
    products in FP32; keep FP32/FP64 reference inputs at their original precision.
    Masking taps also masks their autograd contribution to an exchanged halo.
    Backward reconstructs each masked tap from the shared input rather than
    retaining a full accumulation-precision input for every window position.

    Args:
        inputs: Local projected tokens with shape ``[1, T, C]``.
        convolution: Source depthwise, stride-one convolution attributes and parameters.
        info: True document boundaries and the current local token interval.
        halo: Optional already exchanged causal history; this function adds no communication.
        weight: Optional local parameter view with shape ``[C, 1, W]``.
        bias: Optional local bias; used with an explicit weight override.
        backend: ``eager`` by default, or explicitly ``triton`` / ``ascendc``
            for optional NPU kernels. The selected backend never silently falls back.

    Returns:
        SiLU output with the input shape and dtype.

    Raises:
        ValueError: If metadata, depthwise shape, or the selected kernel contract is invalid.
        ImportError: If an explicitly selected optional kernel package is unavailable.
    """
    validate_conv_backend(backend)
    if inputs.ndim != 3 or inputs.shape[:2] != (1, info.length):
        raise ValueError("Packed convolution expects [1,T,C] matching its token interval.")
    channels = inputs.shape[-1]
    if (
        convolution.groups != convolution.in_channels
        or convolution.out_channels != convolution.in_channels
        or convolution.stride != (1,)
    ):
        raise ValueError("Packed convolution requires a stride-one depthwise Conv1d.")
    if weight is None:
        weight, bias = convolution.weight, convolution.bias
    width, dilation = convolution.kernel_size[0], convolution.dilation[0]
    if weight.shape != (channels, 1, width):
        raise ValueError("Packed convolution weight must match the local head channels.")
    halo_width = (width - 1) * dilation
    if halo is not None and halo.shape != (1, halo_width, channels):
        raise ValueError("Packed convolution halo has an incompatible shape.")
    if backend != "eager":
        return fused_causal_conv1d(
            inputs, weight, bias, info, halo, backend=backend, dilation=dilation,
        )
    if halo is None:
        halo = inputs.new_zeros((1, halo_width, channels))
    extended = torch.cat((halo, inputs), dim=1)
    positions = torch.arange(info.start, info.end, device=inputs.device)
    boundaries = torch.tensor(info.global_offsets, device=inputs.device)
    sample_indices = torch.bucketize(positions, boundaries[1:], right=True)
    starts = boundaries[sample_indices]
    dtype = torch.float32 if inputs.dtype in (torch.bfloat16, torch.float16) else inputs.dtype
    output = torch.zeros_like(inputs, dtype=dtype)
    weights = weight[:, 0].to(dtype)
    for tap in range(width):
        valid = positions - (width - 1 - tap) * dilation >= starts
        output = output + _PackedConvTap.apply(extended, weights[:, tap], valid, tap * dilation, info.length)
    if bias is not None:
        output = output + bias.to(dtype)
    return F.silu(output.to(inputs.dtype))


def run_causal_short_conv(
    tensor: torch.Tensor,
    convolution: nn.Conv1d,
    halo: torch.Tensor | None = None,
    sequence_info: PackedSequenceInfo | None = None,
    backend: str = "eager",
) -> torch.Tensor:
    """Apply one local ShortConv, optionally with a preceding-rank halo.

    Args:
        tensor: Tensor to reshape or redistribute without changing token order.
        convolution: Depthwise causal convolution module providing weights and activation.
        halo: Optional predecessor tokens required by the convolution.
        sequence_info: Immutable document metadata for this local token interval.
        backend: Explicit local operator implementation to dispatch.
    """
    if backend != "eager":
        return fused_causal_conv1d(
            tensor, convolution.weight, convolution.bias, sequence_info, halo,
            backend=backend, dilation=convolution.dilation[0])
    if sequence_info is not None:
        return packed_causal_conv1d(tensor, convolution, sequence_info, halo)
    if halo is None:
        conv_input = tensor.transpose(1, 2)
        padding = convolution.padding
    else:
        conv_input = torch.cat((halo, tensor), dim=1).transpose(1, 2)
        padding = 0
    output = F.conv1d(  # pylint: disable=not-callable
        input=conv_input,
        weight=convolution.weight,
        bias=convolution.bias,
        stride=convolution.stride,
        padding=padding,
        dilation=convolution.dilation,
        groups=tensor.shape[-1],
    )
    if halo is None:
        output = output[:, :, : tensor.shape[1]]
    return F.silu(output).transpose(1, 2)
