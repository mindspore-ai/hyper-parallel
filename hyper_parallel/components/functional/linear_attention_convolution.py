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
from torch.nn import functional as F

if TYPE_CHECKING:
    from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo


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
        """Save preactivation and per-call metadata for joint backward."""
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
        """Differentiate SiLU once inside the native convolution backward."""
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
    """Prepend only the first fragment's true history without inventing a reset."""
    if info is None:
        if halo is None:
            return inputs, None, 0
        # Dense batch rows remain independent; the halo extends each row separately.
        return torch.cat((halo, inputs), dim=1), None, width - 1
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
        _validate_fused_history(inputs, halo, info, weight.shape[-1])


def _validate_fused_history(
    inputs: torch.Tensor,
    halo: torch.Tensor,
    info: PackedSequenceInfo | None,
    width: int,
) -> None:
    """Reject history that cannot represent the same document and activation layout."""
    batch_size = inputs.shape[0] if info is None else 1
    if halo.shape != (batch_size, width - 1, inputs.shape[-1]):
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
        halo: Already exchanged history for independent dense rows or packed tokens; no communication is added.
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
