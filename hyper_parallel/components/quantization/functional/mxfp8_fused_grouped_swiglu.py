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
"""Fused SwiGLU/quantization within one grouped-expert autograd boundary."""

from typing import TYPE_CHECKING, Optional

import torch  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.ops.npu_mxfp8 import mxfp8_grouped_matmul
from hyper_parallel.components.quantization.tensor import MXFP8Tensor

if TYPE_CHECKING:
    from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer


def _save_operands(ctx, counts, projected, operands):
    """Expose physical tensors to saved-tensor hooks, never carrier wrappers."""
    ctx.operand_metadata = tuple((tuple(q.shape), q.dtype, q.quantizer) for q in operands)
    buffers = tuple(buffer for q in operands for buffer in (q.row_data, q.row_scale, q.col_data, q.col_scale))
    ctx.save_for_backward(counts, projected, *buffers)


def _restore_operands(ctx):
    """Use the same physical-buffer reconstruction as the upstream Dense/GMM path."""
    counts, projected, *buffers = ctx.saved_tensors
    operands = []
    for index, (shape, dtype, quantizer) in enumerate(ctx.operand_metadata):
        row, row_scale, col, col_scale = buffers[4 * index:4 * index + 4]
        operands.append(MXFP8Tensor(shape, dtype, quantizer=quantizer, row_data=row,
                                   row_scale=row_scale, col_data=col, col_scale=col_scale))
    return counts, projected, operands


def _fused_quant(projected, quantizer, counts, gradient=None):
    """Keep prequantized operands internal to this Function's GMM calls."""
    options = {"quant_dtype": quantizer.quant_dtype, "group_index": counts.cumsum(0).to(torch.int64)}
    if gradient is None:
        buffers = quantizer.npu_ops.swiglu_mx_quant_dual_axis(projected, **options)
        shape = (projected.shape[0], projected.shape[1] // 2)
    else:
        buffers = quantizer.npu_ops.swiglu_backward_mx_quant_dual_axis(
            projected, gradient.contiguous(), **options)
        shape = projected.shape
    row, row_scale, col, col_scale = buffers
    return MXFP8Tensor(shape, projected.dtype, quantizer=quantizer,
                       row_data=row, row_scale=row_scale, col_data=col, col_scale=col_scale)


def _gmm(left, right, counts, dtype, layout="NN"):
    """Use the existing MXFP8 adapter and count-based expert partitioning."""
    return mxfp8_grouped_matmul(left, right, layout=layout, group_list=counts,
                               group_type=2 if layout == "TN" else 0,
                               group_list_type=1, output_dtype=dtype)


class _FusedGroupedSwiGLU(torch.autograd.Function):
    """Own both GMMs and fusion so external autograd edges remain ordinary tensors."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, inputs: torch.Tensor,
        gate_up: torch.Tensor, down: torch.Tensor, counts: torch.Tensor, quantizer: "MXFP8Quantizer",
    ) -> torch.Tensor:
        """Compute two MXFP8 GMMs with a fused activation/quantization boundary.

        Args:
            ctx: Autograd save context.
            inputs: Expert-major token matrix.
            gate_up: Canonical gate/up weights.
            down: Canonical down weights.
            counts: Per-expert token counts.
            quantizer: MXFP8 quantizer.
        """
        ctx.shapes = (inputs.shape, gate_up.shape, down.shape)
        ctx.dtype, ctx.device, ctx.quantizer = inputs.dtype, inputs.device, quantizer
        if inputs.shape[0] == 0:
            ctx.save_for_backward()
            return inputs.new_empty((0, down.shape[-1]))
        input_q = quantizer.quantize(inputs, group_list=counts, group_list_type=1, rowwise=True, colwise=True)
        gate_up_q = quantizer.quantize(gate_up.contiguous(), rowwise=True, colwise=True)
        down_q = quantizer.quantize(down.contiguous(), rowwise=True, colwise=True)
        projected = _gmm(input_q, gate_up_q, counts, inputs.dtype)
        activated_q = _fused_quant(projected, quantizer, counts)
        output = _gmm(activated_q, down_q, counts, inputs.dtype)
        _save_operands(ctx, counts, projected, (input_q, gate_up_q, activated_q, down_q))
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, gradient: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        """Fuse SwiGLU derivative/quantization and compute requested first gradients.

        Args:
            ctx: Saved forward context.
            gradient: Output gradient.
        """
        needs_input, needs_gate, needs_down = ctx.needs_input_grad[:3]
        if ctx.shapes[0][0] == 0:
            grads = [torch.zeros(shape, device=ctx.device, dtype=ctx.dtype) if needed else None
                     for shape, needed in zip(ctx.shapes, (needs_input, needs_gate, needs_down))]
            return *grads, None, None
        counts, projected, (input_q, gate_up_q, activated_q, down_q) = _restore_operands(ctx)
        grad_q = ctx.quantizer.quantize(
            gradient, group_list=counts if needs_down else None, group_list_type=1,
            rowwise=needs_input or needs_gate, colwise=needs_down)
        grad_down = _gmm(activated_q, grad_q, counts, ctx.dtype, "TN") if needs_down else None
        grad_input = grad_gate = None
        if needs_input or needs_gate:
            grad_activated = _gmm(grad_q, down_q, counts, ctx.dtype, "NT")
            grad_projected_q = _fused_quant(projected, ctx.quantizer, counts, grad_activated)
            if needs_input:
                grad_input = _gmm(grad_projected_q, gate_up_q, counts, ctx.dtype, "NT")
            if needs_gate:
                grad_gate = _gmm(input_q, grad_projected_q, counts, ctx.dtype, "TN")
        return grad_input, grad_gate, grad_down, None, None


def fused_grouped_swiglu(
    inputs: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor,
    counts: torch.Tensor, quantizer: "MXFP8Quantizer",
) -> torch.Tensor:
    """Compute expert-major outputs with canonical [E,H,2I]/[E,I,H] weights.

    Args:
        inputs: BF16 expert-major token matrix.
        gate_up: Canonical packed gate/up weights.
        down: Canonical down weights.
        counts: Per-local-expert token counts, including zero-token experts.
        quantizer: MXFP8 recipe with both fused NPU kernels available.

    Returns:
        Ordinary BF16 outputs before outer routing weights are applied.
    """
    if inputs.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Fused MXFP8 SwiGLU requires BF16 or FP16 inputs")
    if gate_up.dtype != inputs.dtype or down.dtype != inputs.dtype:
        raise ValueError("Fused MXFP8 SwiGLU requires matching input and weight dtypes")
    if inputs.ndim != 2 or gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("Fused MXFP8 SwiGLU requires 2D inputs and 3D expert weights")
    if (gate_up.shape[0] != down.shape[0] or gate_up.shape[1] != inputs.shape[1]
            or gate_up.shape[2] != 2 * down.shape[1] or down.shape[2] != inputs.shape[1]):
        raise ValueError("Fused MXFP8 SwiGLU weight dimensions do not match inputs")
    if counts.ndim != 1 or counts.numel() != gate_up.shape[0] or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("Fused MXFP8 SwiGLU requires one integer count per expert")
    return _FusedGroupedSwiGLU.apply(inputs, gate_up, down, counts, quantizer)
