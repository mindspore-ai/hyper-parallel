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
"""Non-fused MXFP8 projection with physical-tensor autograd saves."""

from typing import TYPE_CHECKING, Optional

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.components.quantization.functional._saved_quantized import (
    save_quantized_operands, restore_quantized_operands,
)
from hyper_parallel.components.quantization.ops.npu_mxfp8 import mxfp8_grouped_matmul

if TYPE_CHECKING:
    from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer


def _gmm(left, right, counts, dtype, layout="NN"):
    """Dispatch the existing MXFP8 grouped operator using per-expert counts."""
    return mxfp8_grouped_matmul(left, right, layout=layout, group_list=counts,
                               group_type=2 if layout == "TN" else 0,
                               group_list_type=1, output_dtype=dtype)


class _MXFP8Projection(torch.autograd.Function):
    """Local projection with physical saves; shared GroupedLinear stays unchanged."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, inputs: torch.Tensor, weight: torch.Tensor,
        counts: torch.Tensor, quantizer: "MXFP8Quantizer",
    ) -> torch.Tensor:
        """Quantize canonical weights and save only requested backward operands.

        Args:
            ctx: Autograd save context.
            inputs: Expert-major token matrix.
            weight: Canonical projection weights.
            counts: Per-expert token counts.
            quantizer: MXFP8 quantizer.
        """
        ctx.shapes = (inputs.shape, weight.shape)
        ctx.dtype, ctx.device, ctx.quantizer = inputs.dtype, inputs.device, quantizer
        needs_input, needs_weight = ctx.needs_input_grad[:2]
        if inputs.shape[0] == 0:
            ctx.save_for_backward()
            return inputs.new_empty((0, weight.shape[-1]))
        xq = quantizer.quantize(inputs, group_list=counts if needs_weight else None, group_list_type=1,
                                rowwise=True, colwise=needs_weight)
        wq = quantizer.quantize(weight.contiguous(), rowwise=True, colwise=True)
        output = _gmm(xq, wq, counts, inputs.dtype)
        save_quantized_operands(ctx, xq if needs_weight else None, wq if needs_input else None, counts)
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, gradient: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        """Reconstruct temporary operands on every backward, including retained graphs.

        Args:
            ctx: Saved forward context.
            gradient: Output gradient.
        """
        needs_input, needs_weight = ctx.needs_input_grad[:2]
        if ctx.shapes[0][0] == 0:
            grads = [torch.zeros(shape, device=ctx.device, dtype=ctx.dtype) if needed else None
                     for shape, needed in zip(ctx.shapes, (needs_input, needs_weight))]
            return *grads, None, None
        counts, xq, wq = restore_quantized_operands(ctx)
        gq = ctx.quantizer.quantize(gradient.contiguous(), group_list=counts if needs_weight else None,
                                    group_list_type=1, rowwise=needs_input, colwise=needs_weight)
        dx = _gmm(gq, wq, counts, ctx.dtype, "NT") if needs_input else None
        dw = _gmm(xq, gq, counts, ctx.dtype, "TN") if needs_weight else None
        return dx, dw, None, None


def mxfp8_projection(inputs: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor,
                      quantizer: "MXFP8Quantizer") -> torch.Tensor:
    """Apply a local expert-major projection with canonical [E,K,N] weights.

    Args:
        inputs: Expert-major token matrix.
        weight: Canonical expert projection weights.
        counts: Token count for each expert.
        quantizer: MXFP8 operand quantizer.

    Returns:
        Unweighted expert-major outputs.
    """
    return _MXFP8Projection.apply(inputs, weight, counts, quantizer)
