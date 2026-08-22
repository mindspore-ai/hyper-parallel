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
"""MXFP8 Dense forward, input-gradient, and weight-gradient."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_models.components.training.low_precision.ops.npu_mxfp8 import (
    mxfp8_matmul,
)
from hyper_models.components.training.low_precision.quantizers.mxfp8 import (
    MXFP8Quantizer,
)


def _as_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten leading dimensions while preserving the contracting axis."""

    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


class _MXFP8LinearFunction(torch.autograd.Function):
    """Run forward, dgrad, and wgrad through A5 MXFP8 matrix multiplies."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        quantizer: MXFP8Quantizer,
    ) -> torch.Tensor:
        """Execute the bias-free MXFP8 forward."""

        input_matrix = _as_matrix(inputs)
        needs_grad_input = inputs.requires_grad
        needs_grad_weight = weight.requires_grad
        input_quant = quantizer.quantize(
            input_matrix,
            rowwise=True,
            colwise=needs_grad_weight,
        )
        weight_quant = quantizer.quantize(
            weight,
            rowwise=True,
            colwise=needs_grad_input,
        )
        output = mxfp8_matmul(
            input_quant,
            weight_quant,
            layout="NT",
            output_dtype=inputs.dtype,
        )
        ctx.input_shape = inputs.shape
        ctx.weight_dtype = weight.dtype
        ctx.quantizer = quantizer
        ctx.input_quant = input_quant if needs_grad_weight else None
        ctx.weight_quant = weight_quant if needs_grad_input else None
        input_quant.update_usage(
            rowwise=False,
            colwise=needs_grad_weight,
        )
        weight_quant.update_usage(
            rowwise=False,
            colwise=needs_grad_input,
        )
        if inputs.ndim != 2:
            output = output.reshape(*inputs.shape[:-1], output.shape[-1])
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        """Execute dgrad and wgrad with one shared grad-output quantization."""

        grad_matrix = _as_matrix(grad_output)
        quantizer = ctx.quantizer
        grad_input = None
        grad_weight = None
        needs_grad_input = ctx.needs_input_grad[0]
        needs_grad_weight = ctx.needs_input_grad[1]
        grad_quant = quantizer.quantize(
            grad_matrix,
            rowwise=needs_grad_input,
            colwise=needs_grad_weight,
        )

        if needs_grad_input:
            grad_input = mxfp8_matmul(
                grad_quant,
                ctx.weight_quant,
                layout="NN",
                output_dtype=grad_output.dtype,
            )
            if len(ctx.input_shape) != 2:
                grad_input = grad_input.reshape(ctx.input_shape)
        if needs_grad_weight:
            grad_weight = mxfp8_matmul(
                grad_quant,
                ctx.input_quant,
                layout="TN",
                output_dtype=ctx.weight_dtype,
            )
        grad_quant.update_usage(rowwise=False, colwise=False)
        ctx.input_quant = None
        ctx.weight_quant = None
        return grad_input, grad_weight, None


def mxfp8_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    quantizer: MXFP8Quantizer,
) -> torch.Tensor:
    """Apply the bias-free Dense MXFP8 autograd function."""

    return _MXFP8LinearFunction.apply(
        inputs,
        weight,
        quantizer,
    )


__all__ = ["mxfp8_linear"]
