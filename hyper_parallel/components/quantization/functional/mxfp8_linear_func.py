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
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.npu_mxfp8 import (
    mxfp8_matmul,
)
from hyper_parallel.components.quantization.quantizers.mxfp8 import (
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
        """Execute the bias-free MXFP8 forward.

        Args:
            ctx: Autograd context owning the saved quantized operands.
            inputs: High-precision input with the contracting dimension last.
            weight: High-precision weight in [out_features, in_features] layout.
            quantizer: MXFP8 quantizer shared by forward and backward.

        Returns:
            High-precision output preserving the input's leading dimensions.
        """

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
        input_quant.update_usage(
            rowwise=False,
            colwise=needs_grad_weight,
        )
        weight_quant.update_usage(
            rowwise=False,
            colwise=needs_grad_input,
        )
        ctx.save_for_backward(
            input_quant if needs_grad_weight else None,
            weight_quant if needs_grad_input else None,
        )
        if inputs.ndim != 2:
            output = output.reshape(*inputs.shape[:-1], output.shape[-1])
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        """Execute first-order dgrad and wgrad; higher derivatives are unsupported.

        Args:
            ctx: Autograd context containing saved operands and shape metadata.
            grad_output: High-precision gradient of the projection output.

        Returns:
            Input and weight gradients, followed by None for the quantizer.
        """

        input_quant, weight_quant = ctx.saved_tensors
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
                weight_quant,
                layout="NN",
                output_dtype=grad_output.dtype,
            )
            if len(ctx.input_shape) != 2:
                grad_input = grad_input.reshape(ctx.input_shape)
        if needs_grad_weight:
            grad_weight = mxfp8_matmul(
                grad_quant,
                input_quant,
                layout="TN",
                output_dtype=ctx.weight_dtype,
            )
        grad_quant.update_usage(rowwise=False, colwise=False)
        return grad_input, grad_weight, None


def mxfp8_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    quantizer: MXFP8Quantizer,
) -> torch.Tensor:
    """Apply the bias-free Dense MXFP8 autograd function.

    Args:
        inputs: High-precision input with the contracting dimension last.
        weight: High-precision weight in [out_features, in_features] layout.
        quantizer: MXFP8 quantizer shared by forward and backward.

    Returns:
        High-precision output preserving the input's leading dimensions.
    """

    return _MXFP8LinearFunction.apply(
        inputs,
        weight,
        quantizer,
    )


__all__ = ["mxfp8_linear"]
