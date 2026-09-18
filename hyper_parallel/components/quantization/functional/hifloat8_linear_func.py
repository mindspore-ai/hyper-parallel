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
"""HiFloat8 Dense forward, input-gradient, and weight-gradient."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.npu_hifloat8 import (
    hifloat8_matmul,
)
from hyper_parallel.components.quantization.quantizers.hifloat8 import (
    HiFloat8Quantizer,
)


def _as_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten leading dimensions while preserving the contracting axis."""

    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


class _HiFloat8LinearFunction(torch.autograd.Function):
    """Run Pangu-compatible HiFloat8 role quantizers through native A5 MMs."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        input_quantizer: HiFloat8Quantizer,
        weight_quantizer: HiFloat8Quantizer,
        grad_output_quantizer: HiFloat8Quantizer,
    ) -> torch.Tensor:
        """Execute bias-free HiFloat8 forward with separate X/W roles.

        Args:
            ctx: Autograd context owning the saved quantized operands.
            inputs: High-precision input with the contracting dimension last.
            weight: High-precision weight in [out_features, in_features] layout.
            input_quantizer: Current-scaling input recipe.
            weight_quantizer: Current-scaling weight recipe.
            grad_output_quantizer: Gradient recipe used by backward.

        Returns:
            High-precision output preserving the input's leading dimensions.
        """

        input_matrix = _as_matrix(inputs)
        needs_grad_input = inputs.requires_grad
        needs_grad_weight = weight.requires_grad
        input_quant = input_quantizer.quantize(
            input_matrix,
            rowwise=True,
            colwise=needs_grad_weight,
        )
        weight_quant = weight_quantizer.quantize(
            weight,
            rowwise=True,
            colwise=needs_grad_input,
        )
        output = hifloat8_matmul(
            input_quant,
            weight_quant,
            layout="NT",
            output_dtype=inputs.dtype,
        )
        ctx.input_shape = inputs.shape
        ctx.weight_dtype = weight.dtype
        ctx.grad_output_quantizer = grad_output_quantizer
        input_quant.update_usage(rowwise=False, colwise=needs_grad_weight)
        weight_quant.update_usage(rowwise=False, colwise=needs_grad_input)
        # Saved tensors are released by autograd, not by the first backward call.
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
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None, None]:
        """Execute first-order HiFloat8 dgrad and wgrad; higher derivatives are unsupported.

        Args:
            ctx: Autograd context containing saved operands and shape metadata.
            grad_output: High-precision gradient of the projection output.

        Returns:
            Input and weight gradients, followed by None for each quantizer.
        """

        input_quant, weight_quant = ctx.saved_tensors
        grad_matrix = _as_matrix(grad_output)
        needs_grad_input = ctx.needs_input_grad[0]
        needs_grad_weight = ctx.needs_input_grad[1]
        grad_quant = ctx.grad_output_quantizer.quantize(
            grad_matrix,
            rowwise=needs_grad_input,
            colwise=needs_grad_weight,
        )
        grad_input = None
        grad_weight = None
        if needs_grad_input:
            grad_input = hifloat8_matmul(
                grad_quant,
                weight_quant,
                layout="NN",
                output_dtype=grad_output.dtype,
            )
            if len(ctx.input_shape) != 2:
                grad_input = grad_input.reshape(ctx.input_shape)
        if needs_grad_weight:
            grad_weight = hifloat8_matmul(
                grad_quant,
                input_quant,
                layout="TN",
                output_dtype=ctx.weight_dtype,
            )
        grad_quant.update_usage(rowwise=False, colwise=False)
        return grad_input, grad_weight, None, None, None


def hifloat8_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    input_quantizer: HiFloat8Quantizer,
    weight_quantizer: HiFloat8Quantizer,
    grad_output_quantizer: HiFloat8Quantizer,
) -> torch.Tensor:
    """Apply the bias-free HiFloat8 Dense autograd function.

    Args:
        inputs: High-precision input with the contracting dimension last.
        weight: High-precision weight in [out_features, in_features] layout.
        input_quantizer: Current-scaling input recipe.
        weight_quantizer: Current-scaling weight recipe.
        grad_output_quantizer: Gradient recipe used by backward.

    Returns:
        Projection output with the input's logical dtype.
    """

    return _HiFloat8LinearFunction.apply(
        inputs,
        weight,
        input_quantizer,
        weight_quantizer,
        grad_output_quantizer,
    )


__all__ = ["hifloat8_linear"]
