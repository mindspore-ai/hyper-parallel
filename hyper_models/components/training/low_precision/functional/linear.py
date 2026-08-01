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
"""Dense MXFP8 forward, input-gradient, and weight-gradient functions."""

from typing import TYPE_CHECKING, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_models.components.training.low_precision.ops import mxfp8_matmul
from hyper_models.components.training.low_precision.quantizers import (
    MXFP8Quantizer,
)
from hyper_models.components.training.low_precision.tensor import MXFP8Tensor

if TYPE_CHECKING:
    from hyper_models.components.training.low_precision.observer.bridge import (
        PrecisionDebugSession,
    )


def _as_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten leading dimensions while preserving the contracting axis."""

    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def _dequantize_rowwise(tensor: MXFP8Tensor) -> torch.Tensor:
    """Rebuild one MXFP8 row-wise operand without invoking another GEMM."""
    if not tensor.is_rowwise():
        raise ValueError("MXFP8 observation requires row-wise operand storage")
    data = tensor.row_data
    scale = tensor.row_scale
    if data is None or scale is None:  # Satisfy static and runtime contracts.
        raise ValueError("MXFP8 observation has incomplete row-wise storage")
    if tensor.quantizer.npu_ops.is_e8m0_dtype(scale.dtype):
        # E8M0 stores an unsigned exponent with bias 127.
        scale = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    else:
        scale = scale.float()
    expanded_scale = scale.repeat_interleave(32, dim=-1)[..., : data.shape[-1]]
    return (data.float() * expanded_scale).to(tensor.dtype)


def _observe_fprop_operand(
    observer: Optional["PrecisionDebugSession"],
    module_fqn: str,
    operand_role: str,
    baseline: torch.Tensor,
    quantized: MXFP8Tensor,
) -> None:
    """Submit a lazy operand reconstruction to the optional debug session."""
    if observer is None:
        return
    observer.observe_quantization(
        module_fqn,
        "fprop",
        operand_role,
        baseline,
        lambda: _dequantize_rowwise(quantized),
    )


class _NpuQuantLinearFn(torch.autograd.Function):
    """Run forward, dgrad, and wgrad through A5 MXFP8 matrix multiplies."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        quantizer: MXFP8Quantizer,
        observer: Optional["PrecisionDebugSession"] = None,
        module_fqn: str = "",
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
        _observe_fprop_operand(observer, module_fqn, "lhs", input_matrix, input_quant)
        _observe_fprop_operand(observer, module_fqn, "rhs", weight, weight_quant)
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
        return grad_input, grad_weight, None, None, None


def npu_quant_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    quantizer: MXFP8Quantizer,
    *,
    observer: Optional["PrecisionDebugSession"] = None,
    module_fqn: str = "",
) -> torch.Tensor:
    """Apply the bias-free Dense MXFP8 autograd function."""

    return _NpuQuantLinearFn.apply(
        inputs,
        weight,
        quantizer,
        observer,
        module_fqn,
    )
