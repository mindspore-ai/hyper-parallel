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
"""Shared Dense low-precision autograd flow and format strategy interface."""

from abc import ABC, abstractmethod
from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.tensor import QuantizedTensorStorage
from hyper_parallel.components.quantization.tensor.quantized_tensor import QuantizedTensor
from hyper_parallel.components.quantization.functional._saved_quantized import (
    restore_quantized_operands,
    save_quantized_operands,
)


def _as_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten leading dimensions while preserving the contracting axis."""

    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


class LinearStrategy(ABC):
    """Own the common Dense forward/backward flow and expose format hooks."""

    @abstractmethod
    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        """Create the format-specific input representation."""

    @abstractmethod
    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        """Create the format-specific weight representation."""

    @abstractmethod
    def quantize_grad_output(
        self,
        grad_output: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        """Create the format-specific gradient-output representation."""

    @abstractmethod
    def matmul(
        self,
        left: QuantizedTensorStorage,
        right: QuantizedTensorStorage,
        *,
        layout: str,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute one format-specific Dense matrix multiplication."""

    def forward(
        self,
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run the complete bias-free Dense forward lifecycle."""

        input_matrix = _as_matrix(inputs)
        needs_grad_input = inputs.requires_grad
        needs_grad_weight = weight.requires_grad
        input_quant = self.quantize_input(
            input_matrix,
            rowwise=True,
            colwise=needs_grad_weight,
        )
        weight_quant = self.quantize_weight(
            weight,
            rowwise=True,
            colwise=needs_grad_input,
        )
        output = self.matmul(
            input_quant,
            weight_quant,
            layout="NT",
            output_dtype=inputs.dtype,
        )

        ctx.input_shape = inputs.shape
        ctx.weight_dtype = weight.dtype
        ctx.strategy = self
        input_quant.update_usage(rowwise=False, colwise=needs_grad_weight)
        weight_quant.update_usage(rowwise=False, colwise=needs_grad_input)
        operands = (
            input_quant if needs_grad_weight else None,
            weight_quant if needs_grad_input else None,
        )
        # Keep physical payloads under PyTorch's saved-tensor lifecycle when
        # the production quantizer returned a wrapper Tensor.  CPU contract
        # strategies may use a light-weight storage double; retain those
        # objects directly so the shared seam remains testable without NPU.
        ctx.quantized_operands_saved = all(
            operand is None or isinstance(operand, QuantizedTensor)
            for operand in operands
        )
        if ctx.quantized_operands_saved:
            save_quantized_operands(ctx, *operands)
            ctx.input_quant = None
            ctx.weight_quant = None
        else:
            ctx.input_quant, ctx.weight_quant = operands
        if inputs.ndim != 2:
            output = output.reshape(*inputs.shape[:-1], output.shape[-1])
        return output

    def backward(
        self,
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the complete Dense dgrad/wgrad lifecycle."""

        grad_matrix = _as_matrix(grad_output)
        needs_grad_input = ctx.needs_input_grad[0]
        needs_grad_weight = ctx.needs_input_grad[1]
        if ctx.quantized_operands_saved:
            _, input_quant, weight_quant = restore_quantized_operands(ctx)
        else:
            input_quant = ctx.input_quant
            weight_quant = ctx.weight_quant
        grad_quant = self.quantize_grad_output(
            grad_matrix,
            rowwise=needs_grad_input,
            colwise=needs_grad_weight,
        )

        grad_input = None
        grad_weight = None
        if needs_grad_input:
            grad_input = self.matmul(
                grad_quant,
                weight_quant,
                layout="NN",
                output_dtype=grad_output.dtype,
            )
            if len(ctx.input_shape) != 2:
                grad_input = grad_input.reshape(ctx.input_shape)
        if needs_grad_weight:
            grad_weight = self.matmul(
                grad_quant,
                input_quant,
                layout="TN",
                output_dtype=ctx.weight_dtype,
            )

        grad_quant.update_usage(rowwise=False, colwise=False)
        ctx.input_quant = None
        ctx.weight_quant = None
        return grad_input, grad_weight


class _LinearFunction(torch.autograd.Function):
    """Bridge PyTorch autograd callbacks to one bound Dense strategy."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        strategy: LinearStrategy,
    ) -> torch.Tensor:
        """Delegate the complete forward lifecycle to the strategy."""

        return strategy.forward(ctx, inputs, weight)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        """Delegate the complete backward lifecycle to the strategy."""

        grad_input, grad_weight = ctx.strategy.backward(ctx, grad_output)
        return grad_input, grad_weight, None


__all__ = ["LinearStrategy", "_LinearFunction"]
