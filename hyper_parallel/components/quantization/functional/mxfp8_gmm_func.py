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
"""Grouped MXFP8 forward, input-gradient, and weight-gradient functions."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.npu_mxfp8 import (
    mxfp8_grouped_matmul,
)
from hyper_parallel.components.quantization.quantizers import (
    MXFP8Quantizer,
)


class _MXFP8GroupedLinearFunction(torch.autograd.Function):
    """Run MoE forward, dgrad, and wgrad through A5 MXFP8 GMMs."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        quantizer: MXFP8Quantizer,
        group_list_type: int,
    ) -> torch.Tensor:
        """Execute one bias-free grouped MXFP8 projection.

        ``weight`` follows the standard expert-linear layout
        ``[experts, out_features, in_features]``. The NPU GMM receives its
        transposed ``[experts, in_features, out_features]`` representation.

        Args:
            ctx: Autograd context owning saved operands and group metadata.
            inputs: High-precision expert-major input matrix.
            weight: High-precision packed expert weights.
            group_list: Token boundaries or counts, one per expert.
            quantizer: MXFP8 quantizer shared by forward and backward.
            group_list_type: Zero for cumulative boundaries, one for counts.

        Returns:
            Expert-major projection output with the input's logical dtype.
        """

        if inputs.ndim != 2:
            raise ValueError(
                "MXFP8 grouped linear inputs must be two-dimensional, "
                f"but got shape {tuple(inputs.shape)}."
            )
        if weight.ndim != 3:
            raise ValueError(
                "MXFP8 grouped linear weight must be three-dimensional, "
                f"but got shape {tuple(weight.shape)}."
            )
        if inputs.shape[-1] != weight.shape[-1]:
            raise ValueError(
                "MXFP8 grouped linear contracting dimensions differ: "
                f"inputs={inputs.shape[-1]}, weight={weight.shape[-1]}."
            )
        if not isinstance(group_list, torch.Tensor) or group_list.ndim != 1:
            raise ValueError(
                "MXFP8 grouped linear group_list must be a one-dimensional Tensor."
            )
        if group_list.shape[0] != weight.shape[0]:
            raise ValueError(
                "MXFP8 grouped linear requires one group per expert: "
                f"groups={group_list.shape[0]}, experts={weight.shape[0]}."
            )
        if group_list_type not in (0, 1):
            raise ValueError(
                "MXFP8 grouped linear group_list_type must be 0 or 1, "
                f"but got {group_list_type}."
            )

        ctx.input_shape = inputs.shape
        ctx.input_dtype = inputs.dtype
        ctx.input_device = inputs.device
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device
        ctx.group_list_type = group_list_type
        ctx.quantizer = quantizer
        ctx.empty_input = inputs.shape[0] == 0
        if ctx.empty_input:
            ctx.save_for_backward(group_list, None, None)
            return inputs.new_empty((0, weight.shape[-2]))

        needs_grad_input = inputs.requires_grad
        needs_grad_weight = weight.requires_grad
        # Only column-wise quantization partitions blocks at expert boundaries.
        input_quant = quantizer.quantize(
            inputs,
            group_list=group_list if needs_grad_weight else None,
            group_list_type=group_list_type,
            rowwise=True,
            colwise=needs_grad_weight,
        )
        weight_for_gmm = weight.transpose(-2, -1).contiguous()
        weight_quant = quantizer.quantize(
            weight_for_gmm,
            rowwise=needs_grad_input,
            colwise=True,
        )
        output = mxfp8_grouped_matmul(
            input_quant,
            weight_quant,
            layout="NN",
            group_list=group_list,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=inputs.dtype,
        )
        input_quant.update_usage(rowwise=False, colwise=needs_grad_weight)
        weight_quant.update_usage(rowwise=needs_grad_input, colwise=False)
        ctx.save_for_backward(
            group_list,
            input_quant if needs_grad_weight else None,
            weight_quant if needs_grad_input else None,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        None,
        None,
    ]:
        """Execute first-order grouped gradients; higher derivatives are unsupported.

        Args:
            ctx: Autograd context containing saved operands and group metadata.
            grad_output: High-precision gradient of the grouped projection output.

        Returns:
            Input and weight gradients, followed by None for the remaining inputs.
        """

        group_list, input_quant, weight_quant = ctx.saved_tensors
        needs_grad_input = ctx.needs_input_grad[0]
        needs_grad_weight = ctx.needs_input_grad[1]
        if ctx.empty_input:
            grad_input = (
                torch.zeros(
                    ctx.input_shape,
                    dtype=ctx.input_dtype,
                    device=ctx.input_device,
                )
                if needs_grad_input
                else None
            )
            grad_weight = (
                torch.zeros(
                    ctx.weight_shape,
                    dtype=ctx.weight_dtype,
                    device=ctx.weight_device,
                )
                if needs_grad_weight
                else None
            )
            return grad_input, grad_weight, None, None, None

        grad_input = None
        grad_weight = None
        grad_quant = ctx.quantizer.quantize(
            grad_output,
            group_list=group_list if needs_grad_weight else None,
            group_list_type=ctx.group_list_type,
            rowwise=needs_grad_input,
            colwise=needs_grad_weight,
        )
        if needs_grad_input:
            grad_input = mxfp8_grouped_matmul(
                grad_quant,
                weight_quant,
                layout="NT",
                group_list=group_list,
                group_type=0,
                group_list_type=ctx.group_list_type,
                output_dtype=ctx.input_dtype,
            )
        if needs_grad_weight:
            grad_weight_for_gmm = mxfp8_grouped_matmul(
                input_quant,
                grad_quant,
                layout="TN",
                group_list=group_list,
                group_type=2,
                group_list_type=ctx.group_list_type,
                output_dtype=ctx.weight_dtype,
            )
            grad_weight = grad_weight_for_gmm.transpose(-2, -1).contiguous()
        grad_quant.update_usage(rowwise=False, colwise=False)
        return grad_input, grad_weight, None, None, None


def npu_quant_grouped_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    quantizer: MXFP8Quantizer,
    *,
    group_list_type: int = 0,
) -> torch.Tensor:
    """Apply a bias-free expert-grouped MXFP8 autograd function.

    Args:
        inputs: Expert-major token matrix ``[tokens, in_features]``.
        weight: Expert weights ``[experts, out_features, in_features]``.
        group_list: Cumulative expert boundaries when ``group_list_type=0``
            or per-expert token counts when ``group_list_type=1``.
        quantizer: MXFP8 quantizer used by forward and backward.
        group_list_type: Interpretation of ``group_list``.

    Returns:
        Expert-major output matrix ``[tokens, out_features]``.
    """

    return _MXFP8GroupedLinearFunction.apply(
        inputs,
        weight,
        group_list,
        quantizer,
        group_list_type,
    )
