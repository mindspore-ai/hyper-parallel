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
"""HiFloat8 grouped forward, input-gradient, and weight-gradient."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_models.components.training.low_precision.ops.npu_hifloat8 import (
    hifloat8_grouped_matmul,
)
from hyper_parallel.auto_models.components.training.low_precision.quantizers.hifloat8 import (
    HiFloat8Quantizer,
)


class _HiFloat8GroupedLinearFunction(torch.autograd.Function):
    """Run one expert projection through three A5 HiFloat8 GMM layouts."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        input_quantizer: HiFloat8Quantizer,
        weight_quantizer: HiFloat8Quantizer,
        grad_output_quantizer: HiFloat8Quantizer,
        group_list_type: int,
    ) -> torch.Tensor:
        """Execute one bias-free expert projection.

        ``weight`` follows ``[experts, out_features, in_features]``. The GMM
        receives its transposed ``[experts, in_features, out_features]`` view.
        """

        if inputs.ndim != 2:
            raise ValueError(
                "HiFloat8 grouped linear inputs must be two-dimensional, "
                f"got shape {tuple(inputs.shape)}."
            )
        if weight.ndim != 3:
            raise ValueError(
                "HiFloat8 grouped linear weight must be three-dimensional, "
                f"got shape {tuple(weight.shape)}."
            )
        if inputs.shape[-1] != weight.shape[-1]:
            raise ValueError(
                "HiFloat8 grouped linear contracting dimensions differ: "
                f"inputs={inputs.shape[-1]}, weight={weight.shape[-1]}."
            )
        if inputs.dtype not in (torch.float16, torch.bfloat16) or (
            weight.dtype not in (torch.float16, torch.bfloat16)
        ):
            raise TypeError(
                "HiFloat8 grouped linear inputs and weight must use "
                "float16 or bfloat16."
            )
        if not isinstance(group_list, torch.Tensor) or group_list.ndim != 1:
            raise ValueError(
                "HiFloat8 grouped linear group_list must be one-dimensional."
            )
        if group_list.dtype != torch.int64:
            raise TypeError(
                "HiFloat8 grouped linear group_list must use torch.int64."
            )
        if group_list.shape[0] != weight.shape[0]:
            raise ValueError(
                "HiFloat8 grouped linear requires one group per expert: "
                f"groups={group_list.shape[0]}, experts={weight.shape[0]}."
            )
        if group_list_type not in (0, 1):
            raise ValueError(
                "HiFloat8 grouped linear group_list_type must be 0 or 1, "
                f"got {group_list_type}."
            )
        if inputs.device != weight.device or group_list.device != inputs.device:
            raise ValueError(
                "HiFloat8 grouped linear inputs, weight, and group_list must "
                "be on the same device."
            )
        if inputs.shape[0] == 0 and torch.any(group_list != 0).item():
            raise ValueError(
                "An empty HiFloat8 grouped input requires an all-zero "
                "group_list."
            )

        ctx.input_shape = inputs.shape
        ctx.input_dtype = inputs.dtype
        ctx.input_device = inputs.device
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device
        ctx.save_for_backward(group_list)
        ctx.group_list_type = group_list_type
        ctx.grad_output_quantizer = grad_output_quantizer
        ctx.empty_input = inputs.shape[0] == 0
        if ctx.empty_input:
            return inputs.new_empty((0, weight.shape[-2]))

        needs_grad_input = inputs.requires_grad
        needs_grad_weight = weight.requires_grad
        input_quant = input_quantizer.quantize(
            inputs,
            group_list=group_list,
            group_list_type=group_list_type,
            rowwise=True,
            colwise=needs_grad_weight,
        )
        weight_for_gmm = weight.transpose(-2, -1).contiguous()
        weight_quant = weight_quantizer.quantize(
            weight_for_gmm,
            rowwise=needs_grad_input,
            colwise=True,
        )
        output = hifloat8_grouped_matmul(
            input_quant,
            weight_quant,
            layout="NN",
            group_list=group_list,
            group_type=0,
            group_list_type=group_list_type,
            output_dtype=inputs.dtype,
        )
        ctx.input_quant = input_quant if needs_grad_weight else None
        ctx.weight_quant = weight_quant if needs_grad_input else None
        input_quant.update_usage(rowwise=False, colwise=needs_grad_weight)
        weight_quant.update_usage(rowwise=needs_grad_input, colwise=False)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        None,
        None,
        None,
        None,
    ]:
        """Execute HiFloat8 dgrad and wgrad with the gradient role recipe."""

        (group_list,) = ctx.saved_tensors
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
            return grad_input, grad_weight, None, None, None, None, None

        grad_quant = ctx.grad_output_quantizer.quantize(
            grad_output,
            group_list=group_list,
            group_list_type=ctx.group_list_type,
            rowwise=needs_grad_input,
            colwise=needs_grad_weight,
        )
        grad_input = None
        grad_weight = None
        if needs_grad_input:
            grad_input = hifloat8_grouped_matmul(
                grad_quant,
                ctx.weight_quant,
                layout="NT",
                group_list=group_list,
                group_type=0,
                group_list_type=ctx.group_list_type,
                output_dtype=ctx.input_dtype,
            )
        if needs_grad_weight:
            grad_weight_for_gmm = hifloat8_grouped_matmul(
                ctx.input_quant,
                grad_quant,
                layout="TN",
                group_list=group_list,
                group_type=2,
                group_list_type=ctx.group_list_type,
                output_dtype=ctx.weight_dtype,
            )
            grad_weight = grad_weight_for_gmm.transpose(-2, -1).contiguous()
        grad_quant.update_usage(rowwise=False, colwise=False)
        ctx.input_quant = None
        ctx.weight_quant = None
        return grad_input, grad_weight, None, None, None, None, None


def hifloat8_grouped_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    input_quantizer: HiFloat8Quantizer,
    weight_quantizer: HiFloat8Quantizer,
    grad_output_quantizer: HiFloat8Quantizer,
    *,
    group_list_type: int = 0,
) -> torch.Tensor:
    """Apply one bias-free expert-grouped HiFloat8 projection."""

    return _HiFloat8GroupedLinearFunction.apply(
        inputs,
        weight,
        group_list,
        input_quantizer,
        weight_quantizer,
        grad_output_quantizer,
        group_list_type,
    )


__all__ = ["hifloat8_grouped_linear"]
