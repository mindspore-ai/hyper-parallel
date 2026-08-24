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
"""NPU grouped matrix multiplication function."""

from typing import Any, Optional, Union

import torch  # pylint: disable=forbidden-backend-import
import torch_npu

GroupList = Optional[Union[torch.Tensor, list[int], tuple[int, ...]]]


class GMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        original_weight: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        group_args: tuple,
    ) -> torch.Tensor:
        """Run grouped matrix multiplication and save its backward state."""
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args

        npu_bias = [bias] if bias is not None else []

        outputs = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[weight],
            bias=npu_bias,
            group_list=group_list,
            split_item=3,
            group_type=group_type,
            group_list_type=group_list_type
        )

        if group_list_data_type == 0:
            ctx.save_for_backward(x, weight)
            ctx.group_list = group_list
        else:
            ctx.save_for_backward(x, weight, group_list)

        ctx.original_weight = original_weight
        ctx.gemm_fusion = gemm_fusion
        ctx.group_list_type = group_list_type
        ctx.group_list_data_type = group_list_data_type
        return outputs[0]

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple:
        """Compute the input and grouped-weight gradients."""
        grad_outputs = grad_outputs.contiguous()

        if ctx.group_list_data_type == 0:
            x, weight = ctx.saved_tensors
            group_list = ctx.group_list
        else:
            x, weight, group_list = ctx.saved_tensors

        original_weight = ctx.original_weight
        group_list_type = ctx.group_list_type

        if isinstance(group_list, (list, tuple)):
            group_list_tensor = torch.tensor(group_list, device=x.device, dtype=torch.int64)
        else:
            group_list_tensor = group_list

        dx_list = torch_npu.npu_grouped_matmul(
            x=[grad_outputs],
            weight=[weight.transpose(-1, -2)],
            bias=[],
            group_list=group_list_tensor,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type
        )
        dx = dx_list[0]

        grad_weight = None
        use_fusion = False
        if ctx.gemm_fusion:
            if hasattr(original_weight, 'main_grad'):
                if original_weight.main_grad is not None:
                    use_fusion = True

        if use_fusion:
            actual_group_list = group_list_tensor

            n_size = original_weight.main_grad.shape[-1]
            main_grad_view = original_weight.main_grad.view(-1, n_size)

            torch_npu.npu_grouped_matmul_add_(
                main_grad_view,
                x,
                grad_outputs,
                actual_group_list,
                transpose_x=True,
                transpose_weight=False,
                group_type=2
            )

            if getattr(weight, 'zero_out_wgrad', False):
                grad_weight = torch.zeros_like(weight)
            else:
                grad_weight = torch.empty_like(weight)

            original_weight.grad_added_to_main_grad = True
        else:
            if x.nelement() > 0:
                dw_list = torch_npu.npu_grouped_matmul(
                    x=[x.transpose(-1, -2)],
                    weight=[grad_outputs],
                    bias=[],
                    group_list=group_list_tensor,
                    split_item=2,
                    group_type=2,
                    group_list_type=group_list_type
                )
                grad_weight = dw_list[0]
            else:
                grad_weight = torch.zeros_like(weight)

        return None, dx, grad_weight, None, None


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list: GroupList = None,
    group_type: int = 0,
    group_list_type: int = 0,
    gemm_fusion: bool = False,
    original_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply NPU grouped matrix multiplication.

    Args:
        x: Expert-major input tensor.
        weight: Grouped weight tensor.
        bias: Optional grouped bias tensor.
        group_list: Cumulative group boundaries as a tensor or integer sequence.
        group_type: Grouping axis accepted by ``npu_grouped_matmul``.
        group_list_type: Group-list representation accepted by the NPU kernel.
        gemm_fusion: Whether to accumulate the weight gradient into ``main_grad``.
        original_weight: Parameter that owns ``main_grad``. Defaults to ``weight``.

    Returns:
        The grouped matrix multiplication output.
    """
    group_list_data_type = 1 if isinstance(group_list, torch.Tensor) else 0
    group_args = (group_list, group_type, gemm_fusion, group_list_type, group_list_data_type)

    if original_weight is None:
        original_weight = weight

    return GMMFunction.apply(original_weight, x, weight, bias, group_args)
