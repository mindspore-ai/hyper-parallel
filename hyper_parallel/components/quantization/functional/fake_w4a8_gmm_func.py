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
"""Fake-QAT W4A8 strategy for the shared grouped-linear lifecycle."""

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.base_gmm_func import GroupedLinear
from hyper_parallel.components.quantization.quantizers.fake_w4a8 import FakeW4A8Quantizer


class FakeW4A8GroupedLinear(GroupedLinear):
    """Use fake MXFP4 weights with the ordinary MXFP8 GMM contract.

    The common flow still owns the three autograd phases.  The format-specific
    hooks below only select operand directions and the two backward quantizers:

    * ``NN`` output and ``NT`` input-gradient (dA) use ``npu_dynamic_mx_quant``;
    * ``TN`` weight-gradient (dX) uses grouped dynamic quantization for block32;
    * block128 uses ordinary dynamic quantization for both backward directions.
    """

    FORMAT_NAME = "Fake W4A8"

    def __init__(self, block_size: int = 32) -> None:
        self.quantizer = FakeW4A8Quantizer(block_size)
        self.block_size = self.quantizer.block_size

    def normalize_group_list(
        self,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> tuple[torch.Tensor, int]:
        if group_list_type == 1:
            return torch.cumsum(group_list, dim=0), 0
        return group_list, 0

    def output_features(self, weight: torch.Tensor) -> int:
        return weight.shape[-2]

    def weight_quantization_directions(self, needs_grad_input: bool) -> tuple[bool, bool]:
        if self.block_size == 128:
            # Block-128 follows the native G-B layout: forward uses the
            # source-direction view and dgrad uses its transposed companion.
            return True, needs_grad_input
        # Follow the ordinary MXFP8 GMM operand contract: forward consumes the
        # column-wise weight view, while dgrad consumes the row-wise view after
        # the right operand is transposed.
        return needs_grad_input, True

    def retain_weight_backward(self, weight_quant, needs_grad_input: bool) -> None:
        if self.block_size == 128:
            weight_quant.update_usage(rowwise=False, colwise=needs_grad_input)
            return
        weight_quant.update_usage(rowwise=needs_grad_input, colwise=False)

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: torch.Tensor,
        group_list_type: int,
    ):
        return self.quantizer.quantize_activation(
            inputs,
            rowwise=rowwise,
            colwise=colwise,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    def quantize_weight(self, weight: torch.Tensor, *, rowwise: bool, colwise: bool):
        return self.quantizer.quantize_weight(
            weight,
            rowwise=rowwise,
            colwise=colwise,
        )

    def quantize_grad_output(
        self,
        grad_output: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: torch.Tensor,
        group_list_type: int,
    ):
        return self.quantizer.quantize_activation(
            grad_output,
            rowwise=rowwise,
            colwise=colwise,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    @staticmethod
    def _select_operands(left, right, layout: str):
        block128_layout = (
            getattr(getattr(right, "quantizer", None), "block_size", None) == 128
        )
        if layout == "NN" and block128_layout:
            left_data, left_scale = left.row_data, left.row_scale
            right_data, right_scale = right.row_data, right.row_scale
        elif layout == "NT" and block128_layout:
            left_data, left_scale = left.row_data, left.row_scale
            right_data, right_scale = right.col_data, right.col_scale
        elif layout == "NN":
            left_data, left_scale = left.row_data, left.row_scale
            right_data, right_scale = right.col_data, right.col_scale
        elif layout == "NT":
            left_data, left_scale = left.row_data, left.row_scale
            right_data = right.row_data.transpose(-1, -2)
            right_scale = right.row_scale.transpose(-3, -2)
        elif layout == "TN":
            left_data = left.col_data.transpose(0, 1)
            left_scale = left.col_scale.transpose(0, 1)
            right_data, right_scale = right.col_data, right.col_scale
        else:
            raise ValueError(
                f"Unsupported fake W4A8 GMM layout {layout!r}; expected NN, NT, or TN."
            )
        if any(value is None for value in (left_data, left_scale, right_data, right_scale)):
            raise ValueError(
                f"Fake W4A8 GMM layout {layout!r} requires both directional operands."
            )
        return left_data, left_scale, right_data, right_scale

    def grouped_matmul(
        self,
        left,
        right,
        *,
        layout: str,
        group_list: torch.Tensor,
        group_type: int,
        group_list_type: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del group_list_type
        left_data, left_scale, right_data, right_scale = self._select_operands(
            left, right, layout
        )
        return self.quantizer.npu_ops.grouped_matmul(
            left_data,
            right_data,
            right_scale,
            left_scale=left_scale,
            group_list=group_list,
            group_type=group_type,
            block_size=self.block_size,
            output_dtype=output_dtype,
        )


__all__ = ["FakeW4A8GroupedLinear"]
