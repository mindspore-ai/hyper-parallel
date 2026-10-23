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
"""W4A8 format hooks for the shared native grouped-linear flow."""

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.config import (
    GMM_SPLIT_ANTIQ,
    GMM_SPLIT_SCALE,
    GROUP_LIST_COUNTS,
    GROUP_LIST_OFFSETS,
    MX_BLOCK_SIZE_32,
)
from hyper_parallel.components.quantization.functional.base_gmm_func import GroupedLinear
from hyper_parallel.components.quantization.quantizers.w4a8 import (
    W4A8Quantizer,
)
from hyper_parallel.components.quantization.tensor import QuantizedTensor


class W4A8GroupedLinear(GroupedLinear):
    """Bind W4A8's target-layout views and native GMM lowering."""

    FORMAT_NAME = "Native W4A8"

    def __init__(self, block_size: int = MX_BLOCK_SIZE_32) -> None:
        """Create one reusable W4A8 strategy and own its quantizer."""

        self.quantizer = W4A8Quantizer(block_size)
        self.block_size = self.quantizer.block_size

    def normalize_group_list(
        self,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> tuple[torch.Tensor, int]:
        """Return cumulative offsets/type 0 for W4A8 quantization and GMM.

        Router output is normally per-expert counts (type 1), so convert it
        once here. Existing cumulative offsets (type 0) pass through unchanged.
        The native W4A8 grouped-matmul adapter uses the offsets contract.
        """
        if group_list_type == GROUP_LIST_COUNTS:
            return torch.cumsum(group_list, dim=0), GROUP_LIST_OFFSETS
        return group_list, GROUP_LIST_OFFSETS

    def output_features(self, weight: torch.Tensor) -> int:
        """Return the output width from the live ``[E, O, K]`` weight."""
        return weight.shape[-2]

    @property
    def weight_forward_is_rowwise(self) -> bool:
        """Use the row-wise weight view in the forward NN GMM."""
        return True

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> QuantizedTensor:
        """Quantize grouped activations for the native W4A8 input role."""
        return self.quantizer.quantize_activation(
            inputs,
            rowwise=rowwise,
            colwise=colwise,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensor:
        """Quantize expert weights for the native W4A8 weight role."""
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
    ) -> QuantizedTensor:
        """Quantize output gradients for the native W4A8 gradient role."""
        return self.quantizer.quantize_activation(
            grad_output,
            rowwise=rowwise,
            colwise=colwise,
            group_list=group_list,
            group_list_type=group_list_type,
        )

    @staticmethod
    def _select_operands(left, right, layout: str):
        """Select directional data and scales for one native GMM layout."""
        if layout == "NN":
            left_data, left_scale = left.row_data, left.row_scale
            right_data, right_scale = right.row_data, right.row_scale
        elif layout == "NT":
            left_data, left_scale = left.row_data, left.row_scale
            right_data, right_scale = right.col_data, right.col_scale
        elif layout == "TN":
            # Preserve transpose strides so the native MX kernel sees the
            # same transpose state on the operand and its scale tensor.
            left_data = left.col_data.transpose(0, 1)
            left_scale = left.col_scale.transpose(0, 1)
            right_data, right_scale = right.col_data, right.col_scale
        else:
            raise ValueError(
                f"Unsupported W4A8 GMM layout {layout!r}; expected NN, NT, or TN."
            )
        if any(value is None for value in (left_data, left_scale, right_data, right_scale)):
            raise ValueError(
                f"W4A8 GMM layout {layout!r} requires both directional operands."
            )
        return left_data, left_scale, right_data, right_scale

    def grouped_matmul(
        self,
        left: QuantizedTensor,
        right: QuantizedTensor,
        *,
        layout: str,
        group_list: torch.Tensor,
        group_type: int,
        group_list_type: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute one native W4A8 grouped matrix multiplication."""
        del group_list_type
        left_data, left_scale, right_data, right_scale = self._select_operands(
            left,
            right,
            layout,
        )
        # NN/NT are native MXFP8 x Packed-MXFP4 W4A8. TN computes the dense
        # master-weight gradient and therefore remains MXFP8 x MXFP8.
        native_w4 = layout in ("NN", "NT")
        # A8W4 grouped matmul consumes E8M0 per-token scales for both MX
        # block sizes.  The TN weight-gradient path uses the same contract.
        use_e8m0_scale = True
        return self.quantizer.npu_ops.grouped_matmul(
            left_data,
            right_data,
            right_scale,
            left_scale=left_scale,
            group_list=group_list,
            group_type=group_type,
            output_dtype=output_dtype,
            use_e8m0_scale=use_e8m0_scale,
            split_item=GMM_SPLIT_SCALE if use_e8m0_scale else GMM_SPLIT_ANTIQ,
            native_w4=native_w4,
        )

__all__ = ["W4A8GroupedLinear"]
