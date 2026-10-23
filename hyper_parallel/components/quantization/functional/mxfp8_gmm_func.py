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
"""MXFP8 format hooks for the shared native grouped-linear flow."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.base_gmm_func import GroupedLinear
from hyper_parallel.components.quantization.ops.npu_mxfp8 import (
    mxfp8_grouped_matmul,
)
from hyper_parallel.components.quantization.quantizers import (
    MXFP8Quantizer,
)
from hyper_parallel.components.quantization.tensor import QuantizedTensor


class MXFP8GroupedLinear(GroupedLinear):
    """Bind MXFP8 quantization and GMM lowering to the common native flow."""

    FORMAT_NAME = "MXFP8"

    def __init__(self, quantizer: Optional[MXFP8Quantizer] = None) -> None:
        """Create one reusable MXFP8 strategy and own its quantizer."""

        self.quantizer = quantizer if quantizer is not None else MXFP8Quantizer()

    def output_features(self, weight: torch.Tensor) -> int:
        """Return the output width from the live ``[E, O, K]`` weight."""
        return weight.shape[-2]

    def normalize_group_list(
        self,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> tuple[torch.Tensor, int]:
        """Preserve counts/type 1 or offsets/type 0 for the MXFP8 path.

        The quantizer derives cumulative offsets when grouped activation
        quantization needs them; the GMM operator receives the original list
        together with its matching type tag.
        """
        return group_list, group_list_type

    @property
    def weight_forward_is_rowwise(self) -> bool:
        """Use the column-wise weight view in the forward NN GMM."""
        return False

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> QuantizedTensor:
        """Quantize grouped activations for the MXFP8 input role."""
        return self.quantizer.quantize(
            inputs,
            group_list=group_list,
            group_list_type=group_list_type,
            rowwise=rowwise,
            colwise=colwise,
        )

    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensor:
        """Quantize expert weights for the MXFP8 weight role."""
        return self.quantizer.quantize(
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
        """Quantize output gradients for the MXFP8 gradient role."""
        return self.quantizer.quantize(
            grad_output,
            group_list=group_list,
            group_list_type=group_list_type,
            rowwise=rowwise,
            colwise=colwise,
        )

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
        """Execute one MXFP8 grouped matrix multiplication."""
        return mxfp8_grouped_matmul(
            left,
            right,
            layout=layout,
            group_list=group_list,
            group_type=group_type,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )


__all__ = ["MXFP8GroupedLinear"]
