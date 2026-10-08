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
"""HiFloat8 format hooks for the shared grouped-linear lifecycle."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.base_gmm_func import (
    GroupedLinear,
)
from hyper_parallel.components.quantization.ops.npu_hifloat8 import (
    hifloat8_grouped_matmul,
)
from hyper_parallel.components.quantization.quantizers.hifloat8 import (
    GRADIENT_FORMAT_MAX,
    INPUT_WEIGHT_FORMAT_MAX,
    HiFloat8Quantizer,
)
from hyper_parallel.components.quantization.tensor import (
    HiFloat8Tensor,
    QuantizedTensorStorage,
)


class HiFloat8GroupedLinear(GroupedLinear):
    """Bind HiFloat8 role quantizers and GMM lowering to the common flow."""

    FORMAT_NAME = "HiFloat8"

    def __init__(
        self,
        input_quantizer: Optional[HiFloat8Quantizer] = None,
        weight_quantizer: Optional[HiFloat8Quantizer] = None,
        grad_output_quantizer: Optional[HiFloat8Quantizer] = None,
    ) -> None:
        """Create or accept the three independent HiFloat8 role quantizers."""

        self.input_quantizer = (
            input_quantizer
            if input_quantizer is not None
            else HiFloat8Quantizer(fp8_max=INPUT_WEIGHT_FORMAT_MAX)
        )
        self.weight_quantizer = (
            weight_quantizer
            if weight_quantizer is not None
            else HiFloat8Quantizer(fp8_max=INPUT_WEIGHT_FORMAT_MAX)
        )
        self.grad_output_quantizer = (
            grad_output_quantizer
            if grad_output_quantizer is not None
            else HiFloat8Quantizer(fp8_max=GRADIENT_FORMAT_MAX)
        )

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> None:
        """Preserve the HiFloat8 dtype, device, and empty-group contract."""

        super()._validate_inputs(
            inputs,
            weight,
            group_list,
            group_list_type,
        )
        if inputs.dtype not in (torch.float16, torch.bfloat16) or weight.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise TypeError(
                "HiFloat8 grouped linear inputs and weight must use float16 or bfloat16."
            )
        if group_list.dtype != torch.int64:
            raise TypeError("HiFloat8 grouped linear group_list must use torch.int64.")
        if inputs.device != weight.device or group_list.device != inputs.device:
            raise ValueError(
                "HiFloat8 grouped linear inputs, weight, and group_list must be on the same device."
            )
        if inputs.shape[0] == 0 and torch.any(group_list != 0).item():
            raise ValueError(
                "An empty HiFloat8 grouped input requires an all-zero group_list."
            )

    def normalize_group_list(
        self,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> tuple[torch.Tensor, int]:
        """Keep the caller's counts or offsets because HiFloat8 accepts both."""

        return group_list, group_list_type

    def output_features(self, weight: torch.Tensor) -> int:
        """Return the output width from the live ``[E, O, K]`` weight."""

        return weight.shape[-2]

    def weight_quantization_directions(
        self,
        needs_grad_input: bool,
    ) -> tuple[bool, bool]:
        """Build the column view for forward and row view only for dgrad."""

        return needs_grad_input, True

    def retain_weight_backward(
        self,
        weight_quant: QuantizedTensorStorage,
        needs_grad_input: bool,
    ) -> None:
        """Release the forward column view and retain the dgrad row view."""

        weight_quant.update_usage(rowwise=needs_grad_input, colwise=False)

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> HiFloat8Tensor:
        """Quantize expert-major activations with the input-role recipe."""

        return self.input_quantizer.quantize(
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
    ) -> HiFloat8Tensor:
        """Quantize GMM-ready expert weights with the weight-role recipe."""

        return self.weight_quantizer.quantize(
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
    ) -> HiFloat8Tensor:
        """Quantize output gradients with the gradient-role recipe."""

        return self.grad_output_quantizer.quantize(
            grad_output,
            group_list=group_list,
            group_list_type=group_list_type,
            rowwise=rowwise,
            colwise=colwise,
        )

    def grouped_matmul(
        self,
        left: HiFloat8Tensor,
        right: HiFloat8Tensor,
        *,
        layout: str,
        group_list: torch.Tensor,
        group_type: int,
        group_list_type: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute one HiFloat8 NN, NT, or TN grouped matrix multiplication."""

        return hifloat8_grouped_matmul(
            left,
            right,
            layout=layout,
            group_list=group_list,
            group_type=group_type,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )


__all__ = ["HiFloat8GroupedLinear"]
