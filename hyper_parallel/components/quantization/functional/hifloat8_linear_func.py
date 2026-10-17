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
"""HiFloat8 hooks for the shared Dense low-precision autograd flow."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.base_linear_func import (
    LinearStrategy,
)
from hyper_parallel.components.quantization.ops.npu_hifloat8 import hifloat8_matmul
from hyper_parallel.components.quantization.quantizers.hifloat8 import (
    GRADIENT_FORMAT_MAX,
    INPUT_WEIGHT_FORMAT_MAX,
    HiFloat8Quantizer,
)
from hyper_parallel.components.quantization.tensor import QuantizedTensorStorage


class HiFloat8LinearStrategy(LinearStrategy):
    """Bind HiFloat8 role quantizers and Dense MM to the shared Linear flow."""

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

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        """Quantize the input with the input-role recipe."""

        return self.input_quantizer.quantize(
            inputs,
            rowwise=rowwise,
            colwise=colwise,
        )

    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        """Quantize the weight with the weight-role recipe."""

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
    ) -> QuantizedTensorStorage:
        """Quantize the gradient output with the gradient-role recipe."""

        return self.grad_output_quantizer.quantize(
            grad_output,
            rowwise=rowwise,
            colwise=colwise,
        )

    def matmul(
        self,
        left: QuantizedTensorStorage,
        right: QuantizedTensorStorage,
        *,
        layout: str,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute the selected HiFloat8 Dense MM layout."""

        return hifloat8_matmul(
            left,
            right,
            layout=layout,
            output_dtype=output_dtype,
        )


__all__ = ["HiFloat8LinearStrategy"]
