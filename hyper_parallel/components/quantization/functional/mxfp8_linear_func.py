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
"""MXFP8 hooks for the shared Dense low-precision autograd flow."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.base_linear_func import (
    LinearStrategy,
)
from hyper_parallel.components.quantization.ops.npu_mxfp8 import mxfp8_matmul
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.components.quantization.tensor import QuantizedTensorStorage


class MXFP8LinearStrategy(LinearStrategy):
    """Bind one MXFP8 quantizer and Dense MM to the shared Linear flow."""

    def __init__(self, quantizer: Optional[MXFP8Quantizer] = None) -> None:
        """Create a strategy with an optional injected quantizer."""

        self.quantizer = quantizer if quantizer is not None else MXFP8Quantizer()

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        """Quantize the input with the shared MXFP8 quantizer."""

        return self.quantizer.quantize(
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
        """Quantize the weight with the shared MXFP8 quantizer."""

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
    ) -> QuantizedTensorStorage:
        """Quantize the gradient output with the shared MXFP8 quantizer."""

        return self.quantizer.quantize(
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
        """Execute the selected MXFP8 Dense MM layout."""

        return mxfp8_matmul(
            left,
            right,
            layout=layout,
            output_dtype=output_dtype,
        )


__all__ = ["MXFP8LinearStrategy"]
