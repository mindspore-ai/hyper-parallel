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
"""NPU low-precision training conversion."""

from hyper_models.components.training.low_precision.config import LowPrecisionConfig
from hyper_models.components.training.low_precision.converter import (
    LowPrecisionConversionError,
    apply_low_precision,
)
from hyper_models.components.training.low_precision.modules import NpuQuantLinear
from hyper_models.components.training.low_precision.ops import (
    NpuCapabilityError,
    mxfp8_matmul,
)
from hyper_models.components.training.low_precision.quantizers import (
    MXFP8Quantizer,
    Quantizer,
)
from hyper_models.components.training.low_precision.tensor import (
    MXFP8Tensor,
    MXFP8TensorData,
    QuantizedTensor,
    QuantizedTensorData,
)

__all__ = [
    "LowPrecisionConfig",
    "LowPrecisionConversionError",
    "MXFP8Quantizer",
    "MXFP8Tensor",
    "MXFP8TensorData",
    "NpuCapabilityError",
    "NpuQuantLinear",
    "QuantizedTensor",
    "QuantizedTensorData",
    "Quantizer",
    "apply_low_precision",
    "mxfp8_matmul",
]
