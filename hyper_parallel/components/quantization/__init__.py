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
"""Low-precision quantization: config, tensors, quantizers, functions, modules."""

from hyper_parallel.components.quantization.config import LowPrecisionConfig
from hyper_parallel.components.quantization.functional import (
    HiFloat8NpuOps,
    LowPrecisionCapabilityError,
    MXFP8NpuOps,
    hifloat8_grouped_matmul,
    hifloat8_matmul,
    mxfp8_grouped_matmul,
    mxfp8_matmul,
)
from hyper_parallel.components.quantization.modules import (
    HiFloat8GroupedExperts,
    HiFloat8Linear,
    MXFP8GroupedExperts,
    MXFP8Linear,
    QuantizedLinearBase,
    replace_hifloat8_linear,
    replace_mxfp8_linear,
)
from hyper_parallel.components.quantization.quantizers import (
    HiFloat8Quantizer,
    MXFP8Quantizer,
    Quantizer,
)
from hyper_parallel.components.quantization.tensor import (
    HiFloat8Tensor,
    HiFloat8TensorStorage,
    MXFP8Tensor,
    MXFP8TensorStorage,
    QuantizedTensor,
    QuantizedTensorStorage,
)

__all__ = [
    "HiFloat8GroupedExperts",
    "HiFloat8Linear",
    "HiFloat8NpuOps",
    "HiFloat8Quantizer",
    "HiFloat8Tensor",
    "HiFloat8TensorStorage",
    "LowPrecisionConfig",
    "LowPrecisionCapabilityError",
    "MXFP8GroupedExperts",
    "MXFP8Linear",
    "MXFP8NpuOps",
    "MXFP8Quantizer",
    "MXFP8Tensor",
    "MXFP8TensorStorage",
    "QuantizedLinearBase",
    "QuantizedTensor",
    "QuantizedTensorStorage",
    "Quantizer",
    "hifloat8_grouped_matmul",
    "hifloat8_matmul",
    "mxfp8_grouped_matmul",
    "mxfp8_matmul",
    "replace_hifloat8_linear",
    "replace_mxfp8_linear",
]
