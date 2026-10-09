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

from hyper_parallel.components.quantization.config import (
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
)
from hyper_parallel.components.quantization.functional import (
    GroupedLinear,
    W4A8GroupedLinear,
    FakeW4A8GroupedLinear,
    build_low_precision_strategy,
)
from hyper_parallel.components.quantization.ops import (
    FakeW4A8CapabilityError,
    FakeW4A8NpuOps,
    HiFloat8NpuOps,
    LowPrecisionCapabilityError,
    MXFP8NpuOps,
    W4A8CapabilityError,
    W4A8NpuOps,
    hifloat8_grouped_matmul,
    hifloat8_matmul,
    mxfp8_grouped_matmul,
    mxfp8_matmul,
    transform_grouped_scale,
    validate_fake_w4a8_gmm_runtime,
    validate_w4a8_gmm_runtime,
)
from hyper_parallel.components.quantization.modules import (
    GroupedExperts,
    HiFloat8GroupedExperts,
    HiFloat8Linear,
    MXFP8Linear,
    QuantizedLinearBase,
    replace_hifloat8_linear,
    replace_mxfp8_linear,
)
from hyper_parallel.components.quantization.quantizers import (
    HiFloat8Quantizer,
    MXFP8Quantizer,
    Quantizer,
    W4A8Quantizer,
    FakeW4A8Quantizer,
)
from hyper_parallel.components.quantization.tensor import (
    HiFloat8Tensor,
    HiFloat8TensorStorage,
    MXFP8Tensor,
    MXFP8TensorStorage,
    QuantizedTensor,
    QuantizedTensorStorage,
    W4A8Tensor,
    W4A8TensorStorage,
)

__all__ = [
    "HiFloat8GroupedExperts",
    "HiFloat8Linear",
    "HiFloat8NpuOps",
    "HiFloat8Quantizer",
    "HiFloat8Tensor",
    "HiFloat8TensorStorage",
    "LowPrecisionConfig",
    "LowPrecisionDtypeScheme",
    "LowPrecisionCapabilityError",
    "GroupedExperts",
    "MXFP8Linear",
    "MXFP8NpuOps",
    "MXFP8Quantizer",
    "MXFP8Tensor",
    "MXFP8TensorStorage",
    "W4A8CapabilityError",
    "W4A8GroupedLinear",
    "W4A8NpuOps",
    "W4A8Quantizer",
    "W4A8Tensor",
    "W4A8TensorStorage",
    "FakeW4A8CapabilityError",
    "FakeW4A8GroupedLinear",
    "FakeW4A8NpuOps",
    "FakeW4A8Quantizer",
    "GroupedLinear",
    "build_low_precision_strategy",
    "QuantizedLinearBase",
    "QuantizedTensor",
    "QuantizedTensorStorage",
    "Quantizer",
    "hifloat8_grouped_matmul",
    "hifloat8_matmul",
    "mxfp8_grouped_matmul",
    "mxfp8_matmul",
    "transform_grouped_scale",
    "validate_w4a8_gmm_runtime",
    "validate_fake_w4a8_gmm_runtime",
    "replace_hifloat8_linear",
    "replace_mxfp8_linear",
]
