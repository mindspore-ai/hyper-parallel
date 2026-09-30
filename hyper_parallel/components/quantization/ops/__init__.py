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
"""NPU operator adapters for low-precision quantization."""

from hyper_parallel.components.quantization.ops.npu_fake_w4a8 import (
    FakeW4A8CapabilityError,
    FakeW4A8NpuOps,
    validate_fake_w4a8_gmm_runtime,
)
from hyper_parallel.components.quantization.ops.npu_hifloat8 import (
    HiFloat8NpuOps,
    hifloat8_grouped_matmul,
    hifloat8_matmul,
    validate_hifloat8_gmm_runtime,
    validate_hifloat8_runtime,
)
from hyper_parallel.components.quantization.ops.npu_mxfp8 import (
    LowPrecisionCapabilityError,
    MXFP8NpuOps,
    mxfp8_grouped_matmul,
    mxfp8_matmul,
    validate_npu_gmm_runtime,
    validate_npu_runtime,
)
from hyper_parallel.components.quantization.ops.npu_w4a8 import (
    W4A8CapabilityError,
    W4A8NpuOps,
    transform_grouped_scale,
    validate_w4a8_gmm_runtime,
)

__all__ = [
    "FakeW4A8CapabilityError",
    "FakeW4A8NpuOps",
    "HiFloat8NpuOps",
    "LowPrecisionCapabilityError",
    "MXFP8NpuOps",
    "W4A8CapabilityError",
    "W4A8NpuOps",
    "hifloat8_grouped_matmul",
    "hifloat8_matmul",
    "mxfp8_grouped_matmul",
    "mxfp8_matmul",
    "transform_grouped_scale",
    "validate_fake_w4a8_gmm_runtime",
    "validate_hifloat8_gmm_runtime",
    "validate_hifloat8_runtime",
    "validate_npu_gmm_runtime",
    "validate_npu_runtime",
    "validate_w4a8_gmm_runtime",
]
