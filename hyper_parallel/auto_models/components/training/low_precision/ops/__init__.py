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
"""NPU low-precision operator adapters."""

from hyper_parallel.auto_models.components.training.low_precision.ops.npu_hifloat8 import (
    HiFloat8NpuOps,
    hifloat8_grouped_matmul,
    hifloat8_matmul,
    validate_hifloat8_gmm_runtime,
    validate_hifloat8_runtime,
)
from hyper_parallel.auto_models.components.training.low_precision.ops.npu_mxfp8 import (
    LowPrecisionCapabilityError,
    MXFP8NpuOps,
    mxfp8_grouped_matmul,
    mxfp8_matmul,
    validate_npu_gmm_runtime,
    validate_npu_runtime,
)

__all__ = [
    "HiFloat8NpuOps",
    "LowPrecisionCapabilityError",
    "MXFP8NpuOps",
    "hifloat8_grouped_matmul",
    "hifloat8_matmul",
    "mxfp8_grouped_matmul",
    "mxfp8_matmul",
    "validate_hifloat8_gmm_runtime",
    "validate_hifloat8_runtime",
    "validate_npu_gmm_runtime",
    "validate_npu_runtime",
]
