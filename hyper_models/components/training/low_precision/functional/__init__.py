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
"""Format-specific autograd functions for low-precision computation."""

from hyper_models.components.training.low_precision.functional.hifloat8_gmm_func import (
    hifloat8_grouped_linear,
)
from hyper_models.components.training.low_precision.functional.hifloat8_linear_func import (
    hifloat8_linear,
)
from hyper_models.components.training.low_precision.functional.mxfp8_gmm_func import (
    npu_quant_grouped_linear,
)
from hyper_models.components.training.low_precision.functional.mxfp8_linear_func import (
    mxfp8_linear,
)

__all__ = [
    "hifloat8_grouped_linear",
    "hifloat8_linear",
    "mxfp8_linear",
    "npu_quant_grouped_linear",
]