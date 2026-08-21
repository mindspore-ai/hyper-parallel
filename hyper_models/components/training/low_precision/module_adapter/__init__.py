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
"""Low-precision adapters for model modules."""

from hyper_models.components.training.low_precision.module_adapter.hifloat8_deepseekv3_adapter import (
    HiFloat8GroupedExperts,
    replace_hifloat8_grouped_experts,
)
from hyper_models.components.training.low_precision.module_adapter.hifloat8_linear import (
    HiFloat8Linear,
    replace_hifloat8_linear,
)
from hyper_models.components.training.low_precision.module_adapter.linear import (
    QuantizedLinearBase,
)
from hyper_models.components.training.low_precision.module_adapter.mxfp8_deepseekv3_adapter import (
    MXFP8GroupedExperts,
    replace_mxfp8_grouped_experts,
)
from hyper_models.components.training.low_precision.module_adapter.mxfp8_linear import (
    MXFP8Linear,
    replace_mxfp8_linear,
)

__all__ = [
    "HiFloat8GroupedExperts",
    "HiFloat8Linear",
    "MXFP8GroupedExperts",
    "MXFP8Linear",
    "QuantizedLinearBase",
    "replace_hifloat8_grouped_experts",
    "replace_hifloat8_linear",
    "replace_mxfp8_grouped_experts",
    "replace_mxfp8_linear",
]
