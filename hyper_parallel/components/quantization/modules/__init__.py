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
"""Quantized module implementations (format-generic, model-agnostic)."""

from hyper_parallel.components.quantization.modules.hifloat8_grouped_linear import (
    HiFloat8GroupedExperts,
)
from hyper_parallel.components.quantization.modules.hifloat8_linear import (
    HiFloat8Linear,
    replace_hifloat8_linear,
)
from hyper_parallel.components.quantization.modules.linear import (
    QuantizedLinearBase,
)
from hyper_parallel.components.quantization.modules.mxfp8_grouped_linear import (
    MXFP8GroupedExperts,
)
from hyper_parallel.components.quantization.modules.mxfp8_linear import (
    MXFP8Linear,
    replace_mxfp8_linear,
)

__all__ = [
    "HiFloat8GroupedExperts",
    "HiFloat8Linear",
    "MXFP8GroupedExperts",
    "MXFP8Linear",
    "QuantizedLinearBase",
    "replace_hifloat8_linear",
    "replace_mxfp8_linear",
]
