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
"""Public Dense Linear module family and low-precision adapter exports."""

from .linear import Linear, LowPrecisionLinear
from .linear_family import (
    replace_hif8_linear,
    replace_mxfp8_linear,
)

__all__ = [
    "Linear",
    "LowPrecisionLinear",
    "replace_hif8_linear",
    "replace_mxfp8_linear",
]
