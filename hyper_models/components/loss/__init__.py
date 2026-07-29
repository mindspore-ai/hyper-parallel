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
"""Loss component interfaces for HyperModels — following design doc §10 + §10.0."""


from .masked_ce import MaskedCrossEntropy
from .utils import calculate_loss, calculate_mtp_loss

# FusedLinearCrossEntropy — optional, requires cut_cross_entropy
try:
    from .linear_ce import FusedLinearCrossEntropy  # noqa: F401
except ImportError:
    pass




__all__ = [
    "MaskedCrossEntropy",
    "FusedLinearCrossEntropy",
    "calculate_loss",
    "calculate_mtp_loss",
]