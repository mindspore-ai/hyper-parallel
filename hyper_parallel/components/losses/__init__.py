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
"""Model-level loss objectives and calling adapters — design doc §10 + §10.0."""

from hyper_parallel.components.losses.dispatcher import calculate_loss
from hyper_parallel.components.losses.masked_ce import MaskedCrossEntropy
from hyper_parallel.components.losses.model_output import ModelOutputLoss
from hyper_parallel.components.losses.mtp import calculate_mtp_loss

# FusedLinearCrossEntropy — optional, requires cut_cross_entropy
try:
    from hyper_parallel.components.losses.linear_ce import FusedLinearCrossEntropy  # noqa: F401
except ImportError:
    pass

__all__ = [
    "MaskedCrossEntropy",
    "ModelOutputLoss",
    "FusedLinearCrossEntropy",
    "calculate_loss",
    "calculate_mtp_loss",
]
