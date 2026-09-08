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
"""Optimizer and learning-rate scheduler interfaces for AutoModels."""

from hyper_parallel.components.optim.builders import AdamW, Muon
from hyper_parallel.components.optim.lr_scheduler import MultiLRScheduler
from hyper_parallel.components.optim.mixed_precision_optimizer import (
    Float16OptimizerWithFloat16Params,
    MixedPrecisionOptimizer,
)
from hyper_parallel.components.optim.parameter_groups import (
    get_adamw_param_groups,
    get_parameter_names,
    split_muon_adamw_params,
)

__all__ = [
    "AdamW",
    "Float16OptimizerWithFloat16Params",
    "MixedPrecisionOptimizer",
    "Muon",
    "MultiLRScheduler",
    "get_adamw_param_groups",
    "get_parameter_names",
    "split_muon_adamw_params",
]
