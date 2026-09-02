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
"""Wan 2.1 AutoModels training components."""

from hyper_parallel.auto_models.components.models.wan.condition import (
    WanConditionModel,
    build_wan_condition_model,
)
from hyper_parallel.auto_models.components.models.wan.configuration import WanTransformer3DTrainingConfig
from hyper_parallel.auto_models.components.models.wan.modeling import (
    WanTransformer3DTrainingModel,
    build_wan_transformer_model,
)

__all__ = [
    "WanConditionModel",
    "WanTransformer3DTrainingConfig",
    "WanTransformer3DTrainingModel",
    "build_wan_condition_model",
    "build_wan_transformer_model",
]
