# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""I4+I5 stub: ShardingPlan / ModuleShardingSpec / NamedPlacement (provided by P4, consumed by everyone).

The canonical definitions live in ``components/distributed/sharding_config.py``;
this stub re-exports them directly, with field names and types exactly matching
sections 3.1-3.2 of design doc 05 (re-export instead of copy, so they can never
drift apart).
"""

from hyper_models.components.distributed.sharding_config import (
    MeshAxisName,
    ModuleShardingSpec,
    NamedPlacement,
    ShardingPlan,
)

__all__ = [
    "MeshAxisName",
    "ModuleShardingSpec",
    "NamedPlacement",
    "ShardingPlan",
]
