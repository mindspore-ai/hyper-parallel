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
"""I4+I5 stub：ShardingPlan / ModuleShardingSpec / NamedPlacement（P4 提供，所有人消费）。

canonical 定义在 ``components/distributed/sharding_config.py``；本 stub 直接
re-export，字段名/类型与 05 §3.1-3.2 完全一致（re-export 而非复制，杜绝漂移）。
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
