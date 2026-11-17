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
"""Hyper-only compatibility helpers for explicitly enabled framework integrations."""

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.api import HSDPModule as FSDPModule
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.integration.torch.dtensor import (
    DTensorAdapter,
    is_dtensor,
)
from hyper_parallel.integration.torch.fsdp import (
    fully_shard,
    get_model_state_dict,
    set_model_state_dict,
)

__all__ = [
    "DTensor",
    "DTensorAdapter",
    "DeviceMesh",
    "FSDPModule",
    "Replicate",
    "Shard",
    "clip_grad_norm_",
    "distribute_tensor",
    "fully_shard",
    "get_model_state_dict",
    "is_dtensor",
    "set_model_state_dict",
]
