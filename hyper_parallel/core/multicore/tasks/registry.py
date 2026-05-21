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
"""
FillConfig registry: maps OpType -> FillConfig class.

Used by the @MultiCore decorator path for automatic task filling.
Manual graph construction (gen_runtime_data.py) instantiates FillConfig
directly in the OperatorNode declaration and does not use this registry.

To register a new operator type:
  1. Add a value to OpType in runtime/graph.py.
  2. Implement a FillConfig subclass in tasks/<op_name>.py.
  3. Add an entry to FILL_CONFIG_REGISTRY below.
"""
from typing import Dict, Type

from hyper_parallel.core.multicore.scheduler.graph import OpType
from hyper_parallel.core.multicore.tasks.task_base import FillConfig
from hyper_parallel.core.multicore.tasks.alltoall import AllToAllFillConfig
from hyper_parallel.core.multicore.tasks.gmm import GmmFillConfig
from hyper_parallel.core.multicore.tasks.swiglu import SwiGLUFillConfig

FILL_CONFIG_REGISTRY: Dict[OpType, Type[FillConfig]] = {
    OpType.ALLTOALL:    AllToAllFillConfig,
    OpType.GMM:         GmmFillConfig,
    OpType.SWIGLU:      SwiGLUFillConfig,
    OpType.SWIGLU_GRAD: SwiGLUFillConfig,
}
