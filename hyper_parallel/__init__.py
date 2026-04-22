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
"""hyper parallel interface"""

__all__ = ["get_platform", "DFunction", "fully_shard", "hsdp_sync_stream", "HSDPModule", "DTensor",
           "Layout", "DeviceMesh", "init_device_mesh", "get_current_mesh", "distribute_module",
           "init_parameters", "init_empty_weights", "init_on_device",
           "shard_module", "custom_shard", "parallelize_value_and_grad", "SkipDTensorDispatch",
           "MetaStep", "MetaStepType", "BatchDimSpec", "PipelineStage", "ScheduleInterleaved1F1B",
           "init_process_group", "destroy_process_group", "get_process_group_ranks", "get_backend", "split_group",
           "get_group_local_rank", "mark_created_groups",
           "ContextParallel", "AsyncContextParallel",
           "ColwiseParallel", "RowwiseParallel", "SequenceParallel",
           "PrepareModuleInput", "PrepareModuleInputOutput", "PrepareModuleOutput",
           "ParallelStyle", "parallelize_module"]

from hyper_parallel.platform import get_platform
from hyper_parallel.core.shard.dfunction import DFunction
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _mesh_resources, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch, distribute_module
from hyper_parallel.core.dtensor.parameter_init import init_parameters
from hyper_parallel.core.dtensor.init_weights import init_empty_weights, init_on_device
from hyper_parallel.core.shard.api import shard_module
from hyper_parallel.core.shard.api import parallelize_value_and_grad
from hyper_parallel.core.shard.custom_shard import custom_shard
from hyper_parallel.core.pipeline_parallel import (PipelineStage, ScheduleInterleaved1F1B, MetaStep, MetaStepType,
                                                   BatchDimSpec)
from hyper_parallel.collectives.cc import (init_process_group, destroy_process_group, get_process_group_ranks,
                                           get_backend, split_group, get_group_local_rank, mark_created_groups)
from hyper_parallel.core.context_parallel import ContextParallel, AsyncContextParallel
from hyper_parallel.core.tensor_parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from hyper_parallel.core.fully_shard.api import fully_shard, hsdp_sync_stream, HSDPModule

get_current_mesh = _mesh_resources.get_current_mesh
