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
# pylint: disable=undefined-all-variable
"""hyper parallel interface"""

__all__ = ["get_platform", "DFunction", "fully_shard", "hsdp_sync_stream", "HSDPModule", "DTensor",
           "Layout", "DeviceMesh", "init_device_mesh", "get_current_mesh", "distribute_module",
           "distribute_tensor", "ones", "zeros", "empty", "full", "rand", "randn",
           "Shard", "RaggedShard", "Replicate", "Partial", "Placement",
           "init_parameters", "init_empty_weights", "init_on_device",
           "shard_module", "custom_shard", "parallelize_value_and_grad", "SkipDTensorDispatch",
           "MetaStep", "MetaStepType", "BatchDimSpec", "PipelineStage", "ScheduleInterleaved1F1B",
           "ScheduleMPipeTranspose",
           "init_process_group", "destroy_process_group", "get_process_group_ranks", "get_backend", "split_group",
           "get_group_local_rank", "mark_created_groups",
           "ContextParallel", "AsyncContextParallel",
           "AsyncDSAIndexerContextParallel", "AsyncDSAIndexerLossContextParallel",
           "AsyncDSASparseAttentionContextParallel",
           "DSAIndexerContextParallel", "DSAIndexerLossContextParallel", "DSASparseAttentionContextParallel",
           "ColwiseParallel", "MC2ColwiseParallel", "MC2RowwiseParallel", "MC2Linear",
           "NoParallel", "RowwiseParallel", "SequenceParallel",
           "PrepareModuleInput", "PrepareModuleInputOutput", "PrepareModuleOutput",
           "ParallelStyle", "parallelize_module", "manual_seed"]

from importlib import import_module as _import_module  # pylint: disable=invalid-name

from hyper_parallel.platform import get_platform
from hyper_parallel.core.shard.dfunction import DFunction
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _mesh_resources, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import (
    DTensor,
    SkipDTensorDispatch,
    distribute_module,
    distribute_tensor,
    ones,
    zeros,
    empty,
    full,
    rand,
    randn,
)
from hyper_parallel.core.dtensor.placement_types import (
    Placement,
    RaggedShard,
    Replicate,
    Partial,
    Shard,
)
from hyper_parallel.core.dtensor.parameter_init import init_parameters
from hyper_parallel.core.dtensor.init_weights import init_empty_weights, init_on_device
from hyper_parallel.core.shard.api import shard_module
from hyper_parallel.core.shard.api import parallelize_value_and_grad
from hyper_parallel.core.shard.custom_shard import custom_shard
from hyper_parallel.core.pipeline_parallel import (PipelineStage, ScheduleInterleaved1F1B, ScheduleMPipeTranspose,
                                                   MetaStep, MetaStepType, BatchDimSpec)
from hyper_parallel.collectives.cc import (init_process_group, destroy_process_group, get_process_group_ranks,
                                           get_backend, split_group, get_group_local_rank, mark_created_groups)
from hyper_parallel.core.context_parallel import (
    AsyncDSAIndexerContextParallel,
    AsyncDSAIndexerLossContextParallel,
    AsyncDSASparseAttentionContextParallel,
    ContextParallel,
    AsyncContextParallel,
    DSAIndexerContextParallel,
    DSAIndexerLossContextParallel,
    DSASparseAttentionContextParallel,
)
from hyper_parallel.core.tensor_parallel import (
    ColwiseParallel,
    NoParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from hyper_parallel.core.dtensor.random import manual_seed
from hyper_parallel.core.fully_shard.api import fully_shard, hsdp_sync_stream, HSDPModule

get_current_mesh = _mesh_resources.get_current_mesh

# MC2 APIs import torch at module load. Resolve them through tensor_parallel's
# lazy __getattr__ so `import hyper_parallel` does not require torch.
_LAZY_EXPORTS = {
    "MC2Linear": "hyper_parallel.core.tensor_parallel",
    "MC2ColwiseParallel": "hyper_parallel.core.tensor_parallel",
    "MC2RowwiseParallel": "hyper_parallel.core.tensor_parallel",
}


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import MC2 symbols that require torch."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = _import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pylint: disable=invalid-name
    """Include lazy MC2 exports in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
