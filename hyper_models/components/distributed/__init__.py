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
"""components.distributed: standalone DTensor sharding components (zero dependency on recipes/_transformers/models)."""

from hyper_models.components.distributed.config import (
    FSDP2Config,
)
from hyper_models.components.distributed.cp_utils import (
    flex_cp_allgather,
    shard_batch_for_cp,
)
from hyper_models.components.distributed.cp_wrappers import (
    INNER_WRAPPER_REGISTRY,
    flex_hf_cp_wrapper,
    flex_qkv_cp_wrapper,
    sdpa_hf_cp_wrapper,
    sdpa_qkv_cp_wrapper,
)
from hyper_models.components.distributed.ep_compute import (
    hf_native_ep_compute_fn,
)
from hyper_models.components.distributed.fsdp2 import (
    FSDP2Manager,
    _instantiate_fsdp2,
)
from hyper_models.components.distributed.function_module import FunctionModule
from hyper_models.components.distributed.dispatch_probe import (
    DispatchProbeReport,
    check_dispatchable,
)
from hyper_models.components.distributed.injection import (
    inner_wrapper,
    local_compute,
)
from hyper_models.components.distributed.local_region import local_region
from hyper_models.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
)
from hyper_models.components.distributed.pipelining import (
    AutoPipeline,
    _instantiate_pipeline,
)
from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
    RedistOp,
)
from hyper_models.components.distributed.sharding_config import (
    TEMPLATES,
    MeshAxisName,
    ModuleShardingSpec,
    NamedPlacement,
    PlacementMismatchError,
    ShardingPlan,
    ShardingTemplate,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_applier import (
    apply_sharding_plan,
    build_expert_mesh,
)
from hyper_models.components.distributed.sharding_planner import (
    ARCH_OVERRIDES,
    SPECIAL_HANDLERS,
    ShardingPlanner,
    validate_model_compatibility,
)
from hyper_models.components.distributed.tp_grad import build_tp_grad_info

__all__ = [
    "ARCH_OVERRIDES",
    "AutoPipeline",
    "DispatchProbeReport",
    "INNER_WRAPPER_REGISTRY",
    "FSDP2Config",
    "FunctionModule",
    "SPECIAL_HANDLERS",
    "TEMPLATES",
    "MeshAxisName",
    "ModuleShardingSpec",
    "NamedPlacement",
    "ParameterClassifier",
    "ParamRole",
    "PlacementMismatchError",
    "PrecompiledBoundary",
    "RedistOp",
    "ShardingPlan",
    "ShardingPlanner",
    "ShardingTemplate",
    "_instantiate_fsdp2",
    "_instantiate_pipeline",
    "apply_sharding_plan",
    "build_expert_mesh",
    "build_tp_grad_info",
    "check_dispatchable",
    "flex_cp_allgather",
    "flex_hf_cp_wrapper",
    "flex_qkv_cp_wrapper",
    "hf_native_ep_compute_fn",
    "inner_wrapper",
    "local_compute",
    "local_region",
    "resolve_placements",
    "sdpa_hf_cp_wrapper",
    "sdpa_qkv_cp_wrapper",
    "shard_batch_for_cp",
    "validate_model_compatibility",
]
