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

from hyper_parallel.auto_models.components.distributed.config import (
    FSDP2Config,
)
from hyper_parallel.auto_models.components.distributed.cp_utils import (
    flex_cp_allgather,
    shard_batch_for_cp,
)
from hyper_parallel.auto_models.components.distributed.cp_wrappers import (
    INNER_WRAPPER_REGISTRY,
    flex_hf_cp_wrapper,
    flex_hf_hybrid_cp_wrapper,
    flex_qkv_cp_wrapper,
    flex_qkv_hybrid_cp_wrapper,
    sdpa_hf_cp_wrapper,
    sdpa_hf_hybrid_cp_wrapper,
    sdpa_hf_load_balance_cp_wrapper,
    sdpa_qkv_cp_wrapper,
    sdpa_qkv_hybrid_cp_wrapper,
    sdpa_qkv_load_balance_cp_wrapper,
    qwen3_moe_async_colossal_cp_wrapper,
    qwen3_moe_async_hybrid_cp_wrapper,
    qwen3_moe_async_ulysses_cp_wrapper,
)
from hyper_parallel.auto_models.components.distributed.ep_compute import (
    EP_ARCHETYPE_SUGGESTIONS,
    EP_ARCHETYPES,
    deepseekv3_ep_compute_fn,
    mixtral_ep_compute_fn,
    qwen2moe_ep_compute_fn,
    qwen3moe_ep_compute_fn,
    routed_only_ep_compute_fn,
)
from hyper_parallel.auto_models.components.distributed.ep_utils import (
    MOE_ROUTER_ADAPTERS,
    bind_local_expert_forward,
    describe_moe_module,
    ep_all_to_all,
    ep_routed_forward,
    require_attrs,
    resolve_swiglu_weights,
)
from hyper_parallel.auto_models.components.distributed.fsdp2 import (
    FSDP2Manager,
    _instantiate_fsdp2,
)
from hyper_parallel.auto_models.components.distributed.function_module import FunctionModule
from hyper_parallel.auto_models.components.distributed.dispatch_probe import (
    DispatchProbeReport,
    check_dispatchable,
)
from hyper_parallel.auto_models.components.distributed.injection import (
    inner_wrapper,
    local_compute,
)
from hyper_parallel.auto_models.components.distributed.local_region import local_region
from hyper_parallel.auto_models.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
)
from hyper_parallel.auto_models.components.distributed.pipelining import (
    AutoPipeline,
    _instantiate_pipeline,
)
from hyper_parallel.auto_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
    RedistOp,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    TEMPLATES,
    MeshAxisName,
    ModuleShardingSpec,
    NamedPlacement,
    PlacementMismatchError,
    ShardingPlan,
    ShardingTemplate,
    resolve_placements,
)
from hyper_parallel.auto_models.components.distributed.sharding_applier import (
    apply_sharding_plan,
    build_expert_mesh,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import (
    ARCH_OVERRIDES,
    SPECIAL_HANDLERS,
    ShardingPlanner,
    validate_model_compatibility,
)
from hyper_parallel.auto_models.components.distributed.source_shard import build_source_shard_info

__all__ = [
    "ARCH_OVERRIDES",
    "AutoPipeline",
    "DispatchProbeReport",
    "EP_ARCHETYPES",
    "EP_ARCHETYPE_SUGGESTIONS",
    "INNER_WRAPPER_REGISTRY",
    "FSDP2Config",
    "FunctionModule",
    "MOE_ROUTER_ADAPTERS",
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
    "bind_local_expert_forward",
    "build_expert_mesh",
    "build_source_shard_info",
    "check_dispatchable",
    "deepseekv3_ep_compute_fn",
    "mixtral_ep_compute_fn",
    "describe_moe_module",
    "ep_all_to_all",
    "ep_routed_forward",
    "flex_cp_allgather",
    "flex_hf_cp_wrapper",
    "flex_hf_hybrid_cp_wrapper",
    "flex_qkv_cp_wrapper",
    "flex_qkv_hybrid_cp_wrapper",
    "inner_wrapper",
    "local_compute",
    "local_region",
    "qwen2moe_ep_compute_fn",
    "qwen3moe_ep_compute_fn",
    "qwen3_moe_async_colossal_cp_wrapper",
    "qwen3_moe_async_hybrid_cp_wrapper",
    "qwen3_moe_async_ulysses_cp_wrapper",
    "require_attrs",
    "resolve_placements",
    "resolve_swiglu_weights",
    "routed_only_ep_compute_fn",
    "sdpa_hf_cp_wrapper",
    "sdpa_hf_hybrid_cp_wrapper",
    "sdpa_hf_load_balance_cp_wrapper",
    "sdpa_qkv_cp_wrapper",
    "sdpa_qkv_hybrid_cp_wrapper",
    "sdpa_qkv_load_balance_cp_wrapper",
    "shard_batch_for_cp",
    "validate_model_compatibility",
]
