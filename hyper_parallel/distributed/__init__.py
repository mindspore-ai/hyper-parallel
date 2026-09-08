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
"""distributed: model-level DTensor sharding plan and recipe contract.

Public surface (migration plan §7.1): the model-level ``ShardingPlan``, the
recipe input contract (``ModuleShardingSpec`` / ``NamedPlacement`` /
``MeshAxisName`` / placement DSL parsers) and the public template type
``ShardingTemplate``, plan construction (``ShardingPlanner`` /
``validate_model_compatibility``) and the injection decorators
(``local_compute`` / ``inner_wrapper``) and plan application
(``apply_sharding_plan``).
"""

from hyper_parallel.distributed.apply import (
    apply_sharding_plan,
)
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed._builder.planner import (
    ShardingPlanner,
    validate_model_compatibility,
)
from hyper_parallel.distributed.recipe_spec import (
    CP,
    DP,
    EP,
    TP,
    MeshAxisName,
    ModuleShardingSpec,
    NamedPlacement,
    PlacementMismatchError,
    inner_wrapper,
    local_compute,
    parse_named_placement,
    parse_placement,
    resolve_placements,
)
from hyper_parallel.distributed._builder.default_templates import (
    ShardingTemplate,
)
from hyper_parallel.distributed.activation_checkpoint import (
    _apply_activation_checkpointing,
    make_selective_checkpoint_context_fn,
)
from hyper_parallel.distributed.attention_swap import (
    apply_attention_swap,
    attention_swap_policy,
    validate_attention_swap,
)
from hyper_parallel.distributed.compile import (
    apply_compile,
    get_compile_layers,
    resolve_compile_kwargs,
)

__all__ = [
    "CP",
    "DP",
    "EP",
    "TP",
    "MeshAxisName",
    "ModuleShardingSpec",
    "NamedPlacement",
    "PlacementMismatchError",
    "ShardingPlan",
    "ShardingPlanner",
    "ShardingTemplate",
    "inner_wrapper",
    "local_compute",
    "parse_named_placement",
    "parse_placement",
    "resolve_placements",
    "apply_sharding_plan",
    "validate_model_compatibility",
    "_apply_activation_checkpointing",
    "make_selective_checkpoint_context_fn",
    "apply_attention_swap",
    "attention_swap_policy",
    "validate_attention_swap",
    "apply_compile",
    "get_compile_layers",
    "resolve_compile_kwargs",
]
