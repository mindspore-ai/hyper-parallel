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
"""Head ownership and shared TP/SP token boundaries for linear attention."""
from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.distributed.module_binding import ParallelModuleSpec


def linear_attention_sharding(*, module: nn.Module, context: Any,
                               roles: dict, tp_divide_attrs: tuple[str, ...]) -> ModuleShardingSpec:
    """Declare head-sharded parameters and reuse the framework's TP/SP token boundary.

    Args:
        module: Complete local or CP attention member.
        context: Logical mesh sizes and the external SP layout.
        roles: Explicit TP placement for each supported parameter name.
        tp_divide_attrs: Cached dimensions not covered by automatic head-count adjustment.
    """
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    params = {}
    for name, _ in module.named_parameters():
        params[name] = dict(placements)
        if context.axis_sizes["tp"] > 1:
            if name not in roles:
                raise ValueError(f"Linear-attention parameter {name!r} has no declared TP role")
            params[name]["tp"] = roles[name]
    tokens = dict(placements)
    token_axis = 0 if module.input_layout == "TH" else 1
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(token_axis)
    if context.axis_sizes["tp"] > 1 and context.sequence_parallel:
        tokens["tp"] = Shard(token_axis)
    compute_input, compute_output = dict(tokens), dict(tokens)
    if context.axis_sizes["tp"] > 1:
        compute_input["tp"], compute_output["tp"] = Replicate(), Partial()
    names = ("output", "attention_weights", "cache") if getattr(module, "_returns_tuple", False) else ("output",)
    return ModuleShardingSpec(params=params, tp_divide_attrs=list(tp_divide_attrs),
                              in_src={"hidden_states": dict(tokens)}, in_dst={"hidden_states": compute_input},
                              out_src={"output": compute_output}, out_dst={"output": dict(tokens)},
                              out_names=names, is_boundary=True)


def validate_linear_attention_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject a user override that changes ownership without an implemented algorithm.

    Args:
        module: Existing source module or resolved family member.
        sharding: Proposed input, output and parameter placements.
        context: Logical parallel sizes or the resolved sharding context.
    """
    member = get_family_build_info(module).resolved
    expected = member.family.sharding(module=module, module_fqn="", member=member, context=context)
    validate_tp_boundary(sharding, expected, name=type(module).__name__,
                         sequence_dim=0 if module.input_layout == "TH" else 1)


def linear_attention_local_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Validate local layouts without borrowing CP resources.

    Args:
        module: Existing source module or resolved family member.
        member: Resolved family member and its implementation arguments.
    """
    del module, member
    return ParallelModuleSpec("attention.linear.local", validate=validate_linear_attention_layout)
