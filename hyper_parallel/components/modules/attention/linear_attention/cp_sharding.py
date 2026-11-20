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
"""Replicated-weight and token-layout contracts for the qualified TP1 families."""
from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.distributed.module_binding import ParallelModuleSpec

def linear_attention_sharding(*, module: nn.Module, module_fqn: str,
                              member: Any, context: Any) -> ModuleShardingSpec:
    """Own replicated weights and the external token shard; CP keeps head changes internal.

    Args:
        module: Existing source module or resolved family member.
        module_fqn: Fully qualified module path in the parent model.
        member: Resolved family member and its implementation arguments.
        context: Logical parallel sizes or the resolved sharding context.
    """
    del module_fqn, member
    if context.axis_sizes["tp"] > 1:
        raise NotImplementedError("Linear-attention family TP is not qualified")
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    tokens = dict(placements)
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(0 if module.input_layout == "TH" else 1)
    names = ("output", "attention_weights", "cache") if getattr(module, "_returns_tuple", False) else ("output",)
    return ModuleShardingSpec(params={name: dict(placements) for name, _ in module.named_parameters()},
                              in_src={"hidden_states": dict(tokens)}, in_dst={"hidden_states": dict(tokens)},
                              out_src={"output": dict(tokens)}, out_dst={"output": dict(tokens)},
                              out_names=names, is_boundary=True)


def validate_linear_attention_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject a user override that changes ownership without an implemented algorithm.

    Args:
        module: Existing source module or resolved family member.
        sharding: Proposed input, output and parameter placements.
        context: Logical parallel sizes or the resolved sharding context.
    """
    expected = linear_attention_sharding(module=module, module_fqn="", member=None, context=context)
    validate_tp_boundary(sharding, expected, name=type(module).__name__)


def linear_attention_local_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Validate local layouts without borrowing CP resources.

    Args:
        module: Existing source module or resolved family member.
        member: Resolved family member and its implementation arguments.
    """
    del module, member
    return ParallelModuleSpec("attention.linear.local", validate=validate_linear_attention_layout)
