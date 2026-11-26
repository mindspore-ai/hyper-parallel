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
import torch.distributed as dist

from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.distributed.module_binding import ParallelModuleSpec


def split_linear_attention_mesh(mesh: DeviceMesh, degree: int, names: tuple[str, str]) -> DeviceMesh:
    """Split chronological CP ranks into outer state and inner head-exchange groups.

    Args:
        mesh: One-dimensional CP mesh retaining its complete root rank map.
        degree: Number of consecutive sequence ranks in each Ulysses group.
        names: Names for the outer state axis and inner Ulysses axis.
    """
    if mesh.ndim != 1 or not isinstance(degree, int) or isinstance(degree, bool):
        raise ValueError("Linear-attention mesh splitting requires a 1-D mesh and integer degree.")
    if degree < 1 or mesh.size() % degree:
        raise ValueError("The split degree must be positive and divide the CP size.")
    ranks = tuple(mesh.rank_list)
    if ranks != tuple(dist.get_process_group_ranks(mesh.get_group())):
        raise ValueError("The CP mesh rank order must match the communication group order.")
    # Group creation must derive the same sibling groups on every root rank.
    root = mesh._get_root_mesh()  # pylint: disable=protected-access
    if root.size() != dist.get_world_size():
        raise ValueError("Hybrid linear attention requires a CP submesh retaining its WORLD root mesh.")
    return mesh._unflatten(0, (mesh.size() // degree, degree), names)  # pylint: disable=protected-access


def validate_state_packets(options: Any, cp_size: int, degree: int, backend: str,
                           local_heads: int | None = None) -> None:
    """Check explicit state packet ownership before creating communication groups.

    Args:
        options: Module-family arguments containing the packet count and protocol.
        cp_size: Total context-parallel degree.
        degree: Ulysses degree within the CP axis.
        backend: Selected local attention backend.
        local_heads: Value heads after TP and Ulysses, when dimensions are known.
    """
    packets = options.get("p2p_head_chunks", 1)
    if isinstance(packets, bool) or not isinstance(packets, int) or packets < 1:
        raise ValueError("p2p_head_chunks must be a positive integer")
    if packets > 1 and (backend != "ascendc" or options.get("state_cp_method", "p2p") != "p2p" or cp_size <= degree):
        raise ValueError("p2p_head_chunks requires Ascend C state P2P")
    if local_heads is not None and packets > local_heads:
        raise ValueError("p2p_head_chunks must not exceed the local value-head count")


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
