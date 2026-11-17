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
"""GQA families: fixed parameter representations and complete TP/SP layout ownership."""

from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.models.replacement import module_replacement
from .grouped_query_attention import GroupedQueryAttentionModule


def validate_gqa_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Reject TP degrees that split KV groups before constructing parameters."""
    del context
    config = config if module is None else module.config
    if config is None:
        raise ValueError("GQA requires configuration")
    heads = (config.num_attention_heads, config.num_key_value_heads)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in heads):
        raise ValueError("GQA head counts must be positive integers")
    if not isinstance(member.arguments.get("fuse_qkv", True), bool):
        raise TypeError("GQA fuse_qkv must be boolean")
    # Declarative replacement requests omit defaults, unlike a direct Python factory call.
    if not isinstance(member.arguments.get("overlap", False), bool):
        raise TypeError("GQA overlap must be boolean")
    _validate_gqa_degrees(config, member)


def _validate_gqa_degrees(config, member):
    """Check TP/CP factors against whole Q and KV groups."""
    degree = member.arguments.get("ulysses_degree")
    cp = member.axis_sizes["cp"]
    degree = cp if degree is None else degree
    if (not isinstance(degree, int) or isinstance(degree, bool)) or degree < 1 or (cp > 1 and cp % degree):
        raise ValueError("GQA Ulysses degree must be a positive integer dividing CP degree")
    tp = member.axis_sizes["tp"]
    if config.num_attention_heads % tp or config.num_key_value_heads % tp:
        raise ValueError("GQA query and KV heads must be divisible by TP degree; KV replication is not implicit")

    if cp > 1 and (config.num_attention_heads // tp % degree or config.num_key_value_heads // tp % degree):
        raise ValueError("GQA Ulysses degree must divide TP-local Q and KV heads")

def gqa_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own all GQA parameters and declare the actual external token layout."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    # Every supported parameter has an explicit role. A new/quantized projection
    # must declare its layout instead of silently inheriting Replicate.
    columns = {f"{name}.{suffix}" for name in ('linear_qkv', 'q_proj', 'k_proj', 'v_proj')
                  for suffix in ("weight", "bias")}
    replicated = {f"{name}.{suffix}" for name in ('q_norm', 'k_norm')
                  for suffix in ("weight", "bias")}
    roles = {**dict.fromkeys(columns, Shard(0)), **dict.fromkeys(replicated, Replicate()),
             "o_proj.weight": Shard(1), "o_proj.bias": Replicate()}
    params = {}
    for name, _ in module.named_parameters():
        if name not in roles:
            raise ValueError(f"GQA parameter {name!r} has no declared sharding role")
        params[name] = dict(placements)
        if context.axis_sizes["tp"] > 1:
            params[name]["tp"] = roles[name]
    tokens = dict(placements)
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(1)
    if context.axis_sizes["tp"] > 1 and context.sequence_parallel:
        tokens["tp"] = Shard(1)
    # CP ownership describes the external token shard. Ulysses changes head /
    # sequence ownership internally and restores this same shard before return.
    compute_input, compute_output = dict(tokens), dict(tokens)
    if context.axis_sizes["tp"] > 1:
        compute_input["tp"], compute_output["tp"] = Replicate(), Partial()
    return ModuleShardingSpec(
        params=params, in_src={"hidden_states": dict(tokens)}, in_dst={"hidden_states": compute_input},
        out_src={"output": compute_output}, out_dst={"output": dict(tokens)},
        out_names=("output", "attention_weights"), is_boundary=True,
    )


def validate_gqa_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject incompatible parameter overrides before any collectives can be bound."""
    expected = gqa_sharding(module=module, module_fqn="", member=None, context=context)
    validate_tp_boundary(sharding, expected, name="GQA")



def gqa_local_parallel_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Retain final layout validation without binding resources to standalone math."""
    del module, member
    return ParallelModuleSpec("attention.gqa.local", validate=validate_gqa_layout)


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    fuse_qkv: bool = True, overlap: bool = False, ulysses_degree: int | None = None,
) -> nn.Module:
    """Build standalone math; the family retains CP options and owns parallel layouts."""
    del module_fqn, context, overlap, ulysses_degree
    return GroupedQueryAttentionModule(config=config, module=module, fuse_qkv=fuse_qkv)


@module_replacement
def GroupedQueryAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, fuse_qkv: bool = True,
    overlap: bool = False, ulysses_degree: int | None = None,
) -> nn.Module:
    """Build one GQA family with fixed projections and independent CP scheduling.

    Args:
        config: Native dimensions/options; mutually exclusive with module. Q/K RMSNorm is opt-in via qk_norm.
        module: Source attention with the supported projection and checkpoint layout.
        module_fqn: Source module path supplied by replacement.
        context: Logical parallel configuration, independent of concrete process groups.
        mesh: Existing job mesh, borrowed by the shared execution binder.
        fuse_qkv: Use grouped linear_qkv (True) or independent Q/K/V Parameters (False); fixed at construction.
        overlap: Submit ready Q/K/V transfers before independent projection stages.
        ulysses_degree: Head-parallel CP factor; 1 selects KV AG, None selects pure Ulysses.

    Returns:
        The local or CP member with fixed parameter names and family provenance.
    """
    return build_family(GroupedQueryAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, fuse_qkv=fuse_qkv,
                        overlap=overlap, ulysses_degree=ulysses_degree)


register_module_family(ModuleFamily(
    name="grouped_query_attention", baseline=GroupedQueryAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.grouped_query_attention."
        "grouped_query_attention_family._build_local"
    ),
    "cp": (
        "hyper_parallel.components.modules.attention.grouped_query_attention."
        "cp_grouped_query_attention.CPGroupedQueryAttentionModule"
    )},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=gqa_sharding, validate=validate_gqa_family, parallel_spec=gqa_local_parallel_spec,
))
