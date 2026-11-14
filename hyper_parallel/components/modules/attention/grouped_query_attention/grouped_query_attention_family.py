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
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def validate_gqa_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Reject TP degrees that split KV groups before constructing parameters."""
    del context
    config = config if module is None else module.config
    if config is None:
        raise ValueError("GQA requires source configuration")
    if not isinstance(member.arguments["overlap"], bool):
        raise TypeError("GQA overlap must be boolean")
    degree = member.arguments["ulysses_degree"]
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
    params = {name: dict(placements) for name, _ in module.named_parameters()}
    if context.axis_sizes["tp"] > 1:
        for name, named in params.items():
            if name.startswith(("linear_qkv.", "q_proj.", "k_proj.", "v_proj.")):
                named["tp"] = Shard(0)
            elif name == "o_proj.weight":
                named["tp"] = Shard(1)
    tokens = dict(placements)
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(1)
    if context.axis_sizes["tp"] > 1 and context.sequence_parallel:
        tokens["tp"] = Shard(1)
    return ModuleShardingSpec(
        params=params, in_src={"hidden_states": dict(tokens)}, in_dst={"hidden_states": dict(tokens)},
        out_src={"output": dict(tokens)}, out_dst={"output": dict(tokens)},
        out_names=("output", "attention_weights"), is_boundary=False,
    )


def validate_gqa_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject incompatible parameter overrides before any collectives can be bound."""
    expected = gqa_sharding(module=module, module_fqn="", member=None, context=context)
    if sharding.params != expected.params:
        raise ValueError("GQA requires replicated per-head Norm, column-sharded QKV and row-sharded output parameters")
    for name, ports in (("input", sharding.in_src), ("output", sharding.out_dst)):
        key = "hidden_states" if name == "input" else "output"
        if ports is None or set(ports) != {key}:
            raise ValueError(f"GQA {name} must declare its named token port")
        expected_tokens = expected.in_src["hidden_states"]
        for axis in set(ports[key]) | set(expected_tokens):
            placement = ports[key].get(axis, Replicate())
            expected_placement = expected_tokens.get(axis, Replicate())
            if axis == "tp":
                if placement not in (Replicate(), Shard(1)):
                    raise ValueError("GQA TP token ports support Replicate or sequence Shard(1)")
            elif placement != expected_placement:
                raise ValueError("GQA cannot change non-TP token ownership at its external boundary")


@module_replacement
def GroupedQueryAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, overlap: bool = False, ulysses_degree: int | None = None,
) -> nn.Module:
    """Build grouped QKV attention; CP topology selects the class, overlap only selects scheduling.

    Args:
        config: Reserved for config-based construction; currently a source module is required.
        module: Source attention with the supported projection and checkpoint layout.
        module_fqn: Source module path supplied by replacement.
        context: Logical parallel configuration, independent of concrete process groups.
        mesh: Existing job mesh, borrowed by the shared execution binder.
        overlap: Submit ready Q/K/V transfers before independent projection stages.
        ulysses_degree: Head-parallel CP factor; 1 selects KV AG, None selects pure Ulysses.

    Returns:
        The local or CP member with fixed parameter names and family provenance.
    """
    return build_family(GroupedQueryAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, overlap=overlap, ulysses_degree=ulysses_degree)


register_module_family(ModuleFamily(
    name="grouped_query_attention", baseline=GroupedQueryAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.grouped_query_attention."
        "grouped_query_attention.GroupedQueryAttentionModule"
    ),
    "cp": (
        "hyper_parallel.components.modules.attention.grouped_query_attention."
        "cp_grouped_query_attention.CPGroupedQueryAttentionModule"
    )},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=gqa_sharding, validate=validate_gqa_family,
))


@module_replacement
def SplitGroupedQueryAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, overlap: bool = False, ulysses_degree: int | None = None,
) -> nn.Module:
    """Build three-projection GQA; CP topology selects the class, overlap only selects scheduling.

    Args:
        config: Reserved for config-based construction; currently a source module is required.
        module: Source attention with the supported projection and checkpoint layout.
        module_fqn: Source module path supplied by replacement.
        context: Logical parallel configuration, independent of concrete process groups.
        mesh: Existing job mesh, borrowed by the shared execution binder.
        overlap: Submit ready Q/K/V transfers before independent projection stages.
        ulysses_degree: Head-parallel CP factor; 1 selects KV AG, None selects pure Ulysses.

    Returns:
        The local or CP member with fixed parameter names and family provenance.
    """
    return build_family(SplitGroupedQueryAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, overlap=overlap, ulysses_degree=ulysses_degree)


register_module_family(ModuleFamily(
    name="split_grouped_query_attention", baseline=SplitGroupedQueryAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.grouped_query_attention."
        "grouped_query_attention.SplitGroupedQueryAttentionModule"
    ),
    "cp": (
        "hyper_parallel.components.modules.attention.grouped_query_attention."
        "cp_grouped_query_attention.CPSplitGroupedQueryAttentionModule"
    )},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=gqa_sharding, validate=validate_gqa_family,
))
