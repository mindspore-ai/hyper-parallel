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
"""MLA family selection and complete parameter/layout ownership."""

from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def validate_mla_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Validate logical ownership before allocating or replacing any parameter."""
    del context
    config = config if module is None else module.config
    if config.num_attention_heads % member.axis_sizes["tp"]:
        raise ValueError("MLA heads must be divisible by TP degree")


def mla_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own all MLA parameters and declare the actual external token layout."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    params = {name: dict(placements) for name, _ in module.named_parameters()}
    if context.axis_sizes["tp"] > 1:
        for name, named in params.items():
            if name.startswith(("q_b_proj.", "kv_b_proj.", "g_proj.")):
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


def validate_mla_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject incompatible parameter overrides before any collectives can be bound."""
    expected = mla_sharding(module=module, module_fqn="", member=None, context=context)
    if sharding.params != expected.params:
        raise ValueError("MLA requires replicated down/Norm, column-sharded up and row-sharded output parameters")
    for name, ports in (("input", sharding.in_src), ("output", sharding.out_dst)):
        key = "hidden_states" if name == "input" else "output"
        if ports is None or set(ports) != {key}:
            raise ValueError(f"MLA {name} must declare its named token port")
        expected_tokens = expected.in_src["hidden_states"]
        for axis in set(ports[key]) | set(expected_tokens):
            placement = ports[key].get(axis, Replicate())
            expected_placement = expected_tokens.get(axis, Replicate())
            if axis == "tp":
                if placement not in (Replicate(), Shard(1)):
                    raise ValueError("MLA TP token ports support Replicate or sequence Shard(1)")
            elif placement != expected_placement:
                raise ValueError("MLA cannot change non-TP token ownership at its external boundary")
    if context.axis_sizes["tp"] > 1 and getattr(module, "use_nope", False):
        raise ValueError("Gated MLA currently supports TP1 only")


@module_replacement
def MultiLatentAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Any = None, mesh: Any = None,
    strategy: str = "expanded_ulysses", overlap: bool = False,
    ulysses_degree: int | None = None, kv_exchange: str = "all_gather",
) -> nn.Module:
    """Build the MLA member from dimensions or a supported source module.

    Args:
        config: Full model dimensions and attention options for direct construction.
        module: Optional source attention; exclusive with config.
        module_fqn: Source path supplied by model replacement.
        context: Logical build constraints. Topology selects the local or CP implementation.
        mesh: Optional existing job mesh used by the shared layout planner.
        strategy: Expanded QKV Ulysses or shared latent-KV communication.
        overlap: Enable existing forward and supported backward scheduling.
        ulysses_degree: CP head-parallel factor; the remaining factor partitions queries.
        kv_exchange: Full AllGather or selective causal-document AllToAllV.

    Returns:
        The concrete MLA algorithm with family provenance and unchanged parameter FQNs.
    """
    return build_family(MultiLatentAttention, config=config, module=module,
                        module_fqn=module_fqn, context=context, mesh=mesh,
                        strategy=strategy, overlap=overlap, ulysses_degree=ulysses_degree, kv_exchange=kv_exchange)


register_module_family(ModuleFamily(
    name="multi_latent_attention", baseline=MultiLatentAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.multi_latent_attention."
        "multi_latent_attention.MultiLatentAttentionModule"
    ), "cp": (
        "hyper_parallel.components.modules.attention.multi_latent_attention."
        "cp_multi_latent_attention.CPMultiLatentAttentionModule"
    )},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=mla_sharding, validate=validate_mla_family,
))
