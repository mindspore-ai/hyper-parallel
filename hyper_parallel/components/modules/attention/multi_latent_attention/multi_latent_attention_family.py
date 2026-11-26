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

from importlib import import_module
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.cp_config import (
    resolve_attention_cp_config, validate_supported_cp_options,
)
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.models.replacement import module_replacement


def validate_mla_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Validate logical ownership before allocating or replacing any parameter."""
    del context
    config = config if module is None else module.config
    if config is None:
        raise ValueError("MLA requires source configuration")
    options = member.arguments
    cp_config = resolve_attention_cp_config(member)
    validate_supported_cp_options(
        cp_config, "MLA", ulysses_strategies=("alltoall", "expanded_ulysses", "latent_kv_head"),
        allow_ulysses=True, allow_kv=True, allow_overlap=True,
    )
    exchange = options.get("kv_exchange", "all_gather")
    if exchange not in ("all_gather", "all_to_all_v"):
        raise ValueError("kv_exchange must be all_gather or all_to_all_v")
    cp, tp = member.axis_sizes["cp"], member.axis_sizes["tp"]
    degree = cp_config["ulysses_cp_size"]
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("MLA ulysses_cp_size must be a positive integer")
    if config.num_attention_heads % tp:
        raise ValueError("MLA heads must be divisible by TP degree")
    _validate_mla_topology(config, cp, tp, degree, exchange)


def _validate_mla_topology(config, cp, tp, degree, exchange):
    """Check active CP decomposition after validating selector syntax."""
    # CP1 may retain a valid future CP request for promotion. Validate syntax
    # now; validate the actual factorization once CP topology becomes active.
    if cp > 1:
        if cp % degree or config.num_attention_heads % (tp * degree):
            raise ValueError("MLA ulysses_cp_size must divide cp_size and TP-local heads")
        if exchange == "all_to_all_v" and cp == degree:
            raise ValueError("selective KV exchange requires query-partitioned MLA CP")
        if tp > 1 and cp != degree:
            raise ValueError("MLA KV AllGather and hybrid CP currently require TP1")

def mla_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own all MLA parameters and declare the actual external token layout."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    # Every supported parameter has an explicit role. A new/quantized projection
    # must declare its layout instead of silently inheriting Replicate.
    columns = {f"{name}.{suffix}" for name in ('q_b_proj', 'kv_b_proj', 'g_proj')
                  for suffix in ("weight", "bias")}
    replicated = {f"{name}.{suffix}" for name in ('linear_qkv', 'q_a_layernorm', 'kv_a_layernorm')
                  for suffix in ("weight", "bias")}
    roles = {**dict.fromkeys(columns, Shard(0)), **dict.fromkeys(replicated, Replicate()),
             "o_proj.weight": Shard(1), "o_proj.bias": Replicate()}
    params = {}
    for name, _ in module.named_parameters():
        if name not in roles:
            raise ValueError(f"MLA parameter {name!r} has no declared sharding role")
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


def validate_mla_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject incompatible parameter overrides before any collectives can be bound."""
    expected = mla_sharding(module=module, module_fqn="", member=None, context=context)
    validate_tp_boundary(sharding, expected, name="MLA")
    if context.axis_sizes["tp"] > 1 and getattr(module, "use_nope", False):
        raise ValueError("Gated MLA currently supports TP1 only")


def mla_local_parallel_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Keep layout validation outside the standalone algorithm."""
    del module, member
    return ParallelModuleSpec("attention.mla.local", validate=validate_mla_layout)


def _build_local(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    kv_exchange: str = "all_gather",
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Consume family-only options; the saved request retains them for CP preparation."""
    # Resolve after family registration; concrete members import this module's layout validators.
    implementation = import_module('.multi_latent_attention', __package__).MultiLatentAttentionModule

    del (module_fqn, context, kv_exchange, ulysses_cp_size,
         ulysses_cp_strategy, kv_cp_size, kv_cp_strategy, kv_cp_load_balance, ring_cp_size,
         ring_cp_strategy, ring_cp_load_balance, cp_comm_overlap)
    return implementation(config=config, module=module)


def _build_cp(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    kv_exchange: str = "all_gather", ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False,
) -> nn.Module:
    """Adapt the public domain configuration to the existing MLA CP executor."""
    del module_fqn
    selected_strategy = ulysses_cp_strategy
    if selected_strategy == "alltoall":
        selected_strategy = "expanded_ulysses"
    from .cp_multi_latent_attention import CPMultiLatentAttentionModule  # pylint: disable=C0415

    return CPMultiLatentAttentionModule(
        config=config, module=module, context=context, ulysses_cp_strategy=selected_strategy,
        cp_comm_overlap=cp_comm_overlap, ulysses_cp_size=ulysses_cp_size,
        kv_exchange=kv_exchange,
    )


@module_replacement
def MultiLatentAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None,
    module_fqn: str = "", context: Any = None, mesh: Any = None,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False, kv_exchange: str = "all_gather",
) -> nn.Module:
    """Build the MLA member from dimensions or a supported source module.

    Args:
        config: Full model dimensions and attention options for direct construction.
        module: Optional source attention; exclusive with config.
        module_fqn: Source path supplied by model replacement.
        context: Logical build constraints. Topology selects the local or CP implementation.
        mesh: Optional existing job mesh used by the shared layout planner.
        ulysses_cp_size: Number of ranks in the Ulysses domain.
        ulysses_cp_strategy: ``alltoall``/``expanded_ulysses`` or ``latent_kv_head``.
        kv_cp_size: Number of ranks in the KV AllGather domain.
        kv_cp_strategy: MLA currently supports ``allgather``; P2P is not implemented.
        kv_cp_load_balance: KV-domain zigzag balance; currently unsupported by MLA.
        ring_cp_size: Number of ranks in the Ring domain; currently must be one.
        ring_cp_strategy: Ring transport selector, reserved for future MLA support.
        ring_cp_load_balance: Ring-domain zigzag balance; currently unsupported by MLA.
        cp_comm_overlap: Enable the existing forward/backward scheduling windows.
        kv_exchange: Full AllGather or selective causal-document AllToAllV.

    Returns:
        The concrete MLA algorithm with family provenance and unchanged parameter FQNs.
    """
    return build_family(MultiLatentAttention, config=config, module=module,
                        module_fqn=module_fqn, context=context, mesh=mesh,
                        ulysses_cp_size=ulysses_cp_size, ulysses_cp_strategy=ulysses_cp_strategy,
                        kv_cp_size=kv_cp_size, kv_cp_strategy=kv_cp_strategy,
                        kv_cp_load_balance=kv_cp_load_balance, ring_cp_size=ring_cp_size,
                        ring_cp_strategy=ring_cp_strategy, ring_cp_load_balance=ring_cp_load_balance,
                        cp_comm_overlap=cp_comm_overlap, kv_exchange=kv_exchange)


register_module_family(ModuleFamily(
    name="multi_latent_attention", baseline=MultiLatentAttention,
    variants={"local": (
        "hyper_parallel.components.modules.attention.multi_latent_attention."
        "multi_latent_attention_family._build_local"
    ), "cp": "hyper_parallel.components.modules.attention.multi_latent_attention."
          "multi_latent_attention_family._build_cp"},
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=mla_sharding, validate=validate_mla_family, parallel_spec=mla_local_parallel_spec,
))
