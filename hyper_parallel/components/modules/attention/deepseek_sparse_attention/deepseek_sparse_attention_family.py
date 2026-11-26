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
"""DeepSeek sparse-attention families, parameter layouts and parallel initialization."""

from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, register_module_family,
)
from hyper_parallel.components.modules.attention.cp_config import (
    resolve_attention_cp_config, validate_supported_cp_options,
)
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def validate_sparse_attention_family(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Check logical topology before allocating native or replacement parameters."""
    # Local/meta construction can retain a CP request before the job topology exists.
    cp_config = resolve_attention_cp_config(member)
    validate_supported_cp_options(
        cp_config, "DeepSeek sparse attention", allow_ulysses=False, allow_kv=True, allow_overlap=True,
    )
    configured_cp_size = context.get("cp_size")
    if configured_cp_size is not None and configured_cp_size != (
            cp_config["ulysses_cp_size"] * cp_config["kv_cp_size"] * cp_config["ring_cp_size"]):
        raise ValueError("DeepSeek sparse attention CP domains must multiply to the configured cp_size")
    config = config if module is None else module.config
    if member.axis_sizes["pp"] > 1:
        raise ValueError("DeepSeek sparse attention does not support pipeline parallelism")
    heads = getattr(module, "num_heads", getattr(config, "num_attention_heads", None))
    if heads % member.axis_sizes["tp"]:
        raise ValueError("Sparse-attention heads must be divisible by TP degree")


def sparse_attention_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Declare every owned parameter and external token port without name fallbacks."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    column_names = ("q_b_proj", "kv_b_proj")
    replica_names = (
        "linear_qkv", "q_a_layernorm", "kv_a_layernorm",
        "indexer.wq_b", "indexer.wk", "indexer.k_norm", "indexer.weights_proj",
    )
    columns = {f"{name}.{suffix}" for name in column_names for suffix in ("weight", "bias")}
    replicas = {
        f"{name}.{suffix}"
        for name in replica_names
        for suffix in ("weight", "bias")
    }
    roles = {
        **dict.fromkeys(columns, Shard(0)), **dict.fromkeys(replicas, Replicate()),
        "o_proj.weight": Shard(1), "o_proj.bias": Replicate(),
    }
    params = {}
    for name, _ in module.named_parameters(remove_duplicate=False):
        if name not in roles:
            raise ValueError(f"Sparse-attention parameter {name!r} has no declared sharding role")
        params[name] = dict(placements)
        if context.axis_sizes["tp"] > 1:
            params[name]["tp"] = roles[name]
    tokens = dict(placements)
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(1)
    if context.axis_sizes["tp"] > 1 and context.sequence_parallel:
        tokens["tp"] = Shard(1)
    compute_input, output = dict(tokens), dict(tokens)
    if context.axis_sizes["tp"] > 1:
        compute_input["tp"] = Replicate()
        output["tp"] = Partial()
    return ModuleShardingSpec(
        params=params, in_src={"hidden_states": dict(tokens)}, in_dst={"hidden_states": compute_input},
        out_src={"output": output}, out_dst={"output": dict(tokens)},
        out_names=("output", "attention_weights"), is_boundary=True,
    )


def validate_sparse_attention_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Validate the declared token and projection layouts after recipe overrides."""
    expected = sparse_attention_sharding(module=module, module_fqn="", member=None, context=context)
    validate_tp_boundary(sharding, expected, name="DeepSeek sparse attention")
    if tuple(sharding.out_names or ()) != tuple(expected.out_names):
        raise ValueError("Sparse-attention output names must preserve the declared return contract")


def sparse_attention_parallel_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Declare shared algorithm stages and validate layouts outside local computation."""
    del module, member
    return ParallelModuleSpec(
        "attention.deepseek_sparse_attention.local", binding_kind="mesh",
        stages={
            "preflight_attention": (
                "hidden_states", "attention_mask", "past_key_values", "position_ids",
                "actual_seq_len", "kwargs", "use_dense",
            ),
            "prepare_attention_inputs": ("hidden_states", "position_embeddings", "metadata", "use_dense"),
            "exchange_attention": ("inputs",),
            "run_attention": ("index_states", "attention_states", "sequence_lengths", "use_dense"),
            "project_output": ("output",),
        },
        validate=validate_sparse_attention_layout,
    )


@module_replacement
def DeepseekSparseAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None,
    ulysses_cp_size: int = 1, ulysses_cp_strategy: str = "alltoall",
    kv_cp_size: int = 1, kv_cp_strategy: str = "allgather", kv_cp_load_balance: bool | str = False,
    ring_cp_size: int = 1, ring_cp_strategy: str = "p2p", ring_cp_load_balance: bool | str = False,
    cp_comm_overlap: bool = False, dense_warmup: bool | None = None,
) -> nn.Module:
    """Build a local or CP member while retaining V3.2 source and checkpoint contracts.

    Args:
        config: Native attention dimensions and options; mutually exclusive with module.
        module: HF attention source or an already loaded unbound local member.
        module_fqn: Source path supplied by the replacement executor.
        context: Logical job topology used for family selection.
        mesh: Optional existing MeshContext for shared layout planning and binding.
        ulysses_cp_size: Ulysses domain size; this family currently requires one.
        ulysses_cp_strategy: Reserved Ulysses strategy; DSA currently has no Ulysses path.
        kv_cp_size: KV domain size; must account for all active CP ranks.
        kv_cp_strategy: KV transport; currently only ``allgather`` is supported.
        kv_cp_load_balance: KV-domain zigzag balance; currently unsupported by DSA.
        ring_cp_size: Ring domain size; this family currently requires one.
        ring_cp_strategy: Reserved Ring strategy; DSA currently has no Ring path.
        ring_cp_load_balance: Ring-domain zigzag balance; currently unsupported by DSA.
        cp_comm_overlap: Submit all key-bank gathers before waiting when true; defaults to False.
        dense_warmup: None follows the mutable model stage; bool pins the stage.

    Returns:
        Concrete member retaining the requested family and parameter layout.
    """
    return build_family(
        DeepseekSparseAttention, config=config, module=module, module_fqn=module_fqn, context=context,
        mesh=mesh, ulysses_cp_size=ulysses_cp_size, ulysses_cp_strategy=ulysses_cp_strategy,
        kv_cp_size=kv_cp_size, kv_cp_strategy=kv_cp_strategy, kv_cp_load_balance=kv_cp_load_balance,
        ring_cp_size=ring_cp_size, ring_cp_strategy=ring_cp_strategy,
        ring_cp_load_balance=ring_cp_load_balance, cp_comm_overlap=cp_comm_overlap,
        dense_warmup=dense_warmup,
    )


register_module_family(ModuleFamily(
    name="deepseek_sparse_attention", baseline=DeepseekSparseAttention,
    variants={
        "local": (
            "hyper_parallel.components.modules.attention.deepseek_sparse_attention."
            "deepseek_sparse_attention.DeepseekSparseAttentionModule"
        ),
        "cp": (
            "hyper_parallel.components.modules.attention.deepseek_sparse_attention."
            "cp_deepseek_sparse_attention.CPDeepseekSparseAttentionModule"
        ),
    },
    default_variant="local", parallel_axis="cp",
    variant_rules=(VariantRule({"parallel": True}, "cp"),),
    sharding=sparse_attention_sharding, validate=validate_sparse_attention_family,
    parallel_spec=sparse_attention_parallel_spec,
))
