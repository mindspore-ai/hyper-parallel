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
"""Public CSA2 factory and its complete local/CP implementation allowlist."""

from __future__ import annotations

from functools import partial
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, SelectorSpec, VariantRule, build_family, register_module_family,
)
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.family_layout import validate_tp_boundary
from hyper_parallel.distributed.context_parallel.indexer_balance import (
    resolve_indexer_balance_block_size, validate_indexer_hybrid_threshold,
)
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement
from .compressed_sparse_attention2 import CompressedSparseAttention2Module
from .cp_compressed_sparse_attention2 import CPCompressedSparseAttention2, _TPCompressedSparseAttention2


def _sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Declare parameter roles and let the framework execute the TP/SP boundary."""
    del module_fqn, member
    axes = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    columns = ("q_b_proj", "indexer.q_b_proj", "indexer.weights_proj", "o_a_proj")
    replicated = ("q_a_proj", "q_a_norm", "kv_proj", "kv_norm", "compressor.wkv", "compressor.wgate",
                  "compressor.norm", "indexer.wk", "indexer.k_norm")
    roles = {f"{name}.{suffix}": Shard(0) for name in columns for suffix in ("weight", "bias")}
    roles.update({f"{name}.{suffix}": Replicate() for name in replicated for suffix in ("weight", "bias")})
    roles.update({"sinks": Shard(0), "o_b_proj.weight": Shard(1), "o_b_proj.bias": Replicate()})
    params = {}
    for name, _ in module.named_parameters():
        if name not in roles:
            raise ValueError(f"CSA2 parameter {name!r} has no declared sharding role")
        params[name] = dict(axes)
        if context.axis_sizes["tp"] > 1:
            params[name]["tp"] = roles[name]
    tokens = dict(axes)
    if context.axis_sizes["cp"] > 1:
        tokens["cp"] = Shard(1)
    if context.axis_sizes["tp"] > 1 and context.sequence_parallel:
        tokens["tp"] = Shard(1)
    compute_input, compute_output = dict(tokens), dict(tokens)
    if context.axis_sizes["tp"] > 1:
        compute_input["tp"], compute_output["tp"] = Replicate(), Partial()
    return ModuleShardingSpec(
        params=params, in_src={"hidden_states": dict(tokens)}, in_dst={"hidden_states": compute_input},
        out_src={"output": compute_output}, out_dst={"output": dict(tokens)},
        out_names=("output", "attention_weights"), is_boundary=True,
    )


def _validate_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject overrides that split heads/groups or bypass the declared TP boundary."""
    expected = _sharding(module=module, module_fqn="", member=None, context=context)
    validate_tp_boundary(sharding, expected, name="CSA2")
    tp_size = context.axis_sizes["tp"]
    if (module.config.o_groups % tp_size or module.num_heads % tp_size
            or (module.is_index_source and module.indexer.global_num_heads % tp_size)):
        raise ValueError("CSA2 main heads, Indexer heads and output groups must be divisible by TP")


def _parallel_spec(*, module: nn.Module, member: Any) -> ParallelModuleSpec:
    """Bind resources only to parallel members; standalone math has no mesh dependency."""
    del member
    if isinstance(module, CPCompressedSparseAttention2):
        return ParallelModuleSpec(
            "attention.csa2.cp", "cp",
            {"query_partition": ("sequence_length", "position_ids"),
             "prepare_attention_inputs": ("hidden_states", "embeddings", "shared_state", "document"),
             "exchange_attention": ("prepared", "shared_state"),
             "launch_bank": ("tensor", "sequence_dim")}, validate=_validate_layout,
        )
    if isinstance(module, _TPCompressedSparseAttention2):
        return ParallelModuleSpec(
            "attention.csa2.tp", "mesh", {"head_reducer": (),
                                          "query_partition": ("sequence_length", "position_ids")},
            validate=_validate_layout,
        )
    return ParallelModuleSpec("attention.csa2.local", validate=_validate_layout)


def _validate(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Validate the source contract and supported fusion, balance and communication options."""
    del context
    if module is not None and not isinstance(module, CompressedSparseAttention2Module):
        raise TypeError("CSA2 source conversion requires the shared V4.1 attention parameter contract")
    if module is None:
        required = ("v41_compress_ratios", "v41_kv_source_layer_ids", "v41_index_source_layer_ids")
        if any(not hasattr(config, name) for name in required):
            raise ValueError("CSA2 config must include compression ratios and Full/Reindex source layers")
    arguments = member.arguments
    mode = arguments.get("indexer_load_balance", "none")
    resolve_indexer_balance_block_size(mode, arguments.get("indexer_balance_block_size"))
    validate_indexer_hybrid_threshold(mode, arguments.get("indexer_zigzag_min_length"))
    if arguments.get("main_kv_comm", "all_gather") not in ("all_gather", "all_to_all_v"):
        raise ValueError("main_kv_comm must be 'all_gather' or 'all_to_all_v'")
    if arguments.get("index_k_comm", "all_gather") not in ("all_gather", "all_to_all_v"):
        raise ValueError("index_k_comm must be 'all_gather' or 'all_to_all_v'")
    paged = arguments.get("indexer_paged_k", False)
    if not isinstance(paged, bool):
        raise ValueError("indexer_paged_k must be a bool")
    if paged and not member.selectors["use_fused_ops"]:
        raise ValueError("indexer_paged_k requires use_fused_ops=True")
    if member.axis_sizes["tp"] > 1 and (member.selectors["use_fused_ops"] or mode != "none"):
        raise NotImplementedError("CSA2 fusion and Indexer load balancing currently require TP=1")


def _build_local(*, config: Any = None, module: nn.Module | None = None, layer_idx: int = 0,
                 module_fqn: str = "", context: Any = None, use_fused_ops: bool = False,
                 indexer_load_balance: str = "none", main_kv_comm: str = "all_gather",
                 indexer_balance_block_size: int | None = None,
                 indexer_zigzag_min_length: int | None = None, index_k_comm: str = "all_gather",
                 indexer_paged_k: bool = False) -> nn.Module:
    """Retain CP options in family provenance without adding them to local math."""
    del module_fqn, indexer_load_balance, main_kv_comm, index_k_comm
    del indexer_balance_block_size, indexer_zigzag_min_length
    implementation = (_TPCompressedSparseAttention2 if context is not None and context.get("tp_size", 1) > 1
                      else CompressedSparseAttention2Module)
    return implementation(module, config=config, layer_idx=layer_idx, use_fused_ops=use_fused_ops,
                          indexer_paged_k=indexer_paged_k)


_reference = partial(_build_local, use_fused_ops=False)
_fused = partial(_build_local, use_fused_ops=True)
_cp_reference = partial(CPCompressedSparseAttention2, use_fused_ops=False)
_cp_fused = partial(CPCompressedSparseAttention2, use_fused_ops=True)


@module_replacement
def CompressedSparseAttention2(  # pylint: disable=invalid-name
        *, config: Any = None, module: nn.Module | None = None, layer_idx: int = 0,
        module_fqn: str = "", context: Any = None, mesh: Any = None,
        use_fused_ops: bool = False, indexer_load_balance: str = "none",
        main_kv_comm: str = "all_gather", indexer_balance_block_size: int | None = None,
        indexer_zigzag_min_length: int | None = None, index_k_comm: str = "all_gather",
        indexer_paged_k: bool = False,
) -> nn.Module:
    """Build complete CSA2, choosing CP automatically from the job's existing mesh.

    Args:
        config: V4.1 architecture config for direct parameter construction.
        module: Compatible source module whose parameters and buffers are retained.
        layer_idx: Layer whose source relationships are read during direct construction.
        module_fqn: Model path supplied by the generic replacement executor.
        context: Job topology and construction facts.
        mesh: Optional prebuilt mesh for direct construction and layout binding.
        use_fused_ops: Select all three native entries; unsupported calls raise explicitly.
        indexer_load_balance: none, zigzag, greedy or hybrid; only selection tasks move.
        main_kv_comm: all_gather or document-prefix all_to_all_v.
        index_k_comm: Independent Index K all_gather or document-prefix all_to_all_v.
        indexer_paged_k: Use shared LI V2 physical pages for logical prefixes; requires fusion.
        indexer_balance_block_size: Greedy/hybrid task size, defaulting to 2048.
        indexer_zigzag_min_length: Hybrid-only long-document threshold in raw tokens.
    """
    return build_family(
        CompressedSparseAttention2, config=config, module=module, layer_idx=layer_idx,
        module_fqn=module_fqn, context=context, mesh=mesh, use_fused_ops=use_fused_ops,
        indexer_load_balance=indexer_load_balance, main_kv_comm=main_kv_comm,
        indexer_balance_block_size=indexer_balance_block_size,
        indexer_zigzag_min_length=indexer_zigzag_min_length,
        index_k_comm=index_k_comm, indexer_paged_k=indexer_paged_k,
    )


_PATH = __name__
register_module_family(ModuleFamily(
    name="compressed_sparse_attention2", baseline=CompressedSparseAttention2,
    variants={"reference": f"{_PATH}._reference", "fused": f"{_PATH}._fused",
              "cp_reference": f"{_PATH}._cp_reference", "cp_fused": f"{_PATH}._cp_fused"},
    selector_specs={"use_fused_ops": SelectorSpec(bool, False)},
    variant_rules=(
        VariantRule({"use_fused_ops": True, "parallel": False}, "fused"),
        VariantRule({"use_fused_ops": False, "parallel": True}, "cp_reference"),
        VariantRule({"use_fused_ops": True, "parallel": True}, "cp_fused"),
    ),
    default_variant="reference", parallel_axis="cp", sharding=_sharding, validate=_validate,
    parallel_spec=_parallel_spec,
))
