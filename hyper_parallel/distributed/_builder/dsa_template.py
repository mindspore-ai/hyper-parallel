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
"""Structurally matched tensor-parallel boundary template for DSA attention."""

import re
from typing import Dict, Optional

from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


_ATTN_ROOT = r"(?:^|\.)layers\.\d+\.(?:mtp_block\.)?self_attention$"
_ATTN = _ATTN_ROOT[:-1] + r"\."
_ATTENTION_TYPES = frozenset({"dsa", "mla", "gqa"})
_DISTINCTIVE_LEAVES = frozenset({
    "linear_qb",
    "linear_kvb",
    "index_linear_qb",
    "index_linear_k",
    "linear_merge_weight",
    "sparse_lightning_indexer_kllloss",
})


def _direct_params(module, placement):
    return {
        name: {"tp": placement}
        for name, _ in module.named_parameters(recurse=False)
    }


def _linear(module, *, param, in_src, in_dst, out_src, out_dst):
    return ModuleShardingSpec(
        params=_direct_params(module, param),
        in_src={"input": {"tp": in_src}},
        in_dst={"input": {"tp": in_dst}},
        out_src={"output": {"tp": out_src}},
        out_dst={"output": {"tp": out_dst}},
    )


def _is_dsa_attention(module) -> bool:
    """Match the DSA/MLA attention contract without using a model name."""
    if getattr(module, "attention_type", None) in _ATTENTION_TYPES:
        return True
    if any(
        name.startswith("param_sink_")
        for name, _ in module.named_parameters(recurse=False)
    ):
        return True
    child_names = {name for name, _ in module.named_children()}
    return bool(_DISTINCTIVE_LEAVES.intersection(child_names))


def _matching_attention_roots(named_modules) -> frozenset[str]:
    return frozenset(
        fqn
        for fqn, module in named_modules.items()
        if re.search(_ATTN_ROOT, fqn) and _is_dsa_attention(module)
    )


def matches_dsa_template(model) -> bool:
    """Return whether *model* exposes the DSA/MLA structural contract."""
    return bool(_matching_attention_roots(dict(model.named_modules())))


def _attention_sink_spec(module) -> Optional[ModuleShardingSpec]:
    """Replicate the shared ``param_sink_*`` parameters of an attention root.

    Sink parameters participate in the head-sharded DSA projections but are
    shared by all TP ranks.  They live directly on the attention module, which
    is intentionally not a communication boundary in this template, so declare
    a parameter-only replicated spec.  Returns None when the module has no sink
    parameters.
    """
    sink_params = {
        name: {"tp": Replicate()}
        for name, _ in module.named_parameters(recurse=False)
        if name.startswith("param_sink_")
    }
    if not sink_params:
        return None
    return ModuleShardingSpec(params=sink_params, is_boundary=False)


def _lm_head_spec(module) -> ModuleShardingSpec:
    """Vocab-parallel LM-head projection consumed after the SP sequence gather.

    The integration gathers the language-model SP output before its
    multi-token prediction vocabulary heads, so the vocab-parallel projection
    consumes a replicated sequence (not Shard(1)).
    """
    return ModuleShardingSpec(
        params=_direct_params(module, Shard(0)),
        in_src={"input": {"tp": Replicate()}},
        in_dst={"input": {"tp": Replicate()}},
        out_src={"output": {"tp": Shard(-1)}},
        out_dst={"output": {"tp": Shard(-1)}},
    )


def _embed_tokens_spec(module) -> ModuleShardingSpec:
    """Vocab-parallel embedding whose output stays replicated for the parent.

    The VL model merges image/audio features into token embeddings before
    entering the language-model SP boundary.  Keep the vocab-parallel embedding
    output replicated here; the parent language-model boundary performs the
    sequence scatter after multimodal fusion.
    """
    return ModuleShardingSpec(
        params=_direct_params(module, Shard(0)),
        in_src={"hidden_states": {"tp": Replicate()}},
        in_dst={"hidden_states": {"tp": Replicate()}},
        out_src={"output": {"tp": Partial()}},
        out_dst={"output": {"tp": Replicate()}},
    )


def _head_sharded_projection_spec(fqn, module, leaf) -> ModuleShardingSpec:
    """Head-sharded DSA projection leaf (linear_qb / index_linear_qb / merge)."""
    spec = _linear(
        module, param=Shard(0), in_src=Shard(1), in_dst=Replicate(),
        out_src=Shard(-1), out_dst=Shard(-1))
    # DSA keeps num_heads/num_index_heads on the parent attention module while
    # the head-sharded weight lives in this leaf boundary.  Tag one canonical
    # Q projection so D-17 adjusts the owner exactly once after TP parameter
    # sharding.
    if leaf == "linear_qb":
        spec._head_count_owner = fqn.rsplit(".", 1)[0]  # pylint: disable=protected-access
    return spec


def _output_projection_spec(fqn, module, named_modules) -> ModuleShardingSpec:
    """Output projection whose reduce-scatter axis follows the runtime layout.

    GQA/DSA feed this projection in BSH layout, while MLA transposes its BSH
    attention result to SBH before the projection and transposes the output
    back afterwards.  Select the reduce-scatter axis from the owning attention
    implementation instead of the FQN: regular decoder layers may mix DSA and
    MLA attention, so an ``mtp_block`` name alone cannot determine the runtime
    layout.
    """
    parent_fqn = fqn.rsplit(".", 1)[0]
    attention_type = getattr(named_modules.get(parent_fqn), "attention_type", None)
    if attention_type == "mla":
        sequence_dim = 0
    elif attention_type in {"gqa", "dsa"}:
        sequence_dim = 1
    else:
        # Retain the legacy fallback for architecture-compatible test doubles
        # or external modules that do not expose attention_type.
        sequence_dim = 0 if ".mtp_block." in fqn else 1
    return _linear(
        module, param=Shard(1), in_src=Shard(-1), in_dst=Shard(-1),
        out_src=Partial(), out_dst=Shard(sequence_dim))


def _layernorm_spec(module) -> ModuleShardingSpec:
    """Sequence-parallel q/k layernorm boundary."""
    return ModuleShardingSpec(
        params=_direct_params(module, Replicate()),
        in_src={"hidden_states": {"tp": Shard(1)}},
        in_dst={"hidden_states": {"tp": Shard(1)}},
        out_src={"output": {"tp": Shard(1)}},
        out_dst={"output": {"tp": Shard(1)}},
    )


def _rotary_spec() -> ModuleShardingSpec:
    """Replicated rotary-embedding input contract."""
    inputs = {name: {"tp": Replicate()} for name in ("t", "cos", "sin")}
    return ModuleShardingSpec(params={}, in_src=inputs, in_dst=inputs, out_src={}, out_dst={})


def _sparse_indexer_spec() -> ModuleShardingSpec:
    """Contract for the sparse-lightning indexer KLL-loss leaf."""
    src = {
        "index_query": {"tp": Replicate()},
        "index_key": {"tp": Replicate()},
        "merge_weight": {"tp": Replicate()},
        "query": {"tp": Shard(1)},
        "key": {"tp": Replicate()},
        "topk_indices": {"tp": Replicate()},
        "softmax_max": {"tp": Shard(2)},
        "softmax_sum": {"tp": Shard(2)},
        "query_rope": {"tp": Shard(1)},
        "key_rope": {"tp": Replicate()},
        "actual_seq_qlen": {"tp": Replicate()},
        "actual_seq_klen": {"tp": Replicate()},
    }
    return ModuleShardingSpec(
        params={}, in_src=src,
        in_dst={name: {"tp": Replicate()} for name in src},
        out_src={}, out_dst={})


def _attention_leaf_spec(fqn, module, leaf, named_modules) -> Optional[ModuleShardingSpec]:
    """Build the boundary spec for one attention-leaf FQN (None if unhandled)."""
    if leaf in {"linear_qb", "index_linear_qb", "linear_merge_weight"}:
        return _head_sharded_projection_spec(fqn, module, leaf)
    if leaf in {"linear_qkv", "index_linear_k"}:
        return _linear(
            module, param=Replicate(), in_src=Shard(1), in_dst=Shard(1),
            out_src=Shard(1), out_dst=Shard(1))
    if leaf == "linear_kvb":
        return _linear(
            module, param=Shard(0), in_src=Replicate(), in_dst=Replicate(),
            out_src=Shard(-1), out_dst=Shard(-1))
    if leaf == "linear_proj":
        return _output_projection_spec(fqn, module, named_modules)
    if leaf in {"q_layernorm", "k_layernorm", "index_k_layernorm"}:
        return _layernorm_spec(module)
    if leaf in {"rotary_emb", "gather_rotary_emb"}:
        return _rotary_spec()
    if leaf == "sparse_lightning_indexer_kllloss":
        return _sparse_indexer_spec()
    return None


def build_dsa_specs(model) -> Dict[str, ModuleShardingSpec]:
    """Materialize DSA leaf-boundary specs for a structurally matched model.

    DSA does not follow a single q/k/v/o projection chain: some projections
    preserve sequence parallelism, some shard index/query heads, and
    ``linear_kvb.weight`` is consumed directly.  The template is selected by
    this module contract instead of ``config.architectures``/``model_type``.
    """
    specs = {}
    named_modules = dict(model.named_modules())
    attention_roots = _matching_attention_roots(named_modules)
    if not attention_roots:
        return specs

    for fqn, module in named_modules.items():
        if fqn in attention_roots:
            sink_spec = _attention_sink_spec(module)
            if sink_spec is not None:
                specs[fqn] = sink_spec

        # The integration gathers the language-model SP output before its
        # multi-token prediction vocabulary heads, so the vocab-parallel
        # projection consumes a replicated sequence (not Shard(1)).
        if fqn == "lm_head" or fqn.endswith(".lm_head"):
            specs[fqn] = _lm_head_spec(module)
            continue
        if fqn.endswith(".language_model.embed_tokens"):
            specs[fqn] = _embed_tokens_spec(module)
            continue
        if not re.search(_ATTN, fqn) or not any(
            fqn.startswith(f"{root}.") for root in attention_roots
        ):
            continue
        leaf = fqn.rsplit(".", 1)[-1]
        leaf_spec = _attention_leaf_spec(fqn, module, leaf, named_modules)
        if leaf_spec is not None:
            specs[fqn] = leaf_spec
    return specs
