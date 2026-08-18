"""Built-in tensor-parallel boundary template for OpenPangu DSA attention."""

import re
from typing import Dict

from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard

from .sharding_config import ModuleShardingSpec


_ATTN = r"(?:^|\.)layers\.\d+\.(?:mtp_block\.)?self_attention\."


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


def build_openpangu_dsa_specs(model) -> Dict[str, ModuleShardingSpec]:
    """Materialize DSA leaf-boundary specs for an OpenPangu model.

    DSA does not follow a single q/k/v/o projection chain: some projections
    preserve sequence parallelism, some shard index/query heads, and
    ``linear_kvb.weight`` is consumed directly.  Keeping these contracts as a
    named architecture template avoids rebuilding them in each trainer.
    """
    specs = {}
    for fqn, module in model.named_modules():
        # The OpenPangu integration gathers the language-model SP output before
        # its multi-token prediction vocabulary heads.  The vocab-parallel
        # projection therefore consumes a replicated sequence (not Shard(1)).
        if fqn == "lm_head" or fqn.endswith(".lm_head"):
            specs[fqn] = ModuleShardingSpec(
                params=_direct_params(module, Shard(0)),
                in_src={"input": {"tp": Replicate()}},
                in_dst={"input": {"tp": Replicate()}},
                out_src={"output": {"tp": Shard(-1)}},
                out_dst={"output": {"tp": Shard(-1)}},
            )
            continue
        # OpenPangu VL merges image/audio features into token embeddings before
        # entering the language-model SP boundary.  Keep the vocab-parallel
        # embedding output replicated here; the parent language-model boundary
        # performs the sequence scatter after multimodal fusion.
        if fqn.endswith(".language_model.embed_tokens"):
            specs[fqn] = ModuleShardingSpec(
                params=_direct_params(module, Shard(0)),
                in_src={"hidden_states": {"tp": Replicate()}},
                in_dst={"hidden_states": {"tp": Replicate()}},
                out_src={"output": {"tp": Partial()}},
                out_dst={"output": {"tp": Replicate()}},
            )
            continue
        if not re.search(_ATTN, fqn):
            continue
        leaf = fqn.rsplit(".", 1)[-1]

        if leaf in {"linear_qb", "index_linear_qb", "linear_merge_weight"}:
            specs[fqn] = _linear(
                module, param=Shard(0), in_src=Shard(1), in_dst=Replicate(),
                out_src=Shard(-1), out_dst=Shard(-1))
            # DSA keeps num_heads/num_index_heads on the parent attention
            # module while the head-sharded weight lives in this leaf
            # boundary.  Tag one canonical Q projection so D-17 adjusts the
            # owner exactly once after TP parameter sharding.
            if leaf == "linear_qb":
                specs[fqn]._head_count_owner = fqn.rsplit(".", 1)[0]
        elif leaf in {"linear_qkv", "index_linear_k"}:
            specs[fqn] = _linear(
                module, param=Replicate(), in_src=Shard(1), in_dst=Shard(1),
                out_src=Shard(1), out_dst=Shard(1))
        elif leaf == "linear_kvb":
            specs[fqn] = _linear(
                module, param=Shard(0), in_src=Replicate(), in_dst=Replicate(),
                out_src=Shard(-1), out_dst=Shard(-1))
        elif leaf == "linear_proj":
            sequence_dim = 0 if ".mtp_block." in fqn else 1
            specs[fqn] = _linear(
                module, param=Shard(1), in_src=Shard(-1), in_dst=Shard(-1),
                out_src=Partial(), out_dst=Shard(sequence_dim))
        elif leaf in {"q_layernorm", "k_layernorm", "index_k_layernorm"}:
            specs[fqn] = ModuleShardingSpec(
                params=_direct_params(module, Replicate()),
                in_src={"hidden_states": {"tp": Shard(1)}},
                in_dst={"hidden_states": {"tp": Shard(1)}},
                out_src={"output": {"tp": Shard(1)}},
                out_dst={"output": {"tp": Shard(1)}},
            )
        elif leaf in {"rotary_emb", "gather_rotary_emb"}:
            inputs = {name: {"tp": Replicate()} for name in ("t", "cos", "sin")}
            specs[fqn] = ModuleShardingSpec(
                params={}, in_src=inputs, in_dst=inputs, out_src={}, out_dst={})
        elif leaf == "sparse_lightning_indexer_kllloss":
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
            specs[fqn] = ModuleShardingSpec(
                params={}, in_src=src,
                in_dst={name: {"tp": Replicate()} for name in src},
                out_src={}, out_dst={})
    return specs


OPENPANGU_DSA_ARCHITECTURES = frozenset({
    "openpanguv2vl", "openpangu_v2_vl", "openpangu_v2_vl_moe",
})
