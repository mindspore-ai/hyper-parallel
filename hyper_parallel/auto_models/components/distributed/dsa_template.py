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
"""Built-in tensor-parallel boundary template for DSA attention."""

import re
from typing import Dict

from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard

from .sharding_config import ModuleShardingSpec


_ATTN_ROOT = r"(?:^|\.)layers\.\d+\.(?:mtp_block\.)?self_attention$"
_ATTN = _ATTN_ROOT[:-1] + r"\."


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


def build_dsa_specs(model) -> Dict[str, ModuleShardingSpec]:
    """Materialize DSA leaf-boundary specs for the target model.

    DSA does not follow a single q/k/v/o projection chain: some projections
    preserve sequence parallelism, some shard index/query heads, and
    ``linear_kvb.weight`` is consumed directly.  Keeping these contracts as a
    named architecture template avoids rebuilding them in each trainer.
    """
    specs = {}
    named_modules = dict(model.named_modules())
    for fqn, module in named_modules.items():
        # Sink parameters participate in the head-sharded DSA projections but
        # are shared by all TP ranks.  They live directly on the attention
        # module, which is intentionally not a communication boundary in this
        # template, so declare a parameter-only replicated spec.
        if re.search(_ATTN_ROOT, fqn):
            sink_params = {
                name: {"tp": Replicate()}
                for name, _ in module.named_parameters(recurse=False)
                if name.startswith("param_sink_")
            }
            if sink_params:
                specs[fqn] = ModuleShardingSpec(
                    params=sink_params,
                    is_boundary=False,
                )

        # The integration gathers the language-model SP output before
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
        # The VL model merges image/audio features into token embeddings before
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
            # GQA/DSA feed this projection in BSH layout, while MLA transposes
            # its BSH attention result to SBH before the projection and
            # transposes the output back afterwards.  Select the reduce-scatter
            # axis from the owning attention implementation instead of the FQN:
            # regular decoder layers may mix DSA and MLA attention, so an
            # ``mtp_block`` name alone cannot determine the runtime layout.
            parent_fqn = fqn.rsplit(".", 1)[0]
            attention_type = getattr(named_modules.get(parent_fqn), "attention_type", None)
            if attention_type == "mla":
                sequence_dim = 0
            elif attention_type in {"gqa", "dsa"}:
                sequence_dim = 1
            else:
                # Retain the legacy fallback for architecture-compatible test
                # doubles or external modules that do not expose attention_type.
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


DSA_ARCHITECTURES = frozenset({
    "v2vl", "v2_vl", "v2_vl_moe",
})
