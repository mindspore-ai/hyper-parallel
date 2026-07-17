# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""TP-local head-count adjustment for non-TP-tolerant modeling code (D-17).

Some HF modeling code reshapes/splits with an explicit (global) head count,
e.g. ``q.view(b, s, self.num_heads, self.head_dim)``, instead of the
TP-tolerant ``-1`` form. After TP colwise sharding each rank holds
``num_heads / tp`` heads locally, so any forward that runs on **local
tensors** needs the local count (AutoModel-style adaptation).

Dual-mode rule -- a module's cached attributes are rewritten only when its
forward sees local tensors in the current mode:

- production: parameters are permanently unwrapped to local tensors, so
  every head-sharded module is rewritten;
- validate: boundary modules run DTensor dispatch on the GLOBAL logical
  shape (a local count would break the reshape), so they are never
  rewritten; local-region modules (``local_compute_fn`` / EP injection
  intent / ``use_local_map``) unwrap to local inside the region in both
  modes, so they are rewritten in validate too.

The config object is never touched (``head_dim`` / RoPE derivations keep
working); only module-instance cached attributes are rewritten, idempotently
-- originals are stashed in ``module._hp_full_head_counts`` and repeat calls
are no-ops.
"""

import logging

from hyper_models.components.distributed.sharding_config import (
    resolve_placements,
)
from hyper_parallel.core.dtensor.placement_types import Shard

logger = logging.getLogger(__name__)

# Attribute list: survey of transformers/src/transformers/models (2026-07),
# counting forward-time reshape/split usages of cached head-count attributes:
# - q head count: num_heads (x393) / num_attention_heads (x122) / n_heads
#   (x50) / num_attn_heads (prophetnet) / n_head (openai, xlnet) /
#   heads, num_head (falcon, qwen2_5_omni, ...);
# - kv head count: num_key_value_heads (widespread) / num_kv_heads, kv_heads
#   (falcon, gpt_bigcode);
# - excluded: head_dim / attention_head_size / head_size (the head dimension,
#   never sharded) and num_key_value_groups (a ratio -- TP-invariant).
Q_HEAD_ATTRS = (
    "num_heads", "num_attention_heads", "n_heads", "num_attn_heads",
    "n_head", "heads", "num_head",
)
KV_HEAD_ATTRS = ("num_key_value_heads", "num_kv_heads", "kv_heads")

_ORIGINALS_ATTR = "_hp_full_head_counts"

# q/k/v projections whose colwise Shard(0) splits the head dimension
# (q_b_proj covers the MLA up-projection, D-14).
_QKV_WEIGHT_SUFFIXES = (
    "q_proj.weight", "q_b_proj.weight", "k_proj.weight", "v_proj.weight",
    "qkv_proj.weight", "qkv.weight",
)


def _is_head_sharded(spec, mesh_dim_names) -> bool:
    """True when the spec column-wise shards any q/k/v projection on the TP axis."""
    if "tp" not in mesh_dim_names:
        return False
    tp_idx = tuple(mesh_dim_names).index("tp")
    for name, placement in spec.params.items():
        if not name.endswith(_QKV_WEIGHT_SUFFIXES):
            continue
        if resolve_placements(placement, mesh_dim_names)[tp_idx] == Shard(0):
            return True
    return False


def _tp_degree(mesh, mesh_dim_names) -> int:
    if "tp" not in mesh_dim_names:
        return 1
    return mesh["tp"].size()


def update_module_head_counts(module, tp_size, module_fqn="") -> int:
    """Divide a module's cached head-count attributes by tp_size (idempotent).

    Only plain-int attributes whose name is in Q_HEAD_ATTRS / KV_HEAD_ATTRS
    are rewritten. Originals are recorded in ``module._hp_full_head_counts``;
    repeat calls (e.g. applying a plan twice) are no-ops. A non-divisible
    value is left unchanged with a loud warning (planner-side
    validate_model_compatibility normally fails first on config-level
    divisibility).

    Returns the number of attributes rewritten on this call.
    """
    if tp_size <= 1:
        return 0
    originals = getattr(module, _ORIGINALS_ATTR, None)
    if originals is None:
        originals = {}
        setattr(module, _ORIGINALS_ATTR, originals)
    n = 0
    for attr in Q_HEAD_ATTRS + KV_HEAD_ATTRS:
        if attr in originals:
            continue
        value = getattr(module, attr, None)
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        originals[attr] = value
        if value % tp_size != 0:
            logger.warning(
                "head-count adjustment: %s.%s = %d is not divisible by "
                "tp_size=%d -- left unchanged",
                module_fqn, attr, value, tp_size)
            continue
        setattr(module, attr, value // tp_size)
        logger.info("head-count adjustment: %s.%s %d -> %d (tp_size=%d)",
                    module_fqn, attr, value, value // tp_size, tp_size)
        n += 1
    return n


def maybe_update_head_counts(module, spec, module_fqn, mesh, mesh_dim_names) -> None:
    """Adjust cached head counts when the spec head-shards the module.

    Called by the applier exactly where the module's forward is guaranteed to
    see local tensors: production (Phase A, all specs) and validate (Phase C,
    local-region specs only).
    """
    if _is_head_sharded(spec, mesh_dim_names):
        update_module_head_counts(
            module, _tp_degree(mesh, mesh_dim_names), module_fqn)
