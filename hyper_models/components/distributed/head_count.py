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
  rewritten; local-region modules (``local_compute_fn`` /
  ``region_dispatch=False``) unwrap to local inside the region in both
  modes, so they are rewritten in validate too.

The config object is never touched (``head_dim`` / RoPE derivations keep
working); only module-instance cached attributes are rewritten, idempotently
-- originals are stashed in ``module._hp_full_head_counts`` and repeat calls
are no-ops.
"""

import logging
from typing import Any, Optional, Sequence

from hyper_models.components.distributed.sharding_config import (
    TpLocalAttrPlan,
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
_TP_LOCAL_ORIGINALS_ATTR = "_hp_full_tp_local_attrs"

_FORBIDDEN_USER_ATTRS = frozenset({
    "head_dim", "attention_head_size", "head_size",
    "num_key_value_groups", "training", "dtype", "device",
})

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


def collect_auto_head_attrs(
    module: Any, spec: Any, mesh_dim_names: Sequence[str],
) -> tuple[str, ...]:
    """Collect canonical head-count attributes derived from TP placements."""
    if not _is_head_sharded(spec, mesh_dim_names):
        return ()
    return tuple(
        attr for attr in Q_HEAD_ATTRS + KV_HEAD_ATTRS
        if isinstance(getattr(module, attr, None), int)
        and not isinstance(getattr(module, attr), bool)
    )


def normalize_tp_divide_attrs(
    attrs: Optional[Sequence[str]], *, module: Any, module_fqn: str,
    tp_size: int,
) -> tuple[str, ...]:
    """Validate and normalize user-declared TP-local integer attributes."""
    if attrs is None:
        return ()
    if not isinstance(attrs, (list, tuple)):
        raise ValueError(
            f"{module_fqn}: tp_divide_attrs must be a list of attribute names, "
            f"got {type(attrs).__name__}")
    normalized = []
    seen = set()
    for attr in attrs:
        if not isinstance(attr, str) or not attr or not attr.isidentifier():
            raise ValueError(
                f"{module_fqn}: tp_divide_attrs entries must be non-empty "
                f"Python identifiers, got {attr!r}")
        if attr.startswith("_hp_") or attr in _FORBIDDEN_USER_ATTRS:
            raise ValueError(
                f"{module_fqn}: tp_divide_attrs cannot adjust protected "
                f"attribute {attr!r}")
        if attr in seen:
            raise ValueError(
                f"{module_fqn}: tp_divide_attrs contains duplicate {attr!r}")
        seen.add(attr)
        value = getattr(module, attr, None)
        if not isinstance(value, int) or isinstance(value, bool):
            actual = type(value).__name__ if hasattr(module, attr) else "missing"
            raise ValueError(
                f"{module_fqn}: tp_divide_attrs attribute {attr!r} must "
                f"exist and be a plain int, got {actual}")
        if value <= 0 or value % tp_size != 0:
            raise ValueError(
                f"{module_fqn}: tp_divide_attrs attribute {attr!r}={value} "
                f"must be positive and divisible by tp_size={tp_size}")
        normalized.append(attr)
    return tuple(normalized)


def build_tp_local_attr_plan(
    module: Any, spec: Any, module_fqn: str, tp_size: int,
    mesh_dim_names: Sequence[str],
) -> TpLocalAttrPlan:
    """Build the internal auto/user TP-local attribute plan for one module."""
    auto_attrs = collect_auto_head_attrs(module, spec, mesh_dim_names)
    user_attrs = normalize_tp_divide_attrs(
        spec.tp_divide_attrs, module=module, module_fqn=module_fqn,
        tp_size=tp_size,
    )
    duplicates = sorted(set(auto_attrs) & set(user_attrs))
    if duplicates:
        raise ValueError(
            f"{module_fqn}: tp_divide_attrs redundantly declares D-17 "
            f"automatic head attributes {duplicates}; remove them from YAML")
    return TpLocalAttrPlan(auto_divide=auto_attrs, user_divide=user_attrs)


def _update_user_tp_attrs(
    module: Any, attrs: Sequence[str], tp_size: int, module_fqn: str,
) -> int:
    """Apply validated user TP-local attribute divisions idempotently."""
    if tp_size <= 1 or not attrs:
        return 0
    originals = getattr(module, _TP_LOCAL_ORIGINALS_ATTR, None)
    if originals is None:
        originals = {}
        setattr(module, _TP_LOCAL_ORIGINALS_ATTR, originals)
    count = 0
    for attr in attrs:
        if attr in originals:
            expected = originals[attr] // tp_size
            current = getattr(module, attr)
            if current != expected:
                raise ValueError(
                    f"{module_fqn}: tp_divide_attrs attribute {attr!r} was "
                    f"already adjusted from {originals[attr]} to {current}, "
                    f"which is incompatible with tp_size={tp_size} "
                    f"(expected {expected})")
            continue
        value = getattr(module, attr)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"{module_fqn}: tp_divide_attrs attribute {attr!r} must "
                f"exist and be a plain int, got {type(value).__name__}")
        if value <= 0 or value % tp_size != 0:
            raise ValueError(
                f"{module_fqn}: tp_divide_attrs attribute {attr!r}={value} "
                f"must be positive and divisible by tp_size={tp_size}")
        originals[attr] = value
        setattr(module, attr, value // tp_size)
        logger.info(
            "TP-local attribute adjustment: %s.%s %d -> %d (tp_size=%d, source=user)",
            module_fqn, attr, value, value // tp_size, tp_size,
        )
        count += 1
    return count


def maybe_update_head_counts(module, spec, module_fqn, mesh, mesh_dim_names) -> None:
    """Adjust cached head counts when the spec head-shards the module.

    Called by the applier exactly where the module's forward is guaranteed to
    see local tensors: production (Phase A, all specs) and validate (Phase C,
    local-region specs only).
    """
    tp_size = _tp_degree(mesh, mesh_dim_names)
    attr_plan = getattr(spec, "_tp_local_attr_plan", None)
    if attr_plan is None:
        # Backward-compatible path for manually assembled plans that did not
        # pass through ShardingPlanner.finalize.
        if _is_head_sharded(spec, mesh_dim_names):
            update_module_head_counts(module, tp_size, module_fqn)
        _update_user_tp_attrs(
            module, getattr(spec, "tp_divide_attrs", ()) or (),
            tp_size, module_fqn,
        )
        return
    if attr_plan.auto_divide:
        update_module_head_counts(module, tp_size, module_fqn)
    _update_user_tp_attrs(
        module, attr_plan.user_divide, tp_size, module_fqn,
    )
