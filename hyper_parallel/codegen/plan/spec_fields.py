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
"""Single source of truth: which ``ModuleShardingSpec`` fields cross the meta boundary.

``freeze_param_plan`` / ``freeze_injections`` (plan -> meta, in
``codegen/plan/freeze.py``) and ``_rebuild_live_plan_from_meta``
(meta -> plan, in ``codegen/runtime.py``) both walk this table, so a spec
field can no longer be frozen-but-never-rebuilt (or the reverse): adding a
field to ``ModuleShardingSpec`` without classifying it here fails the
reflection test in ``tests/codegen/test_spec_field_table.py``.

The table is pure data (no torch, no imports from freeze/runtime — both
consume it, so an import here would be circular).  Each entry names the
spec attribute, the meta key it serializes to, the meta section it belongs
to, and the serializer/deserializer *kind* the two sides dispatch on.  The
kind also fixes the write condition — ``is not None`` (an explicit empty
dict/list crosses the boundary; this is the params={} case).

Divergences this table replaces (found 2026-09-22, all closed here):
- ``tp_divide_attrs`` was never frozen, so a rebuilt spec silently lost the
  user's YAML declaration and ``maybe_update_head_counts`` fell back to the
  head-sharded heuristic only;
- freeze wrote ``params`` under a truthy check, dropping an explicit empty
  dict and leaving the rebuilt spec with ``params=None`` where Phase A
  (``_shard_planned_parameters``) expects a mapping;
- freeze read ``getattr(spec, "needs_cp_attn")`` — the real field is
  ``_needs_cp_attn``, so the lookup was dead code; the field is now
  explicitly EXEMPT (fixing the name would add a key to every existing
  artifact's meta and shift bytes; see the table comment).
"""
from __future__ import annotations

from dataclasses import dataclass

#: Frozen into ``meta.param_plan[fqn]`` entries (freeze_param_plan).
GROUP_PARAM_PLAN = "param_plan"
#: Frozen into ``meta.injections`` rules (freeze_injections).
GROUP_INJECTIONS = "injections"
#: Never crosses the meta boundary (planner-internal / apply-time only).
GROUP_EXEMPT = "exempt"


@dataclass(frozen=True)
class SpecField:
    """One ``ModuleShardingSpec`` field's freeze/rebuild contract.

    Attributes:
        name: Attribute on ``ModuleShardingSpec`` (``"_ep_stack"``).
        key: Key in the frozen meta (``"ep_stack"``); ``""`` for exempt.
        group: Meta section — one of the ``GROUP_*`` constants.
        freeze: Freeze-side serializer kind; ``""`` for exempt fields.
        rebuild: Rebuild-side deserializer kind; ``""`` for exempt fields.
        post_init: The dataclass field is ``init=False`` (planner-internal
            output like ``_deferred_bias_params``) — the rebuild sets it
            via ``setattr`` after construction instead of a constructor
            kwarg.

    Serializer kinds (freeze side, dispatched in freeze.py):

    - ``placement_map``: per-param -> per-axis placements, each via
      ``placement_to_string``.  Written when the value ``is not None`` —
      an explicit ``params={}`` ("shards nothing") survives the boundary.
    - ``named_placements``: ``named_placement_to_dict``.  Written when the
      value ``is not None``.
    - ``name_list`` / ``attr_list``: ``list(value)`` copy.  ``name_list``
      is written only when truthy (legacy out_names behavior);
      ``attr_list`` when ``is not None`` (an explicit ``[]`` clears an
      inherited glob declaration and must not be lost).
    - ``scalar``: value passes through unchanged.  Written when the value
      ``is not None``.
    - ``value_to_dict``: ``_value_to_dict`` (Target/Placement/callable ->
      JSON-safe).  Written when the value ``is not None``.
    - ``plain``: value passes through unchanged (string fields).
    - ``dict_copy``: ``dict(value)``.  Written when truthy.
    - ``int_truthy``: value passes through.  Written when truthy (0 is the
      "inactive" default).

    Deserializer kinds (rebuild side, dispatched in runtime.py):
    ``placement_map`` / ``named_placements`` / ``name_list`` / ``attr_list``
    mirror their freeze kinds; ``bool_default_true`` restores ``is_boundary``
    with a ``True`` default for pre-flag metas; ``opt_scalar`` reads
    ``entry.get(key)``; ``injection_target`` re-resolves a serialized
    Target through ``_injection_target``.
    """

    name: str
    key: str
    group: str
    freeze: str = ""
    rebuild: str = ""
    post_init: bool = False


#: Every ``ModuleShardingSpec`` field, classified.  Order is load-bearing:
#: it is the insertion order of keys in frozen entries, so it must match
#: the legacy hand-written freeze order (params -> in/out placements ->
#: out_names -> tp_divide_attrs -> flags; injections keep the legacy
#: inner_* -> ep_* order) to keep regenerated artifacts byte-stable.
SPEC_FIELDS: tuple[SpecField, ...] = (
    # ── Frozen into meta.param_plan[fqn] ──
    SpecField("params", "params", GROUP_PARAM_PLAN,
              freeze="placement_map", rebuild="placement_map"),
    SpecField("in_src", "in_src", GROUP_PARAM_PLAN,
              freeze="named_placements", rebuild="named_placements"),
    SpecField("in_dst", "in_dst", GROUP_PARAM_PLAN,
              freeze="named_placements", rebuild="named_placements"),
    SpecField("out_src", "out_src", GROUP_PARAM_PLAN,
              freeze="named_placements", rebuild="named_placements"),
    SpecField("out_dst", "out_dst", GROUP_PARAM_PLAN,
              freeze="named_placements", rebuild="named_placements"),
    # out_names: the declared output *order* is separate from the out_src
    #   dict's serialized key order — the runtime resolves tuple indices
    #   against the spec's declared order (declared_out_names prefers
    #   entry["out_names"]), not a sorted order that could swap e.g.
    #   hidden/aux and redistribute the wrong output.
    SpecField("out_names", "out_names", GROUP_PARAM_PLAN,
              freeze="name_list", rebuild="name_list"),
    SpecField("tp_divide_attrs", "tp_divide_attrs", GROUP_PARAM_PLAN,
              freeze="attr_list", rebuild="attr_list"),
    SpecField("is_boundary", "is_boundary", GROUP_PARAM_PLAN,
              freeze="scalar", rebuild="bool_default_true"),
    SpecField("region_dispatch", "region_dispatch", GROUP_PARAM_PLAN,
              freeze="scalar", rebuild="opt_scalar"),
    # _deferred_bias_params: D-22 (rowwise bias defer) — the planner-computed
    #   bias param paths (e.g. ("o_proj.bias",)) whose addition is deferred
    #   until AFTER the boundary exit TP reduction (Megatron
    #   RowParallelLinear semantics).  Frozen so the emitted forward can
    #   call ``self._hyper_deferred_bias(outputs)`` at the exit and the
    #   install path can run the native suppression/restore pair
    #   (``_install_bias_suppression`` / ``_maybe_add_deferred_biases``).
    #   ``init=False`` on the dataclass — rebuilt via setattr (post_init).
    SpecField("_deferred_bias_params", "deferred_bias_params", GROUP_PARAM_PLAN,
              freeze="name_list", rebuild="tuple_empty", post_init=True),
    # ── Frozen into meta.injections rules ──
    SpecField("inner_wrapper", "inner_wrapper", GROUP_INJECTIONS,
              freeze="value_to_dict", rebuild="injection_target"),
    SpecField("inner_target", "inner_target", GROUP_INJECTIONS,
              freeze="plain", rebuild="opt_scalar"),
    SpecField("inner_out_src", "inner_out_src", GROUP_INJECTIONS,
              freeze="value_to_dict", rebuild="injection_target"),
    SpecField("local_compute_fn", "local_compute_fn", GROUP_INJECTIONS,
              freeze="value_to_dict", rebuild="injection_target"),
    SpecField("_ep_stack", "ep_stack", GROUP_INJECTIONS,
              freeze="dict_copy", rebuild="dict_copy"),
    SpecField("_ep_size", "ep_size", GROUP_INJECTIONS,
              freeze="int_truthy", rebuild="int_truthy"),
    # ── Exempt: never crosses the meta boundary ──
    # _tp_local_attr_plan: planner-derived from tp_divide_attrs + module
    #   structure (TpLocalAttrPlan, not JSON-safe).  A rebuilt spec leaves
    #   it None and maybe_update_head_counts takes its backward-compatible
    #   path: auto attrs via the _is_head_sharded heuristic, user attrs via
    #   the rebuilt tp_divide_attrs — both sources survive.
    SpecField("_tp_local_attr_plan", "", GROUP_EXEMPT),
    # _is_terminal: validate-mode propagation marker, not production state.
    SpecField("_is_terminal", "", GROUP_EXEMPT),
    # _needs_cp_attn: apply-time preflight metadata (attention boundary
    #   under cp>1 without inner_wrapper fails fast in apply_sharding_plan
    #   — a path codegen's rebuild does not take).  The legacy freeze loop
    #   read "needs_cp_attn" (no underscore), which never exists, so no
    #   artifact meta carries the key; fixing the name would shift bytes
    #   for every existing artifact, so the field stays exempt.
    SpecField("_needs_cp_attn", "", GROUP_EXEMPT),
    # _resolved_inner_wrapper / _resolved_inner_target: written back by the
    #   applier for introspection; meaningless at generation time.
    SpecField("_resolved_inner_wrapper", "", GROUP_EXEMPT),
    SpecField("_resolved_inner_target", "", GROUP_EXEMPT),
)


def fields_for_group(group: str) -> tuple[SpecField, ...]:
    """Return the table entries for one ``GROUP_*`` section, in table order."""
    return tuple(field for field in SPEC_FIELDS if field.group == group)


#: The param_plan section (``meta.param_plan[fqn]`` entry keys).
PARAM_PLAN_SPEC_FIELDS: tuple[SpecField, ...] = fields_for_group(GROUP_PARAM_PLAN)

#: The injections section (``meta.injections`` rule keys, minus "match").
INJECTION_SPEC_FIELDS: tuple[SpecField, ...] = fields_for_group(GROUP_INJECTIONS)

#: ``init=False`` dataclass fields in the frozen groups — the rebuild sets
#: them via ``setattr`` after constructing the spec.
POST_INIT_SPEC_FIELDS: tuple[SpecField, ...] = tuple(
    field for field in SPEC_FIELDS if field.post_init
)

#: Every meta key that may appear in a frozen entry/rule (both groups).
FROZEN_SPEC_KEYS: frozenset[str] = frozenset(
    field.key for field in SPEC_FIELDS if field.key
)


__all__ = [
    "FROZEN_SPEC_KEYS",
    "GROUP_EXEMPT",
    "GROUP_INJECTIONS",
    "GROUP_PARAM_PLAN",
    "INJECTION_SPEC_FIELDS",
    "PARAM_PLAN_SPEC_FIELDS",
    "POST_INIT_SPEC_FIELDS",
    "SPEC_FIELDS",
    "SpecField",
    "fields_for_group",
]
