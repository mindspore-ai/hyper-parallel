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
"""The spec field table must cover every ``ModuleShardingSpec`` field exactly.

``hyper_parallel/codegen/plan/spec_fields.py`` is the single source of truth
for which spec fields cross the ``codegen_meta.json`` boundary: freeze
(plan -> meta) and rebuild (meta -> plan) both walk it.  These tests keep
the table honest in three ways:

- reflection: every ``ModuleShardingSpec`` dataclass field is classified
  (and the table names no field that does not exist) — an upstream field
  addition without a table entry fails here, not as a silent freeze/rebuild
  gap in production;
- legacy shape: a spec shaped like the committed 129 artifact freezes to
  the exact legacy key order, so regenerated modeling files stay
  byte-stable (only the three intended changes — ``tp_divide_attrs``
  frozen, ``params={}`` kept, ``_deferred_bias_params`` frozen (D-22) —
  may diverge from the pre-table behavior);
- round-trip: the committed artifact's meta rebuilds into a live plan and
  re-freezes to the same param_plan (injections compared per ``match`` —
  their rule order follows ``param_plan``'s sorted JSON order on rebuild,
  which predates the table).
"""

import os
from types import SimpleNamespace
import unittest

from hyper_parallel.codegen.meta import load_codegen_meta
from hyper_parallel.codegen.plan.freeze import (
    freeze_injections,
    freeze_param_plan,
)
from hyper_parallel.codegen.plan.spec_fields import (
    FROZEN_SPEC_KEYS,
    GROUP_EXEMPT,
    GROUP_INJECTIONS,
    GROUP_PARAM_PLAN,
    POST_INIT_SPEC_FIELDS,
    SPEC_FIELDS,
)
from hyper_parallel.codegen.runtime import _rebuild_live_plan_from_meta
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec

_COMMITTED_META = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "examples", "training_demo", "generated", "codegen_meta.json",
)

#: Serializer kinds ``_freeze_spec_field`` dispatches on (freeze side).
_KNOWN_FREEZE_KINDS = frozenset({
    "placement_map", "named_placements", "name_list", "attr_list",
    "scalar", "value_to_dict", "plain", "dict_copy", "int_truthy", "",
})

#: Deserializer kinds ``_rebuild_spec_field`` dispatches on (rebuild side).
_KNOWN_REBUILD_KINDS = frozenset({
    "placement_map", "named_placements", "name_list", "attr_list",
    "bool_default_true", "opt_scalar", "injection_target",
    "dict_copy", "int_truthy", "tuple_empty", "",
})

_S0 = Shard(0)
_SM1 = Shard(1, uneven_shard=True)
_PSUM = Partial("sum")
_R = Replicate()


def _artifact_shaped_spec(with_region_dispatch):
    """A spec shaped like the committed 129 artifact's self_attn boundaries.

    Mirrors the Qwen3-MoE tp2/cp2/ep2 generation: params + in/out placement
    contracts + flags; ``tp_divide_attrs`` absent (that YAML declares none)
    and no injection fields.
    """
    return SimpleNamespace(
        params={"q_proj.weight": {"tp": _S0}, "k_proj.weight": {"tp": _S0}},
        in_src={"hidden_states": {"tp": _SM1}},
        in_dst={"hidden_states": {"tp": _R}},
        out_src={"output": {"tp": _PSUM}},
        out_dst={"output": {"tp": _R}},
        out_names=None,
        tp_divide_attrs=None,
        is_boundary=True,
        region_dispatch=False if with_region_dispatch else None,
        inner_wrapper=None,
        inner_target=None,
        inner_out_src=None,
        local_compute_fn=None,
        _ep_stack=None,
        _ep_size=0,
    )


class TestTableCoversSpecFields(unittest.TestCase):
    """Reflection: the table and the dataclass must stay in lockstep."""

    def test_every_dataclass_field_is_classified(self):
        """A new ModuleShardingSpec field without a table entry fails here."""
        spec_fields = {field.name for field in SPEC_FIELDS}
        dataclass_fields = set(ModuleShardingSpec.__dataclass_fields__)
        self.assertEqual(
            spec_fields, dataclass_fields,
            f"spec_fields.py is out of date with ModuleShardingSpec: "
            f"missing={dataclass_fields - spec_fields} "
            f"unknown={spec_fields - dataclass_fields}",
        )

    def test_non_exempt_fields_carry_unique_meta_keys_and_kinds(self):
        """Frozen entries need a key and a dispatchable kind on both sides."""
        keys = [field.key for field in SPEC_FIELDS if field.group != GROUP_EXEMPT]
        self.assertEqual(len(keys), len(set(keys)), "duplicate meta keys")
        for field in SPEC_FIELDS:
            self.assertIn(field.freeze, _KNOWN_FREEZE_KINDS, field.name)
            self.assertIn(field.rebuild, _KNOWN_REBUILD_KINDS, field.name)
            if field.group == GROUP_EXEMPT:
                self.assertEqual(field.key, "", field.name)
                self.assertEqual(field.freeze, "", field.name)
                self.assertEqual(field.rebuild, "", field.name)
            else:
                self.assertTrue(field.key, field.name)

    def test_exempt_fields_are_deliberate(self):
        """The exempt set is exactly the five documented planner internals."""
        exempt = {field.name for field in SPEC_FIELDS if field.group == GROUP_EXEMPT}
        self.assertEqual(exempt, {
            "_tp_local_attr_plan", "_is_terminal",
            "_needs_cp_attn", "_resolved_inner_wrapper", "_resolved_inner_target",
        })

    def test_post_init_fields_match_dataclass_init_false(self):
        """``post_init`` marks exactly the frozen fields the dataclass cannot
        take as constructor kwargs.

        ``_rebuild_live_plan_from_meta`` constructs ``ModuleShardingSpec(**kwargs)``
        and then sets ``post_init`` fields via ``setattr`` — passing an
        ``init=False`` field as a kwarg is a ``TypeError``, and forgetting to
        mark one silently drops it from the rebuilt spec.  Both directions of
        that drift fail here.
        """
        post_init_names = {field.name for field in POST_INIT_SPEC_FIELDS}
        self.assertEqual(post_init_names, {"_deferred_bias_params"})
        for field in SPEC_FIELDS:
            dataclass_init = ModuleShardingSpec.__dataclass_fields__[field.name].init
            if field.group == GROUP_EXEMPT:
                self.assertFalse(field.post_init, field.name)
            else:
                self.assertEqual(
                    field.post_init, not dataclass_init,
                    f"{field.name}: post_init={field.post_init} but dataclass "
                    f"init={dataclass_init} — the rebuild constructs the spec "
                    "from table kwargs and setattr's post_init fields",
                )


class TestFreezeLegacyShape(unittest.TestCase):
    """Freeze output for artifact-shaped specs must match the legacy bytes."""

    def test_artifact_shaped_entry_key_order(self):
        """Key order (literal-rendering order) matches the legacy freeze."""
        plan = SimpleNamespace(modules={
            "model.layers.0.self_attn": _artifact_shaped_spec(True),
            "model.layers.1.self_attn": _artifact_shaped_spec(False),
        })
        frozen = freeze_param_plan(plan)
        for entry in frozen.values():
            expected = ["params", "in_src", "in_dst", "out_src", "out_dst",
                        "is_boundary"]
            if "region_dispatch" in entry:
                expected.append("region_dispatch")
            self.assertEqual(list(entry.keys()), expected)
        self.assertEqual(
            frozen["model.layers.0.self_attn"],
            {
                "params": {
                    "q_proj.weight": {"tp": "S(0)"},
                    "k_proj.weight": {"tp": "S(0)"},
                },
                "in_src": {"hidden_states": {"tp": "S(-2)"}},
                "in_dst": {"hidden_states": {"tp": "R"}},
                "out_src": {"output": {"tp": "P(sum)"}},
                "out_dst": {"output": {"tp": "R"}},
                "is_boundary": True,
                "region_dispatch": False,
            },
        )

    def test_tp_divide_attrs_freezes_and_rebuilds(self):
        """Fix 2.1: the user's YAML attribute declaration crosses the boundary."""
        spec = SimpleNamespace(
            params={"w.weight": {"tp": _S0}},
            tp_divide_attrs=["num_heads", "num_key_value_heads"],
            is_boundary=True,
        )
        plan = SimpleNamespace(modules={"m.a": spec})
        frozen = freeze_param_plan(plan)
        self.assertEqual(frozen["m.a"]["tp_divide_attrs"],
                         ["num_heads", "num_key_value_heads"])

        meta = SimpleNamespace(
            param_plan={"m.a": {
                "params": {"w.weight": {"tp": "S(0)"}},
                "is_boundary": True,
                "tp_divide_attrs": ["num_heads"],
            }},
            injections=[], special_handlers={}, mesh_dim_names=("tp",),
            tied_pairs=[],
        )
        plan2 = _rebuild_live_plan_from_meta(meta)
        self.assertEqual(plan2.modules["m.a"].tp_divide_attrs, ["num_heads"])

    def test_params_empty_dict_survives_the_boundary(self):
        """Fix 2.3: params={} (shards nothing) must not degrade to None."""
        spec = SimpleNamespace(params={}, is_boundary=True)
        frozen = freeze_param_plan(SimpleNamespace(modules={"m.e": spec}))
        self.assertEqual(frozen["m.e"], {"params": {}, "is_boundary": True})

        meta = SimpleNamespace(
            param_plan={"m.e": {"params": {}, "is_boundary": True}},
            injections=[], special_handlers={}, mesh_dim_names=("tp",),
            tied_pairs=[],
        )
        rebuilt = _rebuild_live_plan_from_meta(meta)
        # Phase A iterates spec.params.items() — None would crash, {} is a
        # legitimate "I/O-stitch-only" boundary.
        self.assertEqual(rebuilt.modules["m.e"].params, {})

    def test_needs_cp_attn_stays_exempt(self):
        """The legacy dead lookup must not resurrect as a frozen key."""
        spec = SimpleNamespace(
            params={}, is_boundary=True,
            _needs_cp_attn=True, needs_cp_attn=True,
        )
        frozen = freeze_param_plan(SimpleNamespace(modules={"m.c": spec}))
        self.assertNotIn("needs_cp_attn", frozen["m.c"])
        self.assertNotIn("_needs_cp_attn", frozen["m.c"])


class TestDeferredBiasRoundTrip(unittest.TestCase):
    """D-22: the deferred-bias param paths cross the boundary as a name list."""

    def test_freeze_writes_truthy_tuple_as_list(self):
        """A planner-computed tuple freezes to a JSON list, after region_dispatch."""
        spec = SimpleNamespace(
            params={"o_proj.weight": {"tp": _S0}},
            is_boundary=True,
            region_dispatch=False,
            _deferred_bias_params=("o_proj.bias",),
        )
        frozen = freeze_param_plan(SimpleNamespace(modules={"m.a": spec}))
        entry = frozen["m.a"]
        self.assertEqual(entry["deferred_bias_params"], ["o_proj.bias"])
        # Table order: the D-22 key lands after region_dispatch (the legacy
        # freeze order plus the new trailing field).
        self.assertEqual(
            list(entry.keys()),
            ["params", "is_boundary", "region_dispatch", "deferred_bias_params"],
        )

    def test_freeze_omits_empty_and_missing_deferred(self):
        """Empty tuple / absent attribute produce NO key (byte-stable metas)."""
        for spec in (
            SimpleNamespace(params={}, is_boundary=True, _deferred_bias_params=()),
            SimpleNamespace(params={}, is_boundary=True),
        ):
            frozen = freeze_param_plan(SimpleNamespace(modules={"m.a": spec}))
            self.assertNotIn("deferred_bias_params", frozen["m.a"])

    def test_rebuild_restores_tuple_and_default_empty(self):
        """Rebuild gives consumers a tuple; absent key restores the () default."""
        meta = SimpleNamespace(
            param_plan={
                "m.a": {
                    "params": {"w.weight": {"tp": "S(0)"}},
                    "is_boundary": True,
                    "deferred_bias_params": ["o_proj.bias"],
                },
                "m.b": {"params": {"w.weight": {"tp": "S(0)"}}, "is_boundary": True},
            },
            injections=[], special_handlers={}, mesh_dim_names=("tp",),
            tied_pairs=[],
        )
        plan = _rebuild_live_plan_from_meta(meta)
        # Consumers (the native suppression/restore pair) iterate the tuple.
        self.assertEqual(plan.modules["m.a"]._deferred_bias_params, ("o_proj.bias",))
        self.assertEqual(plan.modules["m.b"]._deferred_bias_params, ())

    def test_rebuilt_spec_is_constructed_then_setattr(self):
        """The init=False field must not travel as a constructor kwarg."""
        meta = SimpleNamespace(
            param_plan={"m.a": {"params": {}, "is_boundary": True,
                                "deferred_bias_params": ["bias"]}},
            injections=[], special_handlers={}, mesh_dim_names=("tp",),
            tied_pairs=[],
        )
        plan = _rebuild_live_plan_from_meta(meta)
        self.assertEqual(plan.modules["m.a"]._deferred_bias_params, ("bias",))

    def test_full_cycle_preserves_deferred_params(self):
        """freeze(rebuild(meta)) keeps the D-22 key verbatim."""
        meta = SimpleNamespace(
            param_plan={"m.a": {"params": {}, "is_boundary": True,
                                "deferred_bias_params": ["o_proj.bias", "bias"]}},
            injections=[], special_handlers={}, mesh_dim_names=("tp",),
            tied_pairs=[],
        )
        plan = _rebuild_live_plan_from_meta(meta)
        refrozen = freeze_param_plan(plan)
        self.assertEqual(
            refrozen["m.a"]["deferred_bias_params"], ["o_proj.bias", "bias"]
        )


class TestCommittedMetaRoundTrip(unittest.TestCase):
    """The committed 129 artifact must keep rebuilding and re-freezing."""

    @classmethod
    def setUpClass(cls):
        cls.meta = load_codegen_meta(_COMMITTED_META)
        if cls.meta is None:
            raise unittest.SkipTest("committed codegen_meta.json not found")

    def test_meta_keys_are_all_in_the_table(self):
        """Every frozen key the artifact carries is one rebuild knows."""
        for fqn, entry in self.meta.param_plan.items():
            unknown = set(entry) - FROZEN_SPEC_KEYS
            self.assertFalse(unknown, f"{fqn}: keys outside spec_fields table")
        for rule in self.meta.injections:
            unknown = set(rule) - FROZEN_SPEC_KEYS - {"match"}
            self.assertFalse(unknown, f"{rule.get('match')}: keys outside table")

    def test_rebuild_then_freeze_returns_the_same_param_plan(self):
        """freeze(rebuild(meta)) == meta.param_plan — the strongest symmetry."""
        plan = _rebuild_live_plan_from_meta(self.meta)
        self.assertEqual(len(plan.modules), len(self.meta.param_plan))
        refrozen = freeze_param_plan(plan)
        self.assertEqual(refrozen, self.meta.param_plan)

    def test_rebuild_then_freeze_returns_the_same_injection_rules(self):
        """Injection rules match per match key (order is param_plan-driven)."""
        plan = _rebuild_live_plan_from_meta(self.meta)
        refrozen = {rule["match"]: rule for rule in freeze_injections(plan)}
        original = {rule["match"]: rule for rule in self.meta.injections}
        self.assertEqual(refrozen, original)

    def test_rebuilt_specs_carry_no_tp_divide_attrs(self):
        """The committed YAML declares none — rebuild must not invent one."""
        plan = _rebuild_live_plan_from_meta(self.meta)
        for fqn, spec in plan.modules.items():
            self.assertIsNone(spec.tp_divide_attrs, fqn)


if __name__ == "__main__":
    unittest.main()
