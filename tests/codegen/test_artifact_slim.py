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
"""Contract tests for the slimmed artifact literals (A/B/C incremental work).

The emitted ``_HYPER_PARAM_PLAN`` / ``_HYPER_INJECTIONS`` /
``_HYPER_MODULE_OVERRIDES`` literals are pure projections of the frozen
meta fields: R-valued axes stripped, identity-form boundary fields pruned,
same-shape rules grouped.  These tests lock the three contracts that make
the projection safe:

* **Neutrality** — a stripped placement dict resolves to the same placements
  as the original (every reader defaults a missing axis to Replicate).
* **Decision consistency** — a boundary whose forward was pruned to the
  identity form is exactly the boundary whose literal entry loses its
  boundary fields (both decisions come from ``iter_emitted_forms``).
* **Grouping round-trip** — grouped literals expand back to the per-FQN
  shape the runtime helpers already consume.
"""

from __future__ import annotations

from hyper_parallel.codegen.emit.modeling import (
    build_boundary_manifest,
    group_injection_rules,
    inject_param_plan_literals,
    render_python_literal,
)
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen.emit.replacement import group_override_records
from hyper_parallel.codegen.plan.freeze import parse_named_placement
from hyper_parallel.codegen.plan.slim import (
    slim_param_plan_for_emission,
    strip_r_axes,
)
from hyper_parallel.codegen.runtime import (
    _injection_spec,
    _is_boundary_entry,
    hyper_expand_injections,
)
from hyper_parallel.distributed.recipe_spec import resolve_placements
from tests.common.mark_utils import arg_mark

SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward gets rewritten (generic form)."""

    def forward(self, x):
        return x * 2


class Norm(nn.Module):
    """A boundary class whose every transition is identity (pruned)."""

    def forward(self, hidden_states):
        return hidden_states
'''


def _generic_entry() -> dict:
    """A boundary with a cp transition the TP lowerer cannot lower."""
    return {
        "is_boundary": True,
        "in_src": {"x": {"tp": "S(1)", "ep": "R", "cp": "S(1)"}},
        "in_dst": {"x": {"tp": "R", "ep": "R", "cp": "R"}},
        "out_src": {"output": {"tp": "S(-1)", "ep": "R", "cp": "S(1)"}},
        "out_dst": {"output": {"tp": "R", "ep": "R", "cp": "S(1)"}},
        "params": {"weight": {"tp": "S(0)", "ep": "R", "cp": "R"}},
    }


def _identity_entry() -> dict:
    """A boundary whose in/out placements never change on any axis."""
    return {
        "is_boundary": True,
        "in_src": {"hidden_states": {"tp": "S(1)", "ep": "R", "cp": "R"}},
        "in_dst": {"hidden_states": {"tp": "S(1)", "ep": "R", "cp": "R"}},
        "out_src": {"output": {"tp": "S(1)", "ep": "R", "cp": "R"}},
        "out_dst": {"output": {"tp": "S(1)", "ep": "R", "cp": "R"}},
        "params": {"weight": {"tp": "R", "ep": "R", "cp": "R"}},
    }


def _frozen_plan() -> dict:
    """A frozen-plan dict with one generic and one identity boundary class."""
    return {
        "param_plan": {
            "blocks.alpha": _generic_entry(),
            "blocks.norm": _identity_entry(),
        },
        "injections": [],
        "mesh_dim_names": ("cp", "tp"),
        "boundary_classes": {"blocks.alpha": "Alpha", "blocks.norm": "Norm"},
        "source": {"module_name": ""},
    }


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_strip_r_axes_is_placement_neutral():
    """Stripping ``"R"`` axis keys must not change any resolved placement.

    Feature: codegen-lowering
    Description: A frozen entry with mixed R / non-R axes is slimmed; each
        per-name dict goes through the real reader chain —
        ``parse_named_placement`` (leaf ``"R"`` -> ``Replicate()``, what
        ``hyper_shard_params`` / ``_build_rewrap_plan`` run first) then
        ``resolve_placements`` (missing axis -> ``Replicate()``).
    Expectation: The reader chain returns identical placement tuples for the
        original and the stripped dict, on every side and every name.
    """
    def resolved(named: dict) -> tuple:
        return tuple(resolve_placements(parse_named_placement(named), axes))

    axes = ("cp", "tp")
    for entry in (_generic_entry(), _identity_entry()):
        slimmed = slim_param_plan_for_emission({entry and "m": entry}, ())
        slim_entry = slimmed["m"]
        for field in ("params", "in_src", "in_dst", "out_src", "out_dst"):
            for name, named in entry[field].items():
                assert resolved(slim_entry[field][name]) == resolved(named), (
                    field, name,
                )


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_identity_fields_pruned_and_generic_kept():
    """Identity-form boundaries drop boundary fields; generic ones keep them.

    Feature: codegen-lowering
    Description: A plan with one generic (cp transition) and one identity
        boundary is slimmed with the identity FQN collected from the emitter's
        own decision path.
    Expectation: The identity entry keeps only ``params`` (R-stripped) and
        reads as a non-boundary to ``_is_boundary_entry``; the generic entry
        keeps all four placement fields (R-stripped) and ``is_boundary``.
    """
    plan = _frozen_plan()
    slimmed = slim_param_plan_for_emission(
        plan["param_plan"], identity_fqns={"blocks.norm"}
    )

    norm = slimmed["blocks.norm"]
    assert set(norm) == {"params"}
    assert norm["params"] == {"weight": {}}
    assert _is_boundary_entry(norm) is False

    alpha = slimmed["blocks.alpha"]
    assert alpha["is_boundary"] is True
    assert alpha["in_src"] == {"x": {"tp": "S(1)", "cp": "S(1)"}}
    assert alpha["in_dst"] == {"x": {}}
    assert alpha["params"] == {"weight": {"tp": "S(0)"}}
    assert _is_boundary_entry(alpha) is True


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_emitted_literal_prunes_identity_boundary_fields():
    """The emitted literal prunes exactly the pruned-forward boundaries.

    Feature: codegen-lowering
    Description: A full pipeline text (``lower_forward_boundaries`` output)
        plus frozen plan is passed through ``inject_param_plan_literals``.
    Expectation: ``blocks.norm`` (forward untouched — no ``_forward_impl``)
        emits a params-only entry while ``blocks.alpha`` (rewritten) keeps
        its boundary fields; the pruning decision can never disagree with
        the forward rewrite because both come from ``iter_emitted_forms``.
    """
    plan = _frozen_plan()
    text = lower_forward_boundaries(
        SOURCE_TEXT, plan, boundary_classes=plan["boundary_classes"]
    )
    assert "_forward_impl" in text  # Alpha was rewritten
    # Norm's forward was pruned: exactly one rewritten impl (Alpha's) —
    # its def line, the delegate call, and the comment quoting it.
    assert text.count("def _forward_impl") == 1

    literal = inject_param_plan_literals(text, plan)
    assert "'blocks.norm': {'params': {'weight': {}}}" in literal
    assert "'blocks.alpha'" in literal
    assert "'is_boundary': True" in literal
    # The norm entry contributes no placement fields at all.
    norm_block = literal.split("'blocks.norm':")[1].split("},")[0]
    assert "in_src" not in norm_block
    assert "is_boundary" not in norm_block


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_group_override_records_merges_same_spec():
    """Per-target replacement records collapse per spec with a union FQN set.

    Feature: codegen-lowering
    Description: Three records — two from one spec (same factory / type /
        match) and one from another — are grouped.
    Expectation: Two grouped records; the merged one carries the sorted FQN
        union and no ``fqn`` / ``match`` keys; the other stays separate.
    """
    records = [
        {
            "match": ["*.input_layernorm"],
            "fqn": "model.layers.1.input_layernorm",
            "fqns": ["model.layers.1.input_layernorm"],
            "module_type": "m.RMSNorm",
            "factory": "a.replace_rms",
            "exact_type": False,
        },
        {
            "match": ["*.input_layernorm"],
            "fqn": "model.layers.0.input_layernorm",
            "fqns": ["model.layers.0.input_layernorm"],
            "module_type": "m.RMSNorm",
            "factory": "a.replace_rms",
            "exact_type": False,
        },
        {
            "match": ["*.self_attn"],
            "fqn": "model.layers.0.self_attn",
            "fqns": ["model.layers.0.self_attn"],
            "module_type": "m.Attn",
            "factory": "a.replace_attn",
            "exact_type": True,
        },
    ]
    grouped = group_override_records(records)
    assert len(grouped) == 2
    rms = next(r for r in grouped if r["factory"] == "a.replace_rms")
    assert rms["fqns"] == [
        "model.layers.0.input_layernorm",
        "model.layers.1.input_layernorm",
    ]
    assert "fqn" not in rms and "match" not in rms
    attn = next(r for r in grouped if r["factory"] == "a.replace_attn")
    assert attn["exact_type"] is True
    assert attn["fqns"] == ["model.layers.0.self_attn"]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_grouped_injections_expand_roundtrip():
    """Grouped injection literals expand to the per-FQN shape helpers consume.

    Feature: codegen-lowering
    Description: Three frozen rules — two with identical payloads (per-layer
        CP wrappers) and one distinct — are grouped and re-expanded.
    Expectation: Grouping yields two records with a list ``match``;
        ``hyper_expand_injections`` restores exactly the original rules;
        ``_injection_spec`` resolves a per-FQN rule from the expanded list.
    """
    rules = [
        {"inner_wrapper": {"_target_": "m.cp_wrapper"}, "inner_target": "self",
         "match": "blocks.a1"},
        {"inner_wrapper": {"_target_": "m.cp_wrapper"}, "inner_target": "self",
         "match": "blocks.a2"},
        {"local_compute_fn": {"_target_": "m.ep_fn"}, "ep_size": 2,
         "match": "blocks.b"},
    ]
    grouped = group_injection_rules(rules)
    assert len(grouped) == 2
    wrapper = next(r for r in grouped if "inner_wrapper" in r)
    assert wrapper["match"] == ["blocks.a1", "blocks.a2"]

    expanded = hyper_expand_injections(grouped)
    assert sorted(expanded, key=lambda r: r["match"]) == sorted(
        rules, key=lambda r: r["match"]
    )
    spec = _injection_spec(expanded, {}, "blocks.a2")
    assert spec is not None and spec["inner_target"] == "self"
    assert _injection_spec(expanded, {}, "blocks.missing") is None

    # Scalar ``match`` passes through untouched (legacy / direct callers).
    assert hyper_expand_injections(rules) == rules
    assert hyper_expand_injections(None) == []


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_region_forward_carries_declarative_notes():
    """The region template annotates its redistribute steps like generic/TP.

    Feature: codegen-lowering
    Description: A boundary with a ``local_compute_fn`` injection and a
        TP-lowerable input transition (``tp S(1) -> R``) is lowered; a second
        identity-transition region is lowered for contrast.
    Expectation: The first forward carries the per-transition note
        (``x: tp S(1) -> R ==> all_gather(dim=1)``) before its redistribute;
        the identity region emits no transition notes (no noise).
    """
    def region_plan(tp_transition: bool) -> dict:
        entry = {
            "is_boundary": True,
            "in_src": {"x": {"tp": "S(1)" if tp_transition else "R"}},
            "in_dst": {"x": {"tp": "R"}},
            "out_src": {"output": {"tp": "R"}},
            "out_dst": {"output": {"tp": "R"}},
        }
        plan = {
            "param_plan": {"blocks.alpha": entry},
            "injections": [
                {
                    "match": "blocks.alpha",
                    "local_compute_fn": {"_target_": "m.ep_compute"},
                }
            ],
            "mesh_dim_names": ("tp",),
        }
        return plan

    classes = {"blocks.alpha": "Alpha"}

    annotated = lower_forward_boundaries(
        SOURCE_TEXT, region_plan(tp_transition=True), boundary_classes=classes
    )
    assert "x: tp S(1) -> R ==> all_gather(dim=1)" in annotated
    assert annotated.index("x: tp S(1) -> R") < annotated.index(
        "self._hyper_boundary.redistribute_inputs"
    )
    assert "m.ep_compute" in annotated  # compute fn named in the comment
    assert "args = tuple(hyper_to_local_if_dtensor(arg) for arg in args)" in annotated
    assert "key: hyper_to_local_if_dtensor(value)" in annotated

    quiet = lower_forward_boundaries(
        SOURCE_TEXT, region_plan(tp_transition=False), boundary_classes=classes
    )
    assert "==> all_gather" not in quiet
    assert "==> dtensor redistribute" not in quiet


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_render_collapses_empty_container_values():
    """A dict whose values are scalars or empty containers renders on one line.

    Feature: codegen-lowering
    Description: ``render_python_literal`` renders an R-stripped params dict
        whose values are all empty placement dicts.
    Expectation: Single-line output — the slimmed identity entries keep a
        one-entry-per-line shape in the artifact.
    """
    rendered = render_python_literal({"params": {"weight": {}}})
    assert rendered == "{'params': {'weight': {}}}"
    # Non-empty containers still nest (unchanged legacy behavior).
    assert "\n" in render_python_literal({"params": {"weight": {"tp": "S(0)"}}})


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_emitted_parallelize_expands_grouped_injections():
    """The generated entry expands the grouped literal before the helpers run.

    Feature: codegen-lowering
    Description: ``inject_hyper_parallelize`` output is inspected for the
        expansion call and its result being forwarded to the three consumers.
    Expectation: ``hyper_expand_injections(_HYPER_INJECTIONS)`` runs once;
        ``hyper_install_boundaries`` / ``hyper_bind_compute`` /
        ``hyper_apply_inner_wrapper`` receive the expanded local, not the
        grouped literal.
    """
    from hyper_parallel.codegen.emit.modeling import inject_hyper_parallelize

    body = inject_hyper_parallelize("", _frozen_plan())
    assert "injections = hyper_expand_injections(_HYPER_INJECTIONS)" in body
    assert "injections=_HYPER_INJECTIONS" not in body
    assert "model, _HYPER_INJECTIONS," not in body
    assert body.count("model, injections,") == 2


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_boundary_manifest_exposes_production_ops_and_regions():
    """The generated artifact carries a reader-facing communication manifest.

    Feature: codegen-lowering
    Description: A frozen plan containing an imported TP boundary, a CP
        inner-wrapper boundary, and an EP local-compute region is projected to
        ``_CODEGEN_BOUNDARY_MANIFEST``.
    Expectation: The manifest exposes concrete production collectives and the
        CP/EP semantic flow without replacing the machine-readable param plan.
    """
    plan = {
        "param_plan": {
            "lm_head": {
                "is_boundary": True,
                "in_src": {"hidden_states": {"tp": "S(1)"}},
                "in_dst": {"hidden_states": {"tp": "R"}},
                "out_src": {"output": {"tp": "S(-1)"}},
                "out_dst": {"output": {"tp": "S(-1)"}},
            },
            "model.layers.0.self_attn": {
                "is_boundary": True,
                "in_src": {"hidden_states": {"tp": "S(1)"}},
                "in_dst": {"hidden_states": {"tp": "R"}},
                "out_src": {"output": {"tp": "P(sum)"}},
                "out_dst": {"output": {"tp": "S(1)"}},
            },
            "model.layers.0.mlp": {
                "is_boundary": True,
                "in_src": {"x_BLD": {"tp": "S(1)"}},
                "in_dst": {"x_BLD": {"tp": "S(1)"}},
                "out_src": {"output": {"tp": "S(1)"}},
                "out_dst": {"output": {"tp": "S(1)"}},
            },
        },
        "mesh_dim_names": ("tp",),
        "boundary_classes": {
            "lm_head": "Linear",
            "model.layers.0.self_attn": "GQAAttention",
            "model.layers.0.mlp": "Qwen3MoeSparseMoeBlock",
        },
        "injections": [
            {
                "match": "model.layers.0.self_attn",
                "inner_target": "self",
                "inner_wrapper": {"_target_": "m.cp_wrapper"},
            },
            {
                "match": "model.layers.0.mlp",
                "ep_size": 2,
                "local_compute_fn": {"_target_": "m.ep_compute"},
            },
        ],
    }

    manifest = build_boundary_manifest(plan)

    assert manifest["lm_head"]["production_ops"] == ["all_gather"]
    attn = manifest["model.layers.0.self_attn"]
    assert attn["form"] == "cp_inner_wrapper"
    assert attn["production_ops"] == ["all_gather", "reduce_scatter"]
    assert attn["injection"]["flow"] == "attention K/V all-gather and CP-aware causal mask"
    mlp = manifest["model.layers.0.mlp"]
    assert mlp["form"] == "ep_region"
    assert mlp["injection"]["flow"] == (
        "router -> all_to_all dispatch -> local experts -> all_to_all combine"
    )
