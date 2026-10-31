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
"""Contract tests for deterministic ``lower_forward_boundaries`` rewrites.

The model file produced during generation once differed from the file
re-emitted from its own persisted
``meta.param_plan``, because ``meta.param_plan`` is written with
``json.dump(..., sort_keys=True)`` (which reorders the keys alphabetically) and
``lower_forward_boundaries`` iterated the boundary classes in ``param_plan``
insertion order.  Reloading the meta therefore reversed the class order, which
reordered the appended ``_HYPER_BOUNDARY_<CLASS>`` constants and changed the
emitted bytes, so ``verify_drift_intact`` failed consistently.

The lowerer now iterates boundary classes in ``sorted(groups)`` order, making
the byte stream a pure function of *which* classes are boundaries, independent
of how the ``param_plan`` reached us.  This test locks that contract: two plans
carrying the same entries but with opposite key insertion orders must lower to
the identical byte string.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from hyper_parallel.codegen.emit.modeling import _verify_recorded_patch_materialization
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen.emit.recorded import materialize_recorded_forwards
from hyper_parallel.codegen.inline.ir import (
    ForwardExtractPatch,
    ImportPatch,
    InlinePatchSet,
)
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.pipeline import _materialize_imported_forward_hosts
from hyper_parallel.codegen.inline.recognition import inner_wrapper_target
from hyper_parallel.codegen.inline.templates import cp_dispatch_shell_body
from tests.common.mark_utils import arg_mark

SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward gets rewritten."""

    def forward(self, x):
        return x * 2


class Beta(nn.Module):
    """A second boundary class whose forward gets rewritten."""

    def forward(self, x):
        return x + 1
'''


@pytest.mark.parametrize(
    ("factory", "helper"),
    (
        (
            "hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel."
            "kimi_delta_attention_ulysses_cp_wrapper",
            "kimi_delta_attention_cp_forward",
        ),
        (
            "hyper_parallel.models.qwen3_5.adapter.distributed.context_parallel."
            "qwen3_5_gdn_ulysses_cp_wrapper",
            "qwen3_5_gdn_cp_forward",
        ),
        (
            "hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel."
            "mla_cp_wrapper",
            "mla_cp_forward",
        ),
    ),
)
@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_delegating_cp_wrapper_exposes_source_visible_helper(factory, helper):
    """Delegating wrappers retain a module-level helper for code generation."""
    target = inner_wrapper_target(factory)

    assert target is not None
    assert target.helper_name == helper

VARIADIC_SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward takes variadic arguments."""

    def forward(self, x, *args, **kwargs):
        return x * 2
'''

KWARGS_TUPLE_SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward keeps HF-style **kwargs and tuple output."""

    def forward(self, x, y=None, **kwargs):
        hidden = x if y is None else x + y
        return hidden, kwargs.get("aux")
'''


def _boundary(fqn: str, class_name: str) -> tuple[str, dict]:
    """A minimal but real-shaped frozen boundary entry for one class.

    The input side carries a cp transition (``S(1) -> R``) that is not
    TP-lowerable, so with the plan's active axes (``("cp", "tp")``) the
    entry classifies as the generic redistribute form.
    """
    entry = {
        "is_boundary": True,
        "in_src": {"x": {"tp": "S(1)", "ep": "R", "cp": "S(1)"}},
        "in_dst": {"x": {"tp": "R", "ep": "R", "cp": "R"}},
        "out_src": {"output": {"tp": "S(-1)", "ep": "R", "cp": "S(1)"}},
        "out_dst": {"output": {"tp": "R", "ep": "R", "cp": "S(1)"}},
        "params": {"weight": {"tp": "S(0)", "ep": "R", "cp": "R"}},
    }
    return [fqn], {fqn: entry}, {fqn: class_name}


def _recorded_alpha_forward(self, x):
    del self
    return x * 2


def _plan(*, alpha_first: bool) -> dict:
    """A frozen-plan dict whose two boundary classes appear in one of two orders."""
    _, alpha_entry, alpha_map = _boundary("blocks.alpha", "Alpha")
    _, beta_entry, beta_map = _boundary("blocks.beta", "Beta")
    if alpha_first:
        plan = {**alpha_entry, **beta_entry}
        boundary_classes = {**alpha_map, **beta_map}
    else:
        plan = {**beta_entry, **alpha_entry}
        boundary_classes = {**beta_map, **alpha_map}
    return (
        {"param_plan": plan, "injections": [], "mesh_dim_names": ("cp", "tp")},
        boundary_classes,
    )


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_lowering_bytes_are_order_independent():
    """Reversing the ``param_plan`` key order must not change the emitted bytes.

    This is the property the sort fix restores: the two plans carry identical
    entry content, so they must lower to the identical byte string.  Before the
    fix the ``_HYPER_BOUNDARY_Alpha``/``_HYPER_BOUNDARY_Beta`` constants were
    appended in insertion order, so ``alpha_first`` and the reversed plan
    produced different output bytes.

    Feature: codegen-lowering
    Description: Two plans with identical entries but reversed key insertion
        order are lowered through ``lower_forward_boundaries``.
    Expectation: The emitted byte strings are identical (order-independent).
    """
    plan_a, classes_a = _plan(alpha_first=True)
    plan_b, classes_b = _plan(alpha_first=False)

    text_a = lower_forward_boundaries(SOURCE_TEXT, plan_a, boundary_classes=classes_a)
    text_b = lower_forward_boundaries(SOURCE_TEXT, plan_b, boundary_classes=classes_b)

    assert text_a == text_b


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_rewritten_forwards_bind_instance_boundary_plans():
    """No per-boundary global constants are emitted; forwards use the bound plan.

    Explicitly checks that a real rewrite happened, so the byte-identity test
    above cannot pass vacuously (e.g. on a plan that never rewrote a forward).

    Feature: codegen-lowering
    Description: A plan with two boundary classes is lowered and the emitted
        text is inspected for the absence of per-boundary global constants
        and forward-time globals, and the presence of instance-bound calls.
    Expectation: No ``_HYPER_BOUNDARY_`` constants, no ``mesh_context``;
        ``_forward_impl`` and ``self._hp_boundary.redistribute_*`` present.
    """
    plan, classes = _plan(alpha_first=True)
    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    # The rewritten forwards reference the instance-bound compiled plan that
    # ``hp_install_boundaries`` installs — no per-boundary global constants
    # and no forward-time globals.  The artifact carries no plan literals at
    # all: the runtime reads the frozen plan from ``codegen_meta.json``.
    assert "_HYPER_BOUNDARY_" not in text
    assert "mesh_context" not in text
    assert "_forward_impl" in text
    assert text.count("self._hp_boundary.redistribute_inputs") >= 2
    assert text.count("self._hp_boundary.redistribute_outputs") >= 2


def _tp_collective_entry() -> dict:
    """A TP-lowerable boundary: in S(1)->R, out P(sum)->S(1) on the tp axis."""
    return {
        "is_boundary": True,
        "in_src": {"x": {"tp": "S(1)"}},
        "in_dst": {"x": {"tp": "R"}},
        "out_src": {"output": {"tp": "P(sum)"}},
        "out_dst": {"output": {"tp": "S(1)"}},
    }


def _identity_entry() -> dict:
    """An all-identity boundary: no transition on any axis, nothing to lower."""
    return {
        "is_boundary": True,
        "in_src": {"x": {"tp": "R"}},
        "in_dst": {"x": {"tp": "R"}},
        "out_src": {"output": {"tp": "R"}},
        "out_dst": {"output": {"tp": "R"}},
    }


def _single_plan(entry: dict) -> tuple[dict, dict]:
    """A frozen plan whose only boundary is ``blocks.alpha`` (class Alpha)."""
    plan = {"param_plan": {"blocks.alpha": entry}, "injections": []}
    return plan, {"blocks.alpha": "Alpha"}


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_local_region_lowering_requires_builder_recording():
    """A local-compute boundary cannot fall back to an equivalent source rewrite.

    Feature: codegen-lowering
    Description: A boundary whose frozen injection declares ``local_compute_fn``
        reaches ``lower_forward_boundaries`` without the recorder's
        ``skip_source_lowering`` marker.
    Expectation: Generation fails instead of cloning native Phase C and changing
        the numerical execution order.
    """
    plan, classes = _single_plan(_identity_entry())
    plan["injections"] = [
        {"match": "blocks.alpha", "local_compute_fn": {"_target_": "some.factory"}},
    ]

    with pytest.raises(RuntimeError, match="builder recording"):
        lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_runtime_only_local_compute_does_not_poison_shared_source_class():
    """A skipped local-compute injection must not affect another class instance."""
    entry = _identity_entry()
    skipped = {**_identity_entry(), "skip_source_lowering": True}
    plan = {
        "param_plan": {"blocks.alpha": entry, "blocks.runtime": skipped},
        "injections": [{
            "match": "blocks.runtime",
            "local_compute_fn": {"_target_": "some.factory"},
        }],
        "mesh_dim_names": ("tp",),
    }
    classes = {"blocks.alpha": "Alpha", "blocks.runtime": "Alpha"}

    assert lower_forward_boundaries(
        SOURCE_TEXT, plan, boundary_classes=classes
    ) == SOURCE_TEXT


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_recorded_local_compute_rewrites_same_class_with_ep_toggle():
    """A recorded EP patch keeps the original class and forward implementation."""
    alpha_type = type("Alpha", (), {"forward": _recorded_alpha_forward})
    record = SimpleNamespace(
        module_fqn="blocks.alpha",
        module_type=alpha_type,
        original_forward=alpha_type.forward,
        factory="demo.ep.factory",
    )
    meta = SimpleNamespace(
        param_plan={"blocks.alpha": _identity_entry()},
        boundary_classes={"blocks.alpha": "Alpha"},
        external_state_classes=[],
        mesh_dim_names=("ep",),
        injections=[{
            "match": "blocks.alpha",
            "local_compute_fn": {"_target_": "demo.ep.compute"},
        }],
    )
    patches = SimpleNamespace(local_computes=(record,))

    rendered = materialize_recorded_forwards(
        SOURCE_TEXT, meta, patch_plan=patches
    )

    expected_tokens = (
        "class Alpha(nn.Module):",
        "ep_enable = False",
        "def _forward_impl(self, x):",
        "if self.ep_enable:",
    )
    for token in expected_tokens:
        assert token in rendered, (
            f"Expected generated EP source to contain {token!r}, "
            f"got source={rendered!r}"
        )
    for token in ("HPCodegen", "hp_recorded_local_compute_patches"):
        assert token not in rendered, (
            f"Expected generated EP source to omit {token!r}, "
            f"got source={rendered!r}"
        )


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_recorded_ep_toggle_prevents_second_same_class_boundary_lowering():
    """One class-level EP toggle excludes all its FQNs from later lowering."""
    alpha_type = type("Alpha", (), {"forward": _recorded_alpha_forward})
    record = SimpleNamespace(
        module_fqn="blocks.alpha",
        module_type=alpha_type,
        original_forward=alpha_type.forward,
        factory="demo.ep.factory",
    )
    meta = SimpleNamespace(
        param_plan={
            "blocks.alpha": _identity_entry(),
            "blocks.other_alpha": _boundary("unused", "Alpha")[1]["unused"],
        },
        boundary_classes={
            "blocks.alpha": "Alpha",
            "blocks.other_alpha": "Alpha",
        },
        external_state_classes=[],
        mesh_dim_names=("cp", "ep"),
        injections=[{
            "match": "blocks.alpha",
            "local_compute_fn": {"_target_": "demo.ep.compute"},
        }],
    )

    rendered = materialize_recorded_forwards(
        SOURCE_TEXT,
        meta,
        patch_plan=SimpleNamespace(local_computes=(record,)),
    )
    lowered = lower_forward_boundaries(
        rendered, meta, boundary_classes=meta.boundary_classes
    )
    tree = ast.parse(lowered)
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Alpha"
    )
    methods = [
        node.name
        for node in host.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert all(
        entry["skip_source_lowering"]
        for entry in meta.param_plan.values()
    )
    assert methods.count("_forward_impl") == 1
    assert methods.count("forward") == 1
    assert lowered == rendered


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_frozen_local_compute_rewrites_class_without_live_record():
    """Explicit frozen injections remain visible in offline-only snapshots."""
    meta = SimpleNamespace(
        param_plan={"blocks.alpha": _identity_entry()},
        boundary_classes={"blocks.alpha": "Alpha"},
        mesh_dim_names=("ep",),
        materialized_patches=[],
        injections=[{
            "match": "blocks.alpha",
            "local_compute_fn": {"_target_": "demo.ep.compute"},
        }],
    )

    rendered = materialize_recorded_forwards(
        SOURCE_TEXT,
        meta,
        patch_plan=SimpleNamespace(local_computes=()),
    )

    assert "ep_enable = False" in rendered
    assert "def _forward_impl(self, x):" in rendered
    assert "if self.ep_enable:" in rendered
    assert meta.materialized_patches == [{
        "fqn": "blocks.alpha",
        "host_class": "Alpha",
        "kind": "local_compute",
        "source_form": "same_class_toggle_forward",
    }]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_cp_dispatch_rewrites_same_class_without_constructor_swap():
    """CP dispatch is an in-place forward rewrite with the original signature."""
    patches = InlinePatchSet(
        forward_extracts=[
            ForwardExtractPatch(
                class_name="Alpha",
                method_name="forward",
                body=cp_dispatch_shell_body("kimi_cp_forward"),
            )
        ]
    )

    rendered = apply_patch_set(SOURCE_TEXT, patches)

    expected_tokens = (
        "class Alpha(nn.Module):",
        "cp_enable = False",
        "def _forward_impl(self, x):",
        "return self._forward_impl(x)",
        "return kimi_cp_forward(self, x)",
    )
    for token in expected_tokens:
        assert token in rendered, (
            f"Expected generated CP source to contain {token!r}, "
            f"got source={rendered!r}"
        )
    for token in ("class AlphaCP", "HPCodegen"):
        assert token not in rendered, (
            f"Expected generated CP source to omit {token!r}, "
            f"got source={rendered!r}"
        )


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_same_class_cp_and_boundary_merge_into_one_forward_impl():
    """A source-owned CP host keeps one method after boundary lowering."""
    patches = InlinePatchSet(
        forward_extracts=[
            ForwardExtractPatch(
                class_name="Alpha",
                method_name="forward",
                body=cp_dispatch_shell_body("kimi_cp_forward"),
                merge_original_body=True,
            )
        ]
    )
    rendered = apply_patch_set(SOURCE_TEXT, patches)
    _, entries, classes = _boundary("blocks.alpha", "Alpha")
    plan = {
        "param_plan": entries,
        "injections": [],
        "mesh_dim_names": ("cp", "tp"),
    }

    rendered = lower_forward_boundaries(
        rendered, plan, boundary_classes=classes
    )
    tree = ast.parse(rendered)
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Alpha"
    )
    methods = [
        node.name
        for node in host.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert methods.count("_forward_impl") == 1
    assert methods.count("forward") == 1
    assert "return kimi_cp_forward(self, x)" in rendered
    assert "return self._forward_impl(x)" not in rendered
    assert "return x * 2" in rendered


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_same_class_cp_toggle_satisfies_inner_wrapper_verification():
    """The verifier accepts the same-class CP form used by generated models."""
    alpha_type = type("Alpha", (), {})
    wrapper = SimpleNamespace(
        module_fqn="blocks.alpha",
        target_fqn="self",
        target_type=alpha_type,
        request=SimpleNamespace(target=object()),
    )
    patches = SimpleNamespace(
        local_computes=(),
        module_replacements=(),
        inner_wrappers=(wrapper,),
    )
    meta = SimpleNamespace(materialized_patches=[])
    rendered = apply_patch_set(
        SOURCE_TEXT,
        InlinePatchSet(
            forward_extracts=[
                ForwardExtractPatch(
                    class_name="Alpha",
                    method_name="forward",
                    body=cp_dispatch_shell_body("kimi_cp_forward"),
                    merge_original_body=True,
                )
            ]
        ),
    )
    _, entries, classes = _boundary("blocks.alpha", "Alpha")
    rendered = lower_forward_boundaries(
        rendered,
        {
            "param_plan": entries,
            "injections": [],
            "mesh_dim_names": ("cp", "tp"),
        },
        boundary_classes=classes,
    )

    _verify_recorded_patch_materialization(rendered, meta, patch_plan=patches)

    assert meta.materialized_patches == [{
        "fqn": "blocks.alpha",
        "host_class": "Alpha",
        "kind": "inner_wrapper",
        "source_form": "same_class_toggle_forward",
    }]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_replacement_verifier_requires_constructor_call():
    """An imported replacement symbol alone is not a materialized replacement."""
    source_type = type("SourceBlock", (), {})
    replacement_type = type("ReplacementBlock", (), {})
    record = SimpleNamespace(
        module_fqn="blocks.alpha",
        source_type=source_type,
        replacement_type=replacement_type,
    )
    patches = SimpleNamespace(
        local_computes=(),
        module_replacements=(record,),
        inner_wrappers=(),
    )
    meta = SimpleNamespace(materialized_patches=[])
    imported_only = "from demo import ReplacementBlock\n"

    with pytest.raises(RuntimeError, match="were not materialized"):
        _verify_recorded_patch_materialization(
            imported_only, meta, patch_plan=patches
        )

    rendered = imported_only + "\nblock = ReplacementBlock(source)\n"
    _verify_recorded_patch_materialization(rendered, meta, patch_plan=patches)

    assert meta.materialized_patches == [{
        "fqn": "blocks.alpha",
        "host_class": "ReplacementBlock",
        "kind": "module_replacement",
        "source_form": "constructor",
    }]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_bound_method_local_compute_satisfies_verification():
    """A model-owned MethodType wrapper is a visible local-compute patch."""
    module_type = type("ImportedMoe", (), {})

    def source_forward(self, hidden_states):
        return hidden_states

    record = SimpleNamespace(
        module_fqn="blocks.alpha.mlp",
        module_type=module_type,
        original_forward=source_forward,
    )
    patches = SimpleNamespace(
        local_computes=(record,),
        module_replacements=(),
        inner_wrappers=(),
    )
    meta = SimpleNamespace(materialized_patches=[])
    rendered = """
def source_forward_hp_impl(self, hidden_states):
    return hidden_states

def source_forward(self, hidden_states):
    if self.ep_enable:
        return self.__hp_compute__(hidden_states)
    return self._forward_impl(hidden_states)

owner._forward_impl = MethodType(source_forward_hp_impl, owner)
owner.forward = MethodType(source_forward, owner)
"""

    _verify_recorded_patch_materialization(rendered, meta, patch_plan=patches)

    assert meta.materialized_patches == [{
        "fqn": "blocks.alpha.mlp",
        "host_class": "ImportedMoe",
        "kind": "local_compute",
        "source_form": "bound_method_toggle_forward",
    }]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_imported_cp_host_is_rendered_as_same_name_class():
    """An imported replacement gets a visible same-name CP class."""
    patches = InlinePatchSet(
        imports=[ImportPatch("demo.components", ("MLAAttention",))],
        forward_extracts=[
            ForwardExtractPatch(
                class_name="MLAAttention",
                method_name="forward",
                body=cp_dispatch_shell_body("mla_cp_forward"),
            )
        ],
    )

    _materialize_imported_forward_hosts(SOURCE_TEXT, patches)
    rendered = apply_patch_set(SOURCE_TEXT, patches)

    assert "from demo.components import MLAAttention as _MLAAttentionBase" in rendered
    assert "class MLAAttention(_MLAAttentionBase):" in rendered
    assert "class HPCodegen" not in rendered
    assert "if self.cp_enable:" in rendered
    assert "return super().forward(*args, **kwargs)" in rendered
    assert "def _forward_impl" not in rendered


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_imported_cp_host_and_boundary_share_one_forward_impl():
    """CP dispatch and boundary lowering compose into one implementation."""
    patches = InlinePatchSet(
        imports=[ImportPatch("demo.components", ("MLAAttention",))],
        forward_extracts=[
            ForwardExtractPatch(
                class_name="MLAAttention",
                method_name="forward",
                body=cp_dispatch_shell_body("mla_cp_forward"),
            )
        ],
    )
    _materialize_imported_forward_hosts(SOURCE_TEXT, patches)
    rendered = apply_patch_set(SOURCE_TEXT, patches)
    _, entries, classes = _boundary(
        "model.layers.0.self_attn", "MLAAttention"
    )
    plan = {
        "param_plan": entries,
        "injections": [],
        "mesh_dim_names": ("cp", "tp"),
    }

    rendered = lower_forward_boundaries(
        rendered, plan, boundary_classes=classes
    )
    tree = ast.parse(rendered)
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLAAttention"
    )
    methods = [
        node.name
        for node in host.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert methods.count("_forward_impl") == 1
    assert methods.count("forward") == 1
    assert "return mla_cp_forward(self, *args, **kwargs)" in rendered
    assert "return super().forward(*args, **kwargs)" in rendered
    assert "return self._forward_impl(*args, **kwargs)" not in rendered


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_tp_collective_boundary_renders_static_operators():
    """A TP-lowerable boundary with a static signature renders bare operators.

    Feature: codegen-lowering
    Description: A boundary whose declared transitions are all TP-lowerable
        and whose forward passes the structural gates is rewritten to the
        static template: the class marker, ``_forward_impl`` extraction, and
        ``self._hp_tp`` operator calls instead of the generic redistribute.
    Expectation: The emitted text carries the ``tp_collective`` marker, the
        input all_gather call, the extracted impl call, and the output
        reduce_scatter call; no ``self._hp_boundary`` reference remains.
    """
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert '_hp_boundary_form = "tp_collective"' in text
    assert "x = self._hp_tp.all_gather(x, dim=1)" in text
    assert "outputs = self._forward_impl(x)" in text
    assert "outputs = self._hp_tp.reduce_scatter(outputs, dim=1)" in text
    assert "self._hp_boundary" not in text


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_identity_boundary_forward_is_pruned():
    """An all-identity boundary keeps its original forward untouched.

    Feature: codegen-lowering
    Description: A boundary whose declared transitions are all identity is
        pruned: the runtime install path covers the exact no-op / to-local
        semantics with its generic wrapper, so rewriting the forward would
        add indirection without changing behavior.
    Expectation: The emitted text equals the source text (no
        ``_forward_impl``, no marker, no redistribute calls).
    """
    plan, classes = _single_plan(_identity_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert text == SOURCE_TEXT


def _degenerate_entry() -> dict:
    """A degenerate-topology boundary: full per-axis dicts, every axis identity.

    This is the frozen shape a tp=cp=ep=1 regeneration produces: the freeze
    declares every axis (``cp``/``ep``/``tp``) with canonical strings, so an
    identity ``ep: R -> R`` key is present even though the entry is not
    EP-dependent.
    """
    return {
        "is_boundary": True,
        "in_src": {"x": {"cp": "R", "ep": "R", "tp": "R"}},
        "in_dst": {"x": {"cp": "R", "ep": "R", "tp": "R"}},
        "out_src": {"output": {"cp": "R", "ep": "R", "tp": "R"}},
        "out_dst": {"output": {"cp": "R", "ep": "R", "tp": "R"}},
    }


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_degenerate_topology_identity_ep_keys_are_pruned():
    """Empty active axes + identity ``ep`` keys prune the rewrite entirely.

    Regression lock for the tp=cp=ep=1 failure: reading ``ep`` key
    *existence* as EP dependence forced the generic form on boundaries whose
    semantics are an exact no-op, while the install path is skipped entirely
    for such meshes — the rewritten forward's ``_hp_boundary`` reference
    was never bound and training crashed with ``AttributeError``.  Only an
    ``ep`` placement that actually *changes* is an expert-mesh dependency.

    Feature: codegen-lowering
    Description: A plan with no active axes (``mesh_dim_names`` empty/None)
        and a boundary declaring full per-axis dicts with identity ``ep``
        keys is lowered through ``lower_forward_boundaries``.
    Expectation: The emitted text equals the source text — no
        ``_forward_impl``, no marker, no ``_hp_boundary`` reference.
    """
    plan, classes = _single_plan(_degenerate_entry())

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert text == SOURCE_TEXT


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_degenerate_topology_changing_ep_renders_generic():
    """Empty active axes + a changing ``ep`` placement keep the generic rewrite.

    The mirror of the pruning test: an entry whose ``ep`` placement actually
    changes (``R -> S(0)``) is genuinely expert-mesh dependent — not
    statically decidable — so it must keep the rewritten redistribute forward
    that ``hp_install_boundaries`` binds.

    Feature: codegen-lowering
    Description: A plan with no active axes and a boundary whose ``ep``
        placement changes is lowered through ``lower_forward_boundaries``.
    Expectation: The forward is rewritten to the generic form:
        ``_forward_impl`` extraction and ``self._hp_boundary`` calls.
    """
    entry = _degenerate_entry()
    entry["in_dst"]["x"]["ep"] = "S(0)"
    entry["out_dst"]["output"]["ep"] = "S(0)"
    plan, classes = _single_plan(entry)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert "_forward_impl" in text
    assert "self._hp_boundary.redistribute_inputs" in text
    assert "self._hp_boundary.redistribute_outputs" in text


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_tp_collective_falls_back_to_generic_on_variadic_signature():
    """A ``*args``/``**kwargs`` forward cannot carry the static template.

    Feature: codegen-lowering
    Description: The static template re-passes every declared parameter by
        name, so a variadic signature fails the structural gates and the
        boundary renders as the generic redistribute form instead — even
        though the plan's transitions classify as tp_collective.
    Expectation: No marker and no ``_hp_tp`` calls; the forward goes
        through ``self._hp_boundary.redistribute_inputs/outputs``.
    """
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(
        VARIADIC_SOURCE_TEXT, plan, boundary_classes=classes
    )

    assert "_hp_boundary_form" not in text
    assert "_hp_tp" not in text
    assert "_forward_impl" in text
    assert "self._hp_boundary.redistribute_inputs" in text
    assert "self._hp_boundary.redistribute_outputs" in text


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_tp_collective_allows_kwargs_and_tuple_output():
    """HF-style ``**kwargs`` and tuple output can still expose bare TP ops.

    Feature: codegen-lowering
    Description: A TP-lowerable boundary whose forward has ``**kwargs`` and
        returns a tuple is lowered to the static template. The generated body
        re-passes ``**kwargs`` and applies output redistribution to index 0,
        matching the frozen output-plan convention.
    Expectation: The emitted text carries the TP marker, keeps ``**kwargs`` in
        the ``_forward_impl`` call, rebuilds the tuple, and has no generic
        boundary calls.
    """
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(
        KWARGS_TUPLE_SOURCE_TEXT, plan, boundary_classes=classes
    )

    assert '_hp_boundary_form = "tp_collective"' in text
    assert "outputs = self._forward_impl(x, y, **kwargs)" in text
    assert "_hp_output_0 = self._hp_tp.reduce_scatter(outputs[0], dim=1)" in text
    assert "outputs = (_hp_output_0, *outputs[1:])" in text
    assert "self._hp_boundary" not in text


# ---------------------------------------------------------------------------
# D-22: deferred bias exit hook rendering
# ---------------------------------------------------------------------------

_HOOK_LINE = "outputs = self._hp_deferred_bias(outputs)"


def _generic_entry() -> dict:
    """A generic-form boundary (cp transition is not TP-lowerable)."""
    return {
        "is_boundary": True,
        "in_src": {"x": {"tp": "R", "cp": "S(1)"}},
        "in_dst": {"x": {"tp": "R", "cp": "R"}},
        "out_src": {"output": {"tp": "P(sum)", "cp": "S(1)"}},
        "out_dst": {"output": {"tp": "R", "cp": "S(1)"}},
    }


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_generic_deferred_bias_hook_renders_last():
    """D-22: the exit hook is the LAST statement of the generic exit block.

    Feature: codegen-lowering
    Description: A generic-form boundary whose frozen entry carries
        ``deferred_bias_params`` is lowered; the rowwise bias is suppressed
        inside the region and must be re-added once, after the output
        redistribution and the to-local unwrap.
    Expectation: The hook line is emitted after ``redistribute_outputs`` and
        after ``hp_to_local_if_dtensor``.
    """
    entry = _generic_entry()
    entry["deferred_bias_params"] = ["o_proj.bias"]
    plan, classes = _single_plan(entry)
    plan["mesh_dim_names"] = ("cp", "tp")

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    hook_at = text.index(_HOOK_LINE)
    assert hook_at > text.index("self._hp_boundary.redistribute_outputs(outputs)")
    assert hook_at > text.index("hp_to_local_if_dtensor(outputs)")


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_tp_collective_deferred_bias_hook_renders_after_exit_ops():
    """D-22: the static template re-adds the bias after the exit reduction.

    Feature: codegen-lowering
    Description: A tp_collective boundary whose frozen entry carries deferred
        bias params renders the hook after the output-side bare operators (the
        Partial -> R all_reduce / reduce_scatter IS the exit reduction the
        deferral waits for) and before the return.
    Expectation: The hook line sits after the reduce_scatter call and before
        ``return outputs``.
    """
    entry = _tp_collective_entry()
    entry["deferred_bias_params"] = ["o_proj.bias"]
    plan, classes = _single_plan(entry)
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    hook_at = text.index(_HOOK_LINE)
    assert hook_at > text.index(
        "outputs = self._hp_tp.reduce_scatter(outputs, dim=1)"
    )
    assert hook_at < text.index("return outputs")


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_local_region_deferred_bias_still_requires_native_phase_c():
    """D-22 does not permit a local-region source rewrite.

    Feature: codegen-lowering
    Description: A region boundary with deferred bias reaches source lowering.
    Expectation: Generation fails so native Phase C owns both the region and
        its deferred-bias ordering.
    """
    entry = _identity_entry()
    entry["deferred_bias_params"] = ["o_proj.bias"]
    plan, classes = _single_plan(entry)
    plan["injections"] = [
        {"match": "blocks.alpha", "local_compute_fn": {"_target_": "some.factory"}},
    ]

    with pytest.raises(RuntimeError, match="native Phase C wrapper"):
        lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_no_deferred_bias_no_hook_line():
    """Entries without deferred params emit no hook (byte-stable artifacts).

    Feature: codegen-lowering
    Description: Generic and static boundaries whose entries carry no
        ``deferred_bias_params`` key are lowered.
    Expectation: No ``_hp_deferred_bias`` reference appears — the emitted
        bytes for non-deferring plans are unchanged by the D-22 feature.
    """
    for entry in (_generic_entry(), _tp_collective_entry()):
        plan, classes = _single_plan(entry)
        plan["mesh_dim_names"] = ("cp", "tp")

        text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)
        assert "_hp_deferred_bias" not in text


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_identity_boundary_with_deferred_bias_fails_fast():
    """D-22: an identity-pruned boundary cannot carry deferred bias params.

    Feature: codegen-lowering
    Description: An all-identity boundary whose entry declares deferred bias
        params is lowered. The identity form leaves the original forward
        untouched, so the suppressed bias would never be re-added.
    Expectation: ``NotImplementedError`` is raised at generation time instead
        of silently dropping the bias at train time.
    """
    entry = _identity_entry()
    entry["deferred_bias_params"] = ["o_proj.bias"]
    plan, classes = _single_plan(entry)
    plan["mesh_dim_names"] = ("tp",)

    try:
        lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)
    except NotImplementedError as exc:
        assert "identity" in str(exc)
        assert "deferred bias" in str(exc)
    else:
        raise AssertionError("identity + deferred bias must fail fast")
