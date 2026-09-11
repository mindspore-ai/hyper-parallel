# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
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

from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
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

VARIADIC_SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward takes variadic arguments."""

    def forward(self, x, *args, **kwargs):
        return x * 2
'''


def _boundary(fqn: str, class_name: str) -> tuple[str, dict]:
    """A minimal but real-shaped frozen boundary entry for one class."""
    entry = {
        "is_boundary": True,
        "in_src": {"x": {"tp": "S(1)", "ep": "R", "cp": "S(1)"}},
        "in_dst": {"x": {"tp": "R", "ep": "R", "cp": "S(1)"}},
        "out_src": {"output": {"tp": "S(-1)", "ep": "R", "cp": "S(1)"}},
        "out_dst": {"output": {"tp": "R", "ep": "R", "cp": "S(1)"}},
        "params": {"weight": {"tp": "S(0)", "ep": "R", "cp": "R"}},
    }
    return [fqn], {fqn: entry}, {fqn: class_name}


def _plan(*, alpha_first: bool) -> dict:
    """A frozen-plan dict whose two boundary classes appear in one of two orders."""
    alpha_fqn, alpha_entry, alpha_map = _boundary("blocks.alpha", "Alpha")
    beta_fqn, beta_entry, beta_map = _boundary("blocks.beta", "Beta")
    if alpha_first:
        plan = {**alpha_entry, **beta_entry}
        boundary_classes = {**alpha_map, **beta_map}
    else:
        plan = {**beta_entry, **alpha_entry}
        boundary_classes = {**beta_map, **alpha_map}
    return {"param_plan": plan, "injections": []}, boundary_classes


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
        ``_forward_impl`` and ``self._hyper_boundary.redistribute_*`` present.
    """
    plan, classes = _plan(alpha_first=True)
    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    # The rewritten forwards reference the instance-bound compiled plan that
    # ``hyper_install_boundaries`` installs — no per-boundary global constants
    # and no forward-time globals.  Module-level plan literals
    # (``_HYPER_PARAM_PLAN`` and friends) may still exist in the artifact;
    # they are install-time data consumed once by ``hyper_parallelize`` and
    # are never read inside a forward.
    assert "_HYPER_BOUNDARY_" not in text
    assert "mesh_context" not in text
    assert "_forward_impl" in text
    assert text.count("self._hyper_boundary.redistribute_inputs") >= 2
    assert text.count("self._hyper_boundary.redistribute_outputs") >= 2


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_generated_parallelize_installs_boundaries_without_globals():
    """The generated ``hyper_parallelize`` calls install, not per-call runtime.

    Feature: codegen-lowering
    Description: ``inject_hyper_parallelize`` emits a function body that calls
        ``hyper_install_boundaries`` before ``hyper_bind_compute`` and contains
        no ``globals()`` or legacy wrapper calls.
    Expectation: The emitted body references ``hyper_install_boundaries``,
        has no ``globals()``, and no ``hyper_wrap_module_boundaries``.
    """
    from hyper_parallel.codegen.emit.modeling import (
        inject_codegen_imports,
        inject_hyper_parallelize,
    )

    source = "import torch\n"
    text = inject_hyper_parallelize(inject_codegen_imports(source), {"param_plan": {}})

    body = text[text.index("def hyper_parallelize"):]

    assert "hyper_install_boundaries" in body
    assert body.index("hyper_install_boundaries") < body.index("hyper_bind_compute")
    assert "globals()" not in body
    assert "hyper_wrap_module_boundaries" not in body


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
def test_tp_collective_boundary_renders_static_operators():
    """A TP-lowerable boundary with a static signature renders bare operators.

    Feature: codegen-lowering
    Description: A boundary whose declared transitions are all TP-lowerable
        and whose forward passes the structural gates is rewritten to the
        static template: the class marker, ``_forward_impl`` extraction, and
        ``self._hyper_tp`` operator calls instead of the generic redistribute.
    Expectation: The emitted text carries the ``tp_collective`` marker, the
        input all_gather call, the extracted impl call, and the output
        reduce_scatter call; no ``self._hyper_boundary`` reference remains.
    """
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert '_hyper_boundary_form = "tp_collective"' in text
    assert "x = self._hyper_tp.all_gather(x, dim=1)" in text
    assert "outputs = self._forward_impl(x)" in text
    assert "outputs = self._hyper_tp.reduce_scatter(outputs, dim=1)" in text
    assert "self._hyper_boundary" not in text


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


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_tp_collective_falls_back_to_generic_on_variadic_signature():
    """A ``*args``/``**kwargs`` forward cannot carry the static template.

    Feature: codegen-lowering
    Description: The static template re-passes every declared parameter by
        name, so a variadic signature fails the structural gates and the
        boundary renders as the generic redistribute form instead — even
        though the plan's transitions classify as tp_collective.
    Expectation: No marker and no ``_hyper_tp`` calls; the forward goes
        through ``self._hyper_boundary.redistribute_inputs/outputs``.
    """
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(
        VARIADIC_SOURCE_TEXT, plan, boundary_classes=classes
    )

    assert "_hyper_boundary_form" not in text
    assert "_hyper_tp" not in text
    assert "_forward_impl" in text
    assert "self._hyper_boundary.redistribute_inputs" in text
    assert "self._hyper_boundary.redistribute_outputs" in text
