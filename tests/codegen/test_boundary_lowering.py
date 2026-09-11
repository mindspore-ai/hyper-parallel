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


def test_lowering_bytes_are_order_independent():
    """Reversing the ``param_plan`` key order must not change the emitted bytes.

    This is the property the sort fix restores: the two plans carry identical
    entry content, so they must lower to the identical byte string.  Before the
    fix the ``_HYPER_BOUNDARY_Alpha``/``_HYPER_BOUNDARY_Beta`` constants were
    appended in insertion order, so ``alpha_first`` and the reversed plan
    produced different output bytes.
    """
    plan_a, classes_a = _plan(alpha_first=True)
    plan_b, classes_b = _plan(alpha_first=False)

    text_a = lower_forward_boundaries(SOURCE_TEXT, plan_a, boundary_classes=classes_a)
    text_b = lower_forward_boundaries(SOURCE_TEXT, plan_b, boundary_classes=classes_b)

    assert text_a == text_b


def test_boundary_constants_are_sorted_by_class():
    """The appended constants follow the source-defined order (via the sort).

    Explicitly checks that a real rewrite happened, so the byte-identity test
    above cannot pass vacuously (e.g. on a plan that never rewrote a forward).
    """
    plan, classes = _plan(alpha_first=True)
    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert text.index("_HYPER_BOUNDARY_Alpha") < text.index("_HYPER_BOUNDARY_Beta")
    # Both constants are present and both forwards were rewritten.
    assert "_forward_impl" in text
    assert text.count("hyper_redistribute") >= 2


def test_generated_parallelize_wraps_unlowered_module_boundaries():
    from hyper_parallel.codegen.emit.modeling import (
        inject_codegen_imports,
        inject_hyper_parallelize,
    )

    source = "import torch\n"
    text = inject_hyper_parallelize(inject_codegen_imports(source), {"param_plan": {}})

    body = text[text.index("def hyper_parallelize"):]

    assert "hyper_wrap_module_boundaries" in body
    assert body.index("hyper_wrap_module_boundaries") < body.index("hyper_bind_compute")
