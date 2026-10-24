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
"""Contract tests for the preflight boundary-form structure invariant.

``verify_boundary_forms`` re-derives every boundary class's emitted form from
the persisted meta (through ``iter_emitted_forms`` — the emitter's own
decision path) and fails when the generated source does not carry exactly
that structure.  These tests render a two-form bundle (one tp_collective
boundary, one identity boundary), assert it passes, then corrupt the source
in three ways a post-generation edit (or an emitter/check disagreement)
could produce, and assert each corruption fails fast.  The same bundle emitted
by the opt-in toggle template must pass as well: its classes carry the toggle
marker, which the check accepts in place of the static one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyper_parallel.codegen.check.preflight import verify_boundary_forms
from hyper_parallel.codegen.emit.parallel import (
    lower_forward_boundaries,
    lower_forward_boundaries_toggle,
)
from tests.common.mark_utils import arg_mark

SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward gets statically lowered."""

    def forward(self, x):
        return x * 2


class Beta(nn.Module):
    """An identity boundary class whose forward stays untouched."""

    def forward(self, x):
        return x + 1
'''


def _plan() -> tuple[dict, dict]:
    """A frozen plan: Alpha tp_collective (in S(1)->R, out P->S), Beta identity."""
    plan = {
        "param_plan": {
            "blocks.alpha": {
                "is_boundary": True,
                "in_src": {"x": {"tp": "S(1)"}},
                "in_dst": {"x": {"tp": "R"}},
                "out_src": {"output": {"tp": "P(sum)"}},
                "out_dst": {"output": {"tp": "S(1)"}},
            },
            "blocks.beta": {
                "is_boundary": True,
                "in_src": {"x": {"tp": "R"}},
                "in_dst": {"x": {"tp": "R"}},
                "out_src": {"output": {"tp": "R"}},
                "out_dst": {"output": {"tp": "R"}},
            },
        },
        "injections": [],
        "mesh_dim_names": ("tp",),
    }
    return plan, {"blocks.alpha": "Alpha", "blocks.beta": "Beta"}


def _render() -> str:
    """Render the two-form bundle the same way generation does."""
    plan, classes = _plan()
    return lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)


def _render_toggle() -> str:
    """Render the same bundle through the opt-in toggle template."""
    plan, classes = _plan()
    return lower_forward_boundaries_toggle(SOURCE_TEXT, plan, boundary_classes=classes)


def _layout(tmp_path, text):
    """A minimal layout double pointing at a persisted modeling file."""
    path = tmp_path / "modeling_gen.py"
    path.write_text(text, encoding="utf-8")
    return SimpleNamespace(modeling_path=str(path))


INLINE_SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """An inlined component class whose forward owns its orchestration."""

    def forward(self, x):
        if self.tp_enable:
            x = self._hp_tp.all_gather(x, dim=1)
        return x
'''

#: The body an accessor-era artifact carried: a module-level lookup instead of
#: the instance channel the runtime binds at install time.
_INLINE_ORCHESTRATION = (
    "        if self.tp_enable:\n"
    "            x = self._hp_tp.all_gather(x, dim=1)\n"
)


def _inline_meta():
    """The same boundary contract as ``_meta``, over the inlined fixture class."""
    return SimpleNamespace(
        param_plan={
            "blocks.alpha": {
                "is_boundary": True,
                "in_src": {"x": {"tp": "S(1)"}},
                "in_dst": {"x": {"tp": "R"}},
                "out_src": {"output": {"tp": "P(sum)"}},
                "out_dst": {"output": {"tp": "S(1)"}},
            }
        },
        injections=[],
        mesh_dim_names=("tp",),
        boundary_classes={"blocks.alpha": "Alpha"},
        source={"module_name": ""},
    )


def _meta():
    """A meta double carrying the fields ``iter_emitted_forms`` reads."""
    plan, classes = _plan()
    return SimpleNamespace(
        param_plan=plan["param_plan"],
        injections=[],
        mesh_dim_names=("tp",),
        boundary_classes=classes,
        source={"module_name": ""},
    )


def _bias_plan() -> tuple[dict, dict]:
    """The frozen plan, with Alpha deferring its rowwise bias (D-22)."""
    plan, classes = _plan()
    plan["param_plan"]["blocks.alpha"]["deferred_bias_params"] = ["o_proj.bias"]
    return plan, classes


def _render_bias() -> str:
    """Render the bundle with the D-22 exit line on the Alpha boundary."""
    plan, classes = _bias_plan()
    return lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)


def _bias_meta():
    """A meta double whose Alpha entry declares deferred bias params."""
    plan, classes = _bias_plan()
    return SimpleNamespace(
        param_plan=plan["param_plan"],
        injections=[],
        mesh_dim_names=("tp",),
        boundary_classes=classes,
        source={"module_name": ""},
    )


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_emitted_forms_pass_verification(tmp_path):
    """A freshly lowered bundle satisfies the structure invariant.

    Feature: codegen-preflight
    Description: The bundle rendered by ``lower_forward_boundaries`` (one
        tp_collective class with marker/``_forward_impl``, one identity class
        left pristine) is verified with ``verify_boundary_forms``.
    Expectation: No exception — the persisted structure matches the meta's
        re-derived forms exactly.
    """
    verify_boundary_forms(_meta(), _layout(tmp_path, _render()))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_missing_marker_fails_verification(tmp_path):
    """Stripping the form marker from a static class fails the check.

    Feature: codegen-preflight
    Description: A tp_collective class whose marker was removed no longer
        matches the meta's re-derived form.
    Expectation: ``RuntimeError`` naming the class.
    """
    text = _render().replace('_hp_boundary_form = "tp_collective"', "")

    with pytest.raises(RuntimeError, match="Alpha"):
        verify_boundary_forms(_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_stray_attribute_reference_fails_verification(tmp_path):
    """A static forward that gained a generic-engine attribute fails.

    Feature: codegen-preflight
    Description: Replacing ``self._hp_tp`` with ``self._hp_boundary``
        inside the static forward leaves the marker and ``_forward_impl``
        intact but violates the static template's attribute contract — the
        install-time validation would silently replace this forward.
    Expectation: ``RuntimeError`` naming the class.
    """
    text = _render().replace("self._hp_tp.", "self._hp_boundary.")

    with pytest.raises(RuntimeError, match="Alpha"):
        verify_boundary_forms(_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_deferred_bias_boundary_passes_verification(tmp_path):
    """A tp_collective boundary deferring its rowwise bias passes the check.

    Feature: codegen-preflight
    Description: The D-22 exit call ``self._hp_deferred_bias(outputs)``
        is part of the static template for a class whose frozen entry
        declares deferred bias params — the attribute allowlist the emitter
        and the preflight share must accept it.
    Expectation: No exception.
    """
    text = _render_bias()
    assert "self._hp_deferred_bias(outputs)" in text

    verify_boundary_forms(_bias_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_undeclared_deferred_bias_call_fails_verification(tmp_path):
    """A deferred-bias exit call on a non-deferring class fails the check.

    Feature: codegen-preflight
    Description: ``self._hp_deferred_bias`` is allowed only for classes
        whose frozen entry declares deferred bias params; the same call on
        a plain tp_collective boundary is drift.
    Expectation: ``RuntimeError`` naming the class.
    """
    text = _render().replace(
        "        return outputs",
        "        outputs = self._hp_deferred_bias(outputs)\n"
        "        return outputs",
        1,
    )

    with pytest.raises(RuntimeError, match="Alpha"):
        verify_boundary_forms(_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_rewritten_identity_class_fails_verification(tmp_path):
    """An identity class that carries a rewritten forward fails the check.

    Feature: codegen-preflight
    Description: Injecting a ``_forward_impl`` into the identity-pruned
        class (the shape an older emitter produced) contradicts the meta's
        identity verdict.
    Expectation: ``RuntimeError`` naming the class.
    """
    text = _render().replace(
        '    """An identity boundary class whose forward stays untouched."""\n',
        '    """An identity boundary class whose forward stays untouched."""\n\n'
        "    def _forward_impl(self, x):\n"
        "        return x + 1\n",
    )

    with pytest.raises(RuntimeError, match="Beta"):
        verify_boundary_forms(_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_toggle_marked_boundary_passes_verification(tmp_path):
    """A tp_collective boundary emitted by the toggle template passes.

    Feature: codegen-preflight
    Description: The opt-in switch template marks its classes with
        ``TOGGLE_FORM_MARKER`` (``_hp_boundary_form = "toggle"``) instead of
        the static template's ``tp_collective`` value, while the meta's
        re-derived form is still ``tp_collective`` — either marker satisfies
        the structure invariant.
    Expectation: No exception.
    """
    text = _render_toggle()
    assert '_hp_boundary_form = "toggle"' in text

    verify_boundary_forms(_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_inlined_component_body_passes_verification(tmp_path):
    """A fully inlined component body satisfies the structure invariant.

    Feature: codegen-preflight
    Description: The inlined fused-attention / MoE-shell shape — no extracted
        ``_forward_impl``, no form marker, orchestration through the
        instance-bound channel (``self.tp_enable`` / ``self._hp_tp``) — is
        the expected structure for a class whose forward codegen fully inlined,
        even when its boundary contract classifies as ``tp_collective``.
    Expectation: No exception.
    """
    verify_boundary_forms(_inline_meta(), _layout(tmp_path, INLINE_SOURCE_TEXT))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_removed_accessor_in_body_fails_verification(tmp_path):
    """A body still reading the retired accessor fails loudly.

    Feature: codegen-preflight
    Description: The module-level ``get_parallel_state`` lookup was removed when
        the parallel state moved onto the instance; a body that still calls it
        would raise NameError at train time.
    Expectation: ``RuntimeError`` naming the removed accessor.
    """
    text = INLINE_SOURCE_TEXT.replace(
        _INLINE_ORCHESTRATION,
        "        ps = get_parallel_state()\n"
        "        if ps.tp_enabled:\n"
        "            x = ps.tp.all_gather(x, dim=1)\n",
    )

    with pytest.raises(RuntimeError, match="get_parallel_state"):
        verify_boundary_forms(_inline_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_body_without_marker_or_channel_fails_verification(tmp_path):
    """A body that is neither marked nor inlined fails loudly.

    Feature: codegen-preflight
    Description: Stripping the orchestration from the inlined fixture leaves a
        forward with no channel, no form marker and no ``_forward_impl`` — the
        shape the emitted form cannot account for.
    Expectation: ``RuntimeError`` naming the class.
    """
    text = INLINE_SOURCE_TEXT.replace(_INLINE_ORCHESTRATION, "")

    with pytest.raises(RuntimeError, match="Alpha"):
        verify_boundary_forms(_inline_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_toggle_marked_boundary_without_marker_fails_verification(tmp_path):
    """Stripping the toggle marker still fails the check loudly.

    Feature: codegen-preflight
    Description: A toggle-emitted tp_collective class whose marker was removed
        carries neither accepted marker.
    Expectation: ``RuntimeError`` naming the class.
    """
    text = _render_toggle().replace('_hp_boundary_form = "toggle"', "")

    with pytest.raises(RuntimeError, match="Alpha"):
        verify_boundary_forms(_meta(), _layout(tmp_path, text))
