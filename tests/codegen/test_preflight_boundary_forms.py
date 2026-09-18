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
could produce, and assert each corruption fails fast.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyper_parallel.codegen.check.preflight import verify_boundary_forms
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
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


def _layout(tmp_path, text):
    """A minimal layout double pointing at a persisted modeling file."""
    path = tmp_path / "modeling_gen.py"
    path.write_text(text, encoding="utf-8")
    return SimpleNamespace(modeling_path=str(path))


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
    text = _render().replace('_hyper_boundary_form = "tp_collective"', "")

    with pytest.raises(RuntimeError, match="Alpha"):
        verify_boundary_forms(_meta(), _layout(tmp_path, text))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_stray_attribute_reference_fails_verification(tmp_path):
    """A static forward that gained a generic-engine attribute fails.

    Feature: codegen-preflight
    Description: Replacing ``self._hyper_tp`` with ``self._hyper_boundary``
        inside the static forward leaves the marker and ``_forward_impl``
        intact but violates the static template's attribute contract — the
        install-time validation would silently replace this forward.
    Expectation: ``RuntimeError`` naming the class.
    """
    text = _render().replace("self._hyper_tp.", "self._hyper_boundary.")

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
