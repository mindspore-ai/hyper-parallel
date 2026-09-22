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
"""Contract tests for the switch/toggle forward template (emit/parallel).

The toggle template is a SEPARATE, opt-in generation mode from the default four
forms (:func:`lower_forward_boundaries`).  It keeps the original HF forward as
``_forward_impl`` and wraps it with independent ``{tp|cp|ep}_enable`` switches.
These tests lock the emitted switch bodies without disturbing the four existing
contract tests in ``test_boundary_lowering.py``.
"""

from __future__ import annotations

import ast
import importlib.machinery
import sys
import types

# ``classify_tp_transition`` (pulled lazily by the toggle template) probes
# ``torch.npu``; a CPU-only checkout needs the same stub the attention test uses.
import torch  # noqa: E402

if "torch_npu" not in sys.modules:
    try:
        import torch_npu  # noqa: F401  pylint: disable=unused-import
    except ModuleNotFoundError:  # pragma: no cover - CPU-only host
        _torch_npu = types.ModuleType("torch_npu")
        # ``accelerate.is_npu_available`` calls ``importlib.util.find_spec``, which
        # needs a real loader (loader=None makes find_spec raise).
        _torch_npu.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_npu",
            loader=importlib.machinery.SourceFileLoader("torch_npu", "torch_npu_stub.py"),
        )
        sys.modules["torch_npu"] = _torch_npu
elif getattr(sys.modules["torch_npu"].__spec__, "loader", None) is None:
    sys.modules["torch_npu"].__spec__ = importlib.machinery.ModuleSpec(
        name="torch_npu",
        loader=importlib.machinery.SourceFileLoader("torch_npu", "torch_npu_stub.py"),
    )
if not hasattr(torch, "npu"):

    class _NpuMode:  # pragma: no cover - CPU-only host
        """CPU placeholder so ``torch.npu.is_available()`` reads not-available."""

        def is_available(self):
            return False

        def device_count(self):
            return 0

    torch.npu = _NpuMode()

# ``classify_boundary_form`` lazily imports the distributed/accelerate path at
# call time, but the per-test ``torch_npu`` fixture replaces the stub with a
# spec-less module that makes ``accelerate.is_npu_available`` raise.  Resolve the
# accelerate NPU probe once here, under the good stub, so it is cached before the
# fixture runs (same reason the full codegen suite passes only when run together).
import hyper_parallel.distributed  # noqa: E402,F401  pylint: disable=unused-import

from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries_toggle  # noqa: E402
from tests.common.mark_utils import arg_mark  # noqa: E402

SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward gets switched."""

    def forward(self, x):
        return x * 2
'''


def _toggle_plan(axes=("tp",), *, out_reducible: bool = False) -> dict:
    """A dense plan with one boundary class, sharding only the given axes.

    With ``out_reducible`` the output transitions ``P(sum) -> S(1)`` so the tp
    reduce_scatter segment is emitted; otherwise the output is identity.
    """
    out_src = {axis: ("P(sum)" if out_reducible else "S(1)") for axis in axes}
    out_dst = {axis: "S(1)" for axis in axes}
    entry = {
        "is_boundary": True,
        "in_src": {"x": {axis: "S(1)" for axis in axes}},
        "in_dst": {"x": {axis: "R" for axis in axes}},
        "out_src": {"output": out_src},
        "out_dst": {"output": out_dst},
        "params": {"weight": {"tp": "S(0)"}},
    }
    plan = {
        "param_plan": {"blocks.alpha": entry},
        "injections": [],
        "mesh_dim_names": list(axes),
    }
    return plan, {"blocks.alpha": "Alpha"}


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_toggle_keeps_original_forward_impl():
    """The original HF forward body is preserved verbatim as ``_forward_impl``.

    Feature: codegen-toggle-lowering
    Description: A TP boundary is lowered through
        ``lower_forward_boundaries_toggle``.
    Expectation: The original ``forward(x): return x * 2`` survives as
        ``_forward_impl``; ``mesh_context`` need not be imported; the toggle
        marker is present.
    """
    plan, classes = _toggle_plan()
    text = lower_forward_boundaries_toggle(SOURCE_TEXT, plan, boundary_classes=classes)

    ast.parse(text)
    assert '_hyper_boundary_form = "toggle"' in text
    assert "def _forward_impl(self, x):" in text
    assert "return x * 2" in text


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_toggle_emits_tp_switch_segments():
    """TP collectives are emitted, each guarded by its own ``tp_enable``.

    Feature: codegen-toggle-lowering
    Description: A TP boundary (S(1)->R in, S(1)->S(1) out) is lowered through
        the toggle template.
    Expectation: The ``if tp_enable:`` input all_gather and output
        reduce_scatter blocks both appear, and the original forward is called
        in between.
    """
    plan, classes = _toggle_plan(out_reducible=True)
    text = lower_forward_boundaries_toggle(SOURCE_TEXT, plan, boundary_classes=classes)

    assert "if self.tp_enable:" in text
    assert "x = self._hyper_tp.all_gather(x, dim=1)" in text
    assert "outputs = self._forward_impl(x)" in text
    assert "outputs = self._hyper_tp.reduce_scatter(outputs, dim=1)" in text


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_toggle_separates_cp_and_tp_blocks():
    """A cp+tp boundary guards the two dimensions in independent blocks.

    Feature: codegen-toggle-lowering
    Description: A boundary sharding both cp and tp is lowered through the
        toggle template.
    Expectation: Both ``if cp_enable:`` and ``if tp_enable:`` blocks exist and
        carry their own collective calls.
    """
    plan, classes = _toggle_plan(axes=("cp", "tp"))
    text = lower_forward_boundaries_toggle(SOURCE_TEXT, plan, boundary_classes=classes)

    ast.parse(text)
    assert text.count("if self.tp_enable:") == 1
    assert text.count("if self.cp_enable:") == 1
    assert "x = self._hyper_tp.all_gather(x, dim=1)" in text
    assert "outputs = self._forward_impl(x)" in text


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_toggle_bindings_compute_enabler_flags():
    """Toggle modules get per-axis enabler booleans bound from the live mesh.

    Feature: codegen-toggle-binding
    Description: A toggle boundary whose live active axes are only tp is bound
        through ``_install_toggle_bindings``.
    Expectation: ``tp_enable`` is True while ``cp_enable``/``ep_enable`` are
        False; a single-rank mesh yields no ``_hyper_tp`` operator.
    """
    from hyper_parallel.codegen.runtime import _install_toggle_bindings

    module = types.SimpleNamespace()
    module._hyper_boundary_form = "toggle"
    installed = types.SimpleNamespace(active_dim_names=["tp"], dense_mesh=None)
    _install_toggle_bindings(module, installed)

    assert module.tp_enable is True
    assert module.cp_enable is False
    assert module.ep_enable is False
    assert not hasattr(module, "_hyper_tp")


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_toggle_forward_runs_when_switch_off():
    """A toggle forward executes its original body when the axis is disabled.

    Feature: codegen-toggle-binding
    Description: On a single-rank run the installed ``tp_enable=False``, so the
        generated forward must skip the ``self._hyper_tp`` segment (never
        touching an unbound operator) and call the ``_forward_impl`` body.
    Expectation: ``alpha(3) == 6`` (original ``x * 2``), no ``_hyper_tp`` access.
    """
    from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries_toggle

    entry = {
        "is_boundary": True,
        "in_src": {"x": {"tp": "S(1)"}},
        "in_dst": {"x": {"tp": "R"}},
        "out_src": {"output": {"tp": "P(sum)"}},
        "out_dst": {"output": {"tp": "S(1)"}},
        "params": {"weight": {"tp": "S(0)"}},
    }
    plan = {
        "param_plan": {"blocks.alpha": entry},
        "injections": [],
        "mesh_dim_names": ["tp"],
    }
    text = lower_forward_boundaries_toggle(
        SOURCE_TEXT, plan, boundary_classes={"blocks.alpha": "Alpha"}
    )
    ns: dict = {}
    exec(compile(text, "<toggle>", "exec"), ns)  # nosec B102 - controlled fixture
    alpha = ns["Alpha"]()
    # Single-rank install binds the enablers to False on the instance; the
    # `self.<axis>_enable` segments therefore skip the unbound `_hyper_tp`.
    alpha.tp_enable = False
    alpha.cp_enable = False
    alpha.ep_enable = False
    assert alpha.forward(3) == 6