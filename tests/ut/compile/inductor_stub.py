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
"""Executable test double for ``torch.fx.passes.regional_inductor``.

The module only exists in newer PyTorch builds (the project still supports
torch 2.6 / 2.7 / 2.9), so UT would otherwise raise ``ModuleNotFoundError``.
When the real module is absent this stub is installed into ``sys.modules``:
it opaque-wraps every tagged compute node in a function that executes the
original op eagerly. That keeps graph-shape assertions (no top-level ``aten``
targets, collectives/free stay eager in regional mode) and numeric tests
(gradients match the uncompiled graph) meaningful without real Inductor.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any, Callable, Dict

import torch
from torch.fx import passes as fx_passes

_FAKE_OPS: Dict[int, Callable[..., Any]] = {}


def _fake_compiled_op(key: int, *args: Any, **kwargs: Any) -> Any:
    """Execute the original op that ``key`` refers to."""
    return _FAKE_OPS[key](*args, **kwargs)


def _fake_regional_inductor(
    graph_module: torch.fx.GraphModule, example_inputs: Any = None
) -> None:
    """Opaque-wrap tagged compute nodes, keeping comms/free nodes untouched."""
    del example_inputs
    graph = graph_module.graph
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        custom = node.meta.get("custom") or {}
        if "compile_with_inductor" not in custom:
            continue
        # Full mode tags collectives too, but Inductor keeps the async wait at
        # the top level; mirror that so the shape assertions stay meaningful.
        if "_c10d_functional.wait_tensor" in str(node.target):
            continue
        key = len(_FAKE_OPS)
        _FAKE_OPS[key] = node.target
        node.args = (key, *node.args)
        node.target = _fake_compiled_op
    graph.lint()
    graph_module.recompile()


def install_regional_inductor_stub() -> bool:
    """Install the executable stub when the real module is unavailable.

    Returns:
        ``True`` when a real or stubbed regional_inductor is importable.
    """
    module_name = "torch.fx.passes.regional_inductor"
    if module_name in sys.modules:
        return True
    try:
        available = importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        available = False
    if available:
        return True

    stub = types.ModuleType(module_name)
    stub.regional_inductor = _fake_regional_inductor
    sys.modules[module_name] = stub
    setattr(fx_passes, "regional_inductor", stub)
    return True
