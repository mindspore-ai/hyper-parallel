# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Unit tests for OpDispatcher with custom distributed ops (e.g., StackExt).
"""
import importlib
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pytest

from hyper_parallel.core.layout import Layout
from hyper_parallel.core.dtensor import DTensor

_TEST_FILE_DIR = Path(__file__).resolve().parent
_TESTS_ROOT_DIR = _TEST_FILE_DIR.parent.parent.parent.parent
_CUSTOM_OPS_DIR = _TESTS_ROOT_DIR / "tests" / "custom_ops"

HYPER_PARALLEL_OPS_YAML_DIR = str(_CUSTOM_OPS_DIR)
HYPER_PARALLEL_OPS_PYTHON_PATH = str(_CUSTOM_OPS_DIR)

_YAML_DIR_ENV_KEYS = ("HP_TEST_OPS_YAML_DIR", "HYPER_PARALLEL_OPS_YAML_DIR")
_PY_PATH_ENV_KEYS = ("HP_TEST_OPS_PYTHON_PATH", "HYPER_PARALLEL_OPS_PYTHON_PATH")


def _first_env(*keys: str) -> Optional[str]:
    """
    Feature: Read first available environment variable
    Description: Iterate over provided environment variable keys and return the first non-empty value.
    Expectation: Returns a string value when found; otherwise returns None.
    """
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return None


def _normalize_yaml_dir(yaml_dir_or_file: str) -> Path:
    """
    _op_dispatch.safe_load_yaml_from_dir() expects a DIRECTORY containing *.yaml.
    If user passes a YAML file path, accept it and use its parent directory.
    """
    p = Path(yaml_dir_or_file).expanduser().resolve(strict=False)
    if p.suffix.lower() in (".yml", ".yaml"):
        return p.parent
    return p


def _validate_paths(yaml_dir: Path, python_path: str) -> None:
    """
    Feature: Validate custom ops YAML directory and Python path
    Description: Ensure yaml_dir exists and is a directory, and python_path contains at least one
                 existing directory (split by ':').
    Expectation: Raises ValueError when validation fails; otherwise returns None.
    """
    if not yaml_dir.exists() or not yaml_dir.is_dir():
        raise ValueError(
            f"Invalid yaml directory path: {yaml_dir}\n"
            f"Expected a DIRECTORY containing *.yaml files.\n"
        )

    py_dirs = [
        Path(x).expanduser().resolve(strict=False) for x in python_path.split(":") if x
    ]
    if not py_dirs or not any(d.exists() and d.is_dir() for d in py_dirs):
        raise ValueError(
            f"Invalid python path: {python_path}\n"
            f"Expected at least one existing DIRECTORY.\n"
        )


def _get_custom_paths_from_env() -> Tuple[Path, str]:
    """
    Feature: Resolve custom ops search paths from environment variables
    Description: Read YAML directory and Python import search path from the first available
                 environment variables in _YAML_DIR_ENV_KEYS / _PY_PATH_ENV_KEYS, normalize
                 YAML path (file->parent dir), and validate both paths exist.
    Expectation: Returns (yaml_dir, python_path) when env vars are present and valid;
                 otherwise raises RuntimeError/ValueError.
    """
    yaml_raw = _first_env(*_YAML_DIR_ENV_KEYS)
    py_raw = _first_env(*_PY_PATH_ENV_KEYS)

    if not yaml_raw or not py_raw:
        raise RuntimeError(
            "Missing env vars for custom ops paths.\n"
            "Set either:\n"
            "  HP_TEST_OPS_YAML_DIR and HP_TEST_OPS_PYTHON_PATH\n"
            "or:\n"
            "  HYPER_PARALLEL_OPS_YAML_DIR and HYPER_PARALLEL_OPS_PYTHON_PATH\n"
        )

    yaml_dir = _normalize_yaml_dir(yaml_raw)
    python_path = py_raw
    _validate_paths(yaml_dir, python_path)
    return yaml_dir, python_path


def _reload_op_dispatch_with_env(
    monkeypatch: pytest.MonkeyPatch, yaml_dir: str, python_path: str
):
    """
    MUST set env BEFORE importing _op_dispatch because it creates _OP_DISPATCHER at import time.
    """
    monkeypatch.setenv("HYPER_PARALLEL_OPS_YAML_DIR", yaml_dir)
    monkeypatch.setenv("HYPER_PARALLEL_OPS_PYTHON_PATH", python_path)

    target_mod = "hyper_parallel.core.shard._op_dispatch"
    if target_mod in sys.modules:
        del sys.modules[target_mod]

    mod = importlib.import_module(target_mod)
    mod = importlib.reload(mod)
    return mod


def _require_mindspore():
    """
    Feature: Conditional dependency gate for MindSpore
    Description: Check MindSpore availability at runtime; skip tests when not installed.
    Expectation: Test is skipped (pytest.skip) if MindSpore cannot be imported.
    """
    try:
        importlib.import_module("mindspore")
    except Exception as e:
        pytest.skip(f"mindspore not available: {e}")


base_mesh_shape = (1, 1, 1)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = [0]


def _make_layout_2d_replicated():
    """
    2D tensor layout, fully replicated: ("None", "None")
    Uses the same Layout(...) + layout(...) pattern as your elementwise UT.
    """
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)
    return layout("None", "None")


def _make_dtensors_ms():
    """
    Feature: Create MindSpore DTensor inputs for StackExt dispatch tests
    Description: Create two 2x3 MindSpore tensors, wrap them into DTensor with a fully
                 replicated 2D layout.
    Expectation: Returns (d0, d1) where both are DTensor and share the same replicated layout.
    """
    _require_mindspore()

    np = importlib.import_module("numpy")
    ms = importlib.import_module("mindspore")

    x0 = ms.Tensor(np.arange(6).reshape(2, 3), ms.int32)
    x1 = ms.Tensor(np.arange(6, 12).reshape(2, 3), ms.int32)

    x_layout = _make_layout_2d_replicated()
    d0 = DTensor.from_local(x0, x_layout)
    d1 = DTensor.from_local(x1, x_layout)
    return d0, d1


def _stack_ext_local_ms(x0, x1, axis: int):
    """
    Local op implementation simulating StackExt(x0, x1, axis).
    """
    _require_mindspore()
    ops = importlib.import_module("mindspore.ops")

    if hasattr(ops, "stack"):
        return ops.stack([x0, x1], axis)
    return ops.Stack(axis)([x0, x1])


def _to_numpy_ms(x):
    """
    Feature: Convert MindSpore tensor-like to numpy-like
    Description: Use asnumpy() when available; otherwise return the input.
    Expectation: Returns a numpy array for MindSpore tensors, or the original object.
    """
    return x.asnumpy() if hasattr(x, "asnumpy") else x


def _patch_op_name(monkeypatch, op_dispatch_mod):
    """
    Ensure platform.get_op_name(_stack_ext_local_ms) == "StackExt".
    """
    orig_get_op_name = op_dispatch_mod.platform.get_op_name

    def patched_get_op_name(func):
        if func is _stack_ext_local_ms:
            return "StackExt"
        return orig_get_op_name(func)

    monkeypatch.setattr(
        op_dispatch_mod.platform, "get_op_name", patched_get_op_name, raising=True
    )


def test_stack_ext_dispatch_and_layout(monkeypatch):
    """
    Feature: OpDispatcher dispatch and StackExt layout inference integration
    Description: Reload _op_dispatch with custom YAML/Python paths from environment, create two replicated
                 2D DTensor inputs, patch platform.get_op_name so the local function maps to "StackExt",
                 and dispatch via _OP_DISPATCHER.dispatch using axis=0.
    Expectation: dispatch returns a DTensor whose output layout tensor_map inserts a replicated dimension
                 at axis=0 (i.e., from (-1, -1) to (-1, -1, -1)), and the numerical result matches the
                 local MindSpore stack reference output.
    """
    op_dispatch = _reload_op_dispatch_with_env(
        monkeypatch, HYPER_PARALLEL_OPS_YAML_DIR, HYPER_PARALLEL_OPS_PYTHON_PATH
    )

    d0, d1 = _make_dtensors_ms()
    axis = 0

    _patch_op_name(monkeypatch, op_dispatch)

    out = op_dispatch._OP_DISPATCHER.dispatch(  # pylint: disable=protected-access
        _stack_ext_local_ms, (d0, d1, axis), {}
    )

    assert isinstance(out, DTensor)

    # The input 2D replication tensor map should be (-1, -1)
    assert tuple(d0.layout.to_dict()["tensor_map"]) == (-1, -1)

    # axis=0: outputs rank=3, with a replicated inserted at the axis position => (-1, -1, -1)
    assert tuple(out.layout.to_dict()["tensor_map"]) == (-1, -1, -1)

    # The number is correct
    ref = _stack_ext_local_ms(d0.to_local(), d1.to_local(), axis)
    assert (_to_numpy_ms(out.to_local()) == _to_numpy_ms(ref)).all()


def test_stack_ext_layout_cache(monkeypatch):
    """
    Feature: LayoutCacheManager effectiveness for StackExt infer_layout
    Description: Reload _op_dispatch with custom YAML/Python paths, dispatch the same StackExt call twice
                 with identical inputs and axis=1, and wrap the distributed op infer_layout to count calls.
    Expectation: infer_layout is invoked exactly once due to layout cache hit on the second dispatch
                 (call_count["n"] == 1).
    """
    op_dispatch = _reload_op_dispatch_with_env(
        monkeypatch, HYPER_PARALLEL_OPS_YAML_DIR, HYPER_PARALLEL_OPS_PYTHON_PATH
    )

    d0, d1 = _make_dtensors_ms()
    axis = 1

    _patch_op_name(monkeypatch, op_dispatch)

    dist_op = op_dispatch.LayoutCacheManager.get_instance().distributed_op("StackExt")

    call_count = {"n": 0}
    orig_infer = dist_op.infer_layout

    def wrapped_infer(layouts, extra_args):
        call_count["n"] += 1
        return orig_infer(layouts, extra_args)

    monkeypatch.setattr(dist_op, "infer_layout", wrapped_infer, raising=True)

    _ = op_dispatch._OP_DISPATCHER.dispatch(  # pylint: disable=protected-access
        _stack_ext_local_ms, (d0, d1, axis), {}
    )
    _ = op_dispatch._OP_DISPATCHER.dispatch(  # pylint: disable=protected-access
        _stack_ext_local_ms, (d0, d1, axis), {}
    )

    assert call_count["n"] == 1
