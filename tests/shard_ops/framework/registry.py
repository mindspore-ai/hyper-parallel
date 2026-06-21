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
"""Case collection: ``register`` / ``register_op_family`` / loader."""
import importlib
import inspect
import sys
from typing import Dict, List

from tests.shard_ops.framework.case_spec import (
    CaseSpec,
    InputSpec,
    OpShardCase,
    OpSpec,
)

# Module-level collector. cases/*.py call register/register_op_family at
# import time; suite.py drives the collector via load_cases_from_package().
_COLLECTOR: List[OpShardCase] = []

# Per-package cache so the second call to load_cases_from_package() does not
# silently return an empty list just because Python's module cache short-
# circuits the second importlib.import_module.
_PKG_CACHE: Dict[str, List[OpShardCase]] = {}


def _reset() -> None:
    _COLLECTOR.clear()


def _validate(case: OpShardCase) -> None:
    """Validate case name uniqueness and input/placement count match."""
    if not case.name:
        raise ValueError("OpShardCase.name must be non-empty")
    if len(case.inputs) != len(case.placements):
        raise ValueError(
            f"case {case.name!r}: inputs ({len(case.inputs)}) and "
            f"placements ({len(case.placements)}) length mismatch"
        )
    seen = {c.name for c in _COLLECTOR}
    if case.name in seen:
        raise ValueError(f"duplicate case name: {case.name!r}")


def register(case: OpShardCase) -> None:
    """Register a single ``OpShardCase``. Used directly for ad-hoc ops."""
    if not case.source_module:
        # Record the module that defined this case so CLI can later filter
        # by file path (e.g. ``case_sort.py::sort[2d_dp]``).
        frame = inspect.currentframe()
        while frame:
            mod = frame.f_globals.get("__name__", "")
            if mod != __name__:
                case.source_module = mod
                break
            frame = frame.f_back
    _validate(case)
    _COLLECTOR.append(case)


def register_op_family(ops: List[OpSpec], cases: List[CaseSpec]) -> None:
    """Cross-multiply ``ops × cases`` and register each combination.

    A case can opt-out via ``only_for`` / ``skip_for``. The resulting
    ``OpShardCase.name`` is ``"{op.name}[{case.name}]"`` so single-case
    selection via ``-k`` or ``--case`` keeps working.
    """
    for op in ops:
        for case in cases:
            if case.only_for and op.name not in case.only_for:
                continue
            if op.name in case.skip_for:
                continue
            register(_expand(op, case))


def _expand(op: OpSpec, case: CaseSpec) -> OpShardCase:
    merged = dict(op.default_input)
    if case.init_override is not None:
        merged["init"] = case.init_override
    inputs = [
        InputSpec(shape=tuple(s), seed=case.seed, **merged)
        for s in case.shapes
    ]
    return OpShardCase(
        name=f"{op.name}[{case.name}]",
        fn=op.fn,
        inputs=inputs,
        placements=list(case.placements),
        kwargs=dict(case.kwargs),
        extra_inputs=list(case.extra_inputs),
        compare=case.compare_override or op.default_compare,
        tags=op.tags,
    )


def load_cases_from_package(pkg_path: str,
                            force_reload: bool = False) -> List[OpShardCase]:
    """Import the package, triggering register/register_op_family in each
    ``case_*.py``. Returns a list of registered cases.

    Repeat calls hit a per-package cache so they stay cheap and consistent
    even though Python's module cache short-circuits the second import
    (which would otherwise yield an empty collector).
    """
    if not force_reload and pkg_path in _PKG_CACHE:
        return list(_PKG_CACHE[pkg_path])
    _reset()
    if force_reload:
        _drop_cached_submodules(pkg_path)
    importlib.import_module(pkg_path)
    collected = list(_COLLECTOR)
    _PKG_CACHE[pkg_path] = collected
    return list(collected)


def _drop_cached_submodules(pkg_path: str) -> None:
    """Remove ``pkg_path`` and all its descendants from ``sys.modules`` so
    a subsequent ``import_module`` re-executes ``case_*.py``.
    """
    prefix = pkg_path + "."
    stale = [m for m in sys.modules if m == pkg_path or m.startswith(prefix)]
    for m in stale:
        del sys.modules[m]
