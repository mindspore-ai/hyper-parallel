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
"""Child-process entry: this file is launched by torchrun/msrun via pytest.

Pytest collects ``test_suite_entry`` and runs it once per rank. The function
reads the group config from environment variables (set by the parent
``RUNNER.run_group``), executes each case in turn, isolates per-case errors,
and emits a structured jsonl report.
"""
import os
import time
import traceback
from typing import List

from tests.shard_ops.framework.backend import resolve_backend
from tests.shard_ops.framework.case_spec import OpShardCase
from tests.shard_ops.framework.registry import load_cases_from_package
from tests.shard_ops.framework.reporter import Reporter
from tests.shard_ops.framework.utils import (
    ENV_CASE_NAMES,
    ENV_CASES_PKG,
    ENV_DEVICE_TYPE,
    ENV_FAIL_FAST,
    ENV_FRAMEWORK,
    ENV_MESH_NAMES,
    ENV_MESH_SHAPE,
    ENV_REPORT_DIR,
    parse_int_tuple,
    parse_str_tuple,
)


_PLATFORM_FRAMEWORK_PKG = {
    "torch": "tests.torch.shard.ops.framework",
    "mindspore": "tests.mindspore.st.shard.ops.framework",
}


def _ensure_backend_registered(framework: str) -> None:
    """Trigger platform-specific backend registration in the child process."""
    pkg = _PLATFORM_FRAMEWORK_PKG.get(framework)
    if pkg is None:
        raise RuntimeError(f"unknown framework: {framework!r}")
    __import__(pkg)


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"missing env var {key}. Did you launch via "
            f"tests.shard_ops.framework.runner.RUNNER?"
        )
    return val


def _select_cases(cases_pkg: str, names: List[str]) -> List[OpShardCase]:
    all_cases = {c.name: c for c in load_cases_from_package(cases_pkg)}
    missing = [n for n in names if n not in all_cases]
    if missing:
        raise RuntimeError(
            f"cases listed in env not found in {cases_pkg}: {missing}"
        )
    return [all_cases[n] for n in names]


def _rank() -> int:
    """Best-effort rank lookup that works for torchrun and msrun.

    torchrun exports ``RANK``; msrun exports ``RANK_ID`` (and several other
    aliases depending on the launcher version). Fall back to 0 on local
    plain-pytest runs.
    """
    for key in ("RANK", "RANK_ID", "MS_NODE_ID", "OMPI_COMM_WORLD_RANK"):
        val = os.environ.get(key)
        if val is not None and val.strip() != "":
            try:
                return int(val)
            except ValueError:
                continue
    return 0


def _run_one(case: OpShardCase, backend, mesh) -> None:
    """The common flow: build → derive → ref → distribute → run → gather → assert."""
    # 1. Materialise full tensors on device.
    full_tensors = [backend.make_tensor(spec) for spec in case.inputs]

    # 1b. Compute derived inputs ONCE on the full tensors (e.g. attention stats
    # needing global K). They are appended after the primary tensors.
    derived_full = [d.fn(*full_tensors) for d in case.derived_inputs]

    # 2. Standalone reference.
    if case.needs_mesh:
        expected = case.fn(mesh, *full_tensors, *derived_full,
                           *case.extra_inputs, **case.kwargs)
    else:
        expected = case.fn(*full_tensors, *derived_full,
                           *case.extra_inputs, **case.kwargs)

    # 3. Distribute inputs across the mesh.
    dist_inputs = [
        backend.distribute(t, mesh, p)
        for t, p in zip(full_tensors, case.placements)
    ]
    # 3b. Distribute the derived inputs — the same data, only sliced (never
    # recomputed on shards).
    dist_derived = [
        backend.distribute(t, mesh, d.placement)
        for t, d in zip(derived_full, case.derived_inputs)
    ]

    # 4. Parallel execution.
    if case.needs_mesh:
        actual = case.fn(mesh, *dist_inputs, *dist_derived,
                         *case.extra_inputs, **case.kwargs)
    else:
        actual = case.fn(*dist_inputs, *dist_derived,
                         *case.extra_inputs, **case.kwargs)

    # 5. Gather and compare. Some ops return tuples (e.g. sort) — compare
    # element-wise to keep the per-tensor diagnosis intact.
    _gather_and_assert(expected, actual, backend, case)


def _gather_and_assert(expected, actual, backend, case) -> None:
    """Gather distributed output and compare against expected.

    When ``case.compare_outputs`` is set and the output is a tuple/list,
    only the specified indices are compared (others are skipped).
    """
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError(
                f"{case.name}: tuple structure mismatch "
                f"(expected {type(expected).__name__} len={len(expected)}, "
                f"got {type(actual).__name__})"
            )
        indices = case.compare_outputs if case.compare_outputs is not None else range(len(expected))
        for i in indices:
            e, a = expected[i], actual[i]
            if e is None and a is None:
                continue
            gathered = backend.local_to_global(a)
            try:
                backend.assert_close(e, gathered, case.compare)
            except AssertionError as exc:
                raise AssertionError(f"{case.name}[output#{i}]: {exc}") from exc
        return
    gathered = backend.local_to_global(actual)
    backend.assert_close(expected, gathered, case.compare)


def test_suite_entry() -> None:
    """Pytest entry point inside the launcher."""
    framework = _require_env(ENV_FRAMEWORK)
    device_type = _require_env(ENV_DEVICE_TYPE)
    cases_pkg = _require_env(ENV_CASES_PKG)
    case_names = _require_env(ENV_CASE_NAMES).split(",")
    mesh_shape = parse_int_tuple(_require_env(ENV_MESH_SHAPE))
    mesh_names = parse_str_tuple(_require_env(ENV_MESH_NAMES))
    report_dir = _require_env(ENV_REPORT_DIR)
    fail_fast = os.environ.get(ENV_FAIL_FAST, "0") == "1"

    _ensure_backend_registered(framework)
    backend = resolve_backend(framework, device_type)
    backend.maybe_init_dist()
    mesh = backend.get_or_init_mesh(mesh_shape, mesh_names)

    cases = _select_cases(cases_pkg, case_names)
    reporter = Reporter(report_dir, rank=_rank())

    failures: List[str] = []
    group_broken = False
    stopped_early = False
    try:
        for case in cases:
            if group_broken or stopped_early:
                reason = "group broken" if group_broken else "fail_fast: prior failure"
                reporter.skip(case.name, reason)
                continue
            t0 = time.perf_counter()
            try:
                _run_one(case, backend, mesh)
                reporter.pass_(case.name, time.perf_counter() - t0)
            except Exception as exc:  # pylint: disable=W0718
                reporter.fail(case.name, exc, traceback.format_exc())
                failures.append(case.name)
                if fail_fast:
                    stopped_early = True
                    continue
                if not isinstance(exc, AssertionError):
                    # Only non-assertion (likely comm) errors trigger the
                    # group health probe; a numeric mismatch leaves the
                    # comm group fine and the next case can still run.
                    if not backend.recover_after_failure():
                        group_broken = True
    finally:
        reporter.close()

    if failures or group_broken:
        raise AssertionError(
            f"{len(failures)} case(s) failed; group_broken={group_broken}; "
            f"stopped_early={stopped_early}; failed={failures}"
        )
