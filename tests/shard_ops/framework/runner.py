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
"""Parent-side launcher: assemble TorchCase/MindSporeCase and parallel_run."""
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tests.common.parallel_case import (
    MindSporeCase, TorchCase, parallel_run, run_case,
)
from tests.shard_ops.framework.case_spec import OpShardCase
from tests.shard_ops.framework.reporter import summarize
from tests.shard_ops.framework.suite import GroupSpec
from tests.shard_ops.framework.utils import (
    ENV_CASE_NAMES,
    ENV_CASES_PKG,
    ENV_DEVICE_TYPE,
    ENV_FAIL_FAST,
    ENV_FRAMEWORK,
    ENV_MESH_NAMES,
    ENV_MESH_SHAPE,
    ENV_REPORT_DIR,
    find_free_port,
    slugify,
)

# Path to the child-process pytest entry. Pytest selects __suite_entry__ here.
_ENTRY_FILE = str(Path(__file__).resolve().parent / "entry.py")


def _ensure_fork_start_method() -> None:
    """The per-group env protocol relies on ``mp.Process`` forking from
    the parent so each child inherits the env set just before its own
    ``.start()``. Spawn would re-import everything and miss those env
    overrides. Linux defaults to fork, but other code in the test
    process might have flipped it.
    """
    current = mp.get_start_method(allow_none=True)
    if current is None:
        mp.set_start_method("fork", force=False)
        return
    if current != "fork":
        # Don't try to force a change once set — that raises. Surface a
        # warning so the symptom is debuggable if children behave oddly.
        print(
            f"WARNING: multiprocessing start_method is {current!r}; "
            "shard_ops expects 'fork' for per-group env inheritance",
            file=sys.stderr,
        )


def _pack_into_batches(
        groups: Iterable[GroupSpec],
        global_num_proc: int,
) -> List[List[GroupSpec]]:
    """First-fit packing of groups into device-budget-bounded batches.

    Groups are sorted by ``num_proc`` descending before first-fit so that
    larger groups pack together more efficiently (fewer wasted device
    slots). This does not affect intra-group JIT cache ordering — that is
    handled independently by case ordering within each bucket.
    """
    # Materialize and sort: larger num_proc groups first for best-fit packing.
    sorted_groups = sorted(groups, key=lambda g: g.num_proc, reverse=True)
    batches: List[List[GroupSpec]] = []
    sums: List[int] = []
    for g in sorted_groups:
        if g.num_proc > global_num_proc:
            raise ValueError(
                f"group {g.id} num_proc={g.num_proc} exceeds "
                f"global_num_proc={global_num_proc}"
            )
        placed = False
        for batch, used in zip(batches, sums):
            if used + g.num_proc <= global_num_proc:
                batch.append(g)
                sums[batches.index(batch)] = used + g.num_proc
                placed = True
                break
        if not placed:
            batches.append([g])
            sums.append(g.num_proc)
    return batches


def _report_dir_for(group: GroupSpec, framework: str, device_type: str,
                    fail_fast: bool) -> str:
    """Per-group report directory under the system tempdir.

    The ``ff``/``all`` token in the path separates the level0 (fail-fast)
    and level1 (run-all) suites so they never overwrite each other's
    reports inside the same pytest session.
    """
    ff_tag = "ff" if fail_fast else "all"
    base = os.path.join(
        tempfile.gettempdir(),
        f"hp_shard_ops_{os.getuid()}",
        f"{framework}-{device_type}",
        f"{ff_tag}-group_{group.id}_{slugify(group.cases_pkg)}",
    )
    if os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)
    return base


def _set_env(group: GroupSpec, framework: str, device_type: str,
             report_dir: str, fail_fast: bool) -> None:
    """Inject group config into the parent env. parallel_case spawns the
    child via subprocess which inherits this env."""
    os.environ[ENV_FRAMEWORK] = framework
    os.environ[ENV_DEVICE_TYPE] = device_type
    os.environ[ENV_CASES_PKG] = group.cases_pkg
    os.environ[ENV_CASE_NAMES] = ",".join(c.name for c in group.cases)
    os.environ[ENV_MESH_SHAPE] = ",".join(str(x) for x in group.mesh_shape)
    os.environ[ENV_MESH_NAMES] = ",".join(group.mesh_dim_names)
    os.environ[ENV_REPORT_DIR] = report_dir
    os.environ[ENV_FAIL_FAST] = "1" if fail_fast else "0"


class _Runner:
    """Single entry point used by both gate and local-reproduce paths."""

    def run_group(self, group: GroupSpec, framework: str,
                  device_type: str,
                  fail_fast: Optional[bool] = None) -> None:
        """Run all cases in ``group`` inside a single launcher process.

        ``fail_fast`` overrides ``group.fail_fast`` if provided. Typical
        usage: the suite entry sets it explicitly per pytest function so
        the same group plan can serve both level0 and level1 gates.
        """
        effective_ff = group.fail_fast if fail_fast is None else fail_fast
        report_dir = _report_dir_for(group, framework, device_type,
                                     effective_ff)
        _set_env(group, framework, device_type, report_dir, effective_ff)
        case = self._make_launcher_case(framework, group.num_proc)
        try:
            parallel_run([case])
        finally:
            # Always emit the summary, even on failure, so the failing case
            # is visible in pytest stdout.
            try:
                print(summarize(report_dir), file=sys.stderr)
            except Exception:  # pylint: disable=W0703
                pass

    def run_single(self, case: OpShardCase, framework: str,
                   device_type: str, num_proc: int = 4,
                   cases_pkg: str = "") -> None:
        """Local-only single case runner; same kernel as a 1-case group."""
        group = GroupSpec(
            id=abs(hash(case.name)) % 100000,
            cases=[case],
            mesh_shape=case.mesh_shape or (2, 2),
            mesh_dim_names=case.mesh_dim_names or ("dp", "tp"),
            num_proc=case.num_proc or num_proc,
            cases_pkg=cases_pkg,
        )
        self.run_group(group, framework, device_type)

    def run_groups(
            self,
            groups: Sequence[GroupSpec],
            framework: str,
            device_type: str,
            fail_fast: Optional[bool] = None,
            global_num_proc: int = 8,
            timeout: int = 900,
    ) -> None:
        """Run ``groups`` concurrently on disjoint device slices.

        Packs groups into device-budget-bounded batches and starts each
        batch's launchers in parallel. Within a batch every group runs
        in its own ``mp.Process`` (forked from the parent right after
        its env is set), so each child sees its own ENV_* values.

        Empty input is a no-op.
        """
        if not groups:
            return
        _ensure_fork_start_method()
        batches = _pack_into_batches(groups, global_num_proc)
        for batch in batches:
            self._run_batch(batch, framework, device_type, fail_fast, timeout)

    def _run_batch(
            self,
            batch: Sequence[GroupSpec],
            framework: str,
            device_type: str,
            fail_fast_override: Optional[bool],
            timeout: int,
    ) -> None:
        """Spawn one launcher per group in ``batch`` and wait."""
        report_dirs: List[tuple] = []
        processes: List[tuple] = []
        cursor = 0
        for group in batch:
            effective_ff = (
                group.fail_fast if fail_fast_override is None
                else fail_fast_override
            )
            report_dir = _report_dir_for(group, framework, device_type,
                                         effective_ff)
            report_dirs.append((group, report_dir))
            # Set env in the parent right before .start(); fork inherits
            # the current snapshot, so each child sees its own group's
            # configuration.
            _set_env(group, framework, device_type, report_dir, effective_ff)
            launcher_case = self._make_launcher_case(framework, group.num_proc)
            devices = list(range(cursor, cursor + group.num_proc))
            cursor += group.num_proc
            proc = mp.Process(target=run_case, args=(devices, launcher_case))
            proc.start()
            processes.append((proc, group, report_dir))

        failed: List[str] = []
        timed_out: List[str] = []
        try:
            for proc, group, _ in processes:
                proc.join(timeout=timeout)
                if proc.is_alive():
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.join()
                    timed_out.append(f"group_{group.id}")
                elif proc.exitcode != 0:
                    failed.append(f"group_{group.id}(exit={proc.exitcode})")
        finally:
            for _, report_dir in report_dirs:
                try:
                    print(summarize(report_dir), file=sys.stderr)
                except Exception:  # pylint: disable=W0703
                    pass

        if timed_out:
            raise AssertionError(
                f"groups timed out (possible collective deadlock): {timed_out}"
            )
        if failed:
            raise AssertionError(f"groups failed in batch: {failed}")

    @staticmethod
    def _make_launcher_case(framework: str, num_proc: int):
        """Create a TorchCase or MindSporeCase with an OS-assigned free port."""
        case_name = "test_suite_entry"
        # Always allocate an OS-assigned free port instead of relying on the
        # counter-based ``allocate_port``: the counter does not validate that
        # the port is actually free, and the residual TCPStore sockets left
        # behind by killed test runs can sit in the low 10000s range.
        master_port = find_free_port()
        if framework == "torch":
            return TorchCase(
                _ENTRY_FILE, case_name,
                master_port=master_port, num_proc=num_proc,
            )
        if framework == "mindspore":
            return MindSporeCase(
                _ENTRY_FILE, case_name,
                master_port=master_port,
                worker_num=num_proc, local_worker_num=num_proc, glog_v=2,
            )
        raise ValueError(f"unsupported framework: {framework!r}")


RUNNER = _Runner()
