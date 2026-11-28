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
"""Pytest hooks for ``tests/ut``."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import TYPE_CHECKING, Iterator
from unittest import mock

import pytest

from tests.ut import _gate_runner

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter
    from xdist.workermanage import WorkerController

# transformers v4 chains ``modeling_utils -> loss_utils -> loss_d_fine ->
# loss_for_object_detection -> image_transforms`` and imports ``tensorflow``
# whenever it is importable. The UT gate image ships a tensorflow build whose
# native preload (self_check) segfaults the ARM executor, so a plain
# ``from transformers import PreTrainedModel`` kills the whole collection.
# ``USE_TF=0`` makes transformers' ``is_tf_available()`` return False and
# skips that chain; nothing in this suite uses tensorflow.
os.environ.setdefault("USE_TF", "0")

# UT tensors are tiny, so OpenMP fan-out/join dominates the actual math on
# many-core gate executors (a 70s case drops to 3s with 4 threads on a
# 256-core host). Cap intra-op threads before torch initializes its pool.
os.environ.setdefault("OMP_NUM_THREADS", "4")


class NoAcceleratorGuard:
    """Fail fast when a CPU-gated test accidentally reaches for a device.

    Patches CUDA device selection, NCCL/HCCL process-group initialization and
    ``torch_npu`` entry points to raise immediately, so "accidentally on
    accelerator" can never pass as an ordinary CPU test. Opt in via the
    ``no_accelerator`` fixture or use as a context manager.
    """

    _DEVICE_TARGETS = (
        "torch.cuda.init",
        "torch.cuda.set_device",
    )
    _FORBIDDEN_BACKENDS = {"nccl", "hccl"}

    def __init__(self):
        self._patchers = []

    @staticmethod
    def _fail(name):
        def _blocked(*args, **kwargs):
            raise RuntimeError(
                f"{name} reached from a CPU-gated test; Gate-1 forbids real "
                "accelerator devices and process groups"
            )

        return _blocked

    def _check_init_process_group(self, real_init):
        """Wrap ``init_process_group`` so forbidden backends fail fast."""

        def _guarded(*args, **kwargs):
            backend = kwargs.get("backend")
            if backend is None and args:
                backend = args[0]
            if isinstance(backend, str) and backend.lower() in self._FORBIDDEN_BACKENDS:
                raise RuntimeError(
                    f"init_process_group(backend={backend!r}) reached from a "
                    "CPU-gated test; Gate-1 forbids real accelerator process groups"
                )
            return real_init(*args, **kwargs)

        return _guarded

    def start(self) -> None:
        """Install the guard patches."""
        for target in self._DEVICE_TARGETS:
            patcher = mock.patch(target, self._fail(target))
            patcher.start()
            self._patchers.append(patcher)
        import torch.distributed  # pylint: disable=import-outside-toplevel

        if hasattr(torch.distributed, "init_process_group"):
            patcher = mock.patch(
                "torch.distributed.init_process_group",
                self._check_init_process_group(torch.distributed.init_process_group),
            )
            patcher.start()
            self._patchers.append(patcher)

    def stop(self) -> None:
        """Restore every patched entry point."""
        while self._patchers:
            self._patchers.pop().stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


@pytest.fixture
def no_accelerator() -> Iterator[NoAcceleratorGuard]:
    """Activate :class:`NoAcceleratorGuard` for one test."""
    guard = NoAcceleratorGuard()
    guard.start()
    try:
        yield guard
    finally:
        guard.stop()


def _preload_libgomp_early_for_static_tls() -> None:
    """Preload ``libgomp`` so later OpenMP consumers do not exhaust static TLS.

    ``pytest_configure`` imports PyTorch first; without an early global load of
    OpenMP, a later native extension that links OpenMP can fail with:

        ImportError: libgomp.so.1: cannot allocate memory in static TLS block

    Set ``HYPER_PARALLEL_SKIP_LIBGOMP_PRELOAD=1`` to disable.  Optional
    ``HYPER_PARALLEL_LIBGOMP_PATH`` selects a specific shared object (e.g. CI).
    """
    if os.environ.get("HYPER_PARALLEL_SKIP_LIBGOMP_PRELOAD", ""):
        return
    candidates: list[str] = []
    env_path = os.environ.get("HYPER_PARALLEL_LIBGOMP_PATH", "").strip()
    if env_path:
        candidates.append(env_path)
    found = ctypes.util.find_library("gomp")
    if found:
        candidates.append(found)
    candidates.append("libgomp.so.1")
    for path in candidates:
        if not path:
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_libgomp_early_for_static_tls()


def pytest_configure(config: pytest.Config) -> None:  # pylint: disable=unused-argument
    """Initialize the DTensor backend before test collection.

    Args:
        config: Active pytest configuration.
    """
    import torch  # pylint: disable=import-outside-toplevel

    # Enforce the cap even when torch was imported before this conftest
    # (environment variables are only read at pool creation).
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    _restore_cpu_dataloader_iter_init(torch)
    import hyper_parallel.core.dtensor.dtensor  # noqa: F401 pylint: disable=unused-import,C0415


def _restore_cpu_dataloader_iter_init(torch) -> None:
    """Undo torch_npu's DataLoader patch when no NPU device is usable.

    ``torch_npu`` unconditionally replaces ``_MultiProcessingDataLoaderIter.__init__``
    with a wrapper that calls ``_npu_set_thread_affinity``; with the toolkit
    sourced but no visible device (CPU gate, ``ASCEND_RT_VISIBLE_DEVICES=""``)
    that ACL call raises and every multi-worker DataLoader fails. These are
    CPU-gated tests, so put the original initializer back.
    """
    try:
        import torch_npu  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    try:
        if torch_npu.npu.is_available():
            return
        from torch_npu.utils import _module as npu_module  # pylint: disable=import-outside-toplevel
        original = getattr(npu_module, "origin_mpdl_iter_init", None)
        if original is not None:
            torch.utils.data.dataloader._MultiProcessingDataLoaderIter.__init__ = original  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        # Never let an optional accelerator integration break CPU collection.
        return


def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    """Report per-case phase times in verbose runs and on Jenkins executors.

    Args:
        terminalreporter: Reporter containing completed test phases.
    """
    if terminalreporter.verbosity < 1 and not os.environ.get("JENKINS_URL"):
        return

    durations = {}
    for reports in terminalreporter.stats.values():
        for report in reports:
            # Subtest reports share the parent nodeid; the enclosing call already includes their time.
            if isinstance(report, pytest.TestReport) and not hasattr(report, "context"):
                durations.setdefault(report.nodeid, {})[report.when] = report.duration

    terminalreporter.write_sep("=", "UT case timings (seconds): setup call teardown nodeid")
    for nodeid, phases in sorted(durations.items()):
        fields = [f"{phases[phase]:.6f}" if phase in phases else "-" for phase in ("setup", "call", "teardown")]
        terminalreporter.write_line(f"UT_TIMING {' '.join(fields)} {nodeid}")


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> int | None:
    """Run the Jenkins full-UT invocation with the project's hybrid file split.

    Args:
        config: Active pytest configuration.
    """
    return _gate_runner.run_full_gate(config)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Record whether serial and parallel workers are collecting coverage.

    Args:
        session: Active pytest test session.
    """
    phase = os.environ.get(_gate_runner.PHASE)
    if phase:
        details = _gate_runner.coverage_details()
        if hasattr(session.config, "workeroutput"):
            session.config.workeroutput["hp_ut_coverage"] = details
        else:
            print(f"UT_COVERAGE {phase} {details}", flush=True)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: WorkerController, error: object) -> None:  # pylint: disable=unused-argument
    """Expose each xdist worker's coverage collector in the controller log.

    Args:
        node: Worker controller reporting shutdown.
        error: Worker shutdown error, when present.
    """
    if os.environ.get(_gate_runner.PHASE):
        print(f"UT_COVERAGE {node.gateway.id} {getattr(node, 'workeroutput', {}).get('hp_ut_coverage')}", flush=True)
