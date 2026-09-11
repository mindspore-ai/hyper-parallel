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
"""Pytest entry for the SHMEM C++ unit test binaries.

By default a session-scoped fixture (re)builds the four binaries from
``tests/ut/core/multicore/shmem/cpp/CMakeLists.txt`` into
``build/shmem_ut`` on the host (no CANN/NPU required). The build is a plain
CMake incremental build: unchanged sources cost a near-instant no-op, while
edited sources are recompiled by dependency tracking, so stale artifacts can
never mask a regression. Set ``HP_SHMEM_NATIVE_TEST_DIR`` to reuse prebuilt
binaries instead (the automatic build is then skipped). Without CMake these
tests are skipped; a failing build or case is a hard failure.
"""

import contextlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark

# (binary, case) inventory. The first three binaries dispatch via --case; the
# reinit binary takes the case as a positional argument.
_CORE_CASES = [
    "test_make_chunk_plan_chunk_count",
    "test_divide_aligned_capacity",
    "test_get_chunk_tail_rebalance",
    "test_partition_work_remainder_distribution",
    "test_registry_match_for_free_requires_complete",
    "test_registry_double_free_detected",
    "test_registry_resolve_allows_subview",
    "test_registry_diagnostics_are_ordered_and_track_high_watermark",
    "test_config_strict_unsigned_decimal",
    "test_config_endpoint_minimal_and_passthrough",
    "test_config_data_engine_validation",
    "test_runtime_free_device_guard_phase_classification",
    "test_runtime_free_failure_preserves_record",
    "test_runtime_initialize_failure_allows_retry",
    "test_runtime_shutdown_uninitialized_idempotent",
]
_HOST_ADAPTER_CASES = [
    "test_host_init_attr_fields_and_order",
    "test_host_timeout_and_cann_error_mapping",
    "test_host_alloc_nullptr_and_free_passthrough",
    "test_host_stream_operations_forward_arguments",
]
_DEVICE_SURFACE_CASES = [
    "test_put_get_forward_once",
    "test_put_signal_orders_data_before_signal",
    "test_signal_compare_mapping",
]
_REINIT_CASES = [
    "clean_reinit",
    "active_allocation",
    "initialize_failure",
    "finalize_failure",
]

_BINARY_CASES = (
    [("hp_shmem_core_test", case) for case in _CORE_CASES]
    + [("hp_shmem_host_adapter_test", case) for case in _HOST_ADAPTER_CASES]
    + [("hp_shmem_device_surface_test", case) for case in _DEVICE_SURFACE_CASES]
    + [("hp_shmem_reinit_test", case) for case in _REINIT_CASES]
)

_CASE_IDS = [f"{binary}:{case}" for binary, case in _BINARY_CASES]

_BINARY_DIR_ENV = "HP_SHMEM_NATIVE_TEST_DIR"
_CASE_TIMEOUT_SECONDS = 60
_BUILD_TIMEOUT_SECONDS = 600

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CMAKE_SOURCE_DIR = Path(__file__).resolve().parent / "cpp"
_DEFAULT_BUILD_DIR = _REPO_ROOT / "build" / "shmem_ut"


@contextlib.contextmanager
def _exclusive_build_lock(lock_path: Path):
    """Serialize the build across pytest-xdist workers on the same host."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            import fcntl  # pylint: disable=import-outside-toplevel
        except ImportError:
            # Non-POSIX: no flock; local runs are single-process, best-effort.
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_build_command(argv: list[str]) -> None:
    print(f"[BUILD] {shlex.join(argv)}", flush=True)
    completed = subprocess.run(argv, timeout=_BUILD_TIMEOUT_SECONDS, check=False)
    assert completed.returncode == 0, (
        f"SHMEM C++ UT build command must exit with code 0, but got {completed.returncode}\n"
        f"command: {shlex.join(argv)}"
    )


@pytest.fixture(scope="session")
def cpp_binary_dir() -> Path:
    """Return the directory with the four native test binaries.

    With ``HP_SHMEM_NATIVE_TEST_DIR`` set, use those prebuilt binaries as-is.
    Otherwise run one incremental CMake build per pytest session (guarded by
    a file lock so xdist workers do not build concurrently).
    """
    configured = os.environ.get(_BINARY_DIR_ENV)
    if configured:
        return Path(configured)
    if shutil.which("cmake") is None:
        pytest.skip(
            "cmake is unavailable on this host; install CMake and a C++17 "
            "compiler, or point HP_SHMEM_NATIVE_TEST_DIR at prebuilt binaries"
        )

    build_dir = _DEFAULT_BUILD_DIR
    with _exclusive_build_lock(build_dir.with_suffix(".lock")):
        _run_build_command(["cmake", "-S", str(_CMAKE_SOURCE_DIR), "-B", str(build_dir)])
        _run_build_command(["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 4)])
    return build_dir


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize(("binary", "case"), _BINARY_CASES, ids=_CASE_IDS)
def test_cpp_case(binary: str, case: str, cpp_binary_dir: Path) -> None:
    binary_name = f"{binary}.exe" if os.name == "nt" else binary
    binary_path = cpp_binary_dir / binary_name
    assert binary_path.is_file(), (
        f"SHMEM C++ UT binary must exist: {binary_path} "
        f"(build tests/ut/core/multicore/shmem/cpp with CMake first)"
    )

    argv = [str(binary_path), case] if binary == "hp_shmem_reinit_test" else [str(binary_path), "--case", case]
    # Per-case execution log: captured by pytest and shown only on failure,
    # so a failing case report always carries the exact reproducer command.
    print(f"[RUN] {binary}:{case}")
    print(f"[CMD] {shlex.join(argv)}")
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=_CASE_TIMEOUT_SECONDS, check=False)
    print(f"[EXIT] {binary}:{case} exit_code={completed.returncode}")
    if completed.stdout:
        print(f"[STDOUT]\n{completed.stdout}")
    if completed.stderr:
        print(f"[STDERR]\n{completed.stderr}")
    assert completed.returncode == 0, (
        f"{binary}:{case} must exit with code 0, but got {completed.returncode}\n"
        f"reproduce with: {shlex.join(argv)}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
