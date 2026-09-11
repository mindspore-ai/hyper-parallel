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
"""Lightweight torchrun launchers for SHMEM Torch Binding system tests."""

import os

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port


_TEST_DIR = os.path.dirname(__file__)
_LIFECYCLE_WORKER = os.path.join(_TEST_DIR, "_test_lifecycle.py")
_ALLOCATION_WORKER = os.path.join(_TEST_DIR, "_test_allocation.py")
_ONE_SIDED_WORKER = os.path.join(_TEST_DIR, "_test_one_sided.py")
_ALL_GATHER_WORKER = os.path.join(_TEST_DIR, "_test_all_gather.py")


def _run(worker: str, case_name: str, num_proc: int = 1) -> None:
    torchrun_case(worker, case_name, num_proc=num_proc)


def _set_unique_bootstrap_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_binding_allocation_and_release(monkeypatch) -> None:
    """Validate basic Allocation, explicit alignment, free, and final release."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ALLOCATION_WORKER, "test_binding_allocation_and_release")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_binding_single_process_without_distributed(monkeypatch) -> None:
    """Validate the one-PE SHMEM lifecycle without Torch distributed."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_LIFECYCLE_WORKER, "test_binding_single_process_without_distributed")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_binding_inactive_runtime_hint() -> None:
    """Give an actionable acquire hint when SHMEM capabilities are used directly."""
    _run(_LIFECYCLE_WORKER, "test_binding_inactive_runtime_hint")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_all_gather_output_input_overlap_rejected(monkeypatch) -> None:
    """Reject overlapping AllGather input and output ranges."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ALL_GATHER_WORKER, "test_all_gather_output_input_overlap_rejected")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_all_gather_output_size_mismatch_rejected(monkeypatch) -> None:
    """Reject an AllGather output with the wrong element count."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ALL_GATHER_WORKER, "test_all_gather_output_size_mismatch_rejected")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_binding_storage_assoc_double_free(monkeypatch) -> None:
    """Keep released Storage associated with its stale Allocation identity."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ALLOCATION_WORKER, "test_binding_storage_assoc_double_free")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_binding_orphan_storage_warning(monkeypatch) -> None:
    """Report but never implicitly free an active orphaned Allocation."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ALLOCATION_WORKER, "test_binding_orphan_storage_warning")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_binding_stale_tensor_rejected_by_one_sided_ops(monkeypatch) -> None:
    """Reject freed Tensors passed to one-sided put/get/signal before enqueue."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ALLOCATION_WORKER, "test_binding_stale_tensor_rejected_by_one_sided_ops")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_binding_error_projection(monkeypatch) -> None:
    """Preserve the Native Runtime error fields in the Python RuntimeError text."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ALLOCATION_WORKER, "test_binding_error_projection")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_binding_signal_and_pull(monkeypatch) -> None:
    """Validate Signal mapping and the source-notifies-target Pull handshake."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ONE_SIDED_WORKER, "test_binding_signal_and_pull", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_binding_reinit_with_different_heap_sizes(monkeypatch) -> None:
    """Validate clean Runtime reinitialization with independent Heap sizes."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_LIFECYCLE_WORKER, "test_binding_reinit_with_different_heap_sizes", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_binding_final_release_with_active_allocation_is_retryable(monkeypatch) -> None:
    """Validate final Runtime release rejection, retry, and the following lifecycle."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_LIFECYCLE_WORKER, "test_binding_final_release_with_active_allocation_is_retryable", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_binding_put_get(monkeypatch) -> None:
    """Validate Root PE Put and Get."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    _run(_ONE_SIDED_WORKER, "test_binding_put_get", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_binding_barrier_blocking_and_config(monkeypatch) -> None:
    """Validate barrier(blocking=False) same-Stream ordering and the debug_state config projection."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    # Pin non-default values so the worker's config assertions prove the Runtime echoes the
    # environment instead of its defaults.
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", str(64 * 1024 * 1024))
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_TIMEOUT_SEC", "60")
    _run(_ONE_SIDED_WORKER, "test_binding_barrier_blocking_and_config", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_binding_init_timeout_injection(monkeypatch) -> None:
    """Inject an unreachable bootstrap endpoint and a 1s timeout; expect a prompt error."""
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_TIMEOUT_SEC", "1")
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", "tcp://192.0.2.1:8662")
    _run(_LIFECYCLE_WORKER, "test_binding_init_timeout_injection")
