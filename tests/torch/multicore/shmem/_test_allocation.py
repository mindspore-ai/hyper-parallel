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
"""NPU workers for SHMEM Allocation and Tensor lifecycle system tests."""

import gc

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from tests.torch.multicore.shmem._worker_utils import ACQUIRE_HINT, acquire_runtime, release_runtime


def test_binding_allocation_and_release() -> None:
    """Exercise ordinary/aligned Allocation and final Runtime release."""
    acquire_runtime()
    regular = shmem.empty((256,), dtype=torch.uint8)
    aligned = shmem.empty((256,), dtype=torch.uint8, alignment=512)
    assert aligned.data_ptr() % 512 == 0, (
        f"aligned SHMEM pointer is not 512-byte aligned: address={aligned.data_ptr()}"
    )
    shmem.free(regular)
    shmem.free(aligned)
    release_runtime()


def test_binding_storage_assoc_double_free(capfd) -> None:
    """Report DOUBLE_FREE for stale Storage without invalidating a later Allocation."""
    acquire_runtime()
    allocation_a = shmem.empty(64, dtype=torch.uint8)
    shmem.free(allocation_a)
    with pytest.raises(RuntimeError, match="DOUBLE_FREE"):
        shmem.free(allocation_a)
    del allocation_a
    gc.collect()
    assert "orphaned=true" not in capfd.readouterr().err, (
        "Released Storage must not be reported as an active orphan after a repeated free is rejected"
    )

    allocation_b = shmem.empty(64, dtype=torch.uint8)
    shmem.free(allocation_b)
    release_runtime()


def test_binding_orphan_storage_warning(capfd) -> None:
    """Report an active Allocation when its final Tensor Storage reference is lost."""
    acquire_runtime()
    allocation = shmem.empty(64, dtype=torch.uint8)
    active = shmem.debug_state()["active_allocations"]
    assert isinstance(active, list) and len(active) == 1, f"expected one active Allocation, got {active!r}"
    allocation_id = active[0]["allocation_id"]
    allocation_base = active[0]["allocation_base"]

    subview = allocation[1:]
    with pytest.raises(RuntimeError, match="complete active Allocation"):
        shmem.free(subview)
    del subview
    del allocation
    gc.collect()

    error = capfd.readouterr().err
    assert error.count("orphaned=true") == 1, f"expected one orphan warning, got stderr={error!r}"
    assert f"allocation_id={allocation_id}" in error, f"orphan warning omitted Allocation ID: {error!r}"
    assert f"allocation_base=0x{allocation_base:x}" in error, f"orphan warning omitted Allocation base: {error!r}"
    assert "allocation_bytes=64" in error, f"orphan warning omitted Allocation size: {error!r}"
    assert "Call shmem.free(tensor) before overwriting or dropping" in error, (
        f"orphan warning omitted the preventive action: {error!r}"
    )
    assert "restart the process to recover" in error, f"orphan warning omitted the recovery action: {error!r}"

    state = shmem.debug_state()
    assert state["allocated_count"] == 1, f"orphan detection must not free the Allocation: state={state!r}"
    # This worker intentionally exits with the orphan active: there is no Tensor left from which a safe
    # collective free can be issued, and Runtime finalization must remain rejected rather than hide the defect.


def test_binding_stale_tensor_rejected_by_one_sided_ops() -> None:
    """Reject a freed Tensor passed as a one-sided operand before any RMA is enqueued."""
    acquire_runtime()
    rank = dist.get_rank()
    stale = shmem.empty(16, dtype=torch.uint8)
    local = torch.zeros(16, dtype=torch.uint8, device="npu")
    shmem.free(stale)

    # ResolveAllocation consults the active Allocation registry, so the stale Tensor is rejected before
    # any kernel launch; the released identity is reported as DOUBLE_FREE.
    with pytest.raises(RuntimeError, match="DOUBLE_FREE"):
        shmem.put(stale, local, rank)
    with pytest.raises(RuntimeError, match="DOUBLE_FREE"):
        shmem.get(local, stale, rank)
    with pytest.raises(RuntimeError, match="DOUBLE_FREE"):
        shmem.signal(stale[:4].view(torch.int32), 1, rank)

    release_runtime()


def test_binding_error_projection() -> None:
    """Expose all five stable Native error fields without Python rewriting."""
    acquire_runtime()
    with pytest.raises(RuntimeError) as error:
        shmem.empty(1, dtype=torch.uint8, alignment=3)
    message = str(error.value)
    expected_fields = (
        "error_code=INVALID_ARGUMENT",
        "operation=allocate",
        "phase=Validation",
        "cann_error_code=None",
        "message=",
    )
    missing_fields = [field for field in expected_fields if field not in message]
    assert not missing_fields, f"Native error projection missing fields: missing={missing_fields}, message={message}"
    assert ACQUIRE_HINT not in message, f"parameter error must not contain an acquire hint: message={message}"
    release_runtime()
