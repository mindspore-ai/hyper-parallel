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


def test_binding_storage_invalidated_after_free(capfd) -> None:
    """Zero the shared Storage after free so stale Tensors and views stop exposing the address."""
    acquire_runtime()
    allocation_a = shmem.empty(64, dtype=torch.uint8)
    stale_view = allocation_a[1:]
    shmem.free(allocation_a)
    assert allocation_a.untyped_storage().nbytes() == 0
    assert stale_view.untyped_storage().nbytes() == 0
    assert allocation_a.data_ptr() == 0
    with pytest.raises(RuntimeError, match="invalidated by shmem.free"):
        shmem.free(allocation_a)
    del stale_view, allocation_a
    gc.collect()
    assert "orphaned=true" not in capfd.readouterr().err, (
        "Released Storage must not be reported as an active orphan after a repeated free is rejected"
    )
    allocation_b = shmem.empty(64, dtype=torch.uint8)
    shmem.free(allocation_b)
    release_runtime()


def test_binding_stale_tensor_cannot_corrupt_reused_allocation() -> None:
    """Keep a best-fit reused block intact when the stale Tensor is written after free."""
    acquire_runtime()
    stale = shmem.empty(64, dtype=torch.uint8)
    stale_view = stale[1:]
    reused_address = stale.data_ptr()
    shmem.free(stale)
    fresh = shmem.empty(64, dtype=torch.uint8)
    assert fresh.data_ptr() == reused_address, (
        f"best-fit reuse must return the freed block: fresh={fresh.data_ptr:#x}, reused={reused_address:#x}"
    )
    fresh.fill_(7)
    for stale_operand in (stale, stale_view):
        try:
            stale_operand.add_(1)
        except RuntimeError:
            pass
    try:
        torch.npu.synchronize()
    except RuntimeError:
        pass
    assert bool((fresh.cpu() == 7).all()), "stale writes must not corrupt the reused Allocation"
    shmem.free(fresh)
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
    leaks = state["leaked_allocations"]
    assert isinstance(leaks, list) and len(leaks) == 1, f"expected the orphan in leaked_allocations: {state!r}"
    assert leaks[0]["allocation_id"] == allocation_id, f"leak record omitted the Allocation ID: {leaks!r}"
    assert leaks[0]["allocation_base"] == allocation_base, f"leak record omitted the Allocation base: {leaks!r}"
    assert leaks[0]["allocation_bytes"] == 64, f"leak record omitted the Allocation size: {leaks!r}"
    # This worker intentionally exits with the orphan active: there is no Tensor left from which a safe
    # collective free can be issued, and Runtime finalization must remain rejected rather than hide the defect.


def test_binding_stale_tensor_rejected_by_one_sided_ops() -> None:
    """Reject a freed Tensor passed as a one-sided operand before any RMA is enqueued."""
    acquire_runtime()
    rank = dist.get_rank()
    stale = shmem.empty(16, dtype=torch.uint8)
    local = torch.zeros(16, dtype=torch.uint8, device="npu")
    stale_signal_view = stale[:4].view(torch.int32)
    shmem.free(stale)

    # Entry validation detects the zeroed Storage, so the stale Tensor is rejected before any kernel launch.
    with pytest.raises(RuntimeError, match="invalidated by shmem.free"):
        shmem.put(stale, local, rank)
    with pytest.raises(RuntimeError, match="invalidated by shmem.free"):
        shmem.get(local, stale, rank)
    with pytest.raises(RuntimeError, match="invalidated by shmem.free"):
        shmem.signal(stale_signal_view, 1, rank)

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
