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
"""NPU workers for SHMEM Runtime lifecycle system tests."""

import os
import re
import time

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from tests.torch.multicore.shmem._worker_utils import ACQUIRE_HINT, acquire_runtime


_REINIT_RANK_SKEW_SECONDS = 1.0


def test_binding_single_process_without_distributed() -> None:
    """Run a one-PE SHMEM lifecycle without initializing Torch distributed."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    assert not dist.is_initialized()

    shmem.acquire()
    allocation = shmem.empty(32, dtype=torch.uint8)
    shmem.free(allocation)
    shmem.release()


def test_binding_single_process_data_plane_health() -> None:
    """Exercise the SHMEM native data plane on one PE: allocation, self Put/Get,
    Signal/Wait, and one-rank AllGather, then a clean release.

    The device barrier (shmem.barrier) is deliberately excluded: barrier_on_stream_kernel
    crashes with aicore 507015 under gate-level interconnect contention even at team size 1
    with zero remote rounds (gate 9568), while the bounded data-plane kernels above survive
    the same contention. Cross-rank convergence must use shmem.host_barrier().
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    assert not dist.is_initialized()

    shmem.acquire()

    # Self-directed Put/Get through the MTE data plane.
    remote_buffer = shmem.empty(16, dtype=torch.uint8)
    local_source = torch.full((16,), 7, dtype=torch.uint8, device="npu")
    local_destination = torch.empty(16, dtype=torch.uint8, device="npu")
    remote_buffer.zero_()
    shmem.put(remote_buffer, local_source, 0)
    torch.npu.synchronize()
    assert torch.equal(remote_buffer, local_source), "self Put must land in the symmetric buffer"

    remote_buffer.fill_(9)
    shmem.get(local_destination, remote_buffer, 0)
    torch.npu.synchronize()
    assert bool(torch.all(local_destination == 9).item()), "self Get must read the symmetric buffer"

    # Signal/Wait on the local PE.
    signal_storage = shmem.empty(4, dtype=torch.int32, alignment=64)
    signal_word = signal_storage[:1]
    signal_storage.zero_()
    shmem.signal(signal_word, 5, 0, operation="set")
    shmem.wait_signal(signal_word, 5)
    torch.npu.synchronize()

    # One-rank AllGather: output equals input.
    all_gather_input = torch.arange(8, dtype=torch.float32, device="npu")
    all_gather_output = shmem.empty(8, dtype=torch.float32)
    shmem.all_gather(all_gather_output, all_gather_input)
    torch.npu.synchronize()
    assert torch.equal(all_gather_output, all_gather_input), "one-rank AllGather must copy input to output"

    state = shmem.debug_state()
    assert state["state"] == "Ready", f"state must be Ready, but got {state['state']}"
    assert len(state["active_allocations"]) == 3, (
        f"active_allocations must list exactly three Allocations, but got {state['active_allocations']!r}"
    )

    shmem.free(remote_buffer)
    shmem.free(signal_storage)
    shmem.free(all_gather_output)
    shmem.release()


def test_binding_inactive_runtime_hint() -> None:
    """Explain how to acquire SHMEM before a capability call."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    input_tensor = torch.empty(0, dtype=torch.uint8, device="npu")
    output = torch.empty(0, dtype=torch.uint8, device="npu")

    calls = (
        ("allocate", lambda: shmem.empty(1, dtype=torch.uint8)),
        ("all_gather", lambda: shmem.all_gather(output, input_tensor)),
    )
    for operation, capability in calls:
        with pytest.raises(RuntimeError) as error:
            capability()
        message = str(error.value)
        expected_fields = (
            "error_code=INVALID_STATE",
            f"operation={operation}",
            "phase=Validation",
            "cann_error_code=None",
            "message=",
            ACQUIRE_HINT,
        )
        missing_fields = [field for field in expected_fields if field not in message]
        assert not missing_fields, (
            f"inactive Runtime error is not actionable: operation={operation}, "
            f"missing={missing_fields}, message={message}"
        )

    assert shmem.debug_state()["state"] == "Uninitialized"


def test_binding_reinit_with_different_heap_sizes() -> None:
    """Start clean lifecycles while changing the CANN Heap request."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    mebibyte = 1024 * 1024
    heap_sizes = (64 * mebibyte, 128 * mebibyte, 64 * mebibyte)
    allocation_sizes = (16 * mebibyte, 96 * mebibyte, 16 * mebibyte)
    stale_tensor = None

    for cycle, (heap_size, allocation_size) in enumerate(zip(heap_sizes, allocation_sizes)):
        os.environ["HYPER_PARALLEL_SHMEM_HEAP_SIZE"] = str(heap_size)
        if cycle > 0 and local_rank == dist.get_world_size() - 1:
            time.sleep(_REINIT_RANK_SKEW_SECONDS)
        shmem.acquire()
        tensor = shmem.empty((allocation_size,), dtype=torch.uint8)
        tensor.fill_(local_rank + cycle)
        # Cross-rank convergence: the device-side barrier spins without timeout and dies with aicore
        # 507015 under gate-level interconnect contention, so use the HCCL host barrier.
        shmem.host_barrier()

        if stale_tensor is not None:
            with pytest.raises(RuntimeError, match="invalidated by shmem.free"):
                shmem.free(stale_tensor)

        shmem.free(tensor)
        shmem.host_barrier()
        if cycle == 0:
            stale_tensor = tensor
        shmem.release()

    dist.destroy_process_group()


def test_binding_final_release_with_active_allocation_is_retryable() -> None:
    """Keep the Runtime active until the symmetric Allocation is released."""
    acquire_runtime()
    allocation = shmem.empty(1024, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="allocated_count"):
        shmem.release()

    shmem.free(allocation)
    shmem.release()
    shmem.acquire()
    shmem.release()
    dist.destroy_process_group()


def test_binding_init_timeout_injection(capfd: pytest.CaptureFixture[str]) -> None:
    """Surface an unreachable bootstrap endpoint as a prompt RuntimeError, not a hang."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device_index = torch.npu.current_device()
    dist.init_process_group(backend="hccl")

    started = time.monotonic()
    with pytest.raises(RuntimeError) as exc_info:
        shmem.acquire()
    elapsed = time.monotonic() - started
    error = str(exc_info.value)
    captured = capfd.readouterr()
    native_log = captured.out + captured.err
    print(f"[init-timeout] rank={local_rank} elapsed={elapsed:.2f}s error={error}\n{native_log}")
    assert elapsed < 60, f"Init failure must return promptly: rank={local_rank}, elapsed={elapsed:.2f}s"
    assert f"SHMEM initialization failed on NPU device {device_index}" in error, error
    assert "ACLSHMEM_" in error, error
    if "cann_detail=" in error:
        assert "cann_detail=[[SHM_SHMEM" in error, error
        assert 'cann_detail="' not in error, error
        assert "\n" not in error and "\r" not in error, error
    cann_failures = [line for line in native_log.splitlines() if "[SHM_SHMEM " in line]
    cann_prefix = re.compile(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6} error \d+ pid\[\d+\] "
        r"\[SHM_SHMEM [^]]+:\d+\] "
    )
    assert all(cann_prefix.match(line) for line in cann_failures), cann_failures
    assert "[rank -1]" not in native_log and "[rank unknown]" not in native_log, native_log
    runtime_failures = [
        line
        for line in native_log.splitlines()
        if "[HP_SHMEM " in line and "[Initialize|CannCall]" in line
    ]
    assert runtime_failures, f"expected a Runtime failure log, got output={native_log!r}"
    runtime_prefix = re.compile(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6} error \d+ pid\[\d+\] "
        r"\[HP_SHMEM runtime\.cpp:\d+\] \[rank 0\] \[Initialize\|CannCall\] "
    )
    assert all(runtime_prefix.match(line) for line in runtime_failures), runtime_failures
    for line in runtime_failures:
        assert " op=" not in line and " phase=" not in line and " error=" not in line and " msg=" not in line, line
        assert re.search(r"ACLSHMEM_[A-Z_]+\(-?\d+\)", line), line
        assert "CANN_ERROR" not in line, line
        assert "cann_detail=" not in line and "[SHM_SHMEM " not in line, line
        assert '"' not in line, f"expected a single-line Runtime message without quoting, got line={line!r}"
    dist.destroy_process_group()
