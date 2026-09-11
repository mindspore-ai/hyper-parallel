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
        shmem.barrier()

        if stale_tensor is not None:
            with pytest.raises(RuntimeError, match="DOUBLE_FREE"):
                shmem.free(stale_tensor)

        shmem.free(tensor)
        shmem.barrier()
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


def test_binding_init_timeout_injection() -> None:
    """Surface an unreachable bootstrap endpoint as a prompt RuntimeError, not a hang."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    started = time.monotonic()
    with pytest.raises(RuntimeError) as exc_info:
        shmem.acquire()
    elapsed = time.monotonic() - started
    print(f"[init-timeout] rank={local_rank} elapsed={elapsed:.2f}s error={exc_info.value}")
    assert elapsed < 60, f"Init failure must return promptly: rank={local_rank}, elapsed={elapsed:.2f}s"
    dist.destroy_process_group()
