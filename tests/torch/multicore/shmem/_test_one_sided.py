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
"""NPU workers for SHMEM one-sided communication system tests."""

import os

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from tests.torch.multicore.shmem._worker_utils import acquire_runtime, release_runtime


def test_binding_put_get() -> None:
    """Exercise Root PE Put and Get with symmetric remote operands."""
    acquire_runtime()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    next_pe = (rank + 1) % world_size
    previous_pe = (rank - 1) % world_size

    put_destination = shmem.empty(16, dtype=torch.uint8)
    remote_source = shmem.empty(16, dtype=torch.uint8)
    local_source = torch.full((16,), rank + 1, dtype=torch.uint8, device="npu")
    local_destination = torch.empty_like(local_source)
    put_destination.zero_()
    remote_source.fill_(rank + 11)
    # Cross-rank convergence: every rank must finish local zero-init of its put_destination before any rank
    # publishes a remote Put, otherwise an early remote Put lands before a lagging local zero_() and gets
    # overwritten, so the receiver's assert reads zeros.
    shmem.barrier()

    shmem.put(put_destination, local_source, next_pe)
    shmem.barrier()
    assert torch.equal(put_destination, torch.full_like(put_destination, previous_pe + 1))

    shmem.get(local_destination, remote_source, next_pe)
    torch.npu.synchronize()
    assert torch.equal(local_destination, torch.full_like(local_destination, next_pe + 11))

    shmem.barrier()
    shmem.free(put_destination)
    shmem.free(remote_source)
    release_runtime()


def test_binding_barrier_blocking_and_config() -> None:
    """Exercise barrier(blocking=False) Stream ordering and the debug_state config projection."""
    acquire_runtime()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    peer = (rank + 1) % world_size

    # The effective configuration is introspectable while the Runtime is Ready.
    config = shmem.debug_state()["config"]
    assert config is not None, "debug_state()['config'] must be present while the Runtime is Ready"
    assert config["heap_size_bytes"] == 64 * 1024 * 1024, (
        f"heap_size_bytes must be 64MiB, but got {config['heap_size_bytes']}"
    )
    assert config["timeout_seconds"] == 60, f"timeout_seconds must be 60, but got {config['timeout_seconds']}"
    assert config["data_engine"] == "mte", f"data_engine must be mte, but got {config['data_engine']}"
    expected_endpoint = os.environ.get("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", "tcp://127.0.0.1:8662")
    assert config["bootstrap_endpoint_base"] == expected_endpoint, (
        f"bootstrap_endpoint_base must be {expected_endpoint}, but got {config['bootstrap_endpoint_base']}"
    )

    buffer = shmem.empty(2, dtype=torch.int32)
    # The active Allocation table projects the buffer's identity, base address, and byte count.
    state = shmem.debug_state()
    allocations = state["active_allocations"]
    assert isinstance(allocations, list) and len(allocations) == 1, (
        f"active_allocations must list exactly one Allocation, but got {allocations!r}"
    )
    assert allocations[0]["allocation_base"] == buffer.data_ptr(), (
        f"allocation_base must equal buffer.data_ptr(), but got {allocations[0]['allocation_base']}"
    )
    assert allocations[0]["allocation_bytes"] == 2 * 4, (
        f"allocation_bytes must be 8, but got {allocations[0]['allocation_bytes']}"
    )
    # The requested-byte usage and remaining logical budget must equal the configured Heap size.
    assert state["allocated_bytes"] == 8, f"allocated_bytes must be 8, but got {state['allocated_bytes']}"
    assert state["remaining_bytes"] == 64 * 1024 * 1024 - 8, (
        f"remaining_bytes must be heap_size_bytes - allocated_bytes, but got {state['remaining_bytes']}"
    )
    # The snapshot renders one field per line for human reading in the REPL.
    assert repr(state).startswith("state: Ready"), (
        f"debug_state repr must render one field per line, but got {state!r}"
    )
    buffer.zero_()
    # Cross-rank convergence before any remote-visible write (see test_binding_put_get).
    shmem.barrier()

    # Phase 1: each rank Puts its payload into its own slot (target_pe=rank). Phase 2's cross-rank
    # Get reads the peer's slot and is only correct when the non-blocking barrier orders it after
    # every rank's Put: with blocking=False the Host returns immediately, so ordering must come
    # from the shared ACL Stream.
    payload = torch.full((1,), rank + 10, dtype=torch.int32, device="npu")
    shmem.put(buffer[rank:rank + 1], payload, rank)
    shmem.barrier(blocking=False)
    local_destination = torch.empty(1, dtype=torch.int32, device="npu")
    # Read the peer's own slot (index `peer`, which the peer filled): my view of that offset translates
    # to the peer's memory through the symmetric mapping.
    shmem.get(local_destination, buffer[peer:peer + 1], peer)
    torch.npu.synchronize()
    assert local_destination.item() == peer + 10, (
        f"work after barrier(blocking=False) must observe the peer's Put: expected={peer + 10}, "
        f"got={local_destination.item()}"
    )

    shmem.barrier()
    shmem.free(buffer)
    shmem.release()
    final_state = shmem.debug_state()
    assert final_state["state"] == "Uninitialized", f"state must be Uninitialized, but got {final_state['state']}"
    assert final_state["config"] is None, f"config must be cleared after release, but got {final_state['config']}"
    assert final_state["active_allocations"] is None, (
        f"active_allocations must be cleared after release, but got {final_state['active_allocations']}"
    )
    dist.destroy_process_group()


def test_binding_signal_and_pull() -> None:
    """Exercise Signal strings and a source-notifies-target Pull handshake."""
    acquire_runtime()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    next_pe = (rank + 1) % world_size
    previous_pe = (rank - 1) % world_size

    remote_source = shmem.empty(16, dtype=torch.uint8)
    # Keep each Signal word in an independent 64B slot so unrelated data does not share its cacheline.
    signal_storage = shmem.empty(16, dtype=torch.int32, alignment=64)
    signal_word = signal_storage[:1]
    local_destination = torch.empty(16, dtype=torch.uint8, device="npu")
    remote_source.fill_(rank + 11)
    signal_storage.zero_()
    # A sender must not publish a remote Signal until every receiver has completed local Signal initialization;
    # otherwise a late zero_() can overwrite an already delivered SET and leave the receiver waiting forever.
    shmem.barrier()

    shmem.signal(signal_word, rank + 1, next_pe, operation="set")
    shmem.wait_signal(signal_word, previous_pe + 1)
    shmem.get(local_destination, remote_source, previous_pe)
    torch.npu.synchronize()
    assert torch.equal(local_destination, torch.full_like(local_destination, previous_pe + 11))

    comparison_cases = (("eq", 3), ("ne", 4), ("gt", 2), ("ge", 3), ("lt", 4), ("le", 3))
    shmem.signal(signal_word, 3, rank, operation="set")
    for comparison, value in comparison_cases:
        shmem.wait_signal(signal_word, value, comparison=comparison)
    shmem.signal(signal_word, 2, rank, operation="add")
    shmem.wait_signal(signal_word, 5, comparison="eq")
    torch.npu.synchronize()

    with pytest.raises(ValueError, match="set, add"):
        shmem.signal(signal_word, 1, rank, operation="SET")
    with pytest.raises(ValueError, match="eq, ne, gt, ge, lt, le"):
        shmem.wait_signal(signal_word, 1, comparison="==")
    with pytest.raises(TypeError, match="operation must be a string"):
        shmem.signal(signal_word, 1, rank, operation=1)
    with pytest.raises(TypeError, match="comparison must be a string"):
        shmem.wait_signal(signal_word, 1, comparison=1)

    shmem.barrier()
    shmem.free(remote_source)
    shmem.free(signal_storage)
    release_runtime()
