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
"""NPU workers for SHMEM AllGather system tests."""

import os
import time

import pytest
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore import shmem


# The leading 0 guards the zero-byte contract: shmem.all_gather must validate and
# no-op without enqueuing anything, and the following payload calls must stay intact.
_BOUNDARY_BYTES = (
    0,
    1,
    31,
    32,
    33,
    8191,
    8192,
    8193,
    8223,
    8224,
    64 * 1024 - 1,
    64 * 1024,
    # sub-32B misalignment on the medium-block path (>=64K uses kMediumBlockDim).
    64 * 1024 + 31,
    64 * 1024 + 33,
    1024 * 1024 - 1,
    1024 * 1024,
)


def _acquire() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    shmem.acquire()
    return local_rank, dist.get_world_size()


def _random_input(byte_count: int, rank: int) -> torch.Tensor:
    """Build a reproducible per-rank random payload (distinct across ranks)."""
    torch.manual_seed(rank * 1009 + byte_count)
    return torch.randint(0, 256, (byte_count,), dtype=torch.uint8, device="npu")


def _reference_all_gather(input_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Compute the ground-truth AllGather via the HCCL collective."""
    gathered = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, input_tensor)
    return torch.cat(gathered)


def _run_all_gather(byte_count: int, rank: int, world_size: int) -> None:
    input_tensor = _random_input(byte_count, rank)
    if byte_count == 0:
        output = torch.empty(0, dtype=torch.uint8, device="npu")
    else:
        output = shmem.empty(byte_count * world_size, dtype=torch.uint8)
    shmem.all_gather(output, input_tensor)
    torch.npu.synchronize()

    if byte_count == 0:
        assert output.numel() == 0, f"Empty AllGather must stay empty: rank={rank}, got numel={output.numel()}"
    else:
        expected = _reference_all_gather(input_tensor, world_size)
        assert torch.equal(output, expected), (
            f"AllGather result mismatch vs HCCL reference: byte_count={byte_count}, rank={rank}, "
            f"world_size={world_size}, mismatch_count={(output != expected).sum().item()}"
        )
        shmem.free(output)


def test_all_gather_output_input_overlap_rejected() -> None:
    """Reject overlap before launching the AllGather Kernel."""
    _acquire()
    allocation = shmem.empty(32, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="must not overlap"):
        shmem.all_gather(allocation, allocation)
    shmem.free(allocation)
    shmem.release()
    dist.destroy_process_group()


def test_all_gather_output_size_mismatch_rejected() -> None:
    """Reject output.numel inconsistent with input.numel and Root size."""
    _acquire()
    input_tensor = torch.ones(16, dtype=torch.uint8, device="npu")
    output = shmem.empty(15, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match=r"output\.numel\(\)"):
        shmem.all_gather(output, input_tensor)
    shmem.free(output)
    shmem.release()
    dist.destroy_process_group()


def test_all_gather_boundary_matrix() -> None:
    """Validate every frozen byte boundary on the launcher's Root size."""
    rank, world_size = _acquire()
    for byte_count in _BOUNDARY_BYTES:
        _run_all_gather(byte_count, rank, world_size)
    shmem.release()
    dist.destroy_process_group()


def test_all_gather_correctness_large() -> None:
    """Exercise the 24-AIV, multi-Chunk path with a rebalanced tail."""
    rank, world_size = _acquire()
    _run_all_gather(1024 * 1024 + 31, rank, world_size)
    shmem.release()
    dist.destroy_process_group()


def test_all_gather_stream_order_and_visibility() -> None:
    """Verify rank skew and non-default Stream ordering without timing Host return."""
    rank, world_size = _acquire()
    byte_count = 8193
    output = shmem.empty(byte_count * world_size, dtype=torch.uint8)
    stream = torch.npu.Stream()
    time.sleep(rank * 0.05)
    with torch.npu.stream(stream):
        input_tensor = _random_input(byte_count, rank)
        shmem.all_gather(output, input_tensor)
    stream.synchronize()

    expected = _reference_all_gather(input_tensor, world_size)
    assert torch.equal(output, expected), (
        f"Non-default Stream AllGather mismatch vs HCCL reference: rank={rank}, "
        f"mismatch_count={(output != expected).sum().item()}"
    )
    shmem.free(output)
    shmem.release()
    dist.destroy_process_group()


def test_runtime_device_guard_on_real_npu() -> None:
    """Reject Allocation, Free, and Barrier after current-device drift and recover after restoring it."""
    bound_device, _ = _acquire()
    device_count = torch.npu.device_count()
    if device_count < 2:
        pytest.skip(f"device guard ST requires at least two NPUs, got device_count={device_count}")
    other_device = (bound_device + 1) % device_count

    torch.npu.set_device(other_device)
    with pytest.raises(RuntimeError, match="device"):
        shmem.empty(32, dtype=torch.uint8)

    torch.npu.set_device(bound_device)
    allocation = shmem.empty(32, dtype=torch.uint8)
    torch.npu.set_device(other_device)
    with pytest.raises(RuntimeError, match="device"):
        shmem.free(allocation)
    with pytest.raises(RuntimeError, match="device"):
        shmem.barrier()

    torch.npu.set_device(bound_device)
    shmem.free(allocation)
    shmem.release()
    dist.destroy_process_group()
