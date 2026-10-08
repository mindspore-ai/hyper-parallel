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
"""Real collective tests for contiguous halo and shared AG backward."""

from __future__ import annotations

import os
from typing import Any
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.distributed.context_parallel.collectives import async_cp_allgather_launch, flex_cp_allgather
from hyper_parallel.distributed.context_parallel.halo import async_cp_halo_launch


class WorldMesh:
    """Use only the process group initialized by this worker."""

    @staticmethod
    def size() -> int:
        """Return the CP group size."""
        return dist.get_world_size()

    @staticmethod
    def get_local_rank() -> int:
        """Return the rank within the CP group."""
        return dist.get_rank()

    @staticmethod
    def get_group() -> Any:
        """Return the already initialized group."""
        return dist.group.WORLD


def init_group(backend: str) -> torch.device:
    """Initialize an explicitly selected backend and local device.

    Args:
        backend: Distributed backend, gloo or hccl.
    """
    if backend == "hccl":
        pytest.importorskip("torch_npu")
        if not hasattr(torch, "npu") or torch.npu.device_count() < int(os.environ["WORLD_SIZE"]):
            pytest.skip("requires enough visible Ascend NPUs for this process group")
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        torch.use_deterministic_algorithms(True)
    dist.init_process_group(backend, timeout=timedelta(seconds=120))
    if backend == "hccl":
        # CP1 oracles can execute custom metadata before CP8's first gather.
        # Materialize HCCL first: lazy communicator initialization after custom
        # kernels caused metadata failures in the validated CANN environment.
        dist.barrier()
    return torch.device("npu", int(os.environ["LOCAL_RANK"])) if backend == "hccl" else torch.device("cpu")


def check_halo(device: torch.device) -> None:
    """Verify multi-owner, empty-send/receive and remote-only gradient cases.

    Args:
        device: Device on which tensors and prepared geometry are consumed.
    """
    size, rank = dist.get_world_size(), dist.get_rank()
    for length, window in ((8, 1), (8, 2), (8, 8), (2, 8), (2, 128)):
        for dim in (0, 1, 2):
            for dtype in (torch.float32, torch.bfloat16):
                full = (torch.arange(length*size*6).reshape(length*size, 2, 3) % 17).to(dtype)
                local = full[rank*length:(rank+1)*length].movedim(0, dim).to(device).detach().requires_grad_()
                handle = async_cp_halo_launch(local, dim, window, WorldMesh())
                actual = handle.wait()
                start, end = max(0, rank*length-window+1), (rank+1)*length
                expected = full[start:end].movedim(0, dim)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                # Only the final consumer has nonzero loss. Owners still need
                # gradients returned through ranks with empty receive buffers.
                (actual.float().sum() * float(rank == size-1)).backward()
                expected_grad = torch.zeros_like(full)
                expected_grad[max(0, (size-1)*length-window+1):] = 1
                torch.testing.assert_close(local.grad.cpu(),
                    expected_grad[rank*length:(rank+1)*length].movedim(0, dim), rtol=0, atol=0)
                # A second forward weights all consumers differently, so a
                # duplicated owner row must receive a sum, never an overwrite.
                local.grad = None
                actual = async_cp_halo_launch(local, dim, window, WorldMesh()).wait()
                (actual.float().sum() * (rank+1)).backward()
                expected_grad.zero_()
                for consumer in range(size):
                    expected_grad[max(0, consumer*length-window+1):(consumer+1)*length] += consumer+1
                torch.testing.assert_close(local.grad.cpu(),
                    expected_grad[rank*length:(rank+1)*length].movedim(0, dim), rtol=0, atol=0)


def check_allgather(device: torch.device) -> None:
    """The SUM gather adjoint works for non-leading/non-contiguous sequence axes.

    Args:
        device: Device on which tensors and prepared geometry are consumed.
    """
    size, rank, length = dist.get_world_size(), dist.get_rank(), 4
    for dim in (0, 1, 2, -1):
        for dtype in (torch.float32, torch.bfloat16):
            full = (torch.arange(length*size*6).reshape(length*size, 2, 3) % 17).to(dtype)
            local = full[rank*length:(rank+1)*length].movedim(0, dim).to(device).detach().requires_grad_()
            actual = async_cp_allgather_launch(local, dim, WorldMesh()).wait()
            torch.testing.assert_close(actual.cpu(), full.movedim(0, dim), rtol=0, atol=0)
            weight = (torch.arange(full.numel()).reshape_as(full) % 7).movedim(0, dim).to(device)
            # Two consumers of the same gathered tensor exercise accumulation.
            ((actual.float()*weight).sum() * (rank+1) + actual.float().sum()).backward()
            expected = ((torch.arange(full.numel()).reshape_as(full) % 7) * (size*(size+1)//2) + size)
            expected = expected[rank*length:(rank+1)*length].movedim(0, dim).to(dtype)
            torch.testing.assert_close(local.grad.cpu(), expected, rtol=0, atol=0)
            local.grad = None
            synchronous, _ = flex_cp_allgather(local, local, dim, WorldMesh())
            ((synchronous.float()*weight).sum() * (rank+1) + synchronous.float().sum()).backward()
            torch.testing.assert_close(local.grad.cpu(), expected, rtol=0, atol=0)


def run_collectives(backend: str) -> None:
    """Run both primitives, including their full backward paths.

    Args:
        backend: Distributed backend, gloo or hccl.
    """
    device = init_group(backend)
    try:
        check_halo(device)
        check_allgather(device)
        if dist.get_rank() == 0:
            print(f"PASS {backend} CP{dist.get_world_size()}: halo + AG SUM adjoint, FP32/BF16, dim0/1/2", flush=True)
    finally:
        dist.destroy_process_group()


def test_halo_and_gather_adjoint_gloo():
    """CPU transport regression."""
    run_collectives("gloo")


def test_halo_and_gather_adjoint_hccl():
    """NPU transport regression with zero and variable splits."""
    run_collectives("hccl")


if __name__ == "__main__":
    run_collectives(os.environ.get("HALO_TEST_BACKEND", "gloo"))
