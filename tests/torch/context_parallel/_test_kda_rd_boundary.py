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
"""Real-process RD checks against an independent ordered FP64 recurrence."""
from datetime import timedelta
import importlib
import os

import torch
import torch.distributed as dist

import hyper_parallel as hp
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_rd import KDAStateRecursiveDoubling


def _assert_boundary(actual, expected):
    """Compare both relative and absolute error with the independent FP64 result."""
    delta = actual.cpu().double() - expected
    if not torch.isfinite(actual).all() or delta.norm() > 1e-6 * expected.norm() + 1e-12:
        raise AssertionError("RD boundary differs from the independent FP64 recurrence")
    if delta.abs().max() > 2e-6:
        raise AssertionError("RD absolute boundary error exceeds 2e-6")


def _check(mesh, device):
    """Check noncommuting maps, reverse live calls and compact invocation caches."""
    rank, size = mesh.get_local_rank(), mesh.size()
    generator = torch.Generator().manual_seed(724 + sum(mesh.rank_list))
    shape = (size, 1, 96, 128, 128)
    matrices = torch.randn(shape, generator=generator) * .003 + .9 * torch.eye(128)
    states = torch.randn(shape, generator=generator) * .01
    gradients = torch.randn(shape, generator=generator) * .01
    invocation_maps = [matrices, matrices * .95]
    if size > 1:
        # A true document reset removes the dependence across this interval.
        invocation_maps[1][size // 2].zero_()
    protocol = KDAStateRecursiveDoubling(mesh.get_group(), rank, tuple(mesh.rank_list))
    outputs = [protocol.forward((states[rank] * (1 + step)).to(device),
                                invocation_maps[step][rank].to(device)) for step in range(2)]
    for step in (1, 0):
        output, cache = outputs[step]
        snapshots = [tensor.clone() for tensor in cache]
        actual = protocol.backward(gradients[rank].to(device).contiguous(), cache)
        expected = torch.zeros_like(states[0], dtype=torch.float64)
        adjoint = torch.zeros_like(expected)
        maps = invocation_maps[step]
        for index in range(rank):
            expected = maps[index].double() @ expected + (states[index] * (1 + step)).double()
        for index in range(size - 1, rank, -1):
            adjoint = gradients[index].double() + maps[index].double().transpose(-1, -2) @ adjoint
        _assert_boundary(output, expected)
        _assert_boundary(actual, adjoint)
        for before, after in zip(snapshots, cache):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
            if after.untyped_storage().nbytes() != after.numel() * after.element_size():
                raise AssertionError("RD cache retains an oversized producer allocation")


def _run():
    """Run the NPU protocol on the world mesh and an interleaved DP subgroup."""
    torch.set_num_threads(1)
    torch_npu = importlib.import_module("torch_npu")
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    size = dist.get_world_size()
    root = hp.init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
    _check(root, "npu")
    if size % 2 == 0:
        root = hp.init_device_mesh("npu", (size // 2, 2), mesh_dim_names=("cp", "dp"))
        _check(root["cp"], "npu")
    dist.barrier()
    dist.destroy_process_group()


def test_kda_rd_boundary_npu():
    """Check NPU epilogues, public transport and invocation cache ownership."""
    _run()


if __name__ == "__main__":
    _run()
