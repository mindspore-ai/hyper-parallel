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
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_rd import CachedRecursiveDoubling


def _assert_boundary(actual, expected):
    """Compare both relative and absolute error with the independent FP64 result."""
    delta = actual.cpu().double() - expected
    if not torch.isfinite(actual).all() or delta.norm() > 1e-6 * expected.norm() + 1e-12:
        raise AssertionError("RD boundary differs from the independent FP64 recurrence")
    if delta.abs().max() > 2e-6:
        raise AssertionError("RD absolute boundary error exceeds 2e-6")


def _check(mesh, device):
    """Check noncommuting maps, reverse live calls and public/coalesced equivalence."""
    rank, size = mesh.get_local_rank(), mesh.size()
    generator = torch.Generator().manual_seed(724 + sum(mesh.rank_list))
    shape = (size, 1, 96, 128, 128)
    matrices = torch.randn(shape, generator=generator) * .003 + .9 * torch.eye(128)
    states = torch.randn(shape, generator=generator) * .01
    gradients = torch.randn(shape, generator=generator) * .01
    control = None
    for force_public in (True, False):
        protocol = CachedRecursiveDoubling(mesh.get_group(), rank, tuple(mesh.rank_list), force_public)
        outputs = [protocol.forward((states[rank] * (1 + step)).to(device),
                                    (matrices[rank] * (1 - .05 * step)).to(device)) for step in range(2)]
        values = []
        for step in (1, 0):
            output, cache = outputs[step]
            snapshots = [tensor.clone() for tensor in cache]
            actual = protocol.backward(gradients[rank].to(device).contiguous(), cache)
            expected = torch.zeros_like(states[0], dtype=torch.float64)
            adjoint = torch.zeros_like(expected)
            maps = matrices * (1 - .05 * step)
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
            values.append((output.cpu(), actual.cpu()))
        if control is not None:
            for left, right in zip(control, values):
                for expected, actual in zip(left, right):
                    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        control = values


def _run(device):
    """Run the protocol on the world mesh and an interleaved DP subgroup."""
    torch.set_num_threads(1)
    if device == "npu":
        # CPU Gloo tests must remain independent of the optional NPU extension.
        torch_npu = importlib.import_module("torch_npu")

        torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
        torch.npu.matmul.allow_hf32 = False
        torch.npu.config.allow_internal_format = False
    dist.init_process_group("hccl" if device == "npu" else "gloo", timeout=timedelta(seconds=180))
    size = dist.get_world_size()
    root = hp.init_device_mesh(device, (size,), mesh_dim_names=("cp",))
    _check(root, device)
    if size % 2 == 0:
        root = hp.init_device_mesh(device, (size // 2, 2), mesh_dim_names=("cp", "dp"))
        _check(root["cp"], device)
    dist.barrier()
    dist.destroy_process_group()


def test_kda_rd_boundary_cpu():
    """Check non-power-of-two scans, interleaved groups and live caches."""
    _run("cpu")


def test_kda_rd_boundary_npu():
    """Check NPU epilogues and public/coalesced transport equivalence."""
    _run("npu")


if __name__ == "__main__":
    _run(os.environ.get("KDA_TEST_DEVICE", "npu"))
