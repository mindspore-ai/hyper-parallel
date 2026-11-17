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
"""Real HCCL reductions for CPU-offloaded Hyper gradients."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import

from hyper_parallel import DTensor
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Partial, Shard
from hyper_parallel.core.utils import clip_grad_norm_


def _parameter(mesh: DeviceMesh, local: torch.Tensor, placement: object) -> torch.nn.Parameter:
    """Construct an offloaded parameter without importing an external framework."""
    size = local.numel() * (dist.get_world_size() if placement.is_shard() else 1)
    value = DTensor.from_local(torch.zeros_like(local), mesh, [placement], shape=(size,), stride=(1,))
    parameter = torch.nn.Parameter(value)
    parameter.grad = DTensor.from_local(local.clone(), mesh, [placement], shape=(size,), stride=(1,))
    return parameter


def test_cpu_offload_norms() -> None:
    """Compare offloaded clipping with complete-gradient Torch references."""
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        mesh = DeviceMesh.from_group(dist.group.WORLD, "npu", mesh=torch.arange(world), mesh_dim_names=("dp",))
        full = torch.arange(1, world * 2 + 1, dtype=torch.float32)
        orders = (0.0, 1.0, 2.0, 3.0, float("inf"), -float("inf"))
        for missing in (False, True):
            for order in orders:
                if missing and order == -float("inf"):
                    continue
                param = _parameter(mesh, full.chunk(world)[rank], Shard(0))
                reference = torch.nn.Parameter(torch.zeros_like(full))
                reference.grad = full.clone()
                if missing:
                    reference.grad[:2] = 0
                    if rank == 0:
                        param.grad = None
                if order == 0:
                    # Hyper's existing L0 counts gradient elements, not norm-stack entries.
                    expected = torch.count_nonzero(reference.grad).float()
                    reference.grad.mul_(torch.clamp(1.0 / (expected + 1e-6), max=1.0))
                else:
                    expected = torch.nn.utils.clip_grad_norm_([reference], 1.0, norm_type=order)
                observed = clip_grad_norm_([param], 1.0, norm_type=order)
                torch.testing.assert_close(observed, expected)
                assert observed.device.type == "cpu", f"Expected CPU norm, got {observed.device}"
                assert param.to_local().device.type == "cpu", f"Parameter moved to {param.to_local().device}"
                if param.grad is not None:
                    torch.testing.assert_close(param.grad.to_local(), reference.grad.chunk(world)[rank])
        for order in (2.0, float("inf")):
            param = _parameter(mesh, torch.tensor([float(rank + 1), 2.0]), Partial("sum"))
            local = param.grad.to_local().clone()
            summed = torch.tensor([float(world * (world + 1) // 2), 2.0 * world])
            # Pylint cannot infer this Torch extension function in NPU builds.
            expected = torch.linalg.vector_norm(summed, order)  # pylint: disable=not-callable
            observed = clip_grad_norm_([param], 1.0, norm_type=order)
            torch.testing.assert_close(observed, expected)
            torch.testing.assert_close(param.grad.to_local(), local / (expected + 1e-6))
        for nonfinite in (float("inf"), float("nan")):
            param = _parameter(mesh, torch.tensor([nonfinite if rank == 0 else 1.0]), Shard(0))
            try:
                clip_grad_norm_([param], 1.0, error_if_nonfinite=True)
            except RuntimeError as error:
                assert "non-finite" in str(error), f"Unexpected error: {error}"
            else:
                raise AssertionError(f"Rank {rank} did not reject non-finite global norm {nonfinite}")
        print(f"CPU_OFFLOAD_CLIP_PASS rank={rank} cases=15", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_cpu_offload_norms()
