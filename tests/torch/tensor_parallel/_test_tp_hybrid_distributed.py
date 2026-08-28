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
"""Distributed NPU worker tests for TP + FSDP hybrid parallelism.

Uses the real ``ColwiseParallel`` / ``RowwiseParallel`` from
``hyper_parallel.core.tensor_parallel.style`` combined with ``fully_shard``.

All tests compare distributed NPU output against single-device CPU reference.

Port allocation (launched from ``test_tp_hybrid_distributed.py``):
  10800  hybrid fwd+bwd; sequence-parallel 4-card packed alongside in launcher
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import torch_npu  # noqa: F401  # pylint: disable=unused-import  # side effect: register Ascend NPU

from hyper_parallel import (
    ColwiseParallel,
    RowwiseParallel,
    init_device_mesh,
    parallelize_module,
)
from hyper_parallel.core.fully_shard.api import fully_shard
from tests.torch.utils import init_dist


def _npu_precision_close(a: torch.Tensor, b: torch.Tensor) -> None:
    """Assert NPU vs CPU reference within typical float32 tolerance."""
    torch.testing.assert_close(
        a.cpu().float(),
        b.cpu().float(),
        rtol=1.5e-4,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Shared MLP model
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Two-layer MLP used for TP + FSDP composition tests."""

    def __init__(self, in_f: int, hidden_f: int, out_f: int):
        super().__init__()
        self.w1 = nn.Linear(in_f, hidden_f, bias=True)
        self.w2 = nn.Linear(hidden_f, out_f, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two-layer MLP forward pass.

        Args:
            x: Input activations.

        Returns:
            Output activations.
        """
        return self.w2(F.relu(self.w1(x)))


# ---------------------------------------------------------------------------
# TP + FSDP
# ---------------------------------------------------------------------------


def test_tp_fsdp_mlp_fwd_bwd_precision_npu():
    """
    Feature: Colwise+Rowwise MLP with FSDP forward + backward vs CPU (one torchrun)
    Description:
        1. 2D mesh (dp=2, tp=2); TP then fully_shard; compare gathered forward
        2. Rebuild with new seed; compare gathered w1 weight grad vs CPU
    Expectation: Output and grads close to CPU reference
    """
    init_dist()
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        print(f"Skip: need at least 4 ranks, got {world_size}")
        return

    tp_size = 2
    dp_size = world_size // tp_size

    def _meshes():
        root_mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        return root_mesh, root_mesh["dp"], root_mesh["tp"]

    # --- forward ---
    root_mesh, dp_mesh, tp_mesh = _meshes()
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    in_f, hidden_f, out_f, batch = 32, 64, 24, 8
    assert hidden_f % tp_size == 0
    assert in_f % tp_size == 0

    torch.manual_seed(100)
    ref_model = MLP(in_f, hidden_f, out_f)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = ref_model(x)

    torch.manual_seed(100)
    dist_model = MLP(in_f, hidden_f, out_f).npu()
    parallelize_module(
        dist_model,
        tp_mesh,
        {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
    )
    fully_shard(dist_model, mesh=dp_mesh)

    dp_idx = root_mesh.get_coordinate()[0]
    local_batch = batch // dp_size
    x_local = x[dp_idx * local_batch : (dp_idx + 1) * local_batch].npu()
    with torch.no_grad():
        y_local = dist_model(x_local)

    dp_group = root_mesh.get_group("dp")
    gathered = [torch.empty_like(y_local) for _ in range(dp_size)]
    dist.all_gather(gathered, y_local, group=dp_group)
    _npu_precision_close(torch.cat(gathered, dim=0), y_ref)

    # --- backward ---
    root_mesh, dp_mesh, tp_mesh = _meshes()
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    in_f, hidden_f, out_f, batch = 16, 32, 12, 4
    assert hidden_f % tp_size == 0
    assert in_f % tp_size == 0

    torch.manual_seed(200)
    ref_model = MLP(in_f, hidden_f, out_f)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = ref_model(x)
    y_ref.sum().backward()
    ref_w1_grad = ref_model.w1.weight.grad.clone()

    torch.manual_seed(200)
    dist_model = MLP(in_f, hidden_f, out_f).npu()
    parallelize_module(
        dist_model,
        tp_mesh,
        {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
    )
    fully_shard(dist_model, mesh=dp_mesh)

    dp_idx = root_mesh.get_coordinate()[0]
    local_batch = batch // dp_size
    x_local = x[dp_idx * local_batch : (dp_idx + 1) * local_batch].npu()
    y_local = dist_model(x_local)
    y_local.sum().backward()

    local_w1_grad = dist_model.w1.weight.grad
    if hasattr(local_w1_grad, 'to_local'):
        local_w1_grad = local_w1_grad.to_local()

    dp_group = root_mesh.get_group("dp")
    gathered_dp = [torch.empty_like(local_w1_grad) for _ in range(dp_size)]
    dist.all_gather(gathered_dp, local_w1_grad, group=dp_group)
    dp_full_grad = torch.cat(gathered_dp, dim=0)

    tp_group = root_mesh.get_group("tp")
    gathered_tp = [torch.empty_like(dp_full_grad) for _ in range(tp_size)]
    dist.all_gather(gathered_tp, dp_full_grad, group=tp_group)
    # fully_shard defaults to AVG, so gathering the DP shards already reconstructs the reference gradient.
    full_grad = torch.cat(gathered_tp, dim=0).cpu()
    _npu_precision_close(full_grad, ref_w1_grad)
