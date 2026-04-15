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
"""Distributed NPU worker tests for TP + FSDP / TP + CP hybrid parallelism.

Uses the real ``ColwiseParallel`` / ``RowwiseParallel`` from
``hyper_parallel.core.tensor_parallel.style`` combined with ``fully_shard``
and ``ContextParallel``.

All tests compare distributed NPU output against single-device CPU reference.

Port allocation (launched from ``test_tp_hybrid_distributed.py``):
  10600–10601  4-card TP+FSDP
  10602–10603  8-card TP+CP
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import torch_npu  # noqa: F401  -- Ascend NPU

from hyper_parallel import (
    ColwiseParallel,
    ContextParallel,
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
        return self.w2(F.relu(self.w1(x)))


# ---------------------------------------------------------------------------
# TP + FSDP
# ---------------------------------------------------------------------------


def test_tp_fsdp_mlp_forward_precision_npu():
    """
    Feature: ColwiseParallel + RowwiseParallel MLP with FSDP matches CPU reference
    Description:
        1. Create 2D mesh (dp=2, tp=2), total 4 ranks
        2. TP: w1=ColwiseParallel, w2=RowwiseParallel on tp_mesh
        3. FSDP: fully_shard on dp_mesh
        4. Compare output with CPU single-device MLP
    Expectation: NPU output close to CPU reference
    """
    init_dist()
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        print(f"Skip: need at least 4 ranks, got {world_size}")
        return

    tp_size = 2
    dp_size = world_size // tp_size
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = root_mesh["dp"]
    tp_mesh = root_mesh["tp"]

    torch.manual_seed(42)
    torch.npu.manual_seed(42)

    in_f, hidden_f, out_f, batch = 32, 64, 24, 8
    assert hidden_f % tp_size == 0, (
        f"hidden_features {hidden_f} must be divisible by tp_size {tp_size}"
    )
    assert in_f % tp_size == 0, (
        f"in_features {in_f} must be divisible by tp_size {tp_size}"
    )

    # Build same model for CPU reference and NPU distributed
    torch.manual_seed(100)
    ref_model = MLP(in_f, hidden_f, out_f)

    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = ref_model(x)

    # NPU distributed model with same weights
    torch.manual_seed(100)
    dist_model = MLP(in_f, hidden_f, out_f).npu()

    # Step 1: Apply TP on tp_mesh
    parallelize_module(
        dist_model,
        tp_mesh,
        {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
    )

    # Step 2: Apply FSDP on dp_mesh
    fully_shard(dist_model, mesh=dp_mesh)

    # Split input across DP ranks
    dp_idx = root_mesh.get_coordinate()[0]
    local_batch = batch // dp_size
    x_local = x[dp_idx * local_batch : (dp_idx + 1) * local_batch].npu()

    with torch.no_grad():
        y_local = dist_model(x_local)

    # Gather outputs from all DP ranks and compare with corresponding reference slice
    dp_group = root_mesh.get_group("dp")
    gathered = [torch.empty_like(y_local) for _ in range(dp_size)]
    dist.all_gather(gathered, y_local, group=dp_group)
    y_full = torch.cat(gathered, dim=0)

    _npu_precision_close(y_full, y_ref)


def test_tp_fsdp_mlp_backward_gradient_npu():
    """
    Feature: TP + FSDP backward produces correct gradients
    Description:
        1. Same setup as forward test
        2. Forward + backward with scalar loss
        3. Compare gathered w1 weight grad with CPU reference
    Expectation: Gradients match CPU reference
    """
    init_dist()
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        print(f"Skip: need at least 4 ranks, got {world_size}")
        return

    tp_size = 2
    dp_size = world_size // tp_size
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = root_mesh["dp"]
    tp_mesh = root_mesh["tp"]

    torch.manual_seed(42)
    torch.npu.manual_seed(42)

    in_f, hidden_f, out_f, batch = 16, 32, 12, 4
    assert hidden_f % tp_size == 0
    assert in_f % tp_size == 0

    # CPU reference
    torch.manual_seed(200)
    ref_model = MLP(in_f, hidden_f, out_f)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = ref_model(x)
    y_ref.sum().backward()
    ref_w1_grad = ref_model.w1.weight.grad.clone()

    # NPU distributed
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

    # Forward + backward (FSDP handles unshard/reshard automatically via hooks)
    y_local = dist_model(x_local)
    y_local.sum().backward()

    # Gather w1 grad — must undo both FSDP reduce-scatter and TP sharding.
    # FSDP reduce-scatters along dp dim, TP shards along dim 0 (ColwiseParallel).
    # Both slice dim 0, so each rank holds a (hidden_f/(tp*dp), in_f) grad shard.
    # Correct order: DP all-gather first (undo FSDP), then TP all-gather (undo TP).
    local_w1_grad = dist_model.w1.weight.grad
    if hasattr(local_w1_grad, 'to_local'):
        local_w1_grad = local_w1_grad.to_local()

    # Step 1: All-gather across DP dimension (undo FSDP reduce-scatter)
    dp_group = root_mesh.get_group("dp")
    gathered_dp = [torch.empty_like(local_w1_grad) for _ in range(dp_size)]
    dist.all_gather(gathered_dp, local_w1_grad, group=dp_group)
    dp_full_grad = torch.cat(gathered_dp, dim=0)

    # Step 2: All-gather across TP dimension (undo ColwiseParallel Shard(0))
    tp_group = root_mesh.get_group("tp")
    gathered_tp = [torch.empty_like(dp_full_grad) for _ in range(tp_size)]
    dist.all_gather(gathered_tp, dp_full_grad, group=tp_group)
    full_grad = torch.cat(gathered_tp, dim=0).cpu() / dp_size

    _npu_precision_close(full_grad, ref_w1_grad)


# ---------------------------------------------------------------------------
# TP + ContextParallel
# ---------------------------------------------------------------------------


class SimpleAttention(nn.Module):
    """Minimal single-head attention for CP testing.

    Uses ``F.scaled_dot_product_attention`` style computation but
    implemented explicitly so we can compare with CPU reference.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute single-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of same shape.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # (B, S, D) -> (B, S, D)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """One transformer block: attention + MLP for TP+CP composition test."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.attn = SimpleAttention(dim)
        self.mlp = MLP(dim, hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        h = self.attn(x) + x
        return self.mlp(h) + h


def test_tp_cp_transformer_forward_precision_npu():
    """
    Feature: TP + CP on TransformerBlock matches CPU reference
    Description:
        1. Create 2D mesh (cp=2, tp=2), total 4 ranks (or 8 with cp=2, tp=4)
        2. TP: MLP w1=ColwiseParallel, w2=RowwiseParallel on tp_mesh
        3. CP: ContextParallel on attn module using cp_mesh
        4. Compare output with CPU single-device reference
    Expectation: NPU output close to CPU reference
    """
    init_dist()
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        print(f"Skip: need at least 4 ranks, got {world_size}")
        return

    tp_size = 2
    cp_size = world_size // tp_size
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(cp_size, tp_size),
        mesh_dim_names=("cp", "tp"),
    )
    cp_mesh = root_mesh["cp"]
    tp_mesh = root_mesh["tp"]

    torch.manual_seed(50)
    torch.npu.manual_seed(50)

    dim, hidden_dim, batch, seq_len = 32, 64, 2, 16
    assert hidden_dim % tp_size == 0, (
        f"hidden_dim {hidden_dim} must be divisible by tp_size {tp_size}"
    )
    assert dim % tp_size == 0, (
        f"dim {dim} must be divisible by tp_size {tp_size}"
    )
    assert seq_len % cp_size == 0, (
        f"seq_len {seq_len} must be divisible by cp_size {cp_size}"
    )

    # CPU reference model
    torch.manual_seed(300)
    ref_block = TransformerBlock(dim, hidden_dim)
    x = torch.randn(batch, seq_len, dim, dtype=torch.float32)
    y_ref = ref_block(x)

    # NPU distributed model
    torch.manual_seed(300)
    dist_block = TransformerBlock(dim, hidden_dim).npu()

    # Step 1: Apply TP on MLP submodules
    parallelize_module(
        dist_block.mlp,
        tp_mesh,
        {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
    )

    # Step 2: Apply CP on attention module
    parallelize_module(
        dist_block.attn,
        cp_mesh,
        {"": ContextParallel(seq_dim=1)},
    )

    # Split input along sequence dimension for CP
    cp_idx = root_mesh.get_coordinate()[0]
    local_seq = seq_len // cp_size
    x_local = x[:, cp_idx * local_seq : (cp_idx + 1) * local_seq, :].npu()

    with torch.no_grad():
        y_local = dist_block(x_local)

    # Gather outputs from all CP ranks along sequence dimension
    gathered = [torch.empty_like(y_local) for _ in range(cp_size)]
    dist.all_gather(gathered, y_local)
    y_full = torch.cat(gathered, dim=1)  # gather along seq_dim=1

    _npu_precision_close(y_full, y_ref)
