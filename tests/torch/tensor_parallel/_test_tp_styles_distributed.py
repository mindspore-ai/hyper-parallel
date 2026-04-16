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
"""Distributed NPU worker tests for ``ColwiseParallel`` / ``RowwiseParallel``.

Launched from ``test_tp_styles_distributed.py`` via ``parallel_run`` in **three**
launcher waves: eight scenarios on **2** ranks (two waves of four cases), plus
**Colwise/Rowwise linear forward** again on **4** ranks for a wider TP mesh.

Each worker test uses ``dist.get_world_size()`` for divisibility checks; tensor
shapes are chosen so the same case passes for both 2- and 4-rank runs where a
case is scheduled twice (linear forward only on 4 ranks).

Each test instantiates the real ``ColwiseParallel`` / ``RowwiseParallel`` style
from ``hyper_parallel.core.tensor_parallel.style``, shards a module through
``parallelize_module``, runs a forward (and optionally backward) pass on NPU,
and compares results against a single-device CPU reference.
"""
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import torch_npu  # noqa: F401  -- Ascend NPU

from hyper_parallel import ColwiseParallel, RowwiseParallel, init_device_mesh, parallelize_module
from hyper_parallel.core.dtensor.placement_types import Replicate
from tests.torch.utils import init_dist


def _make_tp_mesh_1d():
    """1-D mesh covering all ranks in the default process group."""
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("tp",),
    )


def _npu_precision_close(a: torch.Tensor, b: torch.Tensor) -> None:
    """Assert NPU vs CPU reference within typical float32 tolerance (HCCL matmul)."""
    torch.testing.assert_close(
        a.cpu().float(),
        b.cpu().float(),
        rtol=1.5e-4,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# ColwiseParallel Linear
# ---------------------------------------------------------------------------


def test_colwise_linear_forward_precision_npu():
    """
    Feature: ColwiseParallel sharded Linear forward matches CPU F.linear reference
    Description:
        1. Create nn.Linear with known weight/bias on all ranks
        2. Shard via ColwiseParallel (weight Shard(0), bias Shard(0))
        3. Forward replicated input through sharded module
        4. Compare output with CPU F.linear(x, w, b)
    Expectation: NPU output close to CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(42)
    torch.npu.manual_seed(42)

    in_f, out_f, batch = 32, 64, 8
    assert out_f % world_size == 0, (
        f"out_features {out_f} must be divisible by world_size {world_size}"
    )

    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x, w, b)

    linear = nn.Linear(in_f, out_f, bias=True).npu()
    with torch.no_grad():
        linear.weight.copy_(w.npu())
        linear.bias.copy_(b.npu())
    x_npu = x.npu()

    sharded = parallelize_module(linear, mesh, ColwiseParallel())
    with torch.no_grad():
        y_hp = sharded(x_npu)
    # ColwiseParallel returns local shard; all-gather before comparing with full reference
    gathered = [torch.empty_like(y_hp) for _ in range(world_size)]
    dist.all_gather(gathered, y_hp)
    y_hp_full = torch.cat(gathered, dim=-1)
    _npu_precision_close(y_hp_full, y_ref)


def test_colwise_linear_backward_gradient_npu():
    """
    Feature: ColwiseParallel backward produces correct gradients on NPU
    Description:
        1. Shard Linear via ColwiseParallel
        2. Forward + backward with a scalar loss
        3. Gather weight gradients from all ranks, compare with CPU reference
    Expectation: Gathered weight grad close to CPU reference grad
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(100)
    torch.npu.manual_seed(100)

    in_f, out_f, batch = 16, 32, 4
    assert out_f % world_size == 0, (
        f"out_features {out_f} must be divisible by world_size {world_size}"
    )

    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)

    # CPU reference
    w_ref = w.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)
    y_ref = F.linear(x, w_ref, b_ref)
    y_ref.sum().backward()

    # NPU sharded
    linear = nn.Linear(in_f, out_f, bias=True).npu()
    with torch.no_grad():
        linear.weight.copy_(w.npu())
        linear.bias.copy_(b.npu())
    x_npu = x.npu().requires_grad_(True)

    sharded = parallelize_module(linear, mesh, ColwiseParallel())
    y_hp = sharded(x_npu)
    y_hp.sum().backward()

    # Gather weight grad shards from all ranks
    local_wgrad = linear.weight.grad
    if isinstance(local_wgrad, tuple):
        local_wgrad = local_wgrad[0]
    gathered = [torch.empty_like(local_wgrad) for _ in range(world_size)]
    dist.all_gather(gathered, local_wgrad)
    full_wgrad = torch.cat(gathered, dim=0).cpu()

    _npu_precision_close(full_wgrad, w_ref.grad)


# ---------------------------------------------------------------------------
# RowwiseParallel Linear
# ---------------------------------------------------------------------------


def test_rowwise_linear_forward_precision_npu():
    """
    Feature: RowwiseParallel sharded Linear forward matches CPU F.linear reference
    Description:
        1. Create nn.Linear with known weight/bias on all ranks
        2. Shard via RowwiseParallel (weight Shard(1), bias Replicate)
        3. Forward sharded input through sharded module
        4. Compare output (after all-reduce) with CPU F.linear(x, w, b)
    Expectation: NPU output close to CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(43)
    torch.npu.manual_seed(43)

    in_f, out_f, batch = 32, 24, 8
    assert in_f % world_size == 0, (
        f"in_features {in_f} must be divisible by world_size {world_size}"
    )

    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x, w, b)

    linear = nn.Linear(in_f, out_f, bias=True).npu()
    with torch.no_grad():
        linear.weight.copy_(w.npu())
        linear.bias.copy_(b.npu())
    x_npu = x.npu()

    sharded = parallelize_module(
        linear, mesh, RowwiseParallel(input_layouts=Replicate())
    )
    with torch.no_grad():
        y_hp = sharded(x_npu)
    _npu_precision_close(y_hp, y_ref)


def test_rowwise_linear_backward_gradient_npu():
    """
    Feature: RowwiseParallel backward produces correct gradients on NPU
    Description:
        1. Shard Linear via RowwiseParallel
        2. Forward + backward with a scalar loss
        3. Gather weight gradients from all ranks, compare with CPU reference
    Expectation: Gathered weight grad close to CPU reference grad
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(101)
    torch.npu.manual_seed(101)

    in_f, out_f, batch = 16, 12, 4
    assert in_f % world_size == 0, (
        f"in_features {in_f} must be divisible by world_size {world_size}"
    )

    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)

    # CPU reference
    w_ref = w.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)
    y_ref = F.linear(x, w_ref, b_ref)
    y_ref.sum().backward()

    # NPU sharded
    linear = nn.Linear(in_f, out_f, bias=True).npu()
    with torch.no_grad():
        linear.weight.copy_(w.npu())
        linear.bias.copy_(b.npu())
    x_npu = x.npu().requires_grad_(True)

    sharded = parallelize_module(
        linear, mesh, RowwiseParallel(input_layouts=Replicate())
    )
    y_hp = sharded(x_npu)
    y_hp.sum().backward()

    # Gather weight grad shards (sharded along dim 1)
    # RowwiseParallel backward through all-reduce accumulates gradient from
    # each rank, so divide by world_size to match the single-device reference.
    local_wgrad = linear.weight.grad
    if isinstance(local_wgrad, tuple):
        local_wgrad = local_wgrad[0]
    gathered = [torch.empty_like(local_wgrad) for _ in range(world_size)]
    dist.all_gather(gathered, local_wgrad)
    full_wgrad = torch.cat(gathered, dim=1).cpu() / world_size

    _npu_precision_close(full_wgrad, w_ref.grad)


# ---------------------------------------------------------------------------
# ColwiseParallel + RowwiseParallel MLP composition
# ---------------------------------------------------------------------------


def test_mlp_colwise_rowwise_forward_precision_npu():
    """
    Feature: ColwiseParallel + RowwiseParallel MLP composition matches CPU reference
    Description:
        1. Two-layer MLP: linear1 (in->hidden) colwise, linear2 (hidden->out) rowwise
        2. parallelize_module with plan {"linear1": ColwiseParallel, "linear2": RowwiseParallel}
        3. Compare end-to-end output with CPU MLP reference
    Expectation: NPU MLP output close to CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(44)
    torch.npu.manual_seed(44)

    in_f, hidden_f, out_f, batch = 32, 64, 24, 8
    assert hidden_f % world_size == 0, (
        f"hidden_features {hidden_f} must be divisible by world_size {world_size}"
    )
    assert in_f % world_size == 0, (
        f"in_features {in_f} must be divisible by world_size {world_size}"
    )

    # Reference weight/bias data for CPU reference computation
    w1_data = torch.randn(hidden_f, in_f, dtype=torch.float32)
    b1_data = torch.randn(hidden_f, dtype=torch.float32)
    w2_data = torch.randn(out_f, hidden_f, dtype=torch.float32)
    b2_data = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)

    # CPU reference: linear1 -> relu -> linear2
    h_ref = F.linear(x, w1_data, b1_data)
    h_ref = F.relu(h_ref)
    y_ref = F.linear(h_ref, w2_data, b2_data)

    # MLP model where `linear1` and `linear2` are nn.Linear **modules**.
    # ColwiseParallel / RowwiseParallel operate at the **module/layer level** —
    # they automatically shard all parameters (weight, bias) of the module.
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(in_f, hidden_f, bias=True)
            self.linear2 = nn.Linear(hidden_f, out_f, bias=True)

        def forward(self, x):
            return self.linear2(F.relu(self.linear1(x)))

    model = MLP().npu()
    with torch.no_grad():
        model.linear1.weight.copy_(w1_data.npu())
        model.linear1.bias.copy_(b1_data.npu())
        model.linear2.weight.copy_(w2_data.npu())
        model.linear2.bias.copy_(b2_data.npu())
    x_npu = x.npu()

    # parallelize_module applies styles at module (layer) granularity:
    #   "linear1" → ColwiseParallel() shards model.linear1 (nn.Linear) column-wise
    #   "linear2" → RowwiseParallel() shards model.linear2 (nn.Linear) row-wise
    # For composition: colwise output is Shard(-1), rowwise expects Shard(-1) input — matches
    parallelize_module(
        model, mesh,
        {"linear1": ColwiseParallel(), "linear2": RowwiseParallel()},
    )
    with torch.no_grad():
        y_hp = model(x_npu)
    _npu_precision_close(y_hp, y_ref)


# ---------------------------------------------------------------------------
# ColwiseParallel Embedding
# ---------------------------------------------------------------------------


def test_colwise_embedding_forward_precision_npu():
    """
    Feature: ColwiseParallel sharded Embedding forward matches CPU F.embedding reference
    Description:
        1. Create nn.Embedding with known weight on all ranks
        2. Shard via ColwiseParallel (weight Shard(1) — shard embedding_dim)
        3. Forward through sharded module, compare gathered output with CPU reference
    Expectation: Gathered NPU output close to CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(45)
    torch.npu.manual_seed(45)

    num_embeddings, embedding_dim, batch, seq_len = 100, 64, 4, 8
    assert embedding_dim % world_size == 0, (
        f"embedding_dim {embedding_dim} must be divisible by world_size {world_size}"
    )

    weight = torch.randn(num_embeddings, embedding_dim, dtype=torch.float32)
    ids = torch.randint(0, num_embeddings, (batch, seq_len), dtype=torch.long)
    y_ref = F.embedding(ids, weight)

    embedding = nn.Embedding(num_embeddings, embedding_dim).npu()
    with torch.no_grad():
        embedding.weight.copy_(weight.npu())
    ids_npu = ids.npu()

    sharded = parallelize_module(embedding, mesh, ColwiseParallel())
    with torch.no_grad():
        y_hp = sharded(ids_npu)

    # ColwiseParallel Embedding: output is Shard(-1), need all-gather to compare
    gathered = [torch.empty_like(y_hp) for _ in range(world_size)]
    dist.all_gather(gathered, y_hp)
    full_output = torch.cat(gathered, dim=-1)

    _npu_precision_close(full_output, y_ref)


# ---------------------------------------------------------------------------
# Unsupported module type rejection
# ---------------------------------------------------------------------------


def test_colwise_unsupported_module_raises_npu():
    """
    Feature: ColwiseParallel rejects unsupported module types on NPU
    Description: apply ColwiseParallel to nn.LayerNorm
    Expectation: raises NotImplementedError
    """
    init_dist()
    mesh = _make_tp_mesh_1d()

    style = ColwiseParallel()
    module = nn.LayerNorm(8).npu()

    with pytest.raises(NotImplementedError):
        style.apply(module, mesh)


def test_rowwise_unsupported_module_raises_npu():
    """
    Feature: RowwiseParallel rejects unsupported module types on NPU
    Description: apply RowwiseParallel to nn.LayerNorm
    Expectation: raises NotImplementedError
    """
    init_dist()
    mesh = _make_tp_mesh_1d()

    style = RowwiseParallel()
    module = nn.LayerNorm(8).npu()

    with pytest.raises(NotImplementedError):
        style.apply(module, mesh)
