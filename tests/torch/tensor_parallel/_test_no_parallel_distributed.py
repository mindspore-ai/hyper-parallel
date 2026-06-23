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
"""Distributed NPU worker tests for ``NoParallel`` (vs CPU single-device reference).

Each worker test uses ``dist.get_world_size()`` so the same test body runs on
2- or 4-rank meshes as configured in ``test_no_parallel_distributed.py``.

Scenarios:

1. **Replicated Linear forward** — NoParallel on a plain ``nn.Linear``; every rank
   computes the full matmul and produces a replicated output that matches the CPU
   single-device reference.
2. **Replicated LayerNorm forward** — NoParallel on ``nn.LayerNorm``; identical to
   CPU reference.
3. **Replicated Linear backward** — Forward + backward; gathered weight gradients
   match the CPU reference.
4. **Input redistribution from Shard** — Upstream ``SequenceParallel`` produces a
   ``Shard(1)`` DTensor; ``NoParallel(desired_input_layout=Replicate())`` must
   redistribute the sharded input back to ``Replicate()`` before the replicated
   module runs. Verifies the fix for comparing ``input_tensor.placements`` instead
   of constructor arguments.
5. **Output redistribution to Shard** — ``NoParallel`` followed by a consumer that
   expects ``Shard(1)``; the output hook redistributes the replicated output.
6. **Composition: SequenceParallel → NoParallel → RowwiseParallel** — End-to-end MLP
   pipeline where a norm layer sits between sequence-sharded and row-wise-sharded
   projections.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import torch_npu  # noqa: F401  -- Ascend NPU

from hyper_parallel import (
    ColwiseParallel,
    NoParallel,
    RowwiseParallel,
    SequenceParallel,
    init_device_mesh,
    parallelize_module,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.torch.utils import init_dist


def _make_tp_mesh_1d():
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("tp",),
    )


def _npu_precision_close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(
        a.cpu().float(),
        b.cpu().float(),
        rtol=1.5e-4,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Basic forward: replicated Linear
# ---------------------------------------------------------------------------


def test_no_parallel_linear_forward_precision_npu():
    """
    Feature: NoParallel replicated Linear forward matches CPU reference
    Description:
        1. Create nn.Linear with known weight/bias on all ranks
        2. Apply NoParallel (params become replicated DTensors)
        3. Forward replicated input; output should be replicated DTensor
        4. Compare output (via to_local) with CPU F.linear(x, w, b)
    Expectation: NPU output close to CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    torch.manual_seed(50)
    torch.npu.manual_seed(50)

    in_f, out_f, batch = 32, 64, 8

    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x, w, b)

    linear = nn.Linear(in_f, out_f, bias=True).npu()
    with torch.no_grad():
        linear.weight.copy_(w.npu())
        linear.bias.copy_(b.npu())
    x_npu = x.npu()

    sharded = parallelize_module(linear, mesh, NoParallel(use_local_output=False))
    with torch.no_grad():
        y_hp = sharded(x_npu)

    assert isinstance(y_hp, DTensor), "output should be a DTensor"
    assert y_hp.placements == (Replicate(),), "output should be Replicate()"
    _npu_precision_close(y_hp.to_local(), y_ref)


# ---------------------------------------------------------------------------
# Basic forward: replicated LayerNorm
# ---------------------------------------------------------------------------


def test_no_parallel_layernorm_forward_precision_npu():
    """
    Feature: NoParallel replicated LayerNorm forward matches CPU reference
    Description:
        1. Create nn.LayerNorm with known weight/bias
        2. Apply NoParallel; forward replicated input
        3. Compare output with CPU reference
    Expectation: NPU output close to CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    torch.manual_seed(51)
    torch.npu.manual_seed(51)

    bsz, seq_len, hidden = 4, 16, 32

    ln_cpu = nn.LayerNorm(hidden, elementwise_affine=True)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)
    with torch.no_grad():
        y_ref = ln_cpu(x_cpu)

    ln_npu = nn.LayerNorm(hidden, elementwise_affine=True).npu()
    with torch.no_grad():
        ln_npu.load_state_dict(ln_cpu.state_dict())

    sharded = parallelize_module(ln_npu, mesh, NoParallel(use_local_output=False))
    with torch.no_grad():
        y_hp = sharded(x_cpu.npu())

    assert isinstance(y_hp, DTensor)
    _npu_precision_close(y_hp.to_local(), y_ref)


# ---------------------------------------------------------------------------
# Backward: replicated Linear gradient
# ---------------------------------------------------------------------------


def test_no_parallel_linear_backward_gradient_npu():
    """
    Feature: NoParallel backward produces correct gradients on NPU
    Description:
        1. Apply NoParallel to Linear
        2. Forward + backward with scalar loss
        3. All-reduce replicated weight gradients; compare with CPU reference
    Expectation: Gathered weight grad close to CPU reference grad
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(52)
    torch.npu.manual_seed(52)

    in_f, out_f, batch = 16, 32, 4

    w = torch.randn(out_f, in_f, dtype=torch.float32)
    b = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)

    w_ref = w.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)
    y_ref = F.linear(x, w_ref, b_ref)
    y_ref.sum().backward()

    linear = nn.Linear(in_f, out_f, bias=True).npu()
    with torch.no_grad():
        linear.weight.copy_(w.npu())
        linear.bias.copy_(b.npu())
    x_npu = x.npu().requires_grad_(True)

    sharded = parallelize_module(linear, mesh, NoParallel())
    y_hp = sharded(x_npu)
    y_hp.sum().backward()

    local_wgrad = linear.weight.grad
    if isinstance(local_wgrad, tuple):
        local_wgrad = local_wgrad[0]
    if isinstance(local_wgrad, DTensor):
        local_wgrad = local_wgrad.to_local()

    gathered = [torch.empty_like(local_wgrad) for _ in range(world_size)]
    dist.all_gather(gathered, local_wgrad)
    full_wgrad = torch.stack(gathered).mean(dim=0).cpu()

    _npu_precision_close(full_wgrad, w_ref.grad)


# ---------------------------------------------------------------------------
# Input redistribution from Shard (SequenceParallel → NoParallel)
# ---------------------------------------------------------------------------


def test_no_parallel_redistribute_sharded_input_npu():
    """
    Feature: NoParallel redistibutes sharded input to Replicate before compute
    Description:
        1. Apply SequenceParallel to a LayerNorm (produces Shard(1) output)
        2. Apply NoParallel to a downstream Linear (desired_input_layout=Replicate())
        3. The input hook should redistribute the Shard(1) DTensor to Replicate
        4. Compare end-to-end output with CPU reference
    Expectation: Correct redistribution; output matches CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.manual_seed(53)
    torch.npu.manual_seed(53)

    bsz, seq_len, hidden, out_f = 4, 16, 32, 24
    assert seq_len % world_size == 0

    class NormThenLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden, elementwise_affine=True)
            self.linear = nn.Linear(hidden, out_f, bias=True)

        def forward(self, x):
            return self.linear(self.norm(x))

    norm_cpu = nn.LayerNorm(hidden, elementwise_affine=True)
    linear_cpu = nn.Linear(hidden, out_f, bias=True)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)

    with torch.no_grad():
        y_ref = linear_cpu(norm_cpu(x_cpu))

    model = NormThenLinear().npu()
    with torch.no_grad():
        model.norm.load_state_dict(norm_cpu.state_dict())
        model.linear.weight.copy_(linear_cpu.weight.npu())
        model.linear.bias.copy_(linear_cpu.bias.npu())

    parallelize_module(model, mesh, {
        "norm": SequenceParallel(sequence_dim=1, use_local_output=False),
        "linear": NoParallel(desired_input_layout=Replicate(),
                             use_local_output=True),
    })

    chunk = seq_len // world_size
    sl = slice(rank * chunk, (rank + 1) * chunk)
    x_local = x_cpu[:, sl, :].npu()

    with torch.no_grad():
        y_local = model(x_local)

    _npu_precision_close(y_local, y_ref)


# ---------------------------------------------------------------------------
# Output redistribution to Shard (NoParallel → RowwiseParallel)
# ---------------------------------------------------------------------------


def test_no_parallel_redistribute_output_to_shard_npu():
    """
    Feature: NoParallel output hook redistributes to a non-default output_layout
    Description:
        1. Apply NoParallel(output_layout=Shard(1)) to a LayerNorm
        2. Output DTensor should be redistributed from Replicate to Shard(1)
        3. Verify output placements are Shard(1)
    Expectation: Output is a Shard(1) DTensor
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(54)
    torch.npu.manual_seed(54)

    bsz, seq_len, hidden = 4, 16, 32
    assert hidden % world_size == 0

    ln_cpu = nn.LayerNorm(hidden, elementwise_affine=True)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)

    ln_npu = nn.LayerNorm(hidden, elementwise_affine=True).npu()
    with torch.no_grad():
        ln_npu.load_state_dict(ln_cpu.state_dict())

    sharded = parallelize_module(
        ln_npu, mesh,
        NoParallel(output_layout=Shard(1), use_local_output=False),
    )
    with torch.no_grad():
        y_hp = sharded(x_cpu.npu())

    assert isinstance(y_hp, DTensor)
    assert y_hp.placements == (Shard(1),), (
        f"expected Shard(1) output, got {y_hp.placements}"
    )

    with torch.no_grad():
        y_ref = ln_cpu(x_cpu)
    chunk = seq_len // world_size
    rank = dist.get_rank()
    sl = slice(rank * chunk, (rank + 1) * chunk)
    _npu_precision_close(y_hp.to_local(), y_ref[:, sl, :])


# ---------------------------------------------------------------------------
# Composition: SequenceParallel → NoParallel → RowwiseParallel
# ---------------------------------------------------------------------------


def test_no_parallel_composition_sp_nopar_row_npu():
    """
    Feature: end-to-end composition Colwise → NoParallel → Rowwise matches CPU reference
    Description:
        1. MLP: linear1 (Colwise) → norm (NoParallel) → linear2 (Rowwise)
        2. Norm sits between TP-sharded layers; NoParallel bridges Shard(-1)→Replicate
           on input and Replicate→Shard(-1) on output so RowwiseParallel receives the
           correct layout
        3. Compare end-to-end output with CPU reference
    Expectation: NPU output close to CPU reference
    """
    init_dist()
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(55)
    torch.npu.manual_seed(55)

    in_f, hidden_f, out_f, batch = 32, 64, 24, 8
    assert hidden_f % world_size == 0
    assert in_f % world_size == 0

    w1_data = torch.randn(hidden_f, in_f, dtype=torch.float32)
    b1_data = torch.randn(hidden_f, dtype=torch.float32)
    norm_w = torch.randn(hidden_f, dtype=torch.float32)
    norm_b = torch.randn(hidden_f, dtype=torch.float32)
    w2_data = torch.randn(out_f, hidden_f, dtype=torch.float32)
    b2_data = torch.randn(out_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)

    h_ref = F.linear(x, w1_data, b1_data)
    h_ref = F.relu(h_ref)
    h_ref = F.layer_norm(h_ref, [hidden_f], weight=norm_w, bias=norm_b)
    y_ref = F.linear(h_ref, w2_data, b2_data)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(in_f, hidden_f, bias=True)
            self.norm = nn.LayerNorm(hidden_f, elementwise_affine=True)
            self.linear2 = nn.Linear(hidden_f, out_f, bias=True)

        def forward(self, x):
            return self.linear2(self.norm(F.relu(self.linear1(x))))

    model = MLP().npu()
    with torch.no_grad():
        model.linear1.weight.copy_(w1_data.npu())
        model.linear1.bias.copy_(b1_data.npu())
        model.norm.weight.copy_(norm_w.npu())
        model.norm.bias.copy_(norm_b.npu())
        model.linear2.weight.copy_(w2_data.npu())
        model.linear2.bias.copy_(b2_data.npu())

    parallelize_module(model, mesh, {
        "linear1": ColwiseParallel(),
        "norm": NoParallel(
            input_layout=Shard(-1),
            desired_input_layout=Replicate(),
            output_layout=Shard(-1),
            use_local_output=False,
        ),
        "linear2": RowwiseParallel(),
    })

    x_npu = x.npu()
    with torch.no_grad():
        y_hp = model(x_npu)

    _npu_precision_close(y_hp, y_ref)
