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
"""NPU worker tests for ``PrepareModuleInput`` / ``PrepareModuleOutput`` / ``PrepareModuleInputOutput``.

Launched from ``test_prepare_module_io_distributed.py`` via ``parallel_run``.
Aligns with PyTorch ``test/distributed/tensor/parallel/test_parallelize_api.py`` (prepare I/O)
plus Colwise/Rowwise composition patterns from ``common_dtensor``.

Each 2-card case asserts ``dist.get_world_size() == 2``; each 4-card case asserts ``== 4``.
"""
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import torch_npu  # noqa: F401  -- Ascend NPU

from hyper_parallel import (
    ColwiseParallel,
    DTensor,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    init_device_mesh,
    parallelize_module,
)
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
# 2-card: layout / chain (PyTorch test_parallelize_api parity)
# ---------------------------------------------------------------------------


def test_prepare_module_input_identity_roundtrip_npu():
    """
    Feature: PrepareModuleInput Shard(0) -> Replicate round-trip (PyTorch test_prepare_module_input)
    Description: Identity module, use_local_output=False, redistribute output back to Shard(0) local.
    Expectation: Restored local tensor equals input on every rank.
    """
    init_dist()
    if dist.get_world_size() != 2:
        pytest.skip("launcher uses 2 ranks")
    mesh = _make_tp_mesh_1d()
    rank = dist.get_rank()

    class Dummy(nn.Module):
        def forward(self, x):
            return x

    m = Dummy().npu()
    parallelize_module(
        m,
        mesh,
        PrepareModuleInput(
            input_layouts=Shard(0),
            desired_input_layouts=Replicate(),
            use_local_output=False,
        ),
    )
    torch.manual_seed(10600)
    torch.npu.manual_seed(10600)
    # Dim 0 for Shard(0) and later redistribute must split evenly on world_size.
    inp = torch.rand(4, 8, dtype=torch.float32, device="npu")
    if rank != 0:
        inp = torch.empty_like(inp)
    dist.broadcast(inp, src=0)

    out = m(inp)
    assert isinstance(out, DTensor)
    restored = out.redistribute(mesh, [Shard(0)]).to_local()
    torch.testing.assert_close(inp.cpu(), restored.cpu(), rtol=0, atol=0)


def test_prepare_module_output_replicate_to_shard_npu():
    """
    Feature: PrepareModuleOutput Replicate -> Shard(0) vs CPU chunk reference (PyTorch test_prepare_module_output)
    Description: Identity forward receives DTensor(Replicate); hook outputs Shard(0) local slice.
    Expectation: Local output matches the corresponding dim-0 chunk of the same global tensor (CPU).
    """
    init_dist()
    if dist.get_world_size() != 2:
        pytest.skip("launcher uses 2 ranks")
    mesh = _make_tp_mesh_1d()
    ws = dist.get_world_size()
    rank = dist.get_rank()

    class Dummy(nn.Module):
        def forward(self, x):
            return x

    m = Dummy().npu()
    parallelize_module(
        m,
        mesh,
        PrepareModuleOutput(
            output_layouts=Replicate(),
            desired_output_layouts=Shard(0),
            use_local_output=True,
        ),
    )
    torch.manual_seed(10601)
    inp_cpu = torch.rand(16, 7, dtype=torch.float32)
    inp_npu = inp_cpu.npu()
    dt = DTensor.from_local(inp_npu, mesh, [Replicate()])
    out = m(dt)
    ref_chunk = inp_cpu.chunk(ws, dim=0)[rank]
    torch.testing.assert_close(out.cpu(), ref_chunk, rtol=0, atol=0)


def test_prepare_module_input_output_chain_npu():
    """
    Feature: PrepareModuleInputOutput end-to-end (PyTorch test_prepare_module_input_output)
    Description: Same layout pipeline as PyTorch test; compare to explicit DTensor chain on CPU reference.
    Expectation: Module output equals manual from_local + redistribute + to_local on CPU side semantics.
    """
    init_dist()
    if dist.get_world_size() != 2:
        pytest.skip("launcher uses 2 ranks")
    mesh = _make_tp_mesh_1d()
    rank = dist.get_rank()

    class Dummy(nn.Module):
        def forward(self, x):
            return x

    m = Dummy().npu()
    parallelize_module(
        m,
        mesh,
        PrepareModuleInputOutput(
            input_layouts=Shard(0),
            desired_input_layouts=Replicate(),
            output_layouts=Replicate(),
            desired_output_layouts=Shard(1),
            use_local_output=True,
        ),
    )
    torch.manual_seed(10602)
    torch.npu.manual_seed(10602)
    # Last dim must divide world_size for Replicate -> Shard(1) redistribute.
    inp = torch.rand(4, 8, dtype=torch.float32, device="npu")
    if rank != 0:
        inp = torch.empty_like(inp)
    dist.broadcast(inp, src=0)

    out = m(inp)
    expected = (
        DTensor.from_local(inp, mesh, [Shard(0)])
        .redistribute(mesh, [Shard(1)])
        .to_local()
    )
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-4, atol=1e-5)


def test_prepare_module_input_then_colwise_linear_vs_cpu_npu():
    """
    Feature: PrepareModuleInput(Shard(1)->Replicate) + ColwiseParallel Linear vs CPU F.linear
    Description: Sequence-style shard on dim-1 per rank; prepare replicates full input; Colwise shards output.
    Expectation: all_gather on output last dim matches CPU reference on full input.
    """
    init_dist()
    if dist.get_world_size() != 2:
        pytest.skip("launcher uses 2 ranks")
    mesh = _make_tp_mesh_1d()
    ws = dist.get_world_size()
    rank = dist.get_rank()

    class Block(nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.lin = nn.Linear(in_f, out_f, bias=True)

        def forward(self, x):
            return self.lin(x)

    in_f, out_f, batch = 32, 64, 8
    assert in_f % ws == 0
    assert out_f % ws == 0

    torch.manual_seed(10603)
    torch.npu.manual_seed(10603)
    w_cpu = torch.randn(out_f, in_f, dtype=torch.float32)
    b_cpu = torch.randn(out_f, dtype=torch.float32)
    x_full = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x_full, w_cpu, b_cpu)

    x_local = x_full.chunk(ws, dim=1)[rank].npu()
    block = Block(in_f, out_f).npu()
    with torch.no_grad():
        block.lin.weight.copy_(w_cpu.npu())
        block.lin.bias.copy_(b_cpu.npu())

    parallelize_module(
        block,
        mesh,
        PrepareModuleInput(
            input_layouts=Shard(1),
            desired_input_layouts=Replicate(),
            use_local_output=True,
        ),
    )
    parallelize_module(block.lin, mesh, ColwiseParallel())

    with torch.no_grad():
        y_hp = block(x_local)
    gathered = [torch.empty_like(y_hp) for _ in range(ws)]
    dist.all_gather(gathered, y_hp)
    y_full = torch.cat(gathered, dim=-1)
    _npu_precision_close(y_full, y_ref)


# ---------------------------------------------------------------------------
# 2-card: Rowwise + output hook, kwargs, None slot, tuple output
# ---------------------------------------------------------------------------


def test_prepare_module_output_after_rowwise_vs_cpu_npu():
    """
    Feature: RowwiseParallel Linear + PrepareModuleOutput(Replicate->Shard(0)) vs CPU
    Description: Sharded input on last dim; rowwise produces replicated local output; output hook shards dim 0.
    Expectation: Concatenated shards along dim 0 match CPU F.linear on full input.
    """
    init_dist()
    if dist.get_world_size() != 2:
        pytest.skip("launcher uses 2 ranks")
    mesh = _make_tp_mesh_1d()
    ws = dist.get_world_size()
    rank = dist.get_rank()

    class Wrapped(nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.lin = nn.Linear(in_f, out_f, bias=True)

        def forward(self, x):
            return self.lin(x)

    in_f, out_f, batch = 32, 24, 8
    assert in_f % ws == 0

    torch.manual_seed(10610)
    torch.npu.manual_seed(10610)
    w_cpu = torch.randn(out_f, in_f, dtype=torch.float32)
    b_cpu = torch.randn(out_f, dtype=torch.float32)
    x_full = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x_full, w_cpu, b_cpu)

    x_local = x_full.chunk(ws, dim=-1)[rank].npu()
    w_mod = Wrapped(in_f, out_f).npu()
    with torch.no_grad():
        w_mod.lin.weight.copy_(w_cpu.npu())
        w_mod.lin.bias.copy_(b_cpu.npu())

    parallelize_module(
        w_mod.lin,
        mesh,
        RowwiseParallel(input_layouts=Shard(-1)),
    )
    PrepareModuleOutput(
        output_layouts=Replicate(),
        desired_output_layouts=Shard(0),
        use_local_output=True,
    ).apply(w_mod, mesh)

    with torch.no_grad():
        y_local = w_mod(x_local)
    gathered = [torch.empty_like(y_local) for _ in range(ws)]
    dist.all_gather(gathered, y_local)
    y_cat = torch.cat(gathered, dim=0)
    _npu_precision_close(y_cat, y_ref)


def test_prepare_module_input_with_kwarg_scale_npu():
    """
    Feature: PrepareModuleInput with_kwargs path vs single-device reference
    Description: forward(x, scale=...); both annotated Replicate; use_local_output=True.
    Expectation: x * scale matches CPU reference.
    """
    init_dist()
    if dist.get_world_size() != 2:
        pytest.skip("launcher uses 2 ranks")
    mesh = _make_tp_mesh_1d()

    class Scaled(nn.Module):
        def forward(self, x, scale=None):
            return x * scale

    m = Scaled().npu()
    parallelize_module(
        m,
        mesh,
        PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
            input_kwarg_layouts={"scale": Replicate()},
            desired_input_kwarg_layouts={"scale": Replicate()},
            use_local_output=True,
        ),
    )
    x = torch.ones(2, 3, dtype=torch.float32, device="npu") * 1.5
    scale = torch.tensor(2.0, dtype=torch.float32, device="npu")
    dist.broadcast(x, src=0)
    dist.broadcast(scale, src=0)
    out = m(x, scale=scale)
    ref = (x.cpu() * scale.cpu()).float()
    _npu_precision_close(out, ref)


def test_prepare_module_input_none_placeholder_dual_arg_npu():
    """
    Feature: None placeholder in input_layouts leaves first arg untouched
    Description: Second arg only is wrapped Replicate->Replicate (no-op redistribute).
    Expectation: Output equals second argument tensor.
    """
    init_dist()
    if dist.get_world_size() != 2:
        pytest.skip("launcher uses 2 ranks")
    mesh = _make_tp_mesh_1d()

    class PickSecond(nn.Module):
        def forward(self, x, y):  # pylint: disable=unused-argument
            return y

    m = PickSecond().npu()
    parallelize_module(
        m,
        mesh,
        PrepareModuleInput(
            input_layouts=(None, Replicate()),
            desired_input_layouts=(None, Replicate()),
            use_local_output=True,
        ),
    )
    x = torch.randn(2, 2, dtype=torch.float32, device="npu")
    y = torch.randn(2, 2, dtype=torch.float32, device="npu")
    dist.broadcast(x, src=0)
    dist.broadcast(y, src=0)
    out = m(x, y)
    torch.testing.assert_close(out.cpu(), y.cpu(), rtol=0, atol=0)


def test_prepare_module_output_tuple_with_none_slot_npu():
    """
    Feature: PrepareModuleOutput with tuple output and None placeholder
    Description: First tensor Replicate->Shard(0); second slot None (passthrough scalar tensor).
    Expectation: First matches dim-0 chunk of input; second remains scalar 1.0.
    """
    init_dist()
    if dist.get_world_size() != 2:
        pytest.skip("launcher uses 2 ranks")
    mesh = _make_tp_mesh_1d()
    ws = dist.get_world_size()
    rank = dist.get_rank()

    class DupOut(nn.Module):
        def forward(self, x):
            return x, torch.ones((), dtype=x.dtype, device=x.device)

    m = DupOut().npu()
    parallelize_module(
        m,
        mesh,
        PrepareModuleOutput(
            output_layouts=(Replicate(), None),
            desired_output_layouts=(Shard(0), None),
            use_local_output=True,
        ),
    )
    torch.manual_seed(10613)
    x = torch.rand(10, 4, dtype=torch.float32, device="npu")
    dist.broadcast(x, src=0)
    a, b = m(x)
    ref_a = x.cpu().chunk(ws, dim=0)[rank]
    torch.testing.assert_close(a.cpu(), ref_a, rtol=0, atol=0)
    assert b.shape == ()
    assert abs(b.item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 4-card: wider mesh (same math as selected 2-card cases + MLP block)
# ---------------------------------------------------------------------------


def test_prepare_module_input_colwise_pipeline_vs_cpu_npu():
    """
    Feature: Same PrepareModuleInput + Colwise pipeline as 2-card case on 4 ranks
    Description: in_f / out_f divisible by 4; CPU F.linear reference on full x.
    Expectation: Gathered Colwise outputs match CPU reference.
    """
    init_dist()
    if dist.get_world_size() != 4:
        pytest.skip("launcher uses 4 ranks")
    mesh = _make_tp_mesh_1d()
    ws = dist.get_world_size()
    rank = dist.get_rank()

    class Block(nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.lin = nn.Linear(in_f, out_f, bias=True)

        def forward(self, x):
            return self.lin(x)

    in_f, out_f, batch = 32, 64, 8
    assert in_f % ws == 0
    assert out_f % ws == 0

    torch.manual_seed(10620)
    torch.npu.manual_seed(10620)
    w_cpu = torch.randn(out_f, in_f, dtype=torch.float32)
    b_cpu = torch.randn(out_f, dtype=torch.float32)
    x_full = torch.randn(batch, in_f, dtype=torch.float32)
    y_ref = F.linear(x_full, w_cpu, b_cpu)

    x_local = x_full.chunk(ws, dim=1)[rank].npu()
    block = Block(in_f, out_f).npu()
    with torch.no_grad():
        block.lin.weight.copy_(w_cpu.npu())
        block.lin.bias.copy_(b_cpu.npu())

    parallelize_module(
        block,
        mesh,
        PrepareModuleInput(
            input_layouts=Shard(1),
            desired_input_layouts=Replicate(),
            use_local_output=True,
        ),
    )
    parallelize_module(block.lin, mesh, ColwiseParallel())

    with torch.no_grad():
        y_hp = block(x_local)
    gathered = [torch.empty_like(y_hp) for _ in range(ws)]
    dist.all_gather(gathered, y_hp)
    y_full = torch.cat(gathered, dim=-1)
    _npu_precision_close(y_full, y_ref)


def test_prepare_module_input_output_mlp_block_vs_cpu_npu():
    """
    Feature: Tiny MLP with Colwise+Rowwise on children and PrepareModuleInputOutput on root
    Description: Input sharded on dim-1 per rank; root I/O style mirrors seq-TP pattern; CPU forward on full x.
    Expectation: Concatenated output shards along dim 1 match CPU MLP output.
    """
    init_dist()
    if dist.get_world_size() != 4:
        pytest.skip("launcher uses 4 ranks")
    mesh = _make_tp_mesh_1d()
    ws = dist.get_world_size()
    rank = dist.get_rank()

    class TinyMlp(nn.Module):
        def __init__(self, d_in: int, d_h: int, d_out: int):
            super().__init__()
            self.up = nn.Linear(d_in, d_h, bias=True)
            self.down = nn.Linear(d_h, d_out, bias=True)

        def forward(self, x):
            return self.down(torch.relu(self.up(x)))

    d_in, d_h, d_out, batch = 32, 64, 24, 8
    assert d_in % ws == 0
    assert d_h % ws == 0
    assert d_out % ws == 0

    torch.manual_seed(10621)
    torch.npu.manual_seed(10621)
    mlp_ref = TinyMlp(d_in, d_h, d_out)
    mlp = TinyMlp(d_in, d_h, d_out).npu()
    mlp.load_state_dict(mlp_ref.state_dict())
    x_full = torch.randn(batch, d_in, dtype=torch.float32)
    with torch.no_grad():
        y_ref = mlp_ref(x_full)

    x_local = x_full.chunk(ws, dim=1)[rank].npu()

    parallelize_module(
        mlp,
        mesh,
        {
            "up": ColwiseParallel(),
            "down": RowwiseParallel(),
        },
    )
    parallelize_module(
        mlp,
        mesh,
        PrepareModuleInputOutput(
            input_layouts=Shard(1),
            desired_input_layouts=Replicate(),
            output_layouts=Replicate(),
            desired_output_layouts=Shard(1),
            use_local_input=True,
            use_local_output=True,
        ),
    )

    with torch.no_grad():
        y_local = mlp(x_local).contiguous()
    gathered = [torch.empty_like(y_local) for _ in range(ws)]
    dist.all_gather(gathered, y_local)
    y_full = torch.cat(gathered, dim=-1)
    _npu_precision_close(y_full, y_ref)
