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
"""Distributed NPU worker tests for MC2 fused tensor-parallel Linear styles.

Launched from ``test_mc2_distributed.py`` via ``parallel_run``.

Constraints from Ascend MC2 kernels (``npu_all_gather_base_mm`` /
``npu_mm_reduce_scatter_base``), same as MindFormers:

* dtype: float16 / bfloat16
* contraction dim ``k`` in ``[256, 65535)``
* world_size in {2, 4, 8}

Shape notes for this file (default ``world_size=2``):

* Row forward / Column forward: contraction uses ``in_features`` or
  ``in_features / tp`` → keep ``_K >= 256 * world_size``.
* Column backward (fused MRS): contraction is ``n_local = out_features / tp``
  → keep ``_OUT >= 256 * world_size``.
"""
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from hyper_parallel import (
    ColwiseParallel,
    MC2ColwiseParallel,
    MC2RowwiseParallel,
    RowwiseParallel,
    init_device_mesh,
    parallelize_module,
)
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.tensor_parallel.mc2 import MC2Linear
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device

# Sized so fused forward *and* fused column backward satisfy k/n_local >= 256
# on 2-card (and remain valid if a 4-card launcher is added later with care).
_K = 512
_OUT = 512
_SEQ = 8
_BATCH = 2
_DTYPE = torch.bfloat16


def _make_tp_mesh_1d():
    return init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("tp",),
    )


def _bf16_close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(
        a.cpu().float(),
        b.cpu().float(),
        rtol=1.5e-1,
        atol=2.5e-1,
    )


def _rowwise_bf16_reference(x_full: torch.Tensor, w: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reference for row-parallel MC2: sum local bf16 matmuls then cast.

    Matches the distributed numerics more closely than a single full-precision
    ``F.linear`` (reduce-scatter accumulates bf16 partials).
    """
    in_f = w.shape[1]
    in_chunk = in_f // world_size
    out_f = w.shape[0]
    acc = torch.zeros(x_full.shape[0], x_full.shape[1], out_f, dtype=torch.float32)
    for rank in range(world_size):
        x_r = x_full[:, :, rank * in_chunk:(rank + 1) * in_chunk].to(_DTYPE)
        w_r = w[:, rank * in_chunk:(rank + 1) * in_chunk].to(_DTYPE)
        acc += F.linear(x_r.float(), w_r.float())
    return acc.to(_DTYPE)


def _require_npu():
    if _DEVICE_TYPE != "npu":
        pytest.skip("MC2 fused kernels require Ascend NPU + torch_npu")


# ---------------------------------------------------------------------------
# MC2ColwiseParallel / MC2RowwiseParallel / MLP (fwd+bwd per torchrun)
# ---------------------------------------------------------------------------


def _gather_wgrad(local_wgrad, world_size: int, cat_dim: int):
    if isinstance(local_wgrad, tuple):
        local_wgrad = local_wgrad[0]
    if hasattr(local_wgrad, "to_local"):
        local_wgrad = local_wgrad.to_local()
    gathered = [torch.empty_like(local_wgrad) for _ in range(world_size)]
    dist.all_gather(gathered, local_wgrad.contiguous())
    return torch.cat(gathered, dim=cat_dim)


def test_mc2_colwise_linear_fwd_bwd_precision_npu():
    """
    Feature: MC2ColwiseParallel forward + backward vs reference (one torchrun)
    Description:
        1. Sequence-sharded input; compare local feature shard with F.linear
        2. Backward: all-gather dW and compare with full F.linear grads
    Expectation: Output and weight grads close to reference (bf16 tolerance)
    """
    _require_npu()
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    in_f, out_f = _K, _OUT
    assert out_f % world_size == 0
    assert out_f // world_size >= 256  # fused column backward contracts on n_local
    assert _SEQ % world_size == 0
    out_chunk = out_f // world_size
    seq_local = _SEQ // world_size

    # --- forward ---
    torch.manual_seed(7)
    torch.npu.manual_seed(7)
    w = torch.randn(out_f, in_f, dtype=_DTYPE)
    b = torch.randn(out_f, dtype=_DTYPE)
    x_full = torch.randn(_SEQ, _BATCH, in_f, dtype=_DTYPE)
    y_ref_full = F.linear(x_full.float(), w.float(), b.float()).to(_DTYPE)
    y_ref_local = y_ref_full[:, :, rank * out_chunk:(rank + 1) * out_chunk]
    x_local = x_full[rank * seq_local:(rank + 1) * seq_local].contiguous()

    linear = to_device(nn.Linear(in_f, out_f, bias=True, dtype=_DTYPE), _DEVICE_TYPE)
    with torch.no_grad():
        linear.weight.copy_(to_device(w, _DEVICE_TYPE))
        linear.bias.copy_(to_device(b, _DEVICE_TYPE))
    sharded = parallelize_module(
        linear,
        mesh,
        MC2ColwiseParallel(input_layouts=Shard(0), use_local_output=True),
        src_data_rank=None,
    )
    assert isinstance(sharded, MC2Linear)
    with torch.no_grad():
        y_hp = sharded(to_device(x_local, _DEVICE_TYPE))
    assert y_hp.shape[0] == _SEQ
    assert y_hp.shape[-1] == out_chunk
    _bf16_close(y_hp, to_device(y_ref_local, _DEVICE_TYPE))

    # --- backward ---
    torch.manual_seed(11)
    torch.npu.manual_seed(11)
    w = torch.randn(out_f, in_f, dtype=_DTYPE)
    b = torch.randn(out_f, dtype=_DTYPE)
    x_full = torch.randn(_SEQ, _BATCH, in_f, dtype=_DTYPE)
    w_ref = w.float().clone().requires_grad_(True)
    b_ref = b.float().clone().requires_grad_(True)
    F.linear(x_full.float(), w_ref, b_ref).sum().backward()
    x_local = x_full[rank * seq_local:(rank + 1) * seq_local].contiguous()

    linear = to_device(nn.Linear(in_f, out_f, bias=True, dtype=_DTYPE), _DEVICE_TYPE)
    with torch.no_grad():
        linear.weight.copy_(to_device(w, _DEVICE_TYPE))
        linear.bias.copy_(to_device(b, _DEVICE_TYPE))
    sharded = parallelize_module(
        linear,
        mesh,
        MC2ColwiseParallel(input_layouts=Shard(0), use_local_output=True),
        src_data_rank=None,
    )
    sharded(to_device(x_local, _DEVICE_TYPE).requires_grad_(True)).sum().backward()
    _bf16_close(_gather_wgrad(linear.weight.grad, world_size, 0), w_ref.grad.to(_DTYPE))


def test_mc2_rowwise_linear_fwd_bwd_precision_npu():
    """
    Feature: MC2RowwiseParallel forward + backward vs reference (one torchrun)
    Description:
        1. Forward: local sequence shard vs reduce-scatter reference
        2. Backward: gather dW along in_features vs F.linear
    Expectation: Output and weight grads close to reference
    """
    _require_npu()
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    in_f, out_f = _K, _OUT
    assert in_f % world_size == 0
    assert _SEQ % world_size == 0
    seq_local = _SEQ // world_size
    in_chunk = in_f // world_size
    row_style = MC2RowwiseParallel(
        input_layouts=Shard(-1),
        output_layouts=Shard(0),
        use_local_output=True,
    )

    # --- forward ---
    torch.manual_seed(13)
    torch.npu.manual_seed(13)
    w = torch.randn(out_f, in_f, dtype=_DTYPE)
    b = torch.zeros(out_f, dtype=_DTYPE)
    x_full = torch.randn(_SEQ, _BATCH, in_f, dtype=_DTYPE)
    y_ref_local = _rowwise_bf16_reference(x_full, w, world_size)[
        rank * seq_local:(rank + 1) * seq_local
    ]
    x_local = x_full[:, :, rank * in_chunk:(rank + 1) * in_chunk].contiguous()

    linear = to_device(nn.Linear(in_f, out_f, bias=True, dtype=_DTYPE), _DEVICE_TYPE)
    with torch.no_grad():
        linear.weight.copy_(to_device(w, _DEVICE_TYPE))
        linear.bias.copy_(to_device(b, _DEVICE_TYPE))
    sharded = parallelize_module(linear, mesh, row_style, src_data_rank=None)
    assert isinstance(sharded, MC2Linear)
    with torch.no_grad():
        y_hp = sharded(to_device(x_local, _DEVICE_TYPE))
    assert y_hp.shape[0] == seq_local
    assert y_hp.shape[-1] == out_f
    _bf16_close(y_hp, to_device(y_ref_local, _DEVICE_TYPE))

    # --- backward ---
    torch.manual_seed(17)
    torch.npu.manual_seed(17)
    w = torch.randn(out_f, in_f, dtype=_DTYPE)
    b = torch.zeros(out_f, dtype=_DTYPE)
    x_full = torch.randn(_SEQ, _BATCH, in_f, dtype=_DTYPE)
    w_ref = w.float().clone().requires_grad_(True)
    b_ref = b.float().clone().requires_grad_(True)
    F.linear(x_full.float(), w_ref, b_ref).sum().backward()
    x_local = x_full[:, :, rank * in_chunk:(rank + 1) * in_chunk].contiguous()

    linear = to_device(nn.Linear(in_f, out_f, bias=True, dtype=_DTYPE), _DEVICE_TYPE)
    with torch.no_grad():
        linear.weight.copy_(to_device(w, _DEVICE_TYPE))
        linear.bias.copy_(to_device(b, _DEVICE_TYPE))
    sharded = parallelize_module(linear, mesh, row_style, src_data_rank=None)
    sharded(to_device(x_local, _DEVICE_TYPE).requires_grad_(True)).sum().backward()
    _bf16_close(_gather_wgrad(linear.weight.grad, world_size, 1), w_ref.grad.to(_DTYPE))


def test_mc2_mlp_col_row_fwd_bwd_precision_npu():
    """
    Feature: MC2 Col+Row MLP forward + backward vs unfused TP+SP (one torchrun)
    Description:
        1. Forward: sequence-sharded input; compare local outputs
        2. Backward: compare gathered fc1/fc2 weight grads
    Expectation: MC2 close to unfused baseline
    """
    _require_npu()
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # fc1 is column-sharded: n_local = ff / tp must be >= 256 for fused MRS backward.
    hidden, ff = _K, _OUT
    assert ff % world_size == 0
    assert ff // world_size >= 256
    assert _SEQ % world_size == 0
    seq_local = _SEQ // world_size

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(hidden, ff, bias=False, dtype=_DTYPE)
            self.fc2 = nn.Linear(ff, hidden, bias=False, dtype=_DTYPE)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    def _build(plan, w1, w2):
        mlp = to_device(_MLP(), _DEVICE_TYPE)
        with torch.no_grad():
            mlp.fc1.weight.copy_(to_device(w1, _DEVICE_TYPE))
            mlp.fc2.weight.copy_(to_device(w2, _DEVICE_TYPE))
        parallelize_module(mlp, mesh, plan, src_data_rank=None)
        return mlp

    unfused = {
        "fc1": ColwiseParallel(input_layouts=Shard(0), use_local_output=False),
        "fc2": RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=Shard(0),
            use_local_output=True,
        ),
    }
    fused = {
        "fc1": MC2ColwiseParallel(input_layouts=Shard(0), use_local_output=False),
        "fc2": MC2RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=Shard(0),
            use_local_output=True,
        ),
    }

    # --- forward ---
    torch.manual_seed(19)
    torch.npu.manual_seed(19)
    w1 = torch.randn(ff, hidden, dtype=_DTYPE)
    w2 = torch.randn(hidden, ff, dtype=_DTYPE)
    x_full = torch.randn(_SEQ, _BATCH, hidden, dtype=_DTYPE)
    x_local = x_full[rank * seq_local:(rank + 1) * seq_local].contiguous()
    baseline = _build(unfused, w1, w2)
    mc2 = _build(fused, w1, w2)
    assert isinstance(mc2.fc1, MC2Linear)
    assert isinstance(mc2.fc2, MC2Linear)
    x_npu = to_device(x_local, _DEVICE_TYPE)
    with torch.no_grad():
        y_ref = baseline(x_npu)
        y_hp = mc2(x_npu)
    assert y_hp.shape[0] == seq_local
    _bf16_close(y_hp, y_ref)

    # --- backward ---
    torch.manual_seed(23)
    torch.npu.manual_seed(23)
    w1 = torch.randn(ff, hidden, dtype=_DTYPE)
    w2 = torch.randn(hidden, ff, dtype=_DTYPE)
    x_full = torch.randn(_SEQ, _BATCH, hidden, dtype=_DTYPE)
    x_local = x_full[rank * seq_local:(rank + 1) * seq_local].contiguous()
    baseline = _build(unfused, w1, w2)
    mc2 = _build(fused, w1, w2)
    x_base = to_device(x_local, _DEVICE_TYPE).requires_grad_(True)
    x_mc2 = to_device(x_local, _DEVICE_TYPE).requires_grad_(True)
    baseline(x_base).sum().backward()
    mc2(x_mc2).sum().backward()
    _bf16_close(
        _gather_wgrad(mc2.fc1.weight.grad, world_size, 0),
        _gather_wgrad(baseline.fc1.weight.grad, world_size, 0),
    )
    _bf16_close(
        _gather_wgrad(mc2.fc2.weight.grad, world_size, 1),
        _gather_wgrad(baseline.fc2.weight.grad, world_size, 1),
    )


def test_mc2_colwise_linear_forward_fp16_npu():
    """
    Feature: MC2ColwiseParallel float16 forward matches F.linear reference
    Description:
        1. Same layout as bf16 colwise forward, but dtype=float16
        2. Compare local feature shard with CPU all-gather + F.linear
    Expectation: NPU MC2 fp16 output close to reference
    """
    _require_npu()
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    dtype = torch.float16
    torch.manual_seed(29)
    torch.npu.manual_seed(29)

    in_f, out_f = _K, _OUT
    assert out_f % world_size == 0
    assert _SEQ % world_size == 0
    rank = dist.get_rank()

    w = torch.randn(out_f, in_f, dtype=dtype)
    b = torch.randn(out_f, dtype=dtype)
    x_full = torch.randn(_SEQ, _BATCH, in_f, dtype=dtype)
    y_ref_full = F.linear(x_full.float(), w.float(), b.float()).to(dtype)
    out_chunk = out_f // world_size
    y_ref_local = y_ref_full[:, :, rank * out_chunk:(rank + 1) * out_chunk]

    seq_local = _SEQ // world_size
    x_local = x_full[rank * seq_local:(rank + 1) * seq_local].contiguous()

    linear = to_device(nn.Linear(in_f, out_f, bias=True, dtype=dtype), _DEVICE_TYPE)
    with torch.no_grad():
        linear.weight.copy_(to_device(w, _DEVICE_TYPE))
        linear.bias.copy_(to_device(b, _DEVICE_TYPE))
    x_npu = to_device(x_local, _DEVICE_TYPE)

    sharded = parallelize_module(
        linear,
        mesh,
        MC2ColwiseParallel(input_layouts=Shard(0), use_local_output=True),
        src_data_rank=None,
    )
    with torch.no_grad():
        y_hp = sharded(x_npu)

    assert y_hp.shape[0] == _SEQ
    assert y_hp.shape[-1] == out_chunk
    _bf16_close(y_hp, to_device(y_ref_local, _DEVICE_TYPE))


def test_mc2_colwise_seq_dim1_forward_precision_npu():
    """
    Feature: MC2ColwiseParallel forward with sequence sharding on dim 1
    Description:
        1. Batch-first input ``[B, S, H]`` sharded on sequence dim 1
        2. Compare gathered output with CPU all-gather + F.linear
    Expectation: NPU MC2 output close to reference
    """
    _require_npu()
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh_1d()
    world_size = dist.get_world_size()
    torch.manual_seed(31)
    torch.npu.manual_seed(31)

    in_f, out_f = _K, _OUT
    assert out_f % world_size == 0
    assert _SEQ % world_size == 0
    rank = dist.get_rank()

    w = torch.randn(out_f, in_f, dtype=_DTYPE)
    b = torch.randn(out_f, dtype=_DTYPE)
    # Batch-first: [B, S, H]
    x_full = torch.randn(_BATCH, _SEQ, in_f, dtype=_DTYPE)
    y_ref_full = F.linear(x_full.float(), w.float(), b.float()).to(_DTYPE)
    out_chunk = out_f // world_size
    y_ref_local = y_ref_full[:, :, rank * out_chunk:(rank + 1) * out_chunk]

    seq_local = _SEQ // world_size
    x_local = x_full[:, rank * seq_local:(rank + 1) * seq_local].contiguous()

    linear = to_device(nn.Linear(in_f, out_f, bias=True, dtype=_DTYPE), _DEVICE_TYPE)
    with torch.no_grad():
        linear.weight.copy_(to_device(w, _DEVICE_TYPE))
        linear.bias.copy_(to_device(b, _DEVICE_TYPE))
    x_npu = to_device(x_local, _DEVICE_TYPE)

    sharded = parallelize_module(
        linear,
        mesh,
        MC2ColwiseParallel(input_layouts=Shard(1), use_local_output=True),
        src_data_rank=None,
    )
    with torch.no_grad():
        y_hp = sharded(x_npu)

    # After fused all-gather on sequence dim 1, local output has full sequence.
    assert y_hp.shape[1] == _SEQ
    assert y_hp.shape[-1] == out_chunk
    _bf16_close(y_hp, to_device(y_ref_local, _DEVICE_TYPE))
