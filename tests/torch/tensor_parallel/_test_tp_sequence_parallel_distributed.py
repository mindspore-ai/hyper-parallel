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
"""Distributed NPU worker tests for ``SequenceParallel`` (vs CPU single-device ref).

Six scenarios: **four** on 2 ranks (``world_size == 2``), **two** on 4 ranks
(``world_size == 4``). Each compares NPU sharded execution to a **CPU float32**
reference (analytic single-process baseline, ``torch.no_grad()`` / autograd on CPU).
"""
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import SequenceParallel, init_device_mesh, parallelize_module
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device


def _make_tp_mesh_1d():
    return init_device_mesh(
        device_type=_DEVICE_TYPE,
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


def _gather_seq_dim1(local_y: torch.Tensor) -> torch.Tensor:
    """Concatenate sequence shards from all ranks along dim=1."""
    ws = dist.get_world_size()
    parts = [torch.empty_like(local_y) for _ in range(ws)]
    dist.all_gather(parts, local_y)
    return torch.cat(parts, dim=1)


def _param_grad_npu_tensor(param) -> torch.Tensor:
    """Plain NPU tensor for ``param.grad`` (handles optional DTensor-like grads)."""
    g = param.grad
    if g is None:
        raise AssertionError("expected param.grad to be populated")
    to_local = getattr(g, "to_local", None)
    if callable(to_local):
        return to_local()
    return g


# ---------------------------------------------------------------------------
# 2-GPU cases (world_size == 2)
# ---------------------------------------------------------------------------


def test_sequence_parallel_layernorm_forward_chunk_vs_cpu_2gpu():
    """
    Feature: per-rank sequence shard vs CPU slice (2 ranks)
    Description: CPU full forward; each NPU rank compares its local output to the
        matching ``y_ref`` time slice.
    Expectation: Chunk-wise match within float32 tolerance.
    """
    init_backend(_DEVICE_TYPE)
    assert dist.get_world_size() == 2, "this case is scheduled with num_proc=2"
    mesh = _make_tp_mesh_1d()
    rank = dist.get_rank()
    ws = dist.get_world_size()

    bsz, seq_len, hidden = 2, 16, 32
    assert seq_len % ws == 0

    torch.manual_seed(201)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(201)

    ln_cpu = nn.LayerNorm(hidden, elementwise_affine=True)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)
    with torch.no_grad():
        y_ref = ln_cpu(x_cpu)

    ln_npu = to_device(nn.LayerNorm(hidden, elementwise_affine=True), _DEVICE_TYPE)
    with torch.no_grad():
        ln_npu.load_state_dict(ln_cpu.state_dict())

    sharded = parallelize_module(
        ln_npu, mesh,
        SequenceParallel(sequence_dim=1, use_local_output=True),
    )
    chunk = seq_len // ws
    sl = slice(rank * chunk, (rank + 1) * chunk)
    x_local = to_device(x_cpu[:, sl, :], _DEVICE_TYPE)

    with torch.no_grad():
        y_local = sharded(x_local)

    _npu_precision_close(y_local, y_ref[:, sl, :])


def test_sequence_parallel_layernorm_forward_gather_full_vs_cpu_2gpu():
    """
    Feature: all_gather sequence outputs match full CPU reference (2 ranks)
    Description: Reconstruct global ``(B, S, H)`` on rank 0 path by concatenating
        per-rank outputs; compare to single-device CPU forward.
    Expectation: Full tensor match vs CPU.
    """
    init_backend(_DEVICE_TYPE)
    assert dist.get_world_size() == 2
    mesh = _make_tp_mesh_1d()
    rank = dist.get_rank()
    ws = dist.get_world_size()

    bsz, seq_len, hidden = 3, 12, 24
    assert seq_len % ws == 0

    torch.manual_seed(202)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(202)

    ln_cpu = nn.LayerNorm(hidden, elementwise_affine=True)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)
    with torch.no_grad():
        y_ref = ln_cpu(x_cpu)

    ln_npu = to_device(nn.LayerNorm(hidden, elementwise_affine=True), _DEVICE_TYPE)
    with torch.no_grad():
        ln_npu.load_state_dict(ln_cpu.state_dict())

    sharded = parallelize_module(
        ln_npu, mesh,
        SequenceParallel(sequence_dim=1, use_local_output=True),
    )
    chunk = seq_len // ws
    sl = slice(rank * chunk, (rank + 1) * chunk)
    x_local = to_device(x_cpu[:, sl, :], _DEVICE_TYPE)

    with torch.no_grad():
        y_local = sharded(x_local)

    y_full = _gather_seq_dim1(y_local)
    if rank == 0:
        _npu_precision_close(y_full, y_ref)


def test_sequence_parallel_dropout_identity_vs_cpu_2gpu():
    """
    Feature: Dropout(p=0) under SequenceParallel equals identity vs CPU shard (2 ranks)
    Description: ``eval`` mode; local output should match local input (same as CPU).
    Expectation: ``y_local`` close to ``x_local``.
    """
    init_backend(_DEVICE_TYPE)
    assert dist.get_world_size() == 2
    mesh = _make_tp_mesh_1d()
    rank = dist.get_rank()
    ws = dist.get_world_size()

    bsz, seq_len, hidden = 2, 8, 16
    assert seq_len % ws == 0

    torch.manual_seed(203)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(203)

    drop_cpu = nn.Dropout(p=0.0).eval()
    drop_npu = to_device(nn.Dropout(p=0.0), _DEVICE_TYPE).eval()
    sharded = parallelize_module(
        drop_npu, mesh,
        SequenceParallel(sequence_dim=1, use_local_output=True),
    )

    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)
    chunk = seq_len // ws
    sl = slice(rank * chunk, (rank + 1) * chunk)
    x_shard_cpu = x_cpu[:, sl, :]
    with torch.no_grad():
        y_ref_shard = drop_cpu(x_shard_cpu)

    x_local = to_device(x_shard_cpu, _DEVICE_TYPE)
    with torch.no_grad():
        y_local = sharded(x_local)

    _npu_precision_close(y_local, to_device(y_ref_shard, _DEVICE_TYPE))


def test_sequence_parallel_layernorm_no_affine_forward_vs_cpu_2gpu():
    """
    Feature: LayerNorm (elementwise_affine=False) shard vs CPU (2 ranks)
    Description: No learnable weight/bias; compare chunk and implicit full-graph
        consistency via gather on rank 0.
    Expectation: Matches CPU reference (chunk + gathered full).
    """
    init_backend(_DEVICE_TYPE)
    assert dist.get_world_size() == 2
    mesh = _make_tp_mesh_1d()
    rank = dist.get_rank()
    ws = dist.get_world_size()

    bsz, seq_len, hidden = 2, 14, 20
    assert seq_len % ws == 0

    torch.manual_seed(204)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(204)

    ln_cpu = nn.LayerNorm(hidden, elementwise_affine=False)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)
    with torch.no_grad():
        y_ref = ln_cpu(x_cpu)

    ln_npu = to_device(nn.LayerNorm(hidden, elementwise_affine=False), _DEVICE_TYPE)
    sharded = parallelize_module(
        ln_npu, mesh,
        SequenceParallel(sequence_dim=1, use_local_output=True),
    )
    chunk = seq_len // ws
    sl = slice(rank * chunk, (rank + 1) * chunk)
    x_local = to_device(x_cpu[:, sl, :], _DEVICE_TYPE)

    with torch.no_grad():
        y_local = sharded(x_local)

    _npu_precision_close(y_local, y_ref[:, sl, :])
    y_full = _gather_seq_dim1(y_local)
    if rank == 0:
        _npu_precision_close(y_full, y_ref)


# ---------------------------------------------------------------------------
# 4-GPU cases (world_size == 4)
# ---------------------------------------------------------------------------


def test_sequence_parallel_layernorm_fwd_bwd_vs_cpu_4gpu():
    """
    Feature: SequenceParallel LayerNorm forward + backward on 4 ranks (one torchrun)
    Description:
        1. Forward: gather full output vs CPU LayerNorm
        2. Backward: all-reduce weight/bias grads vs CPU
    Expectation: Forward and grads close to CPU reference
    """
    init_backend(_DEVICE_TYPE)
    assert dist.get_world_size() == 4, "this case is scheduled with num_proc=4"
    mesh = _make_tp_mesh_1d()
    rank = dist.get_rank()
    ws = dist.get_world_size()

    # --- forward ---
    bsz, seq_len, hidden = 2, 32, 40
    assert seq_len % ws == 0
    torch.manual_seed(301)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(301)

    ln_cpu = nn.LayerNorm(hidden, elementwise_affine=True)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)
    with torch.no_grad():
        y_ref = ln_cpu(x_cpu)

    ln_npu = to_device(nn.LayerNorm(hidden, elementwise_affine=True), _DEVICE_TYPE)
    with torch.no_grad():
        ln_npu.load_state_dict(ln_cpu.state_dict())
    sharded = parallelize_module(
        ln_npu, mesh,
        SequenceParallel(sequence_dim=1, use_local_output=True),
    )
    chunk = seq_len // ws
    sl = slice(rank * chunk, (rank + 1) * chunk)
    with torch.no_grad():
        y_local = sharded(to_device(x_cpu[:, sl, :], _DEVICE_TYPE))
    y_full = _gather_seq_dim1(y_local)
    if rank == 0:
        _npu_precision_close(y_full, y_ref)

    # --- backward ---
    bsz, seq_len, hidden = 2, 16, 28
    assert seq_len % ws == 0
    torch.manual_seed(302)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(302)

    ln_cpu = nn.LayerNorm(hidden, elementwise_affine=True)
    x_cpu = torch.randn(bsz, seq_len, hidden, dtype=torch.float32)
    x_cpu_grad = x_cpu.clone().requires_grad_(True)
    y_ref = ln_cpu(x_cpu_grad)
    y_ref.sum().backward()

    ln_npu = to_device(nn.LayerNorm(hidden, elementwise_affine=True), _DEVICE_TYPE)
    with torch.no_grad():
        ln_npu.load_state_dict(ln_cpu.state_dict())
    sharded = parallelize_module(
        ln_npu, mesh,
        SequenceParallel(sequence_dim=1, use_local_output=True),
    )
    chunk = seq_len // ws
    sl = slice(rank * chunk, (rank + 1) * chunk)
    x_local = to_device(x_cpu[:, sl, :].clone(), _DEVICE_TYPE).requires_grad_(True)
    sharded(x_local).sum().backward()

    w_grad = _param_grad_npu_tensor(ln_npu.weight).clone()
    b_grad = _param_grad_npu_tensor(ln_npu.bias).clone()
    dist.all_reduce(w_grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(b_grad, op=dist.ReduceOp.SUM)
    if rank == 0:
        _npu_precision_close(w_grad, ln_cpu.weight.grad)
        _npu_precision_close(b_grad, ln_cpu.bias.grad)
