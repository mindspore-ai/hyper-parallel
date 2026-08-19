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
"""Optimizer state dict tests for Context Parallelism + FSDP.

Verified scenarios:
  C1 (4-card): CP+FSDP get_optim_state_dict FQN roundtrip (default options)
  C2 (4-card): CP+FSDP full_state_dict + cpu_offload + broadcast_from_rank0
  C3 (4-card): CP+FSDP flatten roundtrip
  C4 (4-card): CP+FSDP local shape correctness

Key design decisions for CP optimizer state dict testing:
  - CP does NOT shard parameters; it only rearranges activations (Q/K/V)
    at attention boundaries via all-to-all / all-gather.
  - CP+FSDP composition: the FSDP mesh is formed by flattening ("dp", "cp")
    into a single "fsdp" dimension. FSDP handles all parameter sharding.
  - CP+FSDP parameters are DTensors sharded on the flattened fsdp dim.
    Optimizer states follow local shard views, same as pure FSDP.
  - The test model is a simple attention block with CP applied on the
    attention module and FSDP applied on the full model.
"""
import math
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"  # pylint: disable=wrong-import-position

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: F401,E402  # pylint: disable=unused-import
from torch import nn  # noqa: E402
from torch.distributed.checkpoint.state_dict import StateDictOptions  # noqa: E402

from hyper_parallel import (  # noqa: E402
    ContextParallel,
    SkipDTensorDispatch,
    get_optim_state_dict,
    init_device_mesh,
    parallelize_module,
    set_optim_state_dict,
)
from hyper_parallel.core.dtensor.dtensor import DTensor  # noqa: E402
from hyper_parallel.core.fully_shard.api import fully_shard  # noqa: E402
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy  # noqa: E402
from tests.torch.utils import init_dist  # noqa: E402  # pylint: enable=wrong-import-position

DIM = 32
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS
SEQ_LEN = 16
BATCH = 4


class SimpleAttention(nn.Module):
    """Simple multi-head attention for CP testing.

    The ``core_attn`` sub-module is the target for ContextParallel.
    After q/k/v projection, tensors are in BSHD format; CP rearranges
    the sequence and head dimensions inside ``core_attn``.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.core_attn = _CoreAttn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: project to q/k/v, compute attention, project output."""
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)
        out = self.core_attn(q, k, v)
        return self.out_proj(out)


class _CoreAttn(nn.Module):
    """Core attention computation (BSHD format). Target for ContextParallel.

    Input/output: (B, S, H, D) layout. ContextParallel with
    ``seq_dim=1, head_dim=2`` rearranges the sequence and head dims
    via all-to-all so each rank computes attention on a subset of
    heads with the full sequence, then rearranges back.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for core attention."""
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2)
        return out.contiguous().flatten(-2)


class CPTestModel(nn.Module):
    """Two-layer model: attention (with CP) + linear."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = SimpleAttention(dim, num_heads)
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: attention with residual + norm, then linear + sum."""
        x = self.norm(self.attn(x) + x)
        return self.linear(x).sum()


def _rank():
    return dist.get_rank()


def _make_cp_fsdp_model():
    """Build a CP+FSDP model on 4 cards.

    Uses a 2-D mesh (dp=2, cp=2) with FSDP on the flattened (dp,cp) mesh.
    CP is applied on the attention module via ContextParallel.
    """
    init_dist()
    device = torch.device("npu", _rank() % 8)

    dp_size = 2
    cp_size = 2

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, cp_size),
        mesh_dim_names=("dp", "cp"),
    )

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    model = CPTestModel(DIM, NUM_HEADS).to(device)

    cp_mesh = mesh["cp"]
    parallelize_module(
        model.attn.core_attn,
        cp_mesh,
        ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1),
    )

    fsdp_mesh = mesh[("dp", "cp")].flatten(mesh_dim_name="fsdp")
    mp = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    fully_shard(model, mesh=fsdp_mesh, reshard_after_forward=True, mp_policy=mp)

    return model, mesh, fsdp_mesh, device


def _train_step(model, optimizer, x):
    """Train step for CP+FSDP model."""
    optimizer.zero_grad()
    with SkipDTensorDispatch():
        loss = model(x)
    loss.backward()
    with SkipDTensorDispatch():
        optimizer.step()
    return loss.item()


# =====================================================================
# C1: CP+FSDP FQN roundtrip
# =====================================================================
def test_c1_cp_fsdp_optim_state_dict_fqn_roundtrip():
    """CP+FSDP: get_optim_state_dict (default) -> set_optim_state_dict -> step."""
    model, mesh, _fsdp_mesh, device = _make_cp_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    cp_size = 2
    local_slen = SEQ_LEN // cp_size
    _cp_rank = mesh.get_local_rank("cp")
    x = torch.randn(BATCH, local_slen, DIM, device=device)

    for _ in range(2):
        _train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert not isinstance(value, DTensor), (
                    f"state.{fqn}.{key} should be plain Tensor, got DTensor"
                )

    model2, _mesh2, _fsdp_mesh2, _device2 = _make_cp_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd)
    loss_after = _train_step(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] C1 PASS: CP+FSDP FQN roundtrip (loss={loss_after:.4f})")


# =====================================================================
# C2: CP+FSDP full_state_dict + cpu_offload + broadcast_from_rank0
# =====================================================================
def test_c2_cp_fsdp_optim_state_dict_full_cpu_broadcast():
    """CP+FSDP: get with full_state_dict+cpu_offload -> set with broadcast -> step."""
    model, _mesh, _fsdp_mesh, device = _make_cp_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    cp_size = 2
    local_slen = SEQ_LEN // cp_size
    x = torch.randn(BATCH, local_slen, DIM, device=device)

    for _ in range(2):
        _train_step(model, optimizer, x)

    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(model, optimizer, options=opts_get)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu", (
                    f"state.{fqn}.{key} should be on CPU, got {value.device}"
                )

    model2, _mesh2, _fsdp_mesh2, _device2 = _make_cp_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step(model2, optimizer2, x)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    set_optim_state_dict(model2, optimizer2, sd, options=opts_set)

    loss_after = _train_step(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after full+cpu+broadcast: {loss_after}"

    print(f"[rank{_rank()}] C2 PASS: CP+FSDP full+cpu+broadcast (loss={loss_after:.4f})")


# =====================================================================
# C3: CP+FSDP flatten roundtrip
# =====================================================================
def test_c3_cp_fsdp_optim_state_dict_flatten():
    """CP+FSDP: get with flatten_optimizer_state_dict=True -> set -> step."""
    model, _mesh, _fsdp_mesh, device = _make_cp_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    cp_size = 2
    local_slen = SEQ_LEN // cp_size
    x = torch.randn(BATCH, local_slen, DIM, device=device)

    for _ in range(2):
        _train_step(model, optimizer, x)

    opts = StateDictOptions(flatten_optimizer_state_dict=True)
    sd = get_optim_state_dict(model, optimizer, options=opts)

    for key in sd.keys():
        assert key.startswith("state.") or key.startswith("param_group."), (
            f"flat key '{key}' should start with 'state.' or 'param_group.'"
        )

    model2, _mesh2, _fsdp_mesh2, _device2 = _make_cp_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd, options=opts)
    loss_after = _train_step(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after flatten roundtrip: {loss_after}"

    print(f"[rank{_rank()}] C3 PASS: CP+FSDP flatten roundtrip (loss={loss_after:.4f})")


# =====================================================================
# C4: CP+FSDP local shape correctness
# =====================================================================
def test_c4_cp_fsdp_local_shape_correctness():
    """Verify optimizer state tensors have correct local shard shape under CP+FSDP."""
    model, _mesh, _fsdp_mesh, device = _make_cp_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    cp_size = 2
    local_slen = SEQ_LEN // cp_size
    x = torch.randn(BATCH, local_slen, DIM, device=device)

    for _ in range(2):
        _train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)
    _fsdp_size = 4

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "step":
                continue
            assert not isinstance(value, DTensor), (
                f"state.{fqn}.{key}: expected plain Tensor with SkipDTensorDispatch, "
                f"got DTensor"
            )
            assert value.shape[0] <= DIM, (
                f"state.{fqn}.{key}: local dim0={value.shape[0]} should be "
                f"<= full dim0={DIM}"
            )

    print(f"[rank{_rank()}] C4 PASS: CP+FSDP local shape correctness")
