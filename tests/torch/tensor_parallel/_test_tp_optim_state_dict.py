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
"""Optimizer state dict tests for Tensor Parallelism (TP).

Verified scenarios:
  T1 (4-card): TP-only get_optim_state_dict FQN roundtrip (default options)
  T2 (4-card): TP-only cpu_offload roundtrip
  T3 (4-card): TP+FSDP hybrid get_optim_state_dict FQN roundtrip
  T4 (4-card): TP+FSDP hybrid full_state_dict + cpu_offload + broadcast_from_rank0
  T5 (4-card): TP+FSDP hybrid flatten roundtrip
  T6 (4-card): TP+FSDP hybrid local shape correctness

Key design decisions for TP optimizer state dict testing:
  - Pure TP: parameters become DTensors (ColwiseParallel produces Shard(0),
    RowwiseParallel produces Shard(1) on weight). Optimizer states are local
    shard tensors. get/set_optim_state_dict must handle DTensor parameter
    conversion correctly.
  - TP+FSDP: TP first then FSDP. The FSDP layer manages parameter sharding
    on the DP dimension, and TP manages on the TP dimension. The composed
    parameter is a 2-D DTensor. Optimizer states follow local shard views.
  - For pure TP, all ranks hold the same model (replicated computation after
    all-reduce/all-gather), so optimizer states should be identical across
    ranks. For TP+FSDP, DP ranks within the same TP group shard the model.
"""
import math
import os
import shutil

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"  # pylint: disable=wrong-import-position

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: F401,E402  # pylint: disable=unused-import
from torch import nn  # noqa: E402
from torch.distributed.checkpoint.state_dict import StateDictOptions  # noqa: E402

from hyper_parallel import (  # noqa: E402
    ColwiseParallel,
    RowwiseParallel,
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

IN_F = 32
HIDDEN_F = 64
OUT_F = 32
BATCH = 4

_CKPT_DIR_NESTED = "/tmp/hp_tp_optim_sd_test_nested"


class MLP(nn.Module):
    """Two-layer MLP used for TP tests."""

    def __init__(self, in_f: int, hidden_f: int, out_f: int):
        super().__init__()
        self.w1 = nn.Linear(in_f, hidden_f, bias=True)
        self.w2 = nn.Linear(hidden_f, out_f, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: two-layer MLP with ReLU activation."""
        return self.w2(F.relu(self.w1(x)))


def _rank():
    return dist.get_rank()


def _cleanup_ckpt(path):
    if _rank() == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()


# =====================================================================
# Pure TP helpers
# =====================================================================
def _make_tp_model():
    """Build a TP-only model on a 1-D tp mesh (4 cards)."""
    init_dist()
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4,),
        mesh_dim_names=("tp",),
    )
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    model = MLP(IN_F, HIDDEN_F, OUT_F).npu()
    parallelize_module(
        model,
        mesh,
        {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
    )
    return model, mesh


def _tp_train_step(model, optimizer, x):
    """Train step for pure TP model."""
    optimizer.zero_grad()
    out = model(x)
    loss = out.sum()
    loss.backward()
    with SkipDTensorDispatch():
        optimizer.step()
    return loss.item()


# =====================================================================
# TP+FSDP helpers
# =====================================================================
def _make_tp_fsdp_model():
    """Build a TP+FSDP model on a 2-D mesh (dp=2, tp=2)."""
    init_dist()
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    model = MLP(IN_F, HIDDEN_F, OUT_F).npu()

    parallelize_module(
        model,
        tp_mesh,
        {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
    )

    mp = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    fully_shard(model, mesh=dp_mesh, reshard_after_forward=True, mp_policy=mp)

    return model, mesh, dp_mesh, tp_mesh


def _tp_fsdp_train_step(model, optimizer, x):
    """Train step for TP+FSDP model."""
    optimizer.zero_grad()
    out = model(x)
    loss = out.sum()
    loss.backward()
    with SkipDTensorDispatch():
        optimizer.step()
    return loss.item()


# =====================================================================
# T1: TP-only FQN roundtrip
# =====================================================================
def test_t1_tp_optim_state_dict_fqn_roundtrip():
    """TP-only: get_optim_state_dict (default) -> set_optim_state_dict -> step."""
    model, _ = _make_tp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, IN_F).npu()

    for _ in range(2):
        _tp_train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert not isinstance(value, DTensor), (
                    f"state.{fqn}.{key} should be plain Tensor, got DTensor"
                )

    model2, _ = _make_tp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _tp_train_step(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd)
    loss_after = _tp_train_step(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] T1 PASS: TP-only FQN roundtrip (loss={loss_after:.4f})")


# =====================================================================
# T2: TP-only cpu_offload roundtrip
# =====================================================================
def test_t2_tp_optim_state_dict_cpu_offload():
    """TP-only: get with cpu_offload -> set -> step."""
    model, _ = _make_tp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, IN_F).npu()

    for _ in range(2):
        _tp_train_step(model, optimizer, x)

    opts = StateDictOptions(cpu_offload=True)
    sd = get_optim_state_dict(model, optimizer, options=opts)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert value.device == torch.device("cpu"), (
                    f"state.{fqn}.{key} should be on CPU, got {value.device}"
                )

    model2, _ = _make_tp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _tp_train_step(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd, options=opts)

    raw_sd2 = optimizer2.state_dict()
    for param_id, state in raw_sd2["state"].items():
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "step":
                continue
            assert value.device.type == "npu", (
                f"state[{param_id}].{key} should be on NPU after set, got {value.device}"
            )

    loss_after = _tp_train_step(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after cpu_offload roundtrip: {loss_after}"

    print(f"[rank{_rank()}] T2 PASS: TP-only cpu_offload roundtrip (loss={loss_after:.4f})")


# =====================================================================
# T3: TP+FSDP FQN roundtrip
# =====================================================================
def test_t3_tp_fsdp_optim_state_dict_fqn_roundtrip():
    """TP+FSDP: get_optim_state_dict (default) -> set_optim_state_dict -> step."""
    model, _, _, _ = _make_tp_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, IN_F).npu()

    for _ in range(2):
        _tp_fsdp_train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert not isinstance(value, DTensor), (
                    f"state.{fqn}.{key} should be plain Tensor, got DTensor"
                )

    model2, _, _, _ = _make_tp_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _tp_fsdp_train_step(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd)
    loss_after = _tp_fsdp_train_step(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] T3 PASS: TP+FSDP FQN roundtrip (loss={loss_after:.4f})")


# =====================================================================
# T4: TP+FSDP full_state_dict + cpu_offload + broadcast_from_rank0
# =====================================================================
def test_t4_tp_fsdp_optim_state_dict_full_cpu_broadcast():
    """TP+FSDP: get with full_state_dict+cpu_offload -> set with broadcast -> step."""
    model, _, _, _ = _make_tp_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, IN_F).npu()

    for _ in range(2):
        _tp_fsdp_train_step(model, optimizer, x)

    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(model, optimizer, options=opts_get)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu", (
                    f"state.{fqn}.{key} should be on CPU, got {value.device}"
                )

    model2, _, _, _ = _make_tp_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _tp_fsdp_train_step(model2, optimizer2, x)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    set_optim_state_dict(model2, optimizer2, sd, options=opts_set)

    loss_after = _tp_fsdp_train_step(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after full+cpu+broadcast: {loss_after}"

    print(f"[rank{_rank()}] T4 PASS: TP+FSDP full+cpu+broadcast (loss={loss_after:.4f})")


# =====================================================================
# T5: TP+FSDP flatten roundtrip
# =====================================================================
def test_t5_tp_fsdp_optim_state_dict_flatten():
    """TP+FSDP: get with flatten_optimizer_state_dict=True -> set -> step."""
    model, _, _, _ = _make_tp_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, IN_F).npu()

    for _ in range(2):
        _tp_fsdp_train_step(model, optimizer, x)

    opts = StateDictOptions(flatten_optimizer_state_dict=True)
    sd = get_optim_state_dict(model, optimizer, options=opts)

    for key in sd.keys():
        assert key.startswith("state.") or key.startswith("param_group."), (
            f"flat key '{key}' should start with 'state.' or 'param_group.'"
        )

    model2, _, _, _ = _make_tp_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _tp_fsdp_train_step(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd, options=opts)
    loss_after = _tp_fsdp_train_step(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after flatten roundtrip: {loss_after}"

    print(f"[rank{_rank()}] T5 PASS: TP+FSDP flatten roundtrip (loss={loss_after:.4f})")


# =====================================================================
# T6: TP+FSDP local shape correctness
# =====================================================================
def test_t6_tp_fsdp_local_shape_correctness():
    """Verify optimizer state tensors have correct local shard shape under TP+FSDP."""
    model, _, _, _ = _make_tp_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, IN_F).npu()

    for _ in range(2):
        _tp_fsdp_train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    dp_size = 2

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
            assert value.shape[0] <= HIDDEN_F, (
                f"state.{fqn}.{key}: local dim0={value.shape[0]} should be "
                f"<= full dim0={HIDDEN_F}"
            )
            assert value.shape[0] < HIDDEN_F or dp_size == 1, (
                f"state.{fqn}.{key}: with dp_size={dp_size}, local dim0={value.shape[0]} "
                f"should be less than full {HIDDEN_F}"
            )

    print(f"[rank{_rank()}] T6 PASS: TP+FSDP local shape correctness")
