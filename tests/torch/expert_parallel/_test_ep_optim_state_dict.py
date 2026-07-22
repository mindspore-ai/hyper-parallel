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
"""Optimizer state dict tests for Expert Parallelism (EP) and EP+FSDP.

Verified scenarios:
  E1 (4-card): EP-only get_optim_state_dict FQN roundtrip (default options)
  E2 (4-card): EP-only cpu_offload roundtrip
  E3 (4-card): FSDP+EP hybrid get_optim_state_dict FQN roundtrip
  E4 (4-card): FSDP+EP hybrid full_state_dict + cpu_offload + broadcast_from_rank0
  E5 (4-card): FSDP+EP hybrid flatten roundtrip
  E6 (4-card): FSDP+EP hybrid local shape correctness

Key design decisions for EP optimizer state dict testing:
  - EP shards experts across ranks: each rank holds a subset of experts.
    ExpertParallel partitions all expert parameters via Shard(0) on the EP dim.
  - FSDP+EP: EP first (shard on ep dim), then FSDP (shard on fsdp dim).
    The composed parameter is a 2-D DTensor with placements [Shard(0), Shard(0)].
  - Gradient scaling: FSDP+EP inflates gradients by fsdp_size * ep_size = world_size.
    Training must divide backward loss by world_size.
  - Each EP rank holds different experts, so different FQNs appear in each rank's
    optimizer state dict. broadcast_from_rank0 only makes sense within DP groups
    where the same FQNs are present.
"""
import os
import shutil

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: F401,E402
from torch.distributed.checkpoint.state_dict import StateDictOptions  # noqa: E402

from hyper_parallel import (  # noqa: E402
    get_optim_state_dict,
    init_device_mesh,
    set_optim_state_dict,
)
from hyper_parallel.core.distributed_checkpoint import (  # noqa: E402
    FileSystemReader,
    save,
    load,
)
from hyper_parallel.core.dtensor.dtensor import DTensor  # noqa: E402
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel  # noqa: E402
from hyper_parallel.core.fully_shard.api import fully_shard  # noqa: E402
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy  # noqa: E402
from hyper_parallel.platform.torch.common import MoE  # noqa: E402
from hyper_parallel.platform.torch.fully_shard.optim_state_dict_utils import (  # noqa: E402
    _build_optim_state_dict_load_template,
)
from tests.torch.utils import init_dist  # noqa: E402

DIM = 64
HIDDEN_DIM = 128
NUM_EXPERTS = 4
TOP_K = 2
BS = 4
SLEN = 16

_CKPT_DIR_NESTED = "/tmp/hp_ep_optim_sd_test_nested"


def _rank():
    return dist.get_rank()


def _cleanup_ckpt(path):
    if _rank() == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()


# =====================================================================
# EP-only helpers
# =====================================================================
def _make_ep_model():
    """Build an EP-only MoE model on a 1-D ep mesh (4 cards)."""
    init_dist()
    device = torch.device("npu", _rank() % 8)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4,),
        mesh_dim_names=("ep",),
    )

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    moe = MoE(dim=DIM, hidden_dim=HIDDEN_DIM, num_experts=NUM_EXPERTS, top_k=TOP_K)
    moe = moe.to(device)

    ExpertParallel().apply(moe.experts, mesh)

    return moe, mesh, device


def _ep_train_step(moe, optimizer, x):
    """Train step for EP-only model."""
    optimizer.zero_grad()
    out = moe(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    return loss.item()


# =====================================================================
# FSDP+EP helpers
# =====================================================================
def _make_fsdp_ep_model():
    """Build a FSDP+EP MoE model on a 2-D mesh (fsdp=2, ep=2)."""
    init_dist()
    device = torch.device("npu", _rank() % 8)

    fsdp_size = 2
    ep_size = 2

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(fsdp_size, ep_size),
        mesh_dim_names=("fsdp", "ep"),
    )

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    moe = MoE(dim=DIM, hidden_dim=HIDDEN_DIM, num_experts=NUM_EXPERTS, top_k=TOP_K)
    moe = moe.to(device)

    ExpertParallel().apply(moe.experts, mesh["ep"])

    mp = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    fully_shard(moe, mesh=mesh["fsdp"], reshard_after_forward=True, mp_policy=mp)

    grad_scale = fsdp_size * ep_size
    return moe, mesh, device, grad_scale


def _fsdp_ep_train_step(moe, optimizer, x, grad_scale):
    """Train step for FSDP+EP model."""
    optimizer.zero_grad()
    out = moe(x)
    loss = out.sum()
    (loss / grad_scale).backward()
    optimizer.step()
    return loss.item()


# =====================================================================
# E1: EP-only FQN roundtrip
# =====================================================================
def test_e1_ep_optim_state_dict_fqn_roundtrip():
    """EP-only: get_optim_state_dict (default) -> set_optim_state_dict -> step."""
    moe, mesh, device = _make_ep_model()
    optimizer = torch.optim.SGD(moe.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(BS, SLEN, DIM, device=device)

    for _ in range(2):
        _ep_train_step(moe, optimizer, x)

    sd = get_optim_state_dict(moe, optimizer)
    assert len(sd["state"]) > 0, "optimizer should have non-empty state after 2 steps"

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert not isinstance(value, DTensor), (
                    f"state.{fqn}.{key} should be plain Tensor, got DTensor"
                )

    moe2, mesh2, device2 = _make_ep_model()
    optimizer2 = torch.optim.SGD(moe2.parameters(), lr=0.01, momentum=0.9)
    _ep_train_step(moe2, optimizer2, x)

    set_optim_state_dict(moe2, optimizer2, sd)
    loss_after = _ep_train_step(moe2, optimizer2, x)
    assert not (loss_after != loss_after), f"loss is NaN after roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] E1 PASS: EP-only FQN roundtrip (loss={loss_after:.4f})")


# =====================================================================
# E2: EP-only cpu_offload roundtrip
# =====================================================================
def test_e2_ep_optim_state_dict_cpu_offload():
    """EP-only: get with cpu_offload -> set -> step."""
    moe, mesh, device = _make_ep_model()
    optimizer = torch.optim.SGD(moe.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(BS, SLEN, DIM, device=device)

    for _ in range(2):
        _ep_train_step(moe, optimizer, x)

    opts = StateDictOptions(cpu_offload=True)
    sd = get_optim_state_dict(moe, optimizer, options=opts)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert value.device == torch.device("cpu"), (
                    f"state.{fqn}.{key} should be on CPU, got {value.device}"
                )

    moe2, mesh2, device2 = _make_ep_model()
    optimizer2 = torch.optim.SGD(moe2.parameters(), lr=0.01, momentum=0.9)
    _ep_train_step(moe2, optimizer2, x)

    set_optim_state_dict(moe2, optimizer2, sd, options=opts)

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

    loss_after = _ep_train_step(moe2, optimizer2, x)
    assert not (loss_after != loss_after), f"loss is NaN after cpu_offload roundtrip: {loss_after}"

    print(f"[rank{_rank()}] E2 PASS: EP-only cpu_offload roundtrip (loss={loss_after:.4f})")


# =====================================================================
# E3: FSDP+EP FQN roundtrip
# =====================================================================
def test_e3_fsdp_ep_optim_state_dict_fqn_roundtrip():
    """FSDP+EP: get_optim_state_dict (default) -> set_optim_state_dict -> step."""
    moe, mesh, device, grad_scale = _make_fsdp_ep_model()
    optimizer = torch.optim.SGD(moe.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(BS, SLEN, DIM, device=device)

    for _ in range(2):
        _fsdp_ep_train_step(moe, optimizer, x, grad_scale)

    sd = get_optim_state_dict(moe, optimizer)
    assert len(sd["state"]) > 0, "optimizer should have non-empty state after 2 steps"

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert not isinstance(value, DTensor), (
                    f"state.{fqn}.{key} should be plain Tensor, got DTensor"
                )

    moe2, mesh2, device2, grad_scale2 = _make_fsdp_ep_model()
    optimizer2 = torch.optim.SGD(moe2.parameters(), lr=0.01, momentum=0.9)
    _fsdp_ep_train_step(moe2, optimizer2, x, grad_scale2)

    set_optim_state_dict(moe2, optimizer2, sd)
    loss_after = _fsdp_ep_train_step(moe2, optimizer2, x, grad_scale2)
    assert not (loss_after != loss_after), f"loss is NaN after roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] E3 PASS: FSDP+EP FQN roundtrip (loss={loss_after:.4f})")


# =====================================================================
# E4: FSDP+EP full_state_dict + cpu_offload + broadcast_from_rank0
# =====================================================================
def test_e4_fsdp_ep_optim_state_dict_full_cpu_broadcast():
    """FSDP+EP: get with full_state_dict+cpu_offload -> set with broadcast -> step."""
    moe, mesh, device, grad_scale = _make_fsdp_ep_model()
    optimizer = torch.optim.SGD(moe.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(BS, SLEN, DIM, device=device)

    for _ in range(2):
        _fsdp_ep_train_step(moe, optimizer, x, grad_scale)

    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(moe, optimizer, options=opts_get)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu", (
                    f"state.{fqn}.{key} should be on CPU, got {value.device}"
                )

    moe2, mesh2, device2, grad_scale2 = _make_fsdp_ep_model()
    optimizer2 = torch.optim.SGD(moe2.parameters(), lr=0.01, momentum=0.9)
    _fsdp_ep_train_step(moe2, optimizer2, x, grad_scale2)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    set_optim_state_dict(moe2, optimizer2, sd, options=opts_set)

    loss_after = _fsdp_ep_train_step(moe2, optimizer2, x, grad_scale2)
    assert not (loss_after != loss_after), f"loss is NaN after full+cpu+broadcast: {loss_after}"

    print(f"[rank{_rank()}] E4 PASS: FSDP+EP full+cpu+broadcast (loss={loss_after:.4f})")


# =====================================================================
# E5: FSDP+EP flatten roundtrip
# =====================================================================
def test_e5_fsdp_ep_optim_state_dict_flatten():
    """FSDP+EP: get with flatten_optimizer_state_dict=True -> set -> step."""
    moe, mesh, device, grad_scale = _make_fsdp_ep_model()
    optimizer = torch.optim.SGD(moe.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(BS, SLEN, DIM, device=device)

    for _ in range(2):
        _fsdp_ep_train_step(moe, optimizer, x, grad_scale)

    opts = StateDictOptions(flatten_optimizer_state_dict=True)
    sd = get_optim_state_dict(moe, optimizer, options=opts)

    for key in sd.keys():
        assert key.startswith("state.") or key.startswith("param_group."), (
            f"flat key '{key}' should start with 'state.' or 'param_group.'"
        )

    moe2, mesh2, device2, grad_scale2 = _make_fsdp_ep_model()
    optimizer2 = torch.optim.SGD(moe2.parameters(), lr=0.01, momentum=0.9)
    _fsdp_ep_train_step(moe2, optimizer2, x, grad_scale2)

    set_optim_state_dict(moe2, optimizer2, sd, options=opts)
    loss_after = _fsdp_ep_train_step(moe2, optimizer2, x, grad_scale2)
    assert not (loss_after != loss_after), f"loss is NaN after flatten roundtrip: {loss_after}"

    print(f"[rank{_rank()}] E5 PASS: FSDP+EP flatten roundtrip (loss={loss_after:.4f})")


# =====================================================================
# E6: FSDP+EP local shape correctness
# =====================================================================
def test_e6_fsdp_ep_local_shape_correctness():
    """Verify optimizer state tensors have correct local shard shape under FSDP+EP."""
    moe, mesh, device, grad_scale = _make_fsdp_ep_model()
    optimizer = torch.optim.SGD(moe.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(BS, SLEN, DIM, device=device)

    for _ in range(2):
        _fsdp_ep_train_step(moe, optimizer, x, grad_scale)

    sd = get_optim_state_dict(moe, optimizer)

    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "step":
                continue
            assert not isinstance(value, DTensor), (
                f"state.{fqn}.{key}: expected plain Tensor (local shard), "
                f"got DTensor"
            )
            assert value.numel() > 0, (
                f"state.{fqn}.{key}: local tensor should have numel > 0"
            )
            assert value.shape[0] < NUM_EXPERTS * HIDDEN_DIM, (
                f"state.{fqn}.{key}: local dim0={value.shape[0]} should be "
                f"less than full (experts*hidden_dim={NUM_EXPERTS * HIDDEN_DIM})"
            )

    print(f"[rank{_rank()}] E6 PASS: FSDP+EP local shape correctness")
