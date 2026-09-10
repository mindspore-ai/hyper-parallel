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
"""Optimizer state dict tests for pure FSDP (1-D mesh).

Verified scenarios:
  F1 (4-card): get_optim_state_dict FQN roundtrip (default options)
  F2 (4-card): get_optim_state_dict with full_state_dict + cpu_offload
  F3 (4-card): set_optim_state_dict with broadcast_from_rank0
  F4 (4-card): flatten roundtrip
  F5 (4-card): strict=False
  F6 (4-card): DCP save/load + load template (nested)
  F7 (4-card): DCP load into brand-new optimizer (empty state)
  F8 (4-card): local shape correctness
  F9 (4-card): full_state_dict + cpu_offload restore to correct device
"""
import math
import os
import shutil

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"  # pylint: disable=wrong-import-position

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: F401,E402  # pylint: disable=unused-import
from torch.distributed.checkpoint.state_dict import StateDictOptions  # noqa: E402

from hyper_parallel import (  # noqa: E402
    SkipDTensorDispatch,
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
from hyper_parallel.core.fully_shard.api import fully_shard  # noqa: E402
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy  # noqa: E402
from hyper_parallel.platform.torch.fully_shard.state_dict_utils import (  # noqa: E402
    _build_optim_state_dict_load_template,
)
from tests.torch.common_net import FullyShardTestNet  # noqa: E402
from tests.torch.utils import init_dist  # noqa: E402  # pylint: enable=wrong-import-position

HIDDEN = 32
LAYERS = 2
BATCH = 4
MP = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
    cast_forward_inputs=True,
)

_CKPT_DIR_NESTED = "/tmp/hp_fsdp_optim_sd_test_nested"
_CKPT_DIR_NEW_OPTIM = "/tmp/hp_fsdp_optim_sd_test_new_optim"


def _rank():
    return dist.get_rank()


def _make_fsdp_model():
    """Build a pure FSDP model on 4-card 1-D mesh."""
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4,),
        mesh_dim_names=("fsdp",),
    )
    model = FullyShardTestNet(HIDDEN, LAYERS, has_bias=False)
    for dense_layer in model.dense_layers.layers:
        fully_shard(dense_layer, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    return model, mesh


def _train_step_with_optim(model, optimizer, x):
    optimizer.zero_grad()
    with SkipDTensorDispatch():
        loss = model(x).sum()
    loss.backward()
    with SkipDTensorDispatch():
        optimizer.step()
    return loss.item()


def _cleanup_ckpt(path):
    if _rank() == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()


# =====================================================================
# F1: FQN roundtrip
# =====================================================================
def test_f1_fsdp_optim_state_dict_fqn_roundtrip():
    """get_optim_state_dict (default) -> set_optim_state_dict -> step."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)
    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert not isinstance(value, DTensor), (
                    f"state.{fqn}.{key} should be plain Tensor, got DTensor"
                )

    source_raw_sd = optimizer.state_dict()
    source_step_vals = {}
    source_exp_avg_norms = {}
    for pid, state in source_raw_sd["state"].items():
        if "step" in state:
            source_step_vals[pid] = (
                state["step"].item()
                if isinstance(state["step"], torch.Tensor)
                else float(state["step"])
            )
        if "exp_avg" in state:
            source_exp_avg_norms[pid] = state["exp_avg"].norm().item()

    model2, _ = _make_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd)

    target_raw_sd = optimizer2.state_dict()
    for pid, source_step_val in source_step_vals.items():
        if pid in target_raw_sd["state"]:
            target_step = target_raw_sd["state"][pid]["step"]
            target_step_val = (
                target_step.item()
                if isinstance(target_step, torch.Tensor)
                else float(target_step)
            )
            assert target_step_val == source_step_val, (
                f"step mismatch for pid={pid}: source={source_step_val}, target={target_step_val}"
            )

    for pid, source_exp_avg_norm in source_exp_avg_norms.items():
        if pid in target_raw_sd["state"] and "exp_avg" in target_raw_sd["state"][pid]:
            target_norm = target_raw_sd["state"][pid]["exp_avg"].norm().item()
            assert abs(target_norm - source_exp_avg_norm) < 1e-5, (
                f"exp_avg norm mismatch for pid={pid}: source={source_exp_avg_norm}, target={target_norm}"
            )

    loss_after = _train_step_with_optim(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] F1 PASS: pure FSDP FQN roundtrip (loss={loss_after:.4f})")


# =====================================================================
# F2: full_state_dict + cpu_offload
# =====================================================================
def test_f2_fsdp_optim_state_dict_full_cpu():
    """get with full_state_dict+cpu_offload -> set with broadcast -> step."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(model, optimizer, options=opts)

    if _rank() == 0:
        assert len(sd["state"]) > 0, "rank0 should have non-empty state"
        for fqn, state in sd["state"].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    assert value.device == torch.device("cpu"), (
                        f"state.{fqn}.{key} should be on CPU, got {value.device}"
                    )
    else:
        assert len(sd["state"]) == 0, "non-rank0 should have empty state"

    model2, _ = _make_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    set_optim_state_dict(model2, optimizer2, sd, options=opts_set)

    loss_after = _train_step_with_optim(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after full+cpu roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] F2 PASS: pure FSDP full + cpu_offload -> set -> step (loss={loss_after:.4f})")


# =====================================================================
# F3: broadcast_from_rank0
# =====================================================================
def test_f3_fsdp_optim_state_dict_broadcast_from_rank0():
    """set_optim_state_dict with broadcast_from_rank0=True."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(model, optimizer, options=opts_get)

    model2, _ = _make_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    set_optim_state_dict(model2, optimizer2, sd, options=opts_set)
    _train_step_with_optim(model2, optimizer2, x)

    print(f"[rank{_rank()}] F3 PASS: pure FSDP broadcast_from_rank0")


# =====================================================================
# F4: flatten roundtrip
# =====================================================================
def test_f4_fsdp_optim_state_dict_flatten():
    """get_optim_state_dict with flatten_optimizer_state_dict=True."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    opts = StateDictOptions(flatten_optimizer_state_dict=True)
    sd = get_optim_state_dict(model, optimizer, options=opts)

    for key in sd.keys():
        assert key.startswith("state.") or key.startswith("param_group."), (
            f"flat key '{key}' should start with 'state.' or 'param_group.'"
        )

    model2, _ = _make_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd, options=opts)
    _train_step_with_optim(model2, optimizer2, x)

    print(f"[rank{_rank()}] F4 PASS: pure FSDP flatten roundtrip")


# =====================================================================
# F5: strict=False
# =====================================================================
def test_f5_fsdp_optim_state_dict_strict_false():
    """set_optim_state_dict with strict=False allows extra FQNs."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)
    sd["state"]["nonexistent.param"] = {"step": torch.tensor(1)}

    model2, _ = _make_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    original_pg_count = len(optimizer2.param_groups)
    original_params_per_group = [len(g["params"]) for g in optimizer2.param_groups]

    opts = StateDictOptions(strict=False)
    set_optim_state_dict(model2, optimizer2, sd, options=opts)

    assert len(optimizer2.param_groups) == original_pg_count, (
        "strict=False should not change param_groups count"
    )
    for i, g in enumerate(optimizer2.param_groups):
        assert len(g["params"]) == original_params_per_group[i], (
            f"strict=False should not change param_groups[{i}] params count"
        )

    loss_after = _train_step_with_optim(model2, optimizer2, x)
    assert not math.isnan(loss_after), f"loss is NaN after strict=False step: {loss_after}"

    print(f"[rank{_rank()}] F5 PASS: pure FSDP strict=False -> step (loss={loss_after:.4f})")


# =====================================================================
# F6: DCP save/load + load template (nested)
# =====================================================================
def test_f6_fsdp_dcp_save_load_nested():
    """DCP save optim state dict -> load template -> load -> set_optim_state_dict."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    _cleanup_ckpt(_CKPT_DIR_NESTED)

    optim_sd = get_optim_state_dict(model, optimizer)
    save(optim_sd, checkpoint_id=_CKPT_DIR_NESTED)

    model2, _ = _make_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)

    storage_reader = FileSystemReader(_CKPT_DIR_NESTED)
    template = _build_optim_state_dict_load_template(
        model2, optimizer2, storage_reader,
    )

    assert "state" in template, "template should have 'state' key"
    assert "param_groups" in template, "template should have 'param_groups' key"

    for fqn, state in template["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert value.numel() > 0, (
                    f"template state.{fqn}.{key} should be non-empty"
                )

    load(template, checkpoint_id=_CKPT_DIR_NESTED)
    set_optim_state_dict(model2, optimizer2, template)
    _train_step_with_optim(model2, optimizer2, x)

    print(f"[rank{_rank()}] F6 PASS: pure FSDP DCP save/load + load template (nested)")
    _cleanup_ckpt(_CKPT_DIR_NESTED)


# =====================================================================
# F7: DCP load into brand-new optimizer (empty state)
# =====================================================================
def test_f7_fsdp_dcp_load_new_optimizer():
    """Load into brand new optimizer that has never stepped."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    _cleanup_ckpt(_CKPT_DIR_NEW_OPTIM)

    optim_sd = get_optim_state_dict(model, optimizer)
    save(optim_sd, checkpoint_id=_CKPT_DIR_NEW_OPTIM)

    model2, _ = _make_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)

    assert len(optimizer2.state_dict()["state"]) == 0, (
        "new optimizer should have empty state"
    )

    storage_reader = FileSystemReader(_CKPT_DIR_NEW_OPTIM)
    template = _build_optim_state_dict_load_template(
        model2, optimizer2, storage_reader,
    )

    assert len(template["state"]) > 0, (
        "template should have non-empty state for new optimizer"
    )

    for fqn, state in template["state"].items():
        assert "exp_avg" in state, f"template state.{fqn} should have exp_avg"
        assert "exp_avg_sq" in state, f"template state.{fqn} should have exp_avg_sq"
        assert "step" in state, f"template state.{fqn} should have step"

    load(template, checkpoint_id=_CKPT_DIR_NEW_OPTIM)
    set_optim_state_dict(model2, optimizer2, template)
    _train_step_with_optim(model2, optimizer2, x)

    print(f"[rank{_rank()}] F7 PASS: pure FSDP DCP load into new optimizer (empty state)")
    _cleanup_ckpt(_CKPT_DIR_NEW_OPTIM)


# =====================================================================
# F8: local shape correctness
# =====================================================================
def test_f8_fsdp_local_shape_correctness():
    """Verify optimizer state tensors have correct local shard shape under pure FSDP."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)
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
            assert value.shape[0] == HIDDEN // 4, (
                f"state.{fqn}.{key}: expected local dim0={HIDDEN // 4}, "
                f"got {value.shape[0]}"
            )

    print(f"[rank{_rank()}] F8 PASS: pure FSDP local shape correctness")


# =====================================================================
# F9: full_state_dict + cpu_offload restore to correct device
# =====================================================================
def test_f9_fsdp_full_cpu_restore_to_device():
    """After set_optim_state_dict with full+cpu, states are on correct device."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(model, optimizer, options=opts_get)

    model2, _ = _make_fsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    set_optim_state_dict(model2, optimizer2, sd, options=opts_set)

    raw_sd2 = optimizer2.state_dict()
    for param_id, state in raw_sd2["state"].items():
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "step":
                continue
            assert value.device.type == "npu", (
                f"state[{param_id}].{key} should be on NPU, got {value.device}"
            )

    print(f"[rank{_rank()}] F9 PASS: pure FSDP full+cpu restore to correct device")
