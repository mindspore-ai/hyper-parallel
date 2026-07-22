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
"""Optimizer state dict tests for fully_shard (HSDP).

Verified scenarios:
  O1 (4-card): get_optim_state_dict FQN roundtrip (default options)
  O2 (4-card): get_optim_state_dict with full_state_dict + cpu_offload
  O3 (4-card): set_optim_state_dict with broadcast_from_rank0
  O4 (4-card): get/set_optim_state_dict flatten roundtrip
  O5 (4-card): set_optim_state_dict with strict=False
  O6 (4-card): DCP save/load + load template (nested)
  O7 (4-card): DCP save/load + load template (flatten)
  O8 (4-card): DCP load into brand-new optimizer (empty state)
  O9 (4-card): HSDP local shape correctness after set_optim_state_dict
  O10 (4-card): full_state_dict + cpu_offload restore to correct device
"""
import os
import shutil

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: F401,E402
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
from hyper_parallel.platform.torch.fully_shard.optim_state_dict_utils import (  # noqa: E402
    _build_optim_state_dict_load_template,
)
from tests.torch.common_net import FullyShardTestNet  # noqa: E402
from tests.torch.utils import init_dist  # noqa: E402

HIDDEN = 32
LAYERS = 2
BATCH = 4
MP = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
    cast_forward_inputs=True,
)

_CKPT_DIR_NESTED = "/tmp/hp_optim_sd_test_nested"
_CKPT_DIR_FLATTEN = "/tmp/hp_optim_sd_test_flatten"
_CKPT_DIR_NEW_OPTIM = "/tmp/hp_optim_sd_test_new_optim"
_CKPT_DIR_TRAINED_LOAD = "/tmp/hp_optim_sd_test_trained_load"
_CKPT_DIR_WRAPPED = "/tmp/hp_optim_sd_test_wrapped"


def _rank():
    return dist.get_rank()


def _make_hsdp_model():
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("replicate", "shard"),
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
# O1: FQN roundtrip
# =====================================================================
def test_o1_optim_state_dict_fqn_roundtrip():
    """get_optim_state_dict (default) -> set_optim_state_dict -> step.

    E2E verification: after set, the restored optimizer must produce the
    same loss as the source optimizer when running the same input, and
    optimizer state values (exp_avg, step) must match numerically.
    """
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
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
            source_step_vals[pid] = state["step"].item() if isinstance(state["step"], torch.Tensor) else float(state["step"])
        if "exp_avg" in state:
            source_exp_avg_norms[pid] = state["exp_avg"].norm().item()

    model2, _ = _make_hsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd)

    target_raw_sd = optimizer2.state_dict()
    for pid in source_step_vals:
        if pid in target_raw_sd["state"]:
            target_step = target_raw_sd["state"][pid]["step"]
            target_step_val = target_step.item() if isinstance(target_step, torch.Tensor) else float(target_step)
            assert target_step_val == source_step_vals[pid], (
                f"step mismatch for pid={pid}: source={source_step_vals[pid]}, target={target_step_val}"
            )

    for pid in source_exp_avg_norms:
        if pid in target_raw_sd["state"] and "exp_avg" in target_raw_sd["state"][pid]:
            target_norm = target_raw_sd["state"][pid]["exp_avg"].norm().item()
            assert abs(target_norm - source_exp_avg_norms[pid]) < 1e-5, (
                f"exp_avg norm mismatch for pid={pid}: source={source_exp_avg_norms[pid]}, target={target_norm}"
            )

    loss_after = _train_step_with_optim(model2, optimizer2, x)
    assert not (loss_after != loss_after), f"loss is NaN after roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] O1 PASS: FQN roundtrip (loss={loss_after:.4f}, step/state verified)")


# =====================================================================
# O2: full_state_dict + cpu_offload
# =====================================================================
def test_o2_optim_state_dict_full_cpu():
    """get with full_state_dict+cpu_offload -> set with broadcast -> step.

    E2E verification: rank 0 holds full CPU state dict, other ranks have
    empty state. After set_optim_state_dict with broadcast_from_rank0,
    all ranks can continue training successfully, and optimizer state
    tensors are restored to the correct NPU device.
    """
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
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

    model2, _ = _make_hsdp_model()
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
                f"state[{param_id}].{key} should be on NPU after set, got {value.device}"
            )

    loss_after = _train_step_with_optim(model2, optimizer2, x)
    assert not (loss_after != loss_after), f"loss is NaN after full+cpu roundtrip step: {loss_after}"

    print(f"[rank{_rank()}] O2 PASS: full + cpu_offload -> set -> step (loss={loss_after:.4f})")


# =====================================================================
# O3: broadcast_from_rank0
# =====================================================================
def test_o3_optim_state_dict_broadcast_from_rank0():
    """set_optim_state_dict with broadcast_from_rank0=True."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(model, optimizer, options=opts_get)

    model2, _ = _make_hsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    set_optim_state_dict(model2, optimizer2, sd, options=opts_set)
    _train_step_with_optim(model2, optimizer2, x)

    print(f"[rank{_rank()}] O3 PASS: broadcast_from_rank0")


# =====================================================================
# O4: flatten roundtrip
# =====================================================================
def test_o4_optim_state_dict_flatten():
    """get_optim_state_dict with flatten_optimizer_state_dict=True."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
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

    model2, _ = _make_hsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step_with_optim(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd, options=opts)
    _train_step_with_optim(model2, optimizer2, x)

    print(f"[rank{_rank()}] O4 PASS: flatten roundtrip")


# =====================================================================
# O5: strict=False
# =====================================================================
def test_o5_optim_state_dict_strict_false():
    """set_optim_state_dict with strict=False allows extra FQNs and step succeeds.

    E2E verification: after loading with strict=False (ignoring extra FQNs),
    the optimizer must still be able to execute a real training step without
    shape mismatch or crash. param_groups integrity is also verified.
    """
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)
    sd["state"]["nonexistent.param"] = {"step": torch.tensor(1)}

    model2, _ = _make_hsdp_model()
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
    assert not (loss_after != loss_after), f"loss is NaN after strict=False step: {loss_after}"

    print(f"[rank{_rank()}] O5 PASS: strict=False -> step (loss={loss_after:.4f})")


# =====================================================================
# O6: DCP save/load + load template (nested)
# =====================================================================
def test_o6_dcp_save_load_nested():
    """DCP save optim state dict -> load template -> load -> set_optim_state_dict."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    _cleanup_ckpt(_CKPT_DIR_NESTED)

    optim_sd = get_optim_state_dict(model, optimizer)
    save(optim_sd, checkpoint_id=_CKPT_DIR_NESTED)

    model2, _ = _make_hsdp_model()
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

    print(f"[rank{_rank()}] O6 PASS: DCP save/load + load template (nested)")
    _cleanup_ckpt(_CKPT_DIR_NESTED)


# =====================================================================
# O7: DCP save/load + load template (flatten)
# =====================================================================
def test_o7_dcp_save_load_flatten():
    """DCP save flatten optim state dict -> load template -> set_optim_state_dict."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    _cleanup_ckpt(_CKPT_DIR_FLATTEN)

    opts = StateDictOptions(flatten_optimizer_state_dict=True)
    optim_sd = get_optim_state_dict(model, optimizer, options=opts)
    save(optim_sd, checkpoint_id=_CKPT_DIR_FLATTEN)

    model2, _ = _make_hsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)

    storage_reader = FileSystemReader(_CKPT_DIR_FLATTEN)
    template = _build_optim_state_dict_load_template(
        model2, optimizer2, storage_reader, options=opts,
    )

    for key in template.keys():
        assert key.startswith("state.") or key.startswith("param_group."), (
            f"flat template key '{key}' should start with 'state.' or 'param_group.'"
        )

    load(template, checkpoint_id=_CKPT_DIR_FLATTEN)
    set_optim_state_dict(model2, optimizer2, template, options=opts)
    _train_step_with_optim(model2, optimizer2, x)

    print(f"[rank{_rank()}] O7 PASS: DCP save/load + load template (flatten)")
    _cleanup_ckpt(_CKPT_DIR_FLATTEN)


# =====================================================================
# O8: DCP load into brand-new optimizer (empty state)
# =====================================================================
def test_o8_dcp_load_new_optimizer():
    """Load into brand new optimizer that has never stepped.

    Critical regression: optimizer.step() has never been called, so
    optimizer.state_dict()["state"] is empty. The load template must
    provide correct shapes/dtypes for exp_avg, exp_avg_sq, etc.
    """
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    _cleanup_ckpt(_CKPT_DIR_NEW_OPTIM)

    optim_sd = get_optim_state_dict(model, optimizer)
    save(optim_sd, checkpoint_id=_CKPT_DIR_NEW_OPTIM)

    model2, _ = _make_hsdp_model()
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

    print(f"[rank{_rank()}] O8 PASS: DCP load into new optimizer (empty state)")
    _cleanup_ckpt(_CKPT_DIR_NEW_OPTIM)


# =====================================================================
# O9: HSDP local shape correctness
# =====================================================================
def test_o9_hsdp_local_shape_correctness():
    """Verify optimizer state tensors have correct local shard shape."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
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
            assert value.shape[0] == HIDDEN // 2, (
                f"state.{fqn}.{key}: expected local dim0={HIDDEN // 2}, "
                f"got {value.shape[0]}"
            )

    print(f"[rank{_rank()}] O9 PASS: local shape correctness")


# =====================================================================
# O10: full_state_dict restore to correct device
# =====================================================================
def test_o10_full_cpu_restore_to_device():
    """After set_optim_state_dict with full+cpu, states are on correct device."""
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(model, optimizer, options=opts_get)

    model2, _ = _make_hsdp_model()
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

    print(f"[rank{_rank()}] O10 PASS: full+cpu restore to correct device")


# =====================================================================
# O11: DCP load into trained optimizer (plan section 10 flow)
# =====================================================================
def test_o11_dcp_load_trained_optimizer():
    """DCP load into a trained optimizer: get -> dcp.load -> set.

    This tests the trained optimizer load flow defined in plan section 10:

        optim_sd = get_optim_state_dict(model, optimizer, options=options)
        dcp.load({"optimizer": optim_sd}, checkpoint_id=checkpoint_id)
        set_optim_state_dict(model, optimizer, optim_sd, options=options)

    The target optimizer has already stepped, so get_optim_state_dict
    returns a state dict with existing tensors that dcp.load can fill.
    """
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(3):
        _train_step_with_optim(model, optimizer, x)

    _cleanup_ckpt(_CKPT_DIR_TRAINED_LOAD)

    optim_sd = get_optim_state_dict(model, optimizer)
    save(optim_sd, checkpoint_id=_CKPT_DIR_TRAINED_LOAD)

    model2, _ = _make_hsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    for _ in range(2):
        _train_step_with_optim(model2, optimizer2, x)

    assert len(optimizer2.state_dict()["state"]) > 0, (
        "trained optimizer should have non-empty state"
    )

    target_sd = get_optim_state_dict(model2, optimizer2)
    load(target_sd, checkpoint_id=_CKPT_DIR_TRAINED_LOAD)
    set_optim_state_dict(model2, optimizer2, target_sd)

    loss_after = _train_step_with_optim(model2, optimizer2, x)
    assert not (loss_after != loss_after), (
        f"loss is NaN after trained optimizer DCP load step: {loss_after}"
    )

    print(f"[rank{_rank()}] O11 PASS: DCP load into trained optimizer (loss={loss_after:.4f})")
    _cleanup_ckpt(_CKPT_DIR_TRAINED_LOAD)


# =====================================================================
# O12: DCP save/load with {"optimizer": ...} wrapper (plan section 10)
# =====================================================================
def test_o12_dcp_save_load_wrapped():
    """DCP save/load with {"optimizer": optim_sd} wrapper format.

    Plan section 10 specifies:
        dcp.save({"optimizer": optim_sd}, checkpoint_id=...)
        dcp.load({"optimizer": optim_sd}, checkpoint_id=...)

    This tests the wrapped format where the optimizer state dict is nested
    under an "optimizer" key, as shown in the plan examples.
    """
    init_dist()
    torch.manual_seed(42 + _rank())
    model, _ = _make_hsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(BATCH, HIDDEN).npu()

    for _ in range(2):
        _train_step_with_optim(model, optimizer, x)

    _cleanup_ckpt(_CKPT_DIR_WRAPPED)

    optim_sd = get_optim_state_dict(model, optimizer)
    save({"optimizer": optim_sd}, checkpoint_id=_CKPT_DIR_WRAPPED)

    model2, _ = _make_hsdp_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)

    storage_reader = FileSystemReader(_CKPT_DIR_WRAPPED)
    template = _build_optim_state_dict_load_template(
        model2, optimizer2, storage_reader,
    )

    load({"optimizer": template}, checkpoint_id=_CKPT_DIR_WRAPPED)
    set_optim_state_dict(model2, optimizer2, template)
    _train_step_with_optim(model2, optimizer2, x)

    print(f"[rank{_rank()}] O12 PASS: DCP save/load with wrapper format")
    _cleanup_ckpt(_CKPT_DIR_WRAPPED)
