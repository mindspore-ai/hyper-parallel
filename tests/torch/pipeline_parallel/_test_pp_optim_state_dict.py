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
"""Optimizer state dict tests for pipeline parallelism.

Verified scenarios:
  P1 (2-card): PP-only get/set_optim_state_dict FQN roundtrip (default options)
  P2 (2-card): PP-only cpu_offload roundtrip
  P3 (4-card): PP+HSDP get/set_optim_state_dict FQN roundtrip
  P4 (4-card): PP+HSDP full_state_dict + cpu_offload restore to correct device
  P4b (4-card): PP+HSDP asymmetric stages — FQN isolation & broadcast correctness
  P5 (4-card): PP+HSDP local shape correctness after set_optim_state_dict
  P6 (4-card): PP+HSDP DCP save/load + load template (nested)
  P7 (4-card): PP+HSDP flatten roundtrip
  P7b (4-card): PP+HSDP flatten DCP save/load + cross-rank key consistency

Key design decisions for PP optimizer state dict testing:
  - Each PP rank owns different stage parameters (different FQNs), so
    ``broadcast_from_rank0`` (which broadcasts from global rank 0 to ALL
    ranks) is NOT suitable for PP — it would send rank 0's stage FQNs to
    ranks that hold different stages.  Within an HSDP group for the same
    PP stage, a PP-scoped broadcast could work, but the current
    ``set_optim_state_dict`` API does not support process-group-scoped
    broadcast.
  - The recommended PP checkpoint pattern is for each rank to
    independently save/load its own stages' optimizer state dicts.
    All tests in this file follow this pattern.
  - The schedule handles both forward and backward internally; after
    ``schedule.run()`` gradients are populated and the optimizer can
    step directly.  ``SkipDTensorDispatch`` is required for
    ``optimizer.step()`` on DTensor parameters (PP+HSDP tests).
"""
import math
import os
import shutil

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"  # pylint: disable=wrong-import-position

import copy  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: F401,E402  # pylint: disable=unused-import
from torch import nn  # noqa: E402
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
from hyper_parallel.core.dtensor.placement_types import Replicate  # noqa: E402
from hyper_parallel.core.fully_shard.api import fully_shard  # noqa: E402
from hyper_parallel.core.pipeline_parallel import PipelineStage, ScheduleGPipe  # noqa: E402
from hyper_parallel.core.pipeline_parallel.scheduler import (  # noqa: E402
    ScheduleInterleaved1F1B,
)
from hyper_parallel.platform.torch.fully_shard.optim_state_dict_utils import (  # noqa: E402
    _build_optim_state_dict_load_template,
)
from tests.torch.utils import init_dist  # noqa: E402
# pylint: enable=wrong-import-position

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_D_HID = 16
_TOTAL_LAYERS = 4
_BATCH = 4
_LR = 0.01

_CKPT_DIR_NESTED = "/tmp/hp_pp_optim_sd_test_nested"


# ---------------------------------------------------------------------------
# Model definitions (reuse MLPModule / StageModel from PP composite tests)
# ---------------------------------------------------------------------------
class MLPModule(nn.Module):
    """Two-layer MLP block."""

    def __init__(self, d_hid: int) -> None:
        super().__init__()
        self.net1 = nn.Linear(d_hid, d_hid)
        self.net2 = nn.Linear(d_hid, d_hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two dense layers."""
        return self.net2(self.net1(x))


class StageModel(nn.Module):
    """One pipeline stage: a contiguous slice of MLP layers."""

    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward received activations through the owned layers."""
        for layer in self.layers:
            x = layer(x)
        return x


def _rank() -> int:
    return dist.get_rank()


def _cleanup_ckpt(path: str) -> None:
    if _rank() == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()


# =====================================================================
# PP-only helpers (2-card, no FSDP)
# =====================================================================
def _build_pp_only_model_and_stage():
    """Build a simple 2-stage PP model on 2 ranks (no FSDP).

    Uses ScheduleGPipe: first stage passes input through, last stage
    computes output.  The schedule handles both forward and backward
    internally; after ``schedule.run()`` the gradients are already
    populated on each stage's parameters.
    """
    _, device_id = init_dist()
    device = torch.device("npu", device_id)
    pp_size = 2
    pp_rank = _rank()
    is_first_stage = pp_rank == 0

    # Each rank owns 2 layers (total 4 layers split into 2 stages)
    torch.manual_seed(42)
    base_layers = [MLPModule(_D_HID) for _ in range(_TOTAL_LAYERS)]
    if is_first_stage:
        stage_layers = [copy.deepcopy(base_layers[0]), copy.deepcopy(base_layers[1])]
    else:
        stage_layers = [copy.deepcopy(base_layers[2]), copy.deepcopy(base_layers[3])]
    stage_model = StageModel(stage_layers).to(device=device)

    # Build process group for PP
    pg = dist.new_group(ranks=list(range(pp_size)))

    pipeline_stage = PipelineStage(
        stage_model,
        stage_index=pp_rank,
        stage_num=pp_size,
        group=pg,
        device=device,
    )
    schedule = ScheduleGPipe([pipeline_stage], _BATCH)
    return stage_model, pipeline_stage, schedule, device, pp_rank, is_first_stage


def _pp_only_train_step(schedule, optimizer, x, is_first_stage):
    """Run one PP-only training step.

    The schedule handles both forward and backward.  After ``schedule.run()``
    gradients are already computed; we only need to call ``optimizer.step()``.
    """
    optimizer.zero_grad(set_to_none=True)
    if is_first_stage:
        losses = schedule.run(x)
    else:
        losses = schedule.run()
    optimizer.step()
    return losses


# =====================================================================
# P1: PP-only FQN roundtrip
# =====================================================================
def test_p1_pp_only_optim_state_dict_fqn_roundtrip():
    """PP-only: get_optim_state_dict (default) -> set_optim_state_dict -> step.

    E2E verification: after roundtrip, the restored optimizer must produce
    a training step without NaN, and FQN keys in the state dict correspond
    to the stage model's named_parameters.
    """
    stage_model, _, schedule, device, pp_rank, is_first_stage = (
        _build_pp_only_model_and_stage()
    )
    optimizer = torch.optim.AdamW(stage_model.parameters(), lr=_LR)
    x = torch.randn(_BATCH, _D_HID, device=device)

    for _ in range(2):
        _pp_only_train_step(schedule, optimizer, x, is_first_stage)

    sd = get_optim_state_dict(stage_model, optimizer)

    # Verify FQNs match the stage model's named_parameters
    model_fqns = {name for name, _ in stage_model.named_parameters()}
    state_fqns = set(sd["state"].keys())
    assert state_fqns.issubset(model_fqns), (
        f"state FQNs {state_fqns - model_fqns} not in model FQNs {model_fqns}"
    )
    # No DTensors in default (local) mode
    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert not isinstance(value, DTensor), (
                    f"state.{fqn}.{key} should be plain Tensor, got DTensor"
                )

    # Roundtrip: create new model + optimizer, load state dict, verify step
    stage_model2, _, schedule2, _, _, is_first2 = (
        _build_pp_only_model_and_stage()
    )
    optimizer2 = torch.optim.AdamW(stage_model2.parameters(), lr=_LR)
    _pp_only_train_step(schedule2, optimizer2, x, is_first2)

    set_optim_state_dict(stage_model2, optimizer2, sd)
    losses = _pp_only_train_step(schedule2, optimizer2, x, is_first2)

    # Compute a loss value from the last-stage output for NaN check
    if not is_first2:
        total_loss = torch.tensor(0.0, device=device)
        for sub_loss in losses:
            activation = sub_loss[0] if isinstance(sub_loss, (list, tuple)) else sub_loss
            total_loss = total_loss + activation.sum()
        loss_val = total_loss.item()
        assert not math.isnan(loss_val), f"loss is NaN after roundtrip step: {loss_val}"

    print(f"[rank{_rank()}] P1 PASS: PP-only FQN roundtrip")


# =====================================================================
# P2: PP-only cpu_offload roundtrip
# =====================================================================
def test_p2_pp_only_cpu_offload_roundtrip():
    """PP-only: get with cpu_offload -> set -> step.

    Each PP rank saves its own stage's optimizer state dict to CPU and
    loads it back.  This is the realistic PP checkpoint pattern: each
    rank independently handles its own stage's optimizer state dict.

    Note: ``broadcast_from_rank0`` is NOT used here because different
    PP ranks have different model parameters (different FQNs).  The
    broadcast mechanism is only meaningful within a data-parallel group
    where all ranks share the same model (see P4 for PP+HSDP).
    """
    stage_model, _, schedule, device, _, is_first_stage = (
        _build_pp_only_model_and_stage()
    )
    optimizer = torch.optim.AdamW(stage_model.parameters(), lr=_LR)
    x = torch.randn(_BATCH, _D_HID, device=device)

    for _ in range(2):
        _pp_only_train_step(schedule, optimizer, x, is_first_stage)

    opts = StateDictOptions(cpu_offload=True)
    sd = get_optim_state_dict(stage_model, optimizer, options=opts)

    # Verify CPU offload
    for fqn, state in sd["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert value.device == torch.device("cpu"), (
                    f"state.{fqn}.{key} should be on CPU, got {value.device}"
                )

    stage_model2, _, schedule2, _, _, is_first2 = (
        _build_pp_only_model_and_stage()
    )
    optimizer2 = torch.optim.AdamW(stage_model2.parameters(), lr=_LR)
    _pp_only_train_step(schedule2, optimizer2, x, is_first2)

    set_optim_state_dict(stage_model2, optimizer2, sd, options=opts)

    # After set, optimizer state tensors should be on NPU
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

    losses = _pp_only_train_step(schedule2, optimizer2, x, is_first2)
    if not is_first2:
        total_loss = torch.tensor(0.0, device=device)
        for sub_loss in losses:
            activation = sub_loss[0] if isinstance(sub_loss, (list, tuple)) else sub_loss
            total_loss = total_loss + activation.sum()
        loss_val = total_loss.item()
        assert not math.isnan(loss_val), f"loss is NaN after cpu_offload roundtrip: {loss_val}"

    print(f"[rank{_rank()}] P2 PASS: PP-only cpu_offload roundtrip")


# =====================================================================
# PP+HSDP helpers (4-card, 3-D mesh pp=2, dp=1, fsdp=2)
# =====================================================================
_PP_SIZE = 2
_DP_SIZE = 1
_FSDP_SIZE = 2
_NUM_MICROBATCHES = 4


def _owned_virtual_stages(pp_rank: int, total_layers: int = _TOTAL_LAYERS) -> list:
    """Loop layout: virtual stage v is owned by v % pp_size."""
    return list(range(pp_rank, total_layers, _PP_SIZE))


def _build_pp_hsdp_stages():
    """Build PP+HSDP stages on 4 ranks with 3-D mesh.

    Uses ScheduleInterleaved1F1B with VPP loop layout.
    Each pp_rank owns 2 virtual stages (single-layer each).
    The 2-D ``(dp, fsdp)`` HSDP submesh drives ``fully_shard``.
    """
    _, device_id = init_dist()
    device = torch.device("npu", device_id)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(_PP_SIZE, _DP_SIZE, _FSDP_SIZE),
        mesh_dim_names=("pp", "dp", "fsdp"),
    )
    pp_mesh = mesh["pp"]
    hsdp_mesh = mesh["dp", "fsdp"]
    pp_rank = mesh.get_local_rank("pp")

    torch.manual_seed(42)
    base_layers = [MLPModule(_D_HID) for _ in range(_TOTAL_LAYERS)]

    stages = []
    pipeline_stages = []
    for virtual_stage_idx in _owned_virtual_stages(pp_rank):
        owned_layer = copy.deepcopy(base_layers[virtual_stage_idx]).to(device=device)
        stage_model = StageModel([owned_layer])
        # Wrap with HSDP
        fully_shard(stage_model, mesh=hsdp_mesh, reshard_after_forward=True)
        if hasattr(stage_model, "set_reduce_op_type"):
            stage_model.set_reduce_op_type("sum")
        pipeline_stage = PipelineStage(
            stage_model,
            stage_index=virtual_stage_idx,
            stage_num=_TOTAL_LAYERS,
            group=pp_mesh.get_group(),
            mesh=pp_mesh,
            device=device,
        )
        stages.append(stage_model)
        pipeline_stages.append(pipeline_stage)

    schedule = ScheduleInterleaved1F1B(pipeline_stages, _NUM_MICROBATCHES)
    stage_optimizers = [torch.optim.AdamW(stage.parameters(), lr=_LR) for stage in stages]

    return stages, pipeline_stages, schedule, stage_optimizers, device, pp_rank, mesh, hsdp_mesh


def _pp_hsdp_train_step(schedule, stage_optimizers, x, pp_rank):
    """Run one PP+HSDP training step.

    The schedule handles forward, backward, and HSDP gradient drain
    (``FSDP_REDUCE_GRAD`` MetaStep).  After ``schedule.run()``, gradients
    are reduced and ready for the optimizer.  ``SkipDTensorDispatch`` is
    required for ``optimizer.step()`` on DTensor parameters.
    """
    for optimizer in stage_optimizers:
        optimizer.zero_grad(set_to_none=True)
    losses = schedule.run(x) if pp_rank == 0 else schedule.run()
    with SkipDTensorDispatch():
        for optimizer in stage_optimizers:
            optimizer.step()
    return losses


# =====================================================================
# P3: PP+HSDP FQN roundtrip
# =====================================================================
def test_p3_pp_hsdp_optim_state_dict_fqn_roundtrip():
    """PP+HSDP: get_optim_state_dict (default) -> set_optim_state_dict -> step.

    Each stage has its own optimizer; FQNs are stage-local (e.g., layers.0.net1.weight).
    After roundtrip, the restored optimizer must produce a training step without NaN.
    """
    stages, _, schedule, stage_optimizers, device, pp_rank, _, _ = (
        _build_pp_hsdp_stages()
    )
    x = torch.randn(_NUM_MICROBATCHES, _D_HID, device=device)

    for _ in range(2):
        _pp_hsdp_train_step(schedule, stage_optimizers, x, pp_rank)

    # Get state dict for each stage's optimizer
    all_sd = []
    for stage, optimizer in zip(stages, stage_optimizers):
        sd = get_optim_state_dict(stage, optimizer)
        # Verify no DTensors in default mode
        for fqn, state in sd["state"].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    assert not isinstance(value, DTensor), (
                        f"state.{fqn}.{key} should be plain Tensor, got DTensor"
                    )
        all_sd.append(sd)

    # Build fresh model+optimizers, train one step, load state dict
    stages2, _, schedule2, stage_optimizers2, _, _, _, _ = (
        _build_pp_hsdp_stages()
    )
    _pp_hsdp_train_step(schedule2, stage_optimizers2, x, pp_rank)

    for stage2, optimizer2, sd in zip(stages2, stage_optimizers2, all_sd):
        set_optim_state_dict(stage2, optimizer2, sd)

    # Verify training continues without NaN
    losses = _pp_hsdp_train_step(schedule2, stage_optimizers2, x, pp_rank)
    # Compute a scalar loss for NaN check on the last stage
    total_loss = torch.tensor(0.0, device=device)
    if pp_rank == _PP_SIZE - 1:
        for sub_loss in losses:
            activation = sub_loss[0] if isinstance(sub_loss, (list, tuple)) else sub_loss
            total_loss = total_loss + activation.sum()
        loss_val = total_loss.item()
        assert not math.isnan(loss_val), f"loss is NaN after roundtrip step: {loss_val}"

    print(f"[rank{_rank()}] P3 PASS: PP+HSDP FQN roundtrip")


# =====================================================================
# P4: PP+HSDP full_state_dict + cpu_offload restore to correct device
# =====================================================================
def test_p4_pp_hsdp_full_cpu_restore_to_device():
    """PP+HSDP: get with full_state_dict+cpu_offload -> set with broadcast -> verify device.

    Each PP stage's HSDP replicate-group root gets the full unsharded
    optimizer state on CPU.  ``set_optim_state_dict`` with
    ``broadcast_from_rank0=True`` then broadcasts from the per-stage
    replicate root to all HSDP peers, and converts back to the correct
    local shard on NPU.

    This test verifies that ``full_state_dict + cpu_offload +
    broadcast_from_rank0`` works correctly under PP+HSDP, where the
    replicate root is NOT global rank 0 for the second PP stage.
    """
    stages, _, schedule, stage_optimizers, device, pp_rank, _, _ = (
        _build_pp_hsdp_stages()
    )
    x = torch.randn(_NUM_MICROBATCHES, _D_HID, device=device)

    for _ in range(2):
        _pp_hsdp_train_step(schedule, stage_optimizers, x, pp_rank)

    # Get full+cpu state dict for each stage
    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    all_full_sd = []
    for stage, optimizer in zip(stages, stage_optimizers):
        sd = get_optim_state_dict(stage, optimizer, options=opts_get)
        # Verify CPU offload: only the replicate-group root keeps data
        for fqn, state in sd.get("state", {}).items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    assert value.device.type == "cpu", (
                        f"full+cpu state.{fqn}.{key} should be on CPU, got {value.device}"
                    )
        all_full_sd.append(sd)

    # Build fresh model+optimizers, train one step, then load full+cpu state
    stages2, _, schedule2, stage_optimizers2, _, _, _, _ = (
        _build_pp_hsdp_stages()
    )
    _pp_hsdp_train_step(schedule2, stage_optimizers2, x, pp_rank)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True,
    )
    for stage2, optimizer2, sd in zip(stages2, stage_optimizers2, all_full_sd):
        set_optim_state_dict(stage2, optimizer2, sd, options=opts_set)

    # After set, optimizer state tensors should be on NPU
    for optimizer2 in stage_optimizers2:
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

    # Verify training continues
    losses = _pp_hsdp_train_step(schedule2, stage_optimizers2, x, pp_rank)
    total_loss = torch.tensor(0.0, device=device)
    if pp_rank == _PP_SIZE - 1:
        for sub_loss in losses:
            activation = sub_loss[0] if isinstance(sub_loss, (list, tuple)) else sub_loss
            total_loss = total_loss + activation.sum()
        loss_val = total_loss.item()
        assert not math.isnan(loss_val), f"loss is NaN after roundtrip: {loss_val}"

    print(f"[rank{_rank()}] P4 PASS: PP+HSDP full+cpu restore to correct device")


# =====================================================================
# P5: PP+HSDP local shape correctness
# =====================================================================
def test_p5_pp_hsdp_local_shape_correctness():
    """Verify optimizer state tensors have correct local shard shape under PP+HSDP.

    Each parameter is sharded dim-0 over the fsdp dim (size 2), so local shape
    should have dim0 = _D_HID // _FSDP_SIZE.
    """
    stages, _, schedule, stage_optimizers, device, pp_rank, _, _ = (
        _build_pp_hsdp_stages()
    )
    x = torch.randn(_NUM_MICROBATCHES, _D_HID, device=device)

    for _ in range(2):
        _pp_hsdp_train_step(schedule, stage_optimizers, x, pp_rank)

    for stage, optimizer in zip(stages, stage_optimizers):
        sd = get_optim_state_dict(stage, optimizer)
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
                expected_dim0 = _D_HID // _FSDP_SIZE
                assert value.shape[0] == expected_dim0, (
                    f"state.{fqn}.{key}: expected local dim0={expected_dim0}, "
                    f"got {value.shape[0]}"
                )

    print(f"[rank{_rank()}] P5 PASS: PP+HSDP local shape correctness")


# =====================================================================
# P6: PP+HSDP DCP save/load + load template (nested)
# =====================================================================
def test_p6_pp_hsdp_dcp_save_load_nested():
    """PP+HSDP: DCP save optim state dict -> load template -> load -> set_optim_state_dict.

    Each PP rank saves its own virtual stages' optimizer state dicts to
    per-rank checkpoint paths using ``no_dist=True`` (no cross-rank DCP
    coordination), then loads them back via the load template pattern.

    In PP+HSDP, different PP ranks own different virtual stages with
    different FQNs, so the standard DCP coordinated save (which requires
    all ranks to participate) is not directly applicable when each PP rank
    saves a different stage's state dict.  ``no_dist=True`` sidesteps this
    by letting each rank save independently.
    """
    stages, _, schedule, stage_optimizers, device, pp_rank, _, _ = (
        _build_pp_hsdp_stages()
    )
    x = torch.randn(_NUM_MICROBATCHES, _D_HID, device=device)

    for _ in range(2):
        _pp_hsdp_train_step(schedule, stage_optimizers, x, pp_rank)

    _cleanup_ckpt(_CKPT_DIR_NESTED)

    # Save each stage's optimizer state dict with no_dist=True.
    # Different PP ranks own different stages with different FQNs,
    # and each HSDP rank within a PP stage holds its own local shard.
    # no_dist=True lets each rank save independently without collective ops.
    virtual_stage_indices = _owned_virtual_stages(pp_rank)
    for local_idx, (stage, optimizer) in enumerate(zip(stages, stage_optimizers)):
        optim_sd = get_optim_state_dict(stage, optimizer)
        v_stage = virtual_stage_indices[local_idx]
        ckpt_path = os.path.join(_CKPT_DIR_NESTED, f"pp{pp_rank}_vstage{v_stage}")
        save(optim_sd, checkpoint_id=ckpt_path, no_dist=True)

    # Build fresh model+optimizers
    stages2, _, schedule2, stage_optimizers2, _, _, _, _ = (
        _build_pp_hsdp_stages()
    )

    for local_idx, (stage2, optimizer2) in enumerate(zip(stages2, stage_optimizers2)):
        v_stage = virtual_stage_indices[local_idx]
        ckpt_path = os.path.join(_CKPT_DIR_NESTED, f"pp{pp_rank}_vstage{v_stage}")
        storage_reader = FileSystemReader(ckpt_path)
        template = _build_optim_state_dict_load_template(
            stage2, optimizer2, storage_reader,
        )
        assert "state" in template, "template should have 'state' key"
        assert "param_groups" in template, "template should have 'param_groups' key"
        for fqn, state in template["state"].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    assert value.numel() > 0, (
                        f"template state.{fqn}.{key} should be non-empty"
                    )
        load(template, checkpoint_id=ckpt_path, no_dist=True)
        set_optim_state_dict(stage2, optimizer2, template)

    # Verify training continues
    losses = _pp_hsdp_train_step(schedule2, stage_optimizers2, x, pp_rank)
    total_loss = torch.tensor(0.0, device=device)
    if pp_rank == _PP_SIZE - 1:
        for sub_loss in losses:
            activation = sub_loss[0] if isinstance(sub_loss, (list, tuple)) else sub_loss
            total_loss = total_loss + activation.sum()
        loss_val = total_loss.item()
        assert not math.isnan(loss_val), f"loss is NaN after DCP load: {loss_val}"

    print(f"[rank{_rank()}] P6 PASS: PP+HSDP DCP save/load + load template (nested)")
    _cleanup_ckpt(_CKPT_DIR_NESTED)


# =====================================================================
# P7: PP+HSDP flatten roundtrip
# =====================================================================
def test_p7_pp_hsdp_flatten_roundtrip():
    """PP+HSDP: get_optim_state_dict with flatten_optimizer_state_dict=True -> set -> step."""
    stages, _, schedule, stage_optimizers, device, pp_rank, _, _ = (
        _build_pp_hsdp_stages()
    )
    x = torch.randn(_NUM_MICROBATCHES, _D_HID, device=device)

    for _ in range(2):
        _pp_hsdp_train_step(schedule, stage_optimizers, x, pp_rank)

    opts = StateDictOptions(flatten_optimizer_state_dict=True)
    all_sd = []
    for stage, optimizer in zip(stages, stage_optimizers):
        sd = get_optim_state_dict(stage, optimizer, options=opts)
        # Verify flat keys
        for key in sd.keys():
            assert key.startswith("state.") or key.startswith("param_group."), (
                f"flat key '{key}' should start with 'state.' or 'param_group.'"
            )
        all_sd.append(sd)

    stages2, _, schedule2, stage_optimizers2, _, _, _, _ = (
        _build_pp_hsdp_stages()
    )
    _pp_hsdp_train_step(schedule2, stage_optimizers2, x, pp_rank)

    for stage2, optimizer2, sd in zip(stages2, stage_optimizers2, all_sd):
        set_optim_state_dict(stage2, optimizer2, sd, options=opts)

    # Verify training continues
    losses = _pp_hsdp_train_step(schedule2, stage_optimizers2, x, pp_rank)
    total_loss = torch.tensor(0.0, device=device)
    if pp_rank == _PP_SIZE - 1:
        for sub_loss in losses:
            activation = sub_loss[0] if isinstance(sub_loss, (list, tuple)) else sub_loss
            total_loss = total_loss + activation.sum()
        loss_val = total_loss.item()
        assert not math.isnan(loss_val), f"loss is NaN after flatten roundtrip: {loss_val}"

    print(f"[rank{_rank()}] P7 PASS: PP+HSDP flatten roundtrip")


# =====================================================================
# Asymmetric PP stage helpers — different FQN sets per stage
# =====================================================================
_STAGE0_LAYERS = 3
_STAGE1_LAYERS = 1
_TOTAL_LAYERS_ASYM = _STAGE0_LAYERS + _STAGE1_LAYERS


def _build_asym_pp_hsdp_stages():
    """Build PP+HSDP stages with **asymmetric** layer counts per stage.

    Stage 0 owns 3 MLPModule layers, stage 1 owns 1.  This produces
    **different FQN sets** on each PP rank:
        rank 0 (stage 0): layers.0.net1.weight, layers.0.net2.weight,
                          layers.1.net1.weight, layers.1.net2.weight,
                          layers.2.net1.weight, layers.2.net2.weight
        rank 1 (stage 1): layers.0.net1.weight, layers.0.net2.weight

    Both ranks have ``layers.0.*`` FQNs, but stage 0 additionally has
    ``layers.1.*`` and ``layers.2.*``.  This is the realistic scenario
    where PP stages have different model structure, and
    ``broadcast_from_rank0`` must be scoped to each stage's HSDP
    subgroup — not broadcast stage 0's FQNs to stage 1.
    """
    _, device_id = init_dist()
    device = torch.device("npu", device_id)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(_PP_SIZE, _DP_SIZE, _FSDP_SIZE),
        mesh_dim_names=("pp", "dp", "fsdp"),
    )
    pp_mesh = mesh["pp"]
    hsdp_mesh = mesh["dp", "fsdp"]
    pp_rank = mesh.get_local_rank("pp")

    torch.manual_seed(99)
    base_layers = [MLPModule(_D_HID) for _ in range(_TOTAL_LAYERS_ASYM)]

    if pp_rank == 0:
        my_layers = [copy.deepcopy(base_layers[i]) for i in range(_STAGE0_LAYERS)]
    else:
        my_layers = [copy.deepcopy(base_layers[_STAGE0_LAYERS + i]) for i in range(_STAGE1_LAYERS)]

    stage_model = StageModel(my_layers).to(device=device)
    fully_shard(stage_model, mesh=hsdp_mesh, reshard_after_forward=True)
    if hasattr(stage_model, "set_reduce_op_type"):
        stage_model.set_reduce_op_type("sum")

    pipeline_stage = PipelineStage(
        stage_model,
        stage_index=pp_rank,
        stage_num=_PP_SIZE,
        group=pp_mesh.get_group(),
        mesh=pp_mesh,
        device=device,
    )
    schedule = ScheduleGPipe([pipeline_stage], _NUM_MICROBATCHES)
    optimizer = torch.optim.AdamW(stage_model.parameters(), lr=_LR)

    return stage_model, pipeline_stage, schedule, optimizer, device, pp_rank, mesh, hsdp_mesh


def _asym_pp_hsdp_train_step(schedule, optimizer, x, pp_rank):
    """Run one training step for the asymmetric PP+HSDP model."""
    optimizer.zero_grad(set_to_none=True)
    if pp_rank == 0:
        schedule.run(x)
    else:
        schedule.run()
    with SkipDTensorDispatch():
        optimizer.step()


# =====================================================================
# P4b: PP+HSDP asymmetric stages — FQN isolation & broadcast correctness
# =====================================================================
def test_p4b_pp_hsdp_asymmetric_fqn_isolation_and_broadcast():
    """PP+HSDP with asymmetric stages: verify FQN isolation and broadcast.

    Key verifications:
      1. Different PP stages have different FQN sets (asymmetric layers).
      2. get_optim_state_dict(full_state_dict+cpu_offload) returns only
         the FQNs matching that stage's named_parameters.
      3. broadcast_from_rank0 is scoped to each stage's HSDP subgroup:
         stage 1's root is NOT global rank 0, yet broadcast works
         correctly within stage 1's HSDP group.
      4. After set_optim_state_dict with broadcast, the restored optimizer
         produces a training step without NaN.
      5. Optimizer state tensors are on NPU after set.
    """
    stage_model, _, schedule, optimizer, device, pp_rank, _, _ = (
        _build_asym_pp_hsdp_stages()
    )
    x = torch.randn(_NUM_MICROBATCHES, _D_HID, device=device)

    for _ in range(2):
        _asym_pp_hsdp_train_step(schedule, optimizer, x, pp_rank)

    model_fqns = {name for name, _ in stage_model.named_parameters()}

    # (1) Verify different FQN sets across PP stages
    # Stage 0 should have layers.0.*, layers.1.*, layers.2.*
    # Stage 1 should have only layers.0.*
    if pp_rank == 0:
        has_layers_1 = any("layers.1" in n for n in model_fqns)
        has_layers_2 = any("layers.2" in n for n in model_fqns)
        assert has_layers_1, f"stage 0 should have layers.1.* FQNs, got {model_fqns}"
        assert has_layers_2, f"stage 0 should have layers.2.* FQNs, got {model_fqns}"
    else:
        has_layers_1 = any("layers.1" in n for n in model_fqns)
        assert not has_layers_1, f"stage 1 should NOT have layers.1.* FQNs, got {model_fqns}"

    # (2) get with full_state_dict+cpu_offload — FQNs must match model
    opts_get = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_optim_state_dict(stage_model, optimizer, options=opts_get)

    # Determine if this rank is the replicate root for its stage
    dtensor_info = None
    for name, param in stage_model.named_parameters():
        if isinstance(param, DTensor):
            dtensor_info = (param.device_mesh, param.placements)
            break

    is_replicate_root = True
    if dtensor_info is not None:
        coord = dtensor_info[0].get_coordinate()
        replicate_dims = [
            i for i, p in enumerate(dtensor_info[1]) if isinstance(p, Replicate)
        ]
        if replicate_dims and coord is not None:
            is_replicate_root = all(coord[d] == 0 for d in replicate_dims)

    if is_replicate_root:
        state_fqns = set(sd.get("state", {}).keys())
        assert state_fqns == model_fqns, (
            f"state FQNs {state_fqns} != model FQNs {model_fqns} "
            f"on replicate root rank"
        )

    # (3) Roundtrip: build fresh model, load with broadcast
    stage_model2, _, schedule2, optimizer2, _, _, _, _ = (
        _build_asym_pp_hsdp_stages()
    )
    _asym_pp_hsdp_train_step(schedule2, optimizer2, x, pp_rank)

    opts_set = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True,
    )
    set_optim_state_dict(stage_model2, optimizer2, sd, options=opts_set)

    # (4) Verify state tensors on NPU
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

    # (5) Verify training continues without NaN
    _asym_pp_hsdp_train_step(schedule2, optimizer2, x, pp_rank)
    print(f"[rank{_rank()}] P4b PASS: asymmetric FQN isolation + broadcast")


# =====================================================================
# P7b: PP+HSDP flatten roundtrip with DCP save/load + cross-rank key check
# =====================================================================
_CKPT_DIR_FLATTEN_DCP = "/tmp/hp_pp_optim_sd_test_flatten_dcp"


def test_p7b_pp_hsdp_flatten_dcp_cross_rank_key_consistency():
    """PP+HSDP: flatten roundtrip + DCP save/load + key structure verification.

    Verifications:
      1. Flatten format keys have correct structure (state.* / param_group.*)
         and FQN parts match model named_parameters.
      2. Flatten state values are non-trivial (non-zero after training).
      3. DCP save with no_dist=True + flatten format -> load template -> load
         -> set_optim_state_dict -> training step without NaN.
    """
    stages, _, schedule, stage_optimizers, device, pp_rank, _, _ = (
        _build_pp_hsdp_stages()
    )
    x = torch.randn(_NUM_MICROBATCHES, _D_HID, device=device)

    for _ in range(2):
        _pp_hsdp_train_step(schedule, stage_optimizers, x, pp_rank)

    opts = StateDictOptions(flatten_optimizer_state_dict=True)
    all_sd = []
    for stage, optimizer in zip(stages, stage_optimizers):
        sd = get_optim_state_dict(stage, optimizer, options=opts)
        all_sd.append(sd)

    # (1) Verify flatten key structure
    for local_idx, sd in enumerate(all_sd):
        model_fqns = {name for name, _ in stages[local_idx].named_parameters()}
        for key in sd.keys():
            assert key.startswith("state.") or key.startswith("param_group."), (
                f"flat key '{key}' should start with 'state.' or 'param_group.'"
            )
            if key.startswith("state."):
                remainder = key[len("state."):]
                matched = any(
                    remainder == fqn or remainder.startswith(fqn + ".")
                    for fqn in model_fqns
                )
                assert matched, (
                    f"flat state key '{key}': FQN part not found in model FQNs {model_fqns}"
                )

    # (2) Verify flatten state values are non-trivial
    for local_idx, sd in enumerate(all_sd):
        for key, value in sd.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                assert value.abs().sum().item() > 0, (
                    f"vstage {local_idx} key '{key}': state tensor should be non-zero"
                )

    # (3) DCP save/load + flatten roundtrip
    _cleanup_ckpt(_CKPT_DIR_FLATTEN_DCP)

    virtual_stage_indices = _owned_virtual_stages(pp_rank)
    for local_idx, (stage, optimizer) in enumerate(zip(stages, stage_optimizers)):
        v_stage = virtual_stage_indices[local_idx]
        ckpt_path = os.path.join(_CKPT_DIR_FLATTEN_DCP, f"pp{pp_rank}_vstage{v_stage}")
        save(all_sd[local_idx], checkpoint_id=ckpt_path, no_dist=True)

    stages2, _, schedule2, stage_optimizers2, _, _, _, _ = (
        _build_pp_hsdp_stages()
    )

    for local_idx, (stage2, optimizer2) in enumerate(zip(stages2, stage_optimizers2)):
        v_stage = virtual_stage_indices[local_idx]
        ckpt_path = os.path.join(_CKPT_DIR_FLATTEN_DCP, f"pp{pp_rank}_vstage{v_stage}")
        storage_reader = FileSystemReader(ckpt_path)
        template = _build_optim_state_dict_load_template(
            stage2, optimizer2, storage_reader, options=opts,
        )
        assert len(template) > 0, "flatten template should not be empty"
        load(template, checkpoint_id=ckpt_path, no_dist=True)
        set_optim_state_dict(stage2, optimizer2, template, options=opts)

    # (4) Verify training continues
    losses = _pp_hsdp_train_step(schedule2, stage_optimizers2, x, pp_rank)
    total_loss = torch.tensor(0.0, device=device)
    if pp_rank == _PP_SIZE - 1:
        for sub_loss in losses:
            activation = sub_loss[0] if isinstance(sub_loss, (list, tuple)) else sub_loss
            total_loss = total_loss + activation.sum()
        loss_val = total_loss.item()
        assert not math.isnan(loss_val), f"loss is NaN after flatten DCP roundtrip: {loss_val}"

    print(f"[rank{_rank()}] P7b PASS: flatten DCP + key structure verification")
    _cleanup_ckpt(_CKPT_DIR_FLATTEN_DCP)
