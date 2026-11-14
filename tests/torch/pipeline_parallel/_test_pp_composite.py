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
"""PP/HSDP policy, split backward, checkpointing, and concurrent-overlap parity."""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from functools import partial
from unittest.mock import patch


from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.checkpoint import checkpoint

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.pipeline_parallel.scheduler import ScheduleGPipe, Schedule1F1B, ScheduleInterleaved1F1B
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.pipeline_parallel import PipelineStage

from tests.torch.pipeline_parallel.hsdp_overlap import initialize_device, register_hsdp_overlap

_D_HID = 8
_NUM_STAGES = 4
_NUM_MICROBATCHES = 4
_LR = 0.01
_INIT_SEED = 42
_DATA_SEED = 2026
_RTOL = 1e-5
_ATOL = 1e-6

_PP_SIZE = int(os.environ.get("HSDP_VPP_PP_SIZE", "2"))
_DP_SIZE = int(os.environ.get("HSDP_VPP_DP_SIZE", "2"))      # HSDP replicate dim
_FSDP_SIZE = int(os.environ.get("HSDP_VPP_FSDP_SIZE", "2"))  # HSDP shard dim
_STEPS = int(os.environ.get("HSDP_VPP_STEPS", "5"))


class MLPModule(nn.Module):
    """Two-layer MLP block (mirrors the composite reference)."""

    def __init__(self, d_hid: int) -> None:
        super().__init__()
        self.net1 = nn.Linear(d_hid, d_hid)
        self.net2 = nn.Linear(d_hid, d_hid)
        self.net2.output_scale_weight = nn.Parameter(torch.ones(d_hid + 1, d_hid))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two dense layers."""
        hidden_states = self.net2(self.net1(x))
        return hidden_states * self.net2.output_scale_weight.mean(dim=0)


class StageModel(nn.Module):
    """One pipeline (virtual) stage: a contiguous slice of MLP layers."""

    def __init__(self, layers, checkpointing=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        self.checkpointing = checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward received activations through the owned layers."""
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False) if self.checkpointing else layer(x)
        return x


def _owned_virtual_stages(pp_rank: int) -> List[int]:
    """Loop layout: virtual stage ``v`` is owned by ``v % pp_size``."""
    return list(range(pp_rank, _NUM_STAGES, _PP_SIZE))


def _to_numpy(tensor) -> np.ndarray:
    """Convert a Tensor / DTensor gradient to a host numpy array."""
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.detach().float().cpu().numpy()


def _build_global_inputs(device: torch.device, total_dp: int, step: int) -> torch.Tensor:
    """Deterministic ``(total_dp * num_microbatches, D_HID)`` global batch, identical on all ranks."""
    generator = torch.Generator(device="cpu").manual_seed(_DATA_SEED + step)
    rows = total_dp * _NUM_MICROBATCHES
    return torch.randn(rows, _D_HID, generator=generator).to(device=device)


def _run_reference_step(
    ref_layers: nn.ModuleList,
    global_inputs: torch.Tensor,
    total_microbatches: int,
) -> float:
    """Serial full-model run over every micro-batch under loss ``sum(output)``; returns total loss."""
    micro = global_inputs.shape[0] // total_microbatches
    total_loss = 0.0
    for mb_idx in range(total_microbatches):
        x = global_inputs[mb_idx * micro : (mb_idx + 1) * micro]
        for layer in ref_layers:
            x = layer(x)
        loss = x.sum()
        loss.backward()
        total_loss += float(loss.detach().cpu().item())
    return total_loss


def _wrap_stage_with_hsdp(stage_model: StageModel, hsdp_mesh) -> StageModel:
    """``fully_shard`` the stage over the 2-D HSDP mesh, reducing grads with SUM.

    Multiple layers have independent management units under an empty HSDP root.
    Single-layer stages let their root manage the layer's parameters directly.
    """
    if len(stage_model.layers) > 1:
        for layer in stage_model.layers:
            fully_shard(layer, mesh=hsdp_mesh, reshard_after_forward=True)
            layer.set_reduce_op_type("sum")
    fully_shard(stage_model, mesh=hsdp_mesh, reshard_after_forward=True)
    stage_model.set_reduce_op_type("sum")
    return stage_model


def _build_stages(
    base_layers: List[MLPModule],
    pp_rank: int,
    pp_mesh,
    hsdp_mesh,
    device: torch.device,
    module_settings: dict,
    checkpointing: bool,
    layers_per_stage: int,
) -> Tuple[List[StageModel], List[PipelineStage], Dict[str, torch.nn.Parameter]]:
    """Realise this pp_rank's owned virtual stages into HSDP-wrapped stages + PipelineStages.

    Returns the wrapped stage models, their PipelineStage wrappers (in execution order),
    and a ``{global_layer_idx}.<sub_name> -> param`` map for grad-parity comparison.
    """
    stages: List[StageModel] = []
    pipeline_stages: List[PipelineStage] = []
    param_map: Dict[str, torch.nn.Parameter] = {}
    for virtual_stage_idx in _owned_virtual_stages(pp_rank):
        layer_start = virtual_stage_idx * layers_per_stage
        owned_layers = [copy.deepcopy(layer).to(device=device)
                        for layer in base_layers[layer_start:layer_start + layers_per_stage]]
        stage_model = _wrap_stage_with_hsdp(StageModel(owned_layers, checkpointing), hsdp_mesh)
        stage_model.set_reshard_after_forward(module_settings["reshard_after_forward"])
        stage_model.set_reshard_after_backward(module_settings["reshard_after_backward"])
        stage_model.set_requires_gradient_sync(module_settings["requires_gradient_sync"])
        pipeline_stage = PipelineStage(
            stage_model,
            stage_index=virtual_stage_idx,
            stage_num=_NUM_STAGES,
            group=pp_mesh.get_group(),
            mesh=pp_mesh,
            device=device,
        )
        stages.append(stage_model)
        pipeline_stages.append(pipeline_stage)
        for layer_index, layer in enumerate(stage_model.layers):
            for sub_name, param in layer.named_parameters():
                param_map[f"{layer_start + layer_index}.{sub_name}"] = param
    return stages, pipeline_stages, param_map


def _assert_grad_parity(
    case_name: str,
    rank: int,
    step: int,
    fsdp_rank: int,
    fsdp_size: int,
    stage_param_map: Dict[str, torch.nn.Parameter],
    ref_param_map: Dict[str, torch.nn.Parameter],
) -> None:
    """Compare every owned reduced gradient against the reference gradient shard.

    Each parameter is sharded dim-0 over the ``fsdp`` (shard) dim only; the ``dp``
    (replicate) dim is all-reduced, so the local reduced grad must equal the full
    serial reference grad sliced to ``[fsdp_rank * shard, (fsdp_rank + 1) * shard]``.
    A drain that never ran leaves ``stage_param.grad is None`` and trips here.
    """
    for fqn, stage_param in stage_param_map.items():
        if stage_param.grad is None:
            raise AssertionError(
                f"{case_name}: rank {rank} step {step} param {fqn!r} grad is None "
                "-- HSDP drain did not run for this stage."
            )
        stage_grad_tensor = stage_param.grad
        is_ragged_grad = (
            isinstance(stage_grad_tensor, DTensor)
            and any(placement.is_ragged_shard() for placement in stage_grad_tensor.placements)
        )
        stage_grad = _to_numpy(stage_grad_tensor)
        ref_grad_full = _to_numpy(ref_param_map[fqn].grad)
        dim0_shard_size = (ref_grad_full.shape[0] + fsdp_size - 1) // fsdp_size
        shard_offset = min(fsdp_rank * dim0_shard_size, ref_grad_full.shape[0])
        actual_shard_size = min(dim0_shard_size, ref_grad_full.shape[0] - shard_offset)
        ref_grad = ref_grad_full[shard_offset : shard_offset + actual_shard_size]
        if is_ragged_grad:
            stage_grad = stage_grad.flatten()
            ref_grad = ref_grad.flatten()
            if stage_grad.size != ref_grad.size:
                raise AssertionError(
                    f"{case_name}: rank {rank} step {step} param {fqn!r} ragged grad numel mismatch "
                    f"(actual={stage_grad.size}, expected={ref_grad.size})."
                )
        elif stage_grad.shape != ref_grad.shape:
            raise AssertionError(
                f"{case_name}: rank {rank} step {step} param {fqn!r} grad shape mismatch "
                f"(actual={stage_grad.shape}, expected={ref_grad.shape})."
            )
        if not np.allclose(stage_grad, ref_grad, rtol=_RTOL, atol=_ATOL):
            abs_err = float(np.abs(stage_grad - ref_grad).max())
            raise AssertionError(
                f"{case_name}: rank {rank} step {step} param {fqn!r} grad mismatch "
                f"(max_abs_err={abs_err:.3e}, rtol={_RTOL}, atol={_ATOL})."
            )


def _assert_pp_hsdp_vpp_matches_reference(
    case_name: str, schedule_factory=ScheduleInterleaved1F1B, *,
    module_settings=None, checkpointing=False, concurrent_overlap=False, layers_per_stage=1,
) -> None:
    """Drive PP + HSDP + Interleaved 1F1B and assert loss + grad parity vs single-card."""
    device = initialize_device()
    rank = dist.get_rank()
    if module_settings is None:
        module_settings = {
            "reshard_after_forward": True,
            "reshard_after_backward": True,
            "requires_gradient_sync": True,
        }
    world_size = dist.get_world_size()
    expected_world = _PP_SIZE * _DP_SIZE * _FSDP_SIZE
    if world_size != expected_world:
        if rank == 0:
            print(
                f"Skip {case_name}: requires world_size={expected_world} "
                f"(pp={_PP_SIZE}, dp={_DP_SIZE}, fsdp={_FSDP_SIZE}), got {world_size}."
            )
        return
    if _NUM_STAGES % _PP_SIZE != 0:
        raise AssertionError(f"NUM_STAGES ({_NUM_STAGES}) must be divisible by pp_size ({_PP_SIZE}).")
    if _D_HID % _FSDP_SIZE != 0:
        raise AssertionError(f"D_HID ({_D_HID}) must be divisible by fsdp_size ({_FSDP_SIZE}).")

    mesh = init_device_mesh(
        device_type=device.type,
        mesh_shape=(_PP_SIZE, _DP_SIZE, _FSDP_SIZE),
        mesh_dim_names=("pp", "dp", "fsdp"),
    )
    pp_mesh = mesh["pp"]
    dp_mesh = mesh["dp"]
    fsdp_mesh = mesh["fsdp"]
    hsdp_mesh = mesh["dp", "fsdp"]  # 2-D (replicate, shard) submesh for fully_shard
    pp_rank = mesh.get_local_rank("pp")
    dp_rank = mesh.get_local_rank("dp")
    fsdp_rank = mesh.get_local_rank("fsdp")
    is_last_stage = pp_rank == _PP_SIZE - 1

    total_dp = _DP_SIZE * _FSDP_SIZE
    hsdp_idx = dp_rank * _FSDP_SIZE + fsdp_rank  # flat data-parallel index over the HSDP group
    total_microbatches = total_dp * _NUM_MICROBATCHES

    torch.manual_seed(_INIT_SEED)
    base_layers = [MLPModule(_D_HID) for _ in range(_NUM_STAGES * layers_per_stage)]

    # The public API targets accelerators. An explicit CPU test substitutes only
    # device resolution; process groups, collectives, autograd, and P2P stay real.
    device_resolution = (
        patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=device)
        if device.type == "cpu" else nullcontext()
    )
    with device_resolution:
        stages, pipeline_stages, stage_param_map = _build_stages(
            base_layers, pp_rank, pp_mesh, hsdp_mesh, device, module_settings, checkpointing, layers_per_stage
        )
    schedule = schedule_factory(pipeline_stages, _NUM_MICROBATCHES)
    if concurrent_overlap:
        register_hsdp_overlap(schedule, pipeline_stages, device)
    stage_optimizers = [optim.SGD(stage.parameters(), lr=_LR) for stage in stages]

    ref_layers = nn.ModuleList(copy.deepcopy(layer) for layer in base_layers).to(device=device)
    ref_param_map = {
        f"{layer_idx}.{sub_name}": param
        for layer_idx, layer in enumerate(ref_layers)
        for sub_name, param in layer.named_parameters()
    }
    ref_optimizer = optim.SGD(ref_layers.parameters(), lr=_LR)

    for step in range(_STEPS):
        for optimizer in stage_optimizers:
            optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)

        global_inputs = _build_global_inputs(device, total_dp, step)
        local_inputs = global_inputs[hsdp_idx * _NUM_MICROBATCHES : (hsdp_idx + 1) * _NUM_MICROBATCHES]

        losses = schedule.run(local_inputs) if pp_rank == 0 else schedule.run()
        ref_total_loss = _run_reference_step(ref_layers, global_inputs, total_microbatches)

        if is_last_stage:
            local_loss = torch.zeros(1, dtype=torch.float32, device=device)
            for sub_loss in losses:
                activation = sub_loss[0] if isinstance(sub_loss, (list, tuple)) else sub_loss
                local_loss += activation.sum()
            # Recover the global summed loss over the full HSDP data-parallel group.
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM, group=fsdp_mesh.get_group())
            global_loss = float(local_loss.item())
            if not np.isfinite(global_loss) or not np.isfinite(ref_total_loss):
                raise AssertionError(
                    f"{case_name}: rank {rank} step {step} non-finite loss "
                    f"(global={global_loss}, ref={ref_total_loss})."
                )
            if not np.allclose(global_loss, ref_total_loss, rtol=_RTOL, atol=_ATOL):
                raise AssertionError(
                    f"{case_name}: rank {rank} step {step} loss mismatch "
                    f"global={global_loss:.6f} != ref={ref_total_loss:.6f}."
                )

        _assert_grad_parity(
            case_name, rank, step, fsdp_rank, _FSDP_SIZE, stage_param_map, ref_param_map
        )

        with SkipDTensorDispatch():
            for optimizer in stage_optimizers:
                optimizer.step()
            ref_optimizer.step()
        for fqn, param in stage_param_map.items():
            reference_weight = _to_numpy(ref_param_map[fqn])
            shard_rows = (reference_weight.shape[0] + _FSDP_SIZE - 1) // _FSDP_SIZE
            reference_shard = reference_weight[fsdp_rank * shard_rows:(fsdp_rank + 1) * shard_rows]
            np.testing.assert_allclose(
                _to_numpy(param).reshape(-1), reference_shard.reshape(-1), rtol=_RTOL, atol=_ATOL,
                err_msg=f"{case_name}: rank={rank}, step={step}, parameter={fqn}",
            )

    if rank == 0:
        print(
            f"[Rank {rank}] {case_name} ok "
            f"(pp={_PP_SIZE}, dp={_DP_SIZE}, fsdp={_FSDP_SIZE}, "
            f"virtual_stages={_NUM_STAGES}, layers_per_stage={layers_per_stage}, "
            f"micro_batches={_NUM_MICROBATCHES}, steps={_STEPS})"
        )


def test_pp_hsdp_vpp_interleaved_1f1b_matches_reference() -> None:
    """
    Feature: ``PP + HSDP`` under ``ScheduleInterleaved1F1B`` vs single-card baseline.
    Description:
        1. Build mesh ``(pp=2, dp=2, fsdp=2)`` on 8 ranks; hand the 2-D ``(dp, fsdp)``
           submesh to ``fully_shard`` so each stage runs HSDP (replicate + shard).
        2. Use the VPP loop layout (2 single-layer virtual stages per pp_rank) under
           ``ScheduleInterleaved1F1B``.
        3. Each step asserts the HSDP-reduced global loss and every owned reduced
           gradient match a full-model serial single-card reference (additive loss),
           directly exercising the PP-driven HSDP gradient drain.
    Expectation:
        Every step's global loss and every owned gradient match the reference within
        ``rtol=1e-5`` / ``atol=1e-6``.
    """
    _assert_pp_hsdp_vpp_matches_reference("pp_hsdp_vpp_interleaved_1f1b")


def test_pp_hsdp_vpp_dxdw_split_matches_reference() -> None:
    """Torch PP+HSDP split backward should match the full-model reference."""
    _assert_pp_hsdp_vpp_matches_reference(
        "pp_hsdp_vpp_dxdw_split",
        schedule_factory=partial(ScheduleInterleaved1F1B, overlap_b_f=True, enable_dxdw_split=True),
    )


def test_pp_hsdp_policies_checkpoint_overlap_matches_reference() -> None:
    """Exercise all residency/sync combinations with real concurrent full and split backward."""
    for reshard_after_forward, reshard_after_backward, gradient_sync in (
        (False, False, False), (False, False, True), (True, True, True), (True, True, False),
    ):
        for split in (False, True):
            _assert_pp_hsdp_vpp_matches_reference(
                "pp_hsdp_policies_checkpoint_overlap",
                schedule_factory=partial(ScheduleInterleaved1F1B, overlap_b_f=True, enable_dxdw_split=split),
                module_settings={
                    "reshard_after_forward": reshard_after_forward,
                    "reshard_after_backward": reshard_after_backward,
                    "requires_gradient_sync": gradient_sync,
                },
                checkpointing=True, concurrent_overlap=True,
            )


def test_pp_hsdp_gpipe_1f1b_matches_reference() -> None:
    """Use PP=4/HSDP=(1,2) with each ordinary pipeline schedule on eight ranks."""
    if (_PP_SIZE, _DP_SIZE, _FSDP_SIZE) != (4, 1, 2):
        raise ValueError("Set HSDP_VPP_PP_SIZE=4, HSDP_VPP_DP_SIZE=1, HSDP_VPP_FSDP_SIZE=2")
    for schedule_factory in (ScheduleGPipe, Schedule1F1B):
        _assert_pp_hsdp_vpp_matches_reference("pp_hsdp_gpipe_1f1b", schedule_factory=schedule_factory)


def test_pp_hsdp_nested_units_checkpoint_matches_reference() -> None:
    """Compare real nested-unit HSDP gradients under full and split backward."""
    for split in (False, True):
        _assert_pp_hsdp_vpp_matches_reference(
            "pp_hsdp_nested_units_checkpoint",
            schedule_factory=partial(ScheduleInterleaved1F1B, overlap_b_f=True, enable_dxdw_split=split),
            checkpointing=True, layers_per_stage=2,
        )
