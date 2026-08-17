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
"""PP + HSDP (2-D ``(replicate, shard)`` mesh) under Interleaved 1F1B vs single-card.

Torch counterpart of ``tests/mindspore/st/pipeline_parallel/_test_pp_composite.py``'s
``test_fully_shard_pp_vpp``, but the ``fully_shard`` units run over a **2-D HSDP**
submesh instead of a 1-D FSDP mesh.  The point is to exercise the HSDP gradient
*drain* (reduce-scatter over the shard dim **and** all-reduce over the replicate
dim) when it is driven by the pipeline ``FSDP_REDUCE_GRAD`` MetaStep under
``ScheduleInterleaved1F1B`` -- the path fixed by the unconditional drain in
``_root_backward_hook``.

Topology (default, 8 ranks): 3-D mesh ``(pp=2, dp=2, fsdp=2)``.

* ``pp`` -- pipeline width 2.
* ``(dp, fsdp)`` -- the 2-D HSDP submesh handed to ``fully_shard``: ``dp`` is the
  replicate dim (grads all-reduced), ``fsdp`` is the shard dim (params sharded
  dim-0, grads reduce-scattered).  The combined ``dp * fsdp`` group is the data-
  parallel world, so each of those ranks consumes a distinct micro-batch slice.

VPP loop layout (``TOTAL_LAYERS=4, pp=2``): virtual stage ``v`` is owned by
``pp_rank v % pp_size`` and holds exactly layer ``v`` -- pp_rank 0 -> stages [0, 2],
pp_rank 1 -> stages [1, 3]; ``stage_num = TOTAL_LAYERS``.

Reference: every rank holds the full MLP stack and runs all ``dp*fsdp*num_microbatches``
micro-batches serially under the additive loss ``sum(output)``.  Because the loss is
additive and grads are reduced with SUM, the HSDP-reduced stage gradient must equal
the full serial gradient (sliced to this rank's shard).  Each training step asserts:

1. global summed loss (all-reduced over the HSDP group) == reference total loss;
2. every owned parameter's reduced gradient == the reference gradient shard
   (this is the direct drain check -- a missed drain shows up as ``grad is None``
   or a one-DP-rank-short gradient).

Sizes are env-tunable (``HSDP_VPP_PP_SIZE`` / ``_DP_SIZE`` / ``_FSDP_SIZE`` /
``_STEPS``) so the same worker can run a reduced config on fewer cards.
"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
from __future__ import annotations

import copy
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch import nn, optim

from hyper_parallel import DTensor, ScheduleInterleaved1F1B, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.pipeline_parallel import PipelineStage

from tests.torch.utils import init_dist

_D_HID = 8
_TOTAL_LAYERS = 4
_NUM_MICROBATCHES = 4
_LR = 0.01
_INIT_SEED = 42
_DATA_SEED = 2026
_RTOL = 2e-3
_ATOL = 1e-3

_PP_SIZE = int(os.environ.get("HSDP_VPP_PP_SIZE", "2"))
_DP_SIZE = int(os.environ.get("HSDP_VPP_DP_SIZE", "2"))      # HSDP replicate dim
_FSDP_SIZE = int(os.environ.get("HSDP_VPP_FSDP_SIZE", "2"))  # HSDP shard dim
_STEPS = int(os.environ.get("HSDP_VPP_STEPS", "5"))


class MLPModule(nn.Module):
    """Two-layer MLP block (mirrors the MindSpore composite reference)."""

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

    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward received activations through the owned layers."""
        for layer in self.layers:
            x = layer(x)
        return x


def _owned_virtual_stages(pp_rank: int) -> List[int]:
    """Loop layout: virtual stage ``v`` (== layer ``v``) is owned by ``v % pp_size``."""
    return list(range(pp_rank, _TOTAL_LAYERS, _PP_SIZE))


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

    Single-layer virtual stages (the VPP loop layout) skip a per-layer wrap to avoid
    nesting an FSDP root inside another FSDP root with the same single child -- matching
    ``_wrap_stage_with_fsdp`` in the MindSpore composite reference.
    """
    if len(stage_model.layers) > 1:
        for layer in stage_model.layers:
            fully_shard(layer, mesh=hsdp_mesh, reshard_after_forward=True)
            if hasattr(layer, "set_reduce_op_type"):
                layer.set_reduce_op_type("sum")
    fully_shard(stage_model, mesh=hsdp_mesh, reshard_after_forward=True)
    if hasattr(stage_model, "set_reduce_op_type"):
        stage_model.set_reduce_op_type("sum")
    return stage_model


def _build_stages(
    base_layers: List[MLPModule],
    pp_rank: int,
    pp_mesh,
    hsdp_mesh,
    device: torch.device,
) -> Tuple[List[StageModel], List[PipelineStage], Dict[str, torch.nn.Parameter]]:
    """Realise this pp_rank's owned virtual stages into HSDP-wrapped stages + PipelineStages.

    Returns the wrapped stage models, their PipelineStage wrappers (in execution order),
    and a ``{global_layer_idx}.<sub_name> -> param`` map for grad-parity comparison.
    """
    stages: List[StageModel] = []
    pipeline_stages: List[PipelineStage] = []
    param_map: Dict[str, torch.nn.Parameter] = {}
    for virtual_stage_idx in _owned_virtual_stages(pp_rank):
        owned_layer = copy.deepcopy(base_layers[virtual_stage_idx]).to(device=device)
        stage_model = _wrap_stage_with_hsdp(StageModel([owned_layer]), hsdp_mesh)
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
        for sub_name, param in stage_model.layers[0].named_parameters():
            param_map[f"{virtual_stage_idx}.{sub_name}"] = param
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
        stage_grad = _to_numpy(stage_param.grad)
        ref_grad_full = _to_numpy(ref_param_map[fqn].grad)
        dim0_shard_size = (ref_grad_full.shape[0] + fsdp_size - 1) // fsdp_size
        shard_offset = min(fsdp_rank * dim0_shard_size, ref_grad_full.shape[0])
        actual_shard_size = min(dim0_shard_size, ref_grad_full.shape[0] - shard_offset)
        ref_grad = ref_grad_full[shard_offset : shard_offset + actual_shard_size]
        if stage_grad.shape != ref_grad.shape:
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


def _assert_pp_hsdp_vpp_matches_reference(case_name: str) -> None:
    """Drive PP + HSDP + Interleaved 1F1B and assert loss + grad parity vs single-card."""
    rank, device_id = init_dist()
    world_size = dist.get_world_size()
    expected_world = _PP_SIZE * _DP_SIZE * _FSDP_SIZE
    if world_size != expected_world:
        if rank == 0:
            print(
                f"Skip {case_name}: requires world_size={expected_world} "
                f"(pp={_PP_SIZE}, dp={_DP_SIZE}, fsdp={_FSDP_SIZE}), got {world_size}."
            )
        return
    if _TOTAL_LAYERS % _PP_SIZE != 0:
        raise AssertionError(f"TOTAL_LAYERS ({_TOTAL_LAYERS}) must be divisible by pp_size ({_PP_SIZE}).")
    if _D_HID % _FSDP_SIZE != 0:
        raise AssertionError(f"D_HID ({_D_HID}) must be divisible by fsdp_size ({_FSDP_SIZE}).")

    device = torch.device("npu", device_id)
    mesh = init_device_mesh(
        device_type="npu",
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
    base_layers = [MLPModule(_D_HID) for _ in range(_TOTAL_LAYERS)]

    stages, pipeline_stages, stage_param_map = _build_stages(
        base_layers, pp_rank, pp_mesh, hsdp_mesh, device
    )
    schedule = ScheduleInterleaved1F1B(pipeline_stages, _NUM_MICROBATCHES)
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

    if rank == 0:
        print(
            f"[Rank {rank}] {case_name} ok "
            f"(pp={_PP_SIZE}, dp={_DP_SIZE}, fsdp={_FSDP_SIZE}, "
            f"virtual_stages={_TOTAL_LAYERS}, micro_batches={_NUM_MICROBATCHES}, steps={_STEPS})"
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
        ``rtol=2e-3`` / ``atol=1e-3``.
    """
    _assert_pp_hsdp_vpp_matches_reference("pp_hsdp_vpp_interleaved_1f1b")
