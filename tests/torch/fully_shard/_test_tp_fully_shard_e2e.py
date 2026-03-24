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
"""End-to-end TP + fully_shard training verification."""
# pylint: disable=W0611
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import numpy as np
import torch
from torch import nn
from torch import optim
import torch_npu

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.shard.api import shard_module
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from tests.torch.utils import init_dist


class TPFullyShardNet(nn.Module):
    """Simple linear network used to compare standalone and TP + FSDP training."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_size, output_size).npu())

    def forward(self, x):
        return torch.matmul(x, self.weight)


def mse_loss_sum(y_pred, y_true):
    """Return the summed MSE loss used by both standalone and distributed paths."""
    error = torch.sub(y_pred, y_true)
    square_error = torch.square(error)
    return torch.sum(square_error)


def _build_mesh(world_size: int, tp_size: int = 2):
    """
    Build a 2D mesh for TP + fully_shard.

    Input:
        world_size: total process count from torchrun.
    Expected output:
        A mesh shaped as (dp, tp), where dp > 1 enables fully_shard and tp > 1 enables tensor parallelism.
    """
    if world_size < 4 or world_size % tp_size != 0:
        return None, None, None, None
    dp_size = world_size // tp_size
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    return root_mesh, root_mesh["dp"], root_mesh["tp"], dp_size


def _build_3d_mesh(world_size: int, dp_size: int = 2, tp_size: int = 2, ep_size: int = 2):
    """
    Build a 3D root mesh and expose the DP and TPxEP sub-meshes.

    Input:
        world_size: total process count from torchrun.
    Expected output:
        A root mesh shaped as (dp, tp, ep), together with its 1D ``dp`` sub-mesh
        and 2D ``(tp, ep)`` sub-mesh used by DTensor TP layouts.
    """
    expected_world_size = dp_size * tp_size * ep_size
    if world_size != expected_world_size:
        return None, None, None, None, None
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dp_size, tp_size, ep_size),
        mesh_dim_names=("dp", "tp", "ep"),
    )
    return root_mesh, root_mesh["dp"], root_mesh[("tp", "ep")], dp_size, tp_size


def _build_hsdp_tp_mesh(
    world_size: int,
    replicate_size: int = 2,
    fsdp_size: int = 2,
    tp_size: int = 2,
):
    """
    Build a 3D root mesh for HSDP + TP.

    Input:
        world_size: total process count from torchrun.
    Expected output:
        A root mesh shaped as (dp, fsdp, tp), together with its 2D HSDP mesh
        ``(dp, fsdp)`` and 1D TP mesh ``tp``.
    """
    expected_world_size = replicate_size * fsdp_size * tp_size
    if world_size != expected_world_size:
        return None, None, None, None, None
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(replicate_size, fsdp_size, tp_size),
        mesh_dim_names=("dp", "fsdp", "tp"),
    )
    return root_mesh, root_mesh[("dp", "fsdp")], root_mesh["tp"], replicate_size * fsdp_size, tp_size


def _flatten_coordinate(mesh_shape, coordinate):
    """Flatten an n-D mesh coordinate into a row-major linear index."""
    flat_idx = 0
    for dim_size, dim_idx in zip(mesh_shape, coordinate):
        flat_idx = flat_idx * dim_size + dim_idx
    return flat_idx


def _assert_loss_and_grad_match(
    *,
    case_name: str,
    steps: int,
    batch_size: int,
    input_size: int,
    output_size: int,
    input_seed: int,
    label_seed: int,
    init_seed: int,
    tp_size: int,
):
    """
    Run one TP + fully_shard end-to-end case and compare standalone vs distributed.

    Input:
        steps/batch_size/input_size/output_size describe the training shape,
        input_seed/label_seed/init_seed make the case deterministic,
        tp_size controls the TP mesh dimension.
    Expected output:
        1. Each training step produces the same loss as standalone.
        2. Each rank's local gradient matches the corresponding standalone [dp, tp] shard.
    """
    rank, _ = init_dist()
    world_size = torch.distributed.get_world_size()
    root_mesh, dp_mesh, tp_mesh, dp_size = _build_mesh(world_size, tp_size=tp_size)
    if root_mesh is None or dp_size <= 1:
        print(
            f"[Rank {rank}] Skip {case_name} because world_size={world_size} "
            f"cannot form TP + FSDP mesh with tp_size={tp_size}."
        )
        return

    if input_size % dp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because input_size={input_size} "
            f"is not divisible by dp_size={dp_size}."
        )
        return
    if output_size % tp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by tp_size={tp_size}."
        )
        return

    torch.manual_seed(1)
    np.random.seed(1)

    # Input: build one global batch and split it across DP ranks inside the
    # distributed path so SUM-based gradient communication matches standalone.
    torch.manual_seed(input_seed)
    input_data = torch.randn(batch_size, input_size).npu()
    torch.manual_seed(label_seed)
    label_data = torch.randn(batch_size, output_size).npu()

    # Input: standalone and distributed models share the same initialization seed.
    # Expected output: any loss/grad difference comes from the parallel path, not init drift.
    torch.manual_seed(init_seed)
    standalone_model = TPFullyShardNet(input_size, output_size)
    torch.manual_seed(init_seed)
    dist_model = TPFullyShardNet(input_size, output_size)

    dp_idx = root_mesh.get_coordinate()[0]
    standalone_grads, standalone_local_losses = _run_standalone_training(
        standalone_model,
        input_data,
        label_data,
        steps,
        dp_size,
        dp_idx,
    )
    dist_losses, dist_grads = _run_tp_fully_shard_training(
        dist_model,
        dp_mesh,
        tp_mesh,
        input_data,
        label_data,
        steps,
        tp_reduce_size=tp_size,
    )

    for step_idx in range(steps):
        assert np.allclose(
            standalone_local_losses[step_idx].numpy(),
            dist_losses[step_idx].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected local loss "
            f"{standalone_local_losses[step_idx].item()}, got {dist_losses[step_idx].item()}"
        )

        expected_grad = _get_expected_grad_slice(
            standalone_grads[step_idx],
            root_mesh,
            dp_size,
            tp_size,
            dp_mesh_ndim=1,
            tp_mesh_dim=1,
        )
        assert np.allclose(
            expected_grad.numpy(),
            dist_grads[step_idx].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected grad slice shape "
            f"{tuple(expected_grad.shape)}, got {tuple(dist_grads[step_idx].shape)}"
        )

    print(
        f"[Rank {rank}] {case_name} passed with mesh={root_mesh.mesh_shape}, "
        f"steps={steps}, local_grad_shape={tuple(dist_grads[-1].shape)}"
    )


def _assert_loss_and_grad_match_3d_root_mesh(
    *,
    case_name: str,
    steps: int,
    batch_size: int,
    input_size: int,
    output_size: int,
    input_seed: int,
    label_seed: int,
    init_seed: int,
    dp_size: int = 2,
    tp_size: int = 2,
    ep_size: int = 2,
):
    """
    Run one TP(+EP replicate) + fully_shard end-to-end case on a 3D root mesh.

    Input:
        steps/batch_size/input_size/output_size describe the training shape,
        input_seed/label_seed/init_seed make the case deterministic,
        ``(dp_size, tp_size, ep_size)`` defines the exact root mesh shape.
    Expected output:
        1. Each training step produces the same per-rank DP-local loss as standalone.
        2. Each rank's local gradient matches the corresponding standalone [dp, tp] shard.
        3. The extra ``ep`` dimension is only replicate and does not change numerical results.
    """
    rank, _ = init_dist()
    world_size = torch.distributed.get_world_size()
    root_mesh, dp_mesh, tp_ep_mesh, built_dp_size, built_tp_size = _build_3d_mesh(
        world_size,
        dp_size=dp_size,
        tp_size=tp_size,
        ep_size=ep_size,
    )
    if root_mesh is None:
        print(
            f"[Rank {rank}] Skip {case_name} because world_size={world_size} "
            f"cannot form root mesh ({dp_size}, {tp_size}, {ep_size})."
        )
        return

    if input_size % built_dp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because input_size={input_size} "
            f"is not divisible by dp_size={built_dp_size}."
        )
        return
    if output_size % built_tp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by tp_size={built_tp_size}."
        )
        return

    torch.manual_seed(1)
    np.random.seed(1)

    torch.manual_seed(input_seed)
    input_data = torch.randn(batch_size, input_size).npu()
    torch.manual_seed(label_seed)
    label_data = torch.randn(batch_size, output_size).npu()

    torch.manual_seed(init_seed)
    standalone_model = TPFullyShardNet(input_size, output_size)
    torch.manual_seed(init_seed)
    dist_model = TPFullyShardNet(input_size, output_size)

    dp_idx = root_mesh.get_coordinate()[0]
    standalone_grads, standalone_local_losses = _run_standalone_training(
        standalone_model,
        input_data,
        label_data,
        steps,
        built_dp_size,
        dp_idx,
    )
    dist_losses, dist_grads = _run_tp_fully_shard_training(
        dist_model,
        dp_mesh,
        tp_ep_mesh,
        input_data,
        label_data,
        steps,
        tp_reduce_size=int(np.prod(tp_ep_mesh.mesh_shape)),
    )

    for step_idx in range(steps):
        assert np.allclose(
            standalone_local_losses[step_idx].numpy(),
            dist_losses[step_idx].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected local loss "
            f"{standalone_local_losses[step_idx].item()}, got {dist_losses[step_idx].item()}"
        )

        expected_grad = _get_expected_grad_slice(
            standalone_grads[step_idx],
            root_mesh,
            built_dp_size,
            built_tp_size,
            dp_mesh_ndim=1,
            tp_mesh_dim=1,
        )
        assert np.allclose(
            expected_grad.numpy(),
            dist_grads[step_idx].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected grad slice shape "
            f"{tuple(expected_grad.shape)}, got {tuple(dist_grads[step_idx].shape)}"
        )

    print(
        f"[Rank {rank}] {case_name} passed with root_mesh={root_mesh.mesh_shape}, "
        f"tp_ep_mesh={tp_ep_mesh.mesh_shape}, steps={steps}, "
        f"local_grad_shape={tuple(dist_grads[-1].shape)}"
    )


def _assert_loss_and_grad_match_hsdp_tp_root_mesh(
    *,
    case_name: str,
    steps: int,
    batch_size: int,
    input_size: int,
    output_size: int,
    input_seed: int,
    label_seed: int,
    init_seed: int,
    replicate_size: int = 2,
    fsdp_size: int = 2,
    tp_size: int = 2,
):
    """
    Run one HSDP(2D) + TP(1D) end-to-end case on a 3D root mesh.

    Input:
        steps/batch_size/input_size/output_size describe the training shape,
        input_seed/label_seed/init_seed make the case deterministic,
        ``(replicate_size, fsdp_size, tp_size)`` defines the exact root mesh shape.
    Expected output:
        1. Each training step produces the same DP-domain local loss as standalone.
        2. Each rank's local gradient matches the corresponding standalone [dp*fsdp, tp] shard.
    """
    rank, _ = init_dist()
    world_size = torch.distributed.get_world_size()
    root_mesh, hsdp_mesh, tp_mesh, dp_domain_size, built_tp_size = _build_hsdp_tp_mesh(
        world_size,
        replicate_size=replicate_size,
        fsdp_size=fsdp_size,
        tp_size=tp_size,
    )
    if root_mesh is None:
        print(
            f"[Rank {rank}] Skip {case_name} because world_size={world_size} "
            f"cannot form root mesh ({replicate_size}, {fsdp_size}, {tp_size})."
        )
        return

    if input_size % dp_domain_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because input_size={input_size} "
            f"is not divisible by dp_domain_size={dp_domain_size}."
        )
        return
    if output_size % built_tp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by tp_size={built_tp_size}."
        )
        return

    torch.manual_seed(1)
    np.random.seed(1)

    torch.manual_seed(input_seed)
    input_data = torch.randn(batch_size, input_size).npu()
    torch.manual_seed(label_seed)
    label_data = torch.randn(batch_size, output_size).npu()

    torch.manual_seed(init_seed)
    standalone_model = TPFullyShardNet(input_size, output_size)
    torch.manual_seed(init_seed)
    dist_model = TPFullyShardNet(input_size, output_size)

    dp_coord = root_mesh.get_coordinate()[:2]
    dp_idx = _flatten_coordinate((replicate_size, fsdp_size), dp_coord)
    standalone_grads, standalone_local_losses = _run_standalone_training(
        standalone_model,
        input_data,
        label_data,
        steps,
        dp_domain_size,
        dp_idx,
    )
    dist_losses, dist_grads = _run_tp_fully_shard_training(
        dist_model,
        hsdp_mesh,
        tp_mesh,
        input_data,
        label_data,
        steps,
        tp_reduce_size=built_tp_size,
    )

    for step_idx in range(steps):
        assert np.allclose(
            standalone_local_losses[step_idx].numpy(),
            dist_losses[step_idx].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected local loss "
            f"{standalone_local_losses[step_idx].item()}, got {dist_losses[step_idx].item()}"
        )

        expected_grad = _get_expected_hsdp_tp_grad_slice(
            standalone_grads[step_idx],
            root_mesh,
            fsdp_size,
            built_tp_size,
        )
        assert np.allclose(
            expected_grad.numpy(),
            dist_grads[step_idx].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected grad slice shape "
            f"{tuple(expected_grad.shape)}, got {tuple(dist_grads[step_idx].shape)}"
        )

    print(
        f"[Rank {rank}] {case_name} passed with root_mesh={root_mesh.mesh_shape}, "
        f"hsdp_mesh={hsdp_mesh.mesh_shape}, steps={steps}, "
        f"local_grad_shape={tuple(dist_grads[-1].shape)}"
    )


def _run_standalone_training(model, x, y, steps, dp_size, dp_idx):
    """
    Run standalone training.

    Input:
        Full weight matrix, full input, full label.
    Expected output:
        Per-step current-rank local-batch losses and full dense gradients from
        the standalone model.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    local_losses = []
    grads = []
    for _ in range(steps):
        loss = mse_loss_sum(model(x), y)
        local_x, local_y = _get_local_batch_slice(x, y, dp_size, dp_idx)
        with torch.no_grad():
            local_loss = mse_loss_sum(model(local_x), local_y)
        loss.backward()
        local_losses.append(local_loss.detach().cpu())
        grads.append(model.weight.grad.data.cpu().clone())
        optimizer.step()
        optimizer.zero_grad()
    return grads, local_losses


def _run_tp_fully_shard_training(model, dp_mesh, tp_mesh, x, y, steps, tp_reduce_size):
    """
    Run TP + fully_shard training.

    Input:
        TP-sharded weight on ``tp_mesh`` and fully_shard on explicit ``dp_mesh``.
    Expected output:
        Per-step global losses and local shard gradients whose values match standalone slices.
    """
    x_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    w_placements = (Shard(1),) + tuple(Replicate() for _ in range(tp_mesh.ndim - 1))

    sharding_plan = ShardingPlan(
        plan={"weight": w_placements},
        input_plan={"input": x_placements},
        output_plan={"output": w_placements},
    )
    model = shard_module(model, device_mesh=tp_mesh, sharding_plan=sharding_plan)
    model = fully_shard(
        model,
        mesh=dp_mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
    )
    # DTensor-based fully_shard uses SUM gradient communication in this path.
    # Split the batch along DP so the reduced gradient equals standalone.
    model.set_reduce_op_type("sum")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dp_idx = _flatten_coordinate(dp_mesh.mesh_shape, dp_mesh.get_coordinate())
    local_x, local_y = _get_local_batch_slice(x, y, dp_mesh.size(), dp_idx)
    dist_x = DTensor.from_local(local_x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(local_y, tp_mesh, x_placements)
    losses = []
    grads = []
    for _ in range(steps):
        y_pred = model(dist_x)
        y_shard = dist_y.redistribute(tp_mesh, y_pred.placements)
        loss = mse_loss_sum(y_pred, y_shard)
        loss = loss.reduce_partial()
        # reduce_partial() makes the scalar loss replicated on tp_mesh, so the
        # backward seed still needs TP-side normalization to match standalone.
        loss.backward(torch.tensor(1.0 / tp_reduce_size, device=local_x.device))

        losses.append(loss.to_local().detach().cpu())
        grads.append(model.weight.grad.data.cpu().clone())

        with SkipDTensorDispatch():
            optimizer.step()
            optimizer.zero_grad()
    return losses, grads


def _get_expected_grad_slice(standalone_grad, mesh, dp_size, tp_size, dp_mesh_ndim=1, tp_mesh_dim=1):
    """
    Slice the standalone gradient into the shard expected on the current rank.

    Input:
        Full standalone gradient with shape [input_size, output_size].
    Expected output:
        Local gradient slice with shape [input_size / dp_size, output_size / tp_size].
    """
    coordinate = mesh.get_coordinate()
    dp_idx = _flatten_coordinate(mesh.mesh_shape[:dp_mesh_ndim], coordinate[:dp_mesh_ndim])
    tp_idx = coordinate[tp_mesh_dim]
    local_input = standalone_grad.shape[0] // dp_size
    local_output = standalone_grad.shape[1] // tp_size
    row_start = dp_idx * local_input
    col_start = tp_idx * local_output
    return standalone_grad[
        row_start: row_start + local_input,
        col_start: col_start + local_output,
    ]


def _get_expected_hsdp_tp_grad_slice(standalone_grad, mesh, fsdp_size, tp_size):
    """
    Slice the standalone gradient into the local shard expected for HSDP(2D) + TP(1D).

    Input:
        Full standalone gradient with shape [input_size, output_size] on a root mesh
        shaped as (dp, fsdp, tp).
    Expected output:
        Local gradient slice with shape [input_size / fsdp_size, output_size / tp_size].
        The DP dimension is replicate-only for parameter storage, so it does not further
        shrink the local gradient shape.
    """
    coordinate = mesh.get_coordinate()
    fsdp_idx = coordinate[1]
    tp_idx = coordinate[2]
    local_input = standalone_grad.shape[0] // fsdp_size
    local_output = standalone_grad.shape[1] // tp_size
    row_start = fsdp_idx * local_input
    col_start = tp_idx * local_output
    return standalone_grad[
        row_start: row_start + local_input,
        col_start: col_start + local_output,
    ]


def _get_local_batch_slice(x, y, dp_size, dp_idx):
    """Return the DP-local input and label slices for ``dp_idx``."""
    x_chunks = torch.chunk(x, dp_size, dim=0)
    y_chunks = torch.chunk(y, dp_size, dim=0)
    return x_chunks[dp_idx].contiguous(), y_chunks[dp_idx].contiguous()


def test_tp_plus_fully_shard_loss_and_grad_match_standalone():
    """
    Feature: TP + fully_shard end-to-end training.
    Description:
        1. Build a standalone model and a TP + fully_shard distributed model from the same initialization.
        2. Feed identical input/label tensors for two optimizer steps.
        3. Compare per-step forward loss and backward local gradient shard.
    Expectation:
        1. Distributed replicated loss equals standalone loss at each step.
        2. Each rank's local gradient equals the corresponding standalone [dp, tp] slice.
    """
    _assert_loss_and_grad_match(
        case_name="tp2_square_case",
        steps=2,
        batch_size=16,
        input_size=8,
        output_size=8,
        input_seed=2,
        label_seed=3,
        init_seed=4,
        tp_size=2,
    )


def test_tp_plus_fully_shard_rectangular_inputs_match_standalone():
    """
    Feature: TP + fully_shard end-to-end training.
    Description:
        1. Use a rectangular weight shape and odd batch size.
        2. Train multiple steps with the same TP + fully_shard path.
        3. Compare per-step forward loss and local gradients with standalone.
    Expectation:
        1. Distributed loss equals standalone loss at each step.
        2. Local gradient shape is [input_size / dp_size, output_size / tp_size] and values match.
    """
    _assert_loss_and_grad_match(
        case_name="tp2_rectangular_case",
        steps=3,
        batch_size=11,
        input_size=12,
        output_size=6,
        input_seed=12,
        label_seed=13,
        init_seed=14,
        tp_size=2,
    )


def test_tp_plus_fully_shard_wide_output_match_standalone():
    """
    Feature: TP + fully_shard end-to-end training.
    Description:
        1. Use a wider output dimension to stress TP column sharding.
        2. Train for two steps and compare forward/backward results.
    Expectation:
        1. Distributed loss equals standalone loss at each step.
        2. Each rank receives the correct standalone gradient slice on the wider TP dimension.
    """
    _assert_loss_and_grad_match(
        case_name="tp2_wide_output_case",
        steps=2,
        batch_size=7,
        input_size=16,
        output_size=12,
        input_seed=22,
        label_seed=23,
        init_seed=24,
        tp_size=2,
    )


def test_tp4_plus_fully_shard_loss_and_grad_match_standalone():
    """
    Feature: TP + fully_shard end-to-end training.
    Description:
        1. Build a mesh with tp_size=4 when the world size supports it.
        2. Validate the same standalone-vs-distributed loss and gradient checks.
    Expectation:
        1. On supported meshes, distributed loss equals standalone loss.
        2. Local gradients match the standalone [dp, tp] slice for tp_size=4.
    """
    _assert_loss_and_grad_match(
        case_name="tp4_case",
        steps=2,
        batch_size=9,
        input_size=8,
        output_size=16,
        input_seed=32,
        label_seed=33,
        init_seed=34,
        tp_size=4,
    )


def test_tp_plus_fully_shard_on_3d_root_mesh_match_standalone():
    """
    Feature: TP + fully_shard end-to-end training on a 3D root mesh.
    Description:
        1. Build a root mesh shaped as (dp, tp, ep) = (2, 2, 2).
        2. Use ``dp`` for fully_shard and ``(tp, ep)`` for DTensor layout.
        3. Compare per-step loss and local gradients with standalone.
    Expectation:
        1. Distributed loss equals standalone on each DP-local batch.
        2. Each rank's local gradient equals the standalone [dp, tp] slice.
        3. The extra EP replicate dimension does not change numerical correctness.
    """
    _assert_loss_and_grad_match_3d_root_mesh(
        case_name="tp_ep_3d_square_case",
        steps=2,
        batch_size=16,
        input_size=8,
        output_size=8,
        input_seed=42,
        label_seed=43,
        init_seed=44,
    )


def test_tp_plus_fully_shard_rectangular_3d_root_mesh_match_standalone():
    """
    Feature: TP + fully_shard end-to-end training on a 3D root mesh.
    Description:
        1. Use a rectangular weight shape on root mesh (2, 2, 2).
        2. Keep TP sharding on the output dimension and EP as a replicate dimension.
        3. Compare loss and gradient slices with standalone.
    Expectation:
        1. Distributed loss equals standalone at each step.
        2. Local gradient values match the standalone [dp, tp] slice on all EP replicas.
    """
    _assert_loss_and_grad_match_3d_root_mesh(
        case_name="tp_ep_3d_rectangular_case",
        steps=3,
        batch_size=10,
        input_size=12,
        output_size=6,
        input_seed=52,
        label_seed=53,
        init_seed=54,
    )


def test_hsdp_plus_tp_on_3d_root_mesh_match_standalone():
    """
    Feature: HSDP + TP end-to-end training on a 3D root mesh.
    Description:
        1. Build a root mesh shaped as (dp, fsdp, tp) = (2, 2, 2).
        2. Use ``(dp, fsdp)`` for fully_shard and ``tp`` for DTensor layout.
        3. Compare per-step loss and local gradients with standalone.
    Expectation:
        1. Distributed loss equals standalone on each DP-domain local batch.
        2. Each rank's local gradient equals the standalone [dp*fsdp, tp] slice.
    """
    _assert_loss_and_grad_match_hsdp_tp_root_mesh(
        case_name="hsdp_tp_3d_square_case",
        steps=2,
        batch_size=16,
        input_size=8,
        output_size=8,
        input_seed=62,
        label_seed=63,
        init_seed=64,
    )
