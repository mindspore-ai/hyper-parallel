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


class TPFullyShardMixedReplicateGroupNet(nn.Module):
    """Linear network mixing a TP-sharded weight with a TP-replicated input scale parameter."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_size, output_size).npu())
        self.scale = nn.Parameter(torch.ones(input_size).npu())

    def forward(self, x):
        return torch.matmul(x * self.scale, self.weight)


class TPFullyShardComplexReplicateParamNet(nn.Module):
    """Nonlinear block mixing one TP-sharded DTensor weight with several replicate-only params."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size).npu())
        self.in_scale = nn.Parameter(torch.randn(input_size).npu())
        self.in_bias = nn.Parameter(torch.randn(input_size).npu())
        self.gate_scale = nn.Parameter(torch.randn(input_size).npu())
        self.gate_bias = nn.Parameter(torch.randn(input_size).npu())

    def forward(self, x):
        hidden = torch.relu(x * self.in_scale + self.in_bias)
        hidden = torch.sigmoid(hidden * self.gate_scale + self.gate_bias) * hidden
        return torch.matmul(hidden, self.weight)


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
    comm_fusion: bool = False,
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
        comm_fusion=comm_fusion,
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


def _assert_same_dim_non_dim0_loss_and_grad_match(
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
    comm_fusion: bool = False,
):
    """
    Run one same-dim TP + fully_shard(dim!=0) end-to-end case and compare standalone vs distributed.

    Input:
        A standard linear weight uses TP sharding on dim 1 and fully_shard also shards dim 1.
    Expected output:
        1. Each training step produces the same loss as standalone.
        2. Each rank's local gradient matches the TP-first, then FSDP-local standalone column slice.
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

    if output_size % (dp_size * tp_size) != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by dp_size * tp_size = {dp_size * tp_size}."
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
        dp_size,
        dp_idx,
    )
    dist_losses, dist_grads = _run_tp_fully_shard_same_dim_non_dim0_training(
        dist_model,
        dp_mesh,
        tp_mesh,
        input_data,
        label_data,
        steps,
        tp_reduce_size=tp_size,
        comm_fusion=comm_fusion,
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

        expected_grad = _get_expected_same_dim_non_dim0_grad_slice(
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


def _assert_mixed_replicate_group_loss_and_grad_match_hsdp_tp_root_mesh(
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
    comm_fusion: bool = False,
):
    """Run one HSDP + TP case mixing TP-sharded and TP-replicated params in one fused group."""
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

    if input_size % fsdp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because input_size={input_size} "
            f"is not divisible by fsdp_size={fsdp_size}."
        )
        return
    if output_size % built_tp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by tp_size = {built_tp_size}."
        )
        return

    torch.manual_seed(1)
    np.random.seed(1)

    torch.manual_seed(input_seed)
    input_data = torch.randn(batch_size, input_size).npu()
    torch.manual_seed(label_seed)
    label_data = torch.randn(batch_size, output_size).npu()

    torch.manual_seed(init_seed)
    standalone_model = TPFullyShardMixedReplicateGroupNet(input_size, output_size)
    torch.manual_seed(init_seed)
    dist_model = TPFullyShardMixedReplicateGroupNet(input_size, output_size)

    dp_coord = root_mesh.get_coordinate()[:2]
    dp_idx = _flatten_coordinate((replicate_size, fsdp_size), dp_coord)
    standalone_grads, standalone_local_losses = _run_standalone_training_with_replicated_scale(
        standalone_model,
        input_data,
        label_data,
        steps,
        dp_domain_size,
        dp_idx,
    )
    dist_losses, dist_grads = _run_hsdp_tp_mixed_replicate_group_training(
        dist_model,
        hsdp_mesh,
        tp_mesh,
        input_data,
        label_data,
        steps,
        dp_domain_size=dp_domain_size,
        tp_reduce_size=built_tp_size,
        comm_fusion=comm_fusion,
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

        expected_weight_grad = _get_expected_hsdp_tp_grad_slice(
            standalone_grads[step_idx]["weight"],
            root_mesh,
            fsdp_size,
            built_tp_size,
        )
        expected_scale_grad = _get_expected_hsdp_tp_replicated_grad_slice(
            standalone_grads[step_idx]["scale"],
            root_mesh,
            fsdp_size,
        )

        assert np.allclose(
            expected_weight_grad.numpy(),
            dist_grads[step_idx]["weight"].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected weight grad slice shape "
            f"{tuple(expected_weight_grad.shape)}, got {tuple(dist_grads[step_idx]['weight'].shape)}"
        )
        assert np.allclose(
            expected_scale_grad.numpy(),
            dist_grads[step_idx]["scale"].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected scale grad slice shape "
            f"{tuple(expected_scale_grad.shape)}, got {tuple(dist_grads[step_idx]['scale'].shape)}"
        )

    print(
        f"[Rank {rank}] {case_name} passed with root_mesh={root_mesh.mesh_shape}, "
        f"hsdp_mesh={hsdp_mesh.mesh_shape}, steps={steps}, "
        f"weight_grad_shape={tuple(dist_grads[-1]['weight'].shape)}, "
        f"scale_grad_shape={tuple(dist_grads[-1]['scale'].shape)}"
    )


def _assert_replicate_param_comm_fusion_loss_and_grad_match_hsdp_tp_root_mesh(
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
    comm_fusion: bool = False,
):
    """Run one HSDP + TP case where a TP-replicated parameter is configured via replicate_params."""
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

    if input_size % fsdp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because input_size={input_size} "
            f"is not divisible by fsdp_size={fsdp_size}."
        )
        return
    if output_size % built_tp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by tp_size = {built_tp_size}."
        )
        return

    torch.manual_seed(1)
    np.random.seed(1)

    torch.manual_seed(input_seed)
    input_data = torch.randn(batch_size, input_size).npu()
    torch.manual_seed(label_seed)
    label_data = torch.randn(batch_size, output_size).npu()

    torch.manual_seed(init_seed)
    standalone_model = TPFullyShardMixedReplicateGroupNet(input_size, output_size)
    torch.manual_seed(init_seed)
    dist_model = TPFullyShardMixedReplicateGroupNet(input_size, output_size)

    dp_coord = root_mesh.get_coordinate()[:2]
    dp_idx = _flatten_coordinate((replicate_size, fsdp_size), dp_coord)
    standalone_grads, standalone_local_losses = _run_standalone_training_with_replicated_scale(
        standalone_model,
        input_data,
        label_data,
        steps,
        dp_domain_size,
        dp_idx,
    )
    dist_losses, dist_grads = _run_hsdp_tp_replicate_param_training(
        dist_model,
        hsdp_mesh,
        tp_mesh,
        input_data,
        label_data,
        steps,
        dp_domain_size=dp_domain_size,
        tp_reduce_size=built_tp_size,
        comm_fusion=comm_fusion,
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

        expected_weight_grad = _get_expected_hsdp_tp_grad_slice(
            standalone_grads[step_idx]["weight"],
            root_mesh,
            fsdp_size,
            built_tp_size,
        )
        expected_scale_grad = standalone_grads[step_idx]["scale"]

        assert np.allclose(
            expected_weight_grad.numpy(),
            dist_grads[step_idx]["weight"].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected weight grad slice shape "
            f"{tuple(expected_weight_grad.shape)}, got {tuple(dist_grads[step_idx]['weight'].shape)}"
        )
        assert np.allclose(
            expected_scale_grad.numpy(),
            dist_grads[step_idx]["scale"].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected scale grad slice shape "
            f"{tuple(expected_scale_grad.shape)}, got {tuple(dist_grads[step_idx]['scale'].shape)}"
        )

    print(
        f"[Rank {rank}] {case_name} passed with root_mesh={root_mesh.mesh_shape}, "
        f"hsdp_mesh={hsdp_mesh.mesh_shape}, steps={steps}, "
        f"weight_grad_shape={tuple(dist_grads[-1]['weight'].shape)}, "
        f"scale_grad_shape={tuple(dist_grads[-1]['scale'].shape)}"
    )


def _assert_complex_replicate_param_comm_fusion_match_hsdp_tp_root_mesh(
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
    comm_fusion: bool = False,
):
    """Run one nonlinear HSDP + TP case with multiple replicate_params under comm_fusion."""
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

    if input_size % fsdp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because input_size={input_size} "
            f"is not divisible by fsdp_size={fsdp_size}."
        )
        return
    if output_size % built_tp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by tp_size = {built_tp_size}."
        )
        return

    torch.manual_seed(1)
    np.random.seed(1)

    torch.manual_seed(input_seed)
    input_data = torch.randn(batch_size, input_size).npu()
    torch.manual_seed(label_seed)
    label_data = torch.randn(batch_size, output_size).npu()

    torch.manual_seed(init_seed)
    standalone_model = TPFullyShardComplexReplicateParamNet(input_size, output_size)
    torch.manual_seed(init_seed)
    dist_model = TPFullyShardComplexReplicateParamNet(input_size, output_size)

    dp_coord = root_mesh.get_coordinate()[:2]
    dp_idx = _flatten_coordinate((replicate_size, fsdp_size), dp_coord)
    standalone_grads, standalone_local_losses = _run_standalone_training_with_named_grads(
        standalone_model,
        x=input_data,
        y=label_data,
        steps=steps,
        dp_size=dp_domain_size,
        dp_idx=dp_idx,
        grad_names=("weight", "in_scale", "in_bias", "gate_scale", "gate_bias"),
    )
    dist_losses, dist_grads = _run_hsdp_tp_complex_replicate_param_training(
        dist_model,
        hsdp_mesh,
        tp_mesh,
        input_data,
        label_data,
        steps,
        dp_domain_size=dp_domain_size,
        tp_reduce_size=built_tp_size,
        comm_fusion=comm_fusion,
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

        expected_weight_grad = _get_expected_hsdp_tp_grad_slice(
            standalone_grads[step_idx]["weight"],
            root_mesh,
            fsdp_size,
            built_tp_size,
        )
        assert np.allclose(
            expected_weight_grad.numpy(),
            dist_grads[step_idx]["weight"].numpy(),
            rtol=1e-3,
            atol=1e-3,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected weight grad slice shape "
            f"{tuple(expected_weight_grad.shape)}, got {tuple(dist_grads[step_idx]['weight'].shape)}"
        )

        for grad_name in ("in_scale", "in_bias", "gate_scale", "gate_bias"):
            expected_grad = standalone_grads[step_idx][grad_name]
            assert np.allclose(
                expected_grad.numpy(),
                dist_grads[step_idx][grad_name].numpy(),
                rtol=1e-3,
                atol=1e-3,
            ), (
                f"{case_name}, rank {rank}, step {step_idx}: expected {grad_name} grad shape "
                f"{tuple(expected_grad.shape)}, got {tuple(dist_grads[step_idx][grad_name].shape)}"
            )

    print(
        f"[Rank {rank}] {case_name} passed with root_mesh={root_mesh.mesh_shape}, "
        f"hsdp_mesh={hsdp_mesh.mesh_shape}, steps={steps}, "
        f"weight_grad_shape={tuple(dist_grads[-1]['weight'].shape)}, "
        f"in_scale_grad_shape={tuple(dist_grads[-1]['in_scale'].shape)}, "
        f"gate_bias_grad_shape={tuple(dist_grads[-1]['gate_bias'].shape)}"
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
        with SkipDTensorDispatch():
            optimizer.step()
            optimizer.zero_grad()
    return grads, local_losses


def _run_standalone_training_with_replicated_scale(model, x, y, steps, dp_size, dp_idx):
    """Run standalone training for the mixed TP-sharded/TP-replicated parameter network."""
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
        grads.append(
            {
                "weight": model.weight.grad.data.cpu().clone(),
                "scale": model.scale.grad.data.cpu().clone(),
            }
        )
        with SkipDTensorDispatch():
            optimizer.step()
            optimizer.zero_grad()
    return grads, local_losses


def _run_standalone_training_with_named_grads(model, x, y, steps, dp_size, dp_idx, grad_names):
    """Run standalone training and capture a configurable set of parameter grads each step."""
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
        grads.append(
            {
                grad_name: getattr(model, grad_name).grad.data.cpu().clone()
                for grad_name in grad_names
            }
        )
        with SkipDTensorDispatch():
            optimizer.step()
            optimizer.zero_grad()
    return grads, local_losses


def _run_tp_fully_shard_training(
    model,
    dp_mesh,
    tp_mesh,
    x,
    y,
    steps,
    tp_reduce_size,
    comm_fusion: bool = False,
):
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
        comm_fusion=comm_fusion,
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


def _run_hsdp_tp_mixed_replicate_group_training(
    model,
    hsdp_mesh,
    tp_mesh,
    x,
    y,
    steps,
    dp_domain_size,
    tp_reduce_size,
    comm_fusion: bool = False,
):
    """Run HSDP + TP training with one TP-sharded weight and one TP-replicated scale parameter."""
    x_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    w_placements = (Shard(1),) + tuple(Replicate() for _ in range(tp_mesh.ndim - 1))
    scale_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))

    sharding_plan = ShardingPlan(
        plan={"weight": w_placements, "scale": scale_placements},
        input_plan={"input": x_placements},
        output_plan={"output": w_placements},
    )
    model = shard_module(model, device_mesh=tp_mesh, sharding_plan=sharding_plan)
    model = fully_shard(
        model,
        mesh=hsdp_mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
        comm_fusion=comm_fusion,
    )
    model.set_reduce_op_type("sum")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dp_idx = _flatten_coordinate(hsdp_mesh.mesh_shape, hsdp_mesh.get_coordinate())
    local_x, local_y = _get_local_batch_slice(x, y, dp_domain_size, dp_idx)
    dist_x = DTensor.from_local(local_x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(local_y, tp_mesh, x_placements)
    losses = []
    grads = []
    for _ in range(steps):
        y_pred = model(dist_x)
        y_shard = dist_y.redistribute(tp_mesh, y_pred.placements)
        loss = mse_loss_sum(y_pred, y_shard)
        loss = loss.reduce_partial()
        loss.backward(torch.tensor(1.0 / tp_reduce_size, device=local_x.device))

        losses.append(loss.to_local().detach().cpu())
        weight_grad = model.weight.grad
        scale_grad = model.scale.grad
        if isinstance(weight_grad, DTensor):
            weight_grad = weight_grad.to_local()
        if isinstance(scale_grad, DTensor):
            scale_grad = scale_grad.to_local()
        grads.append(
            {
                "weight": weight_grad.detach().cpu().clone(),
                "scale": scale_grad.detach().cpu().clone(),
            }
        )

        with SkipDTensorDispatch():
            optimizer.step()
            optimizer.zero_grad()
    return losses, grads


def _run_hsdp_tp_replicate_param_training(
    model,
    hsdp_mesh,
    tp_mesh,
    x,
    y,
    steps,
    dp_domain_size,
    tp_reduce_size,
    comm_fusion: bool = False,
):
    """Run HSDP + TP training where the TP-replicated scale parameter uses replicate_params."""
    x_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    w_placements = (Shard(1),) + tuple(Replicate() for _ in range(tp_mesh.ndim - 1))
    scale_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))

    sharding_plan = ShardingPlan(
        plan={"weight": w_placements, "scale": scale_placements},
        input_plan={"input": x_placements},
        output_plan={"output": w_placements},
    )
    model = shard_module(model, device_mesh=tp_mesh, sharding_plan=sharding_plan)
    model = fully_shard(
        model,
        mesh=hsdp_mesh,
        reshard_after_forward=True,
        replicate_params=(model.scale,),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
        comm_fusion=comm_fusion,
    )
    model.set_reduce_op_type("sum")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dp_idx = _flatten_coordinate(hsdp_mesh.mesh_shape, hsdp_mesh.get_coordinate())
    local_x, local_y = _get_local_batch_slice(x, y, dp_domain_size, dp_idx)
    dist_x = DTensor.from_local(local_x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(local_y, tp_mesh, x_placements)
    losses = []
    grads = []
    for _ in range(steps):
        y_pred = model(dist_x)
        y_shard = dist_y.redistribute(tp_mesh, y_pred.placements)
        loss = mse_loss_sum(y_pred, y_shard)
        loss = loss.reduce_partial()
        loss.backward(torch.tensor(1.0 / tp_reduce_size, device=local_x.device))

        losses.append(loss.to_local().detach().cpu())
        weight_grad = model.weight.grad
        scale_grad = model.scale.grad
        if isinstance(weight_grad, DTensor):
            weight_grad = weight_grad.to_local()
        if isinstance(scale_grad, DTensor):
            scale_grad = scale_grad.to_local()
        grads.append(
            {
                "weight": weight_grad.detach().cpu().clone(),
                "scale": scale_grad.detach().cpu().clone(),
            }
        )

        with SkipDTensorDispatch():
            optimizer.step()
            optimizer.zero_grad()
    return losses, grads


def _run_hsdp_tp_complex_replicate_param_training(
    model,
    hsdp_mesh,
    tp_mesh,
    x,
    y,
    steps,
    dp_domain_size,
    tp_reduce_size,
    comm_fusion: bool = False,
):
    """Run nonlinear HSDP + TP training where several TP-replicated params use replicate_params."""
    x_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    w_placements = (Shard(1),) + tuple(Replicate() for _ in range(tp_mesh.ndim - 1))
    replicate_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))

    sharding_plan = ShardingPlan(
        plan={
            "weight": w_placements,
            "in_scale": replicate_placements,
            "in_bias": replicate_placements,
            "gate_scale": replicate_placements,
            "gate_bias": replicate_placements,
        },
        input_plan={"input": x_placements},
        output_plan={"output": w_placements},
    )
    model = shard_module(model, device_mesh=tp_mesh, sharding_plan=sharding_plan)

    model = fully_shard(
        model,
        mesh=hsdp_mesh,
        reshard_after_forward=True,
        replicate_params=(
            model.in_scale,
            model.in_bias,
            model.gate_scale,
            model.gate_bias,
        ),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
        comm_fusion=comm_fusion,
    )
    model.set_reduce_op_type("sum")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dp_idx = _flatten_coordinate(hsdp_mesh.mesh_shape, hsdp_mesh.get_coordinate())
    local_x, local_y = _get_local_batch_slice(x, y, dp_domain_size, dp_idx)
    dist_x = DTensor.from_local(local_x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(local_y, tp_mesh, x_placements)
    losses = []
    grads = []
    for _ in range(steps):
        y_pred = model(dist_x)
        y_shard = dist_y.redistribute(tp_mesh, y_pred.placements)
        loss = mse_loss_sum(y_pred, y_shard)
        loss = loss.reduce_partial()
        loss.backward(torch.tensor(1.0 / tp_reduce_size, device=local_x.device))

        losses.append(loss.to_local().detach().cpu())
        grad_names = ("weight", "in_scale", "in_bias", "gate_scale", "gate_bias")
        grad_record = {}
        for grad_name in grad_names:
            grad = getattr(model, grad_name).grad
            if isinstance(grad, DTensor):
                grad = grad.to_local()
            grad_record[grad_name] = grad.detach().cpu().clone()
        grads.append(grad_record)

        with SkipDTensorDispatch():
            optimizer.step()
            optimizer.zero_grad()
    return losses, grads


def _run_tp_fully_shard_same_dim_non_dim0_training(
    model,
    dp_mesh,
    tp_mesh,
    x,
    y,
    steps,
    tp_reduce_size,
    comm_fusion: bool = False,
):
    """
    Run same-dim TP + fully_shard training where both shard the weight on dim 1.

    Input:
        TP shards the weight on dim 1, and fully_shard shards the same dim on ``dp_mesh``.
    Expected output:
        Per-step losses and local gradients match the TP-first, then FSDP-local column slices.
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
        shard_placement_fn=lambda param: Shard(1),  # pylint: disable=unused-argument
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
        comm_fusion=comm_fusion,
    )
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
        loss.backward(torch.tensor(1.0 / tp_reduce_size, device=local_x.device))

        losses.append(loss.to_local().detach().cpu())
        grad = model.weight.grad
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        grads.append(grad.detach().cpu().clone())

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


def _get_expected_hsdp_tp_replicated_grad_slice(standalone_grad, mesh, fsdp_size):
    """Slice a TP-replicated parameter's standalone grad for HSDP(2D) + TP(1D)."""
    coordinate = mesh.get_coordinate()
    fsdp_idx = coordinate[1]
    local_numel = standalone_grad.shape[0] // fsdp_size
    start = fsdp_idx * local_numel
    return standalone_grad[start: start + local_numel]


def _get_expected_same_dim_non_dim0_grad_slice(
    standalone_grad,
    mesh,
    dp_size,
    tp_size,
    dp_mesh_ndim=1,
    tp_mesh_dim=1,
):
    """
    Slice the standalone gradient for TP-first, then fully_shard dim-1 sharding.

    Input:
        Full standalone gradient with shape [input_size, output_size].
    Expected output:
        Local gradient slice with shape [input_size, output_size / (dp_size * tp_size)].
    """
    coordinate = mesh.get_coordinate()
    dp_idx = _flatten_coordinate(mesh.mesh_shape[:dp_mesh_ndim], coordinate[:dp_mesh_ndim])
    tp_idx = coordinate[tp_mesh_dim]
    tp_chunk = standalone_grad.shape[1] // tp_size
    local_cols = standalone_grad.shape[1] // (dp_size * tp_size)
    col_start = tp_idx * tp_chunk + dp_idx * local_cols
    return standalone_grad[:, col_start: col_start + local_cols]


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


def test_tp_plus_fully_shard_same_dim_non_dim0_comm_fusion_loss_and_grad_match_standalone():
    """
    Feature: TP + fully_shard + comm_fusion end-to-end training with same-dim sharding on weight dim 1.
    Description:
        1. Build a standard linear weight whose dim 1 is sharded by both TP and fully_shard.
        2. Enable ``comm_fusion`` on the distributed path.
        3. Compare per-step local loss and local sharded gradients with standalone.
    Expectation:
        1. Distributed loss equals standalone loss at each step.
        2. Each rank's local gradient equals the TP-first, then FSDP-local standalone column slice.
    """
    _assert_same_dim_non_dim0_loss_and_grad_match(
        case_name="tp2_same_dim_non_dim0_comm_fusion_case",
        steps=2,
        batch_size=12,
        input_size=8,
        output_size=8,
        input_seed=232,
        label_seed=233,
        init_seed=234,
        tp_size=2,
        comm_fusion=True,
    )

def test_hsdp_plus_tp_comm_fusion_mixed_replicate_groups_match_standalone():
    """
    Feature: HSDP + TP + comm_fusion end-to-end training with mixed replicate groups.
    Description:
        1. Build a root mesh shaped as (dp, fsdp, tp) = (2, 2, 2).
        2. Keep ``weight`` TP-sharded while ``scale`` stays TP-replicated in the same fully_shard unit.
        3. Enable ``comm_fusion`` and compare loss plus both local grads with standalone.
    Expectation:
        1. Distributed loss equals standalone on each DP-domain local batch.
        2. ``weight.grad`` matches the standalone [fsdp, tp] slice.
        3. ``scale.grad`` matches the standalone fsdp-only slice and is independent of TP rank.
    """
    _assert_mixed_replicate_group_loss_and_grad_match_hsdp_tp_root_mesh(
        case_name="hsdp_tp_3d_comm_fusion_mixed_replicate_groups_case",
        steps=2,
        batch_size=16,
        input_size=8,
        output_size=8,
        input_seed=162,
        label_seed=163,
        init_seed=164,
        comm_fusion=True,
    )


def test_hsdp_plus_tp_comm_fusion_replicate_params_match_standalone():
    """
    Feature: HSDP + TP + comm_fusion end-to-end training with replicate_params.
    Description:
        1. Build a root mesh shaped as (dp, fsdp, tp) = (2, 2, 2).
        2. Keep ``weight`` in hsdp_params while ``scale`` is marked as ``replicate_params``.
        3. Enable ``comm_fusion`` and compare loss plus both local grads with standalone.
    Expectation:
        1. Distributed loss equals standalone on each DP-domain local batch.
        2. ``weight.grad`` still matches the standalone [fsdp, tp] slice.
        3. ``scale.grad`` stays full-sized and matches the standalone full gradient even though it bypasses fused RS.
    """
    _assert_replicate_param_comm_fusion_loss_and_grad_match_hsdp_tp_root_mesh(
        case_name="hsdp_tp_3d_comm_fusion_replicate_params_case",
        steps=2,
        batch_size=16,
        input_size=8,
        output_size=8,
        input_seed=165,
        label_seed=166,
        init_seed=167,
        comm_fusion=True,
    )


def test_hsdp_plus_tp_comm_fusion_complex_replicate_params_match_standalone():
    """
    Feature: HSDP + TP + comm_fusion end-to-end training with multiple replicate_params.
    Description:
        1. Build a nonlinear block with one TP-sharded DTensor weight.
        2. Keep four TP-replicated params in ``replicate_params`` around the matmul path.
        3. Enable ``comm_fusion`` and compare loss plus all tracked grads with standalone.
    Expectation:
        1. Distributed loss matches standalone on each DP-domain local batch.
        2. ``weight.grad`` matches the standalone [fsdp, tp] slice.
        3. All replicate-only grads match the standalone full gradients.
    """
    _assert_complex_replicate_param_comm_fusion_match_hsdp_tp_root_mesh(
        case_name="hsdp_tp_3d_comm_fusion_complex_replicate_params_case",
        steps=2,
        batch_size=16,
        input_size=8,
        output_size=8,
        input_seed=168,
        label_seed=169,
        init_seed=170,
        comm_fusion=True,
    )

# Keep the e2e surface intentionally small here.
# The wrapper entrypoints under ``test_tp_fully_shard_e2e.py`` only exercise the
# representative networks below; shape variants and stress permutations can be
# reintroduced later if a specific regression needs dedicated coverage.
