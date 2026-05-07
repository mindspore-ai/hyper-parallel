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
"""End-to-end TP + fully_shard verification for MindSpore."""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms
import mindspore.communication.management as D
import numpy as np
from mindspore import Tensor, nn

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh, init_empty_weights, shard_module
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.hsdp_utils import FullyShardParamMode
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

ms.set_seed(1)
ms.set_deterministic(True)
enable_mindspore_backward_compat()


class TPFullyShardNet(nn.Cell):
    """Small gated network used for standalone vs distributed comparison."""

    def __init__(self, state_dict_np: dict[str, np.ndarray]):
        super().__init__()
        self.proj_weight = ms.Parameter(Tensor(state_dict_np["proj_weight"], ms.float32), name="proj_weight")
        self.gate_weight = ms.Parameter(Tensor(state_dict_np["gate_weight"], ms.float32), name="gate_weight")
        self.residual_weight = ms.Parameter(
            Tensor(state_dict_np["residual_weight"], ms.float32),
            name="residual_weight",
        )
        self.relu = ms.mint.nn.ReLU()
        self.sigmoid = ms.mint.nn.Sigmoid()

    def construct(self, x):
        proj = ms.mint.matmul(x, self.proj_weight)
        gate = ms.mint.matmul(x, self.gate_weight)
        residual = ms.mint.matmul(x, self.residual_weight)
        hidden = self.relu(proj)
        gated = ms.mint.mul(hidden, self.sigmoid(gate))
        return ms.mint.add(gated, residual)


def mse_loss_sum(y_pred, y_true):
    """Return summed MSE loss for both standalone and distributed paths."""
    error = ms.mint.sub(y_pred, y_true)
    square_error = ms.mint.square(error)
    return ms.mint.sum(square_error)


def _build_mesh(world_size: int, tp_size: int = 2):
    """Build a 2D (dp, tp) mesh for TP + fully_shard."""
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
    """Build a 3D (dp, tp, ep) root mesh and expose the DP and TPxEP sub-meshes."""
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
    """Build a 3D (replicate, fsdp, tp) root mesh for HSDP + TP."""
    expected_world_size = replicate_size * fsdp_size * tp_size
    if world_size != expected_world_size:
        return None, None, None, None, None
    root_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(replicate_size, fsdp_size, tp_size),
        mesh_dim_names=("dp", "fsdp", "tp"),
    )
    return root_mesh, root_mesh[("dp", "fsdp")], root_mesh["tp"], replicate_size * fsdp_size, tp_size


def _build_tp_only_mesh(world_size: int, tp_size: int = 2):
    """Build a 1D TP mesh for fully_shard compatibility mode."""
    if world_size != tp_size:
        return None
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(tp_size,),
        mesh_dim_names=("tp",),
    )


def _flatten_coordinate(mesh_shape, coordinate):
    """Flatten an n-D mesh coordinate into a row-major linear index."""
    flat_idx = 0
    for dim_size, dim_idx in zip(mesh_shape, coordinate):
        flat_idx = flat_idx * dim_size + dim_idx
    return flat_idx


def _get_local_batch_slice(x: Tensor, y: Tensor, dp_size: int, dp_idx: int):
    """Return the DP-local input/label slices for one rank."""
    local_batch = x.shape[0] // dp_size
    start = dp_idx * local_batch
    end = start + local_batch
    return x[start:end], y[start:end]


def _get_expected_grad_slice(
    standalone_grad: np.ndarray,
    mesh,
    dp_size: int,
    tp_size: int,
    dp_mesh_ndim: int = 1,
    tp_mesh_dim: int = 1,
) -> np.ndarray:
    """Slice standalone gradient into the [dp, tp] shard expected on the current rank."""
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


def _get_expected_hsdp_tp_grad_slice(
    standalone_grad: np.ndarray,
    root_mesh,
    fsdp_size: int,
    tp_size: int,
) -> np.ndarray:
    """Slice standalone gradient into the local shard expected for HSDP + TP."""
    coordinate = root_mesh.get_coordinate()
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


def _get_expected_tp_grad_slice(standalone_grad: np.ndarray, tp_mesh, tp_size: int) -> np.ndarray:
    """Slice standalone gradient into the TP shard expected on the current rank."""
    tp_idx = tp_mesh.get_coordinate()[0]
    local_output = standalone_grad.shape[1] // tp_size
    col_start = tp_idx * local_output
    return standalone_grad[:, col_start: col_start + local_output]


def _to_numpy_loss(loss):
    """Convert local/DTensor loss to numpy scalar for comparison."""
    if isinstance(loss, DTensor):
        return loss.to_local().asnumpy()
    return loss.asnumpy()


def _build_tp_sharding_plan(tp_mesh):
    """Build the TP sharding plan shared by explicit-mesh and mesh=None paths."""
    x_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    w_placements = (Shard(1),) + tuple(Replicate() for _ in range(tp_mesh.ndim - 1))
    sharding_plan = ShardingPlan(
        plan={
            "proj_weight": w_placements,
            "gate_weight": w_placements,
            "residual_weight": w_placements,
        },
        input_plan={"input": x_placements},
        output_plan={"output": w_placements},
    )
    return x_placements, sharding_plan


def _build_tp_replicate_sharding_plan(tp_mesh):
    """Build a pure-TP plan where inputs and weights are replicated on the TP mesh."""
    x_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    w_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    sharding_plan = ShardingPlan(
        plan={
            "proj_weight": w_placements,
            "gate_weight": w_placements,
            "residual_weight": w_placements,
        },
        input_plan={"input": x_placements},
        output_plan={"output": w_placements},
    )
    return x_placements, sharding_plan


def _build_fp32_mp_policy():
    """Build the float32 mixed-precision policy shared by TP + fully_shard tests."""
    return MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )


def _wrap_tp_fully_shard_model(model, tp_mesh, *, mesh, shard_placement_fn=None):
    """Apply TP sharding then fully_shard to the test model."""
    x_placements, sharding_plan = _build_tp_sharding_plan(tp_mesh)
    model = shard_module(model, device_mesh=tp_mesh, sharding_plan=sharding_plan)
    model = fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        shard_placement_fn=shard_placement_fn,
        mp_policy=_build_fp32_mp_policy(),
    )
    model.set_reduce_op_type("sum")
    return model, x_placements


def _wrap_pure_tp_replicate_fully_shard_model(model, tp_mesh):
    """Apply replicated TP sharding then fully_shard(mesh=None)."""
    x_placements, sharding_plan = _build_tp_replicate_sharding_plan(tp_mesh)
    model = shard_module(model, device_mesh=tp_mesh, sharding_plan=sharding_plan)
    model = fully_shard(
        model,
        mesh=None,
        reshard_after_forward=True,
        mp_policy=_build_fp32_mp_policy(),
    )
    model.set_reduce_op_type("sum")
    return model, x_placements


def _build_empty_init_tp_fully_shard_model(
    state_dict_np: dict[str, np.ndarray],
    tp_mesh,
    *,
    mesh,
):
    """Create a TP + fully_shard model through init_empty_weights and then load the full state."""
    with init_empty_weights():
        model = TPFullyShardNet(state_dict_np)
    with ms.DeviceCtx("meta"):
        model, x_placements = _wrap_tp_fully_shard_model(model, tp_mesh, mesh=mesh)
    model.load_state_dict(
        {name: Tensor(weight, ms.float32) for name, weight in state_dict_np.items()},
        strict=True,
    )
    return model, x_placements


def _zero_param_grads(params):
    """Reset parameter gradients for the next training step."""
    for param in params:
        param.grad = None


def _collect_local_grads(params):
    """Collect parameter gradients as local numpy arrays."""
    grads = []
    for param in params:
        grad = param.grad
        if isinstance(grad, DTensor):
            grads.append(grad.to_local())
        else:
            grads.append(grad)
    return grads


def _build_model_state_dict(input_size: int, output_size: int, init_seed: int) -> dict[str, np.ndarray]:
    """Build a deterministic full-precision state dict for the TP test network."""
    weight_rng = np.random.default_rng(init_seed)
    scale = 0.1
    return {
        "proj_weight": (weight_rng.standard_normal((input_size, output_size)) * scale).astype(np.float32),
        "gate_weight": (weight_rng.standard_normal((input_size, output_size)) * scale).astype(np.float32),
        "residual_weight": (weight_rng.standard_normal((input_size, output_size)) * scale).astype(np.float32),
    }


def _run_standalone_training(model, x, y, steps, dp_size, dp_idx):
    """Run standalone training and record per-step local-batch losses + full gradients."""
    params = tuple(model.trainable_params())
    optimizer = nn.Adam(params, learning_rate=0.01)
    enable_mindspore_backward_compat()
    local_losses = []
    grads = []
    for _ in range(steps):
        _zero_param_grads(params)
        loss = mse_loss_sum(model(x), y)
        loss.backward()
        local_x, local_y = _get_local_batch_slice(x, y, dp_size, dp_idx)
        local_loss = mse_loss_sum(model(local_x), local_y)
        grad_list = _collect_local_grads(params)
        optimizer(tuple(grad_list))
        local_losses.append(local_loss.asnumpy())
        grads.append(tuple(grad.asnumpy().copy() for grad in grad_list))
    return grads, local_losses


def _run_accumulated_tp_fully_shard_step(model, local_x, local_y, tp_mesh, x_placements, micro_step: int):
    """Run one TP + fully_shard step through multiple micro-batches before the optimizer update."""
    micro_batch = local_x.shape[0] // micro_step
    assert micro_batch > 0, f"micro_batch must be positive, got local_x.shape[0]={local_x.shape[0]}"
    total_loss = None
    for micro_idx in range(micro_step):
        start = micro_idx * micro_batch
        end = start + micro_batch
        x_micro = DTensor.from_local(local_x[start:end], tp_mesh, x_placements)
        y_micro = DTensor.from_local(local_y[start:end], tp_mesh, x_placements)
        y_pred = model(x_micro)
        y_shard = y_micro.redistribute(tp_mesh, y_pred.placements)
        loss = mse_loss_sum(y_pred, y_shard)
        if isinstance(loss, DTensor):
            loss = loss.reduce_partial()
        loss.backward(Tensor(1.0 / tp_mesh.size(), ms.float32))
        total_loss = loss if total_loss is None else total_loss + loss
    return total_loss


def _run_tp_fully_shard_training(
    model,
    dp_mesh,
    tp_mesh,
    x,
    y,
    steps,
    *,
    use_empty_weight: bool = False,
    accumulate_grad: bool = False,
    micro_step: int = 4,
):
    """Run TP + fully_shard training and record per-step loss + local gradients."""
    state_dict_np = {
        name: param.asnumpy().copy() for name, param in model.parameters_dict().items()
    }
    if use_empty_weight:
        model, x_placements = _build_empty_init_tp_fully_shard_model(
            state_dict_np,
            tp_mesh,
            mesh=dp_mesh,
        )
    else:
        model, x_placements = _wrap_tp_fully_shard_model(model, tp_mesh, mesh=dp_mesh)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)
    dp_idx = _flatten_coordinate(dp_mesh.mesh_shape, dp_mesh.get_coordinate())
    local_x, local_y = _get_local_batch_slice(x, y, dp_mesh.size(), dp_idx)
    dist_x = DTensor.from_local(local_x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(local_y, tp_mesh, x_placements)

    def forward_fn(data, label):
        y_pred = model(data)
        y_shard = label.redistribute(tp_mesh, y_pred.placements)
        loss = mse_loss_sum(y_pred, y_shard)
        if isinstance(loss, DTensor):
            loss = loss.reduce_partial()
        return loss

    losses = []
    grads = []
    for _ in range(steps):
        model.zero_grad()
        if accumulate_grad:
            assert local_x.shape[0] % micro_step == 0, (
                f"Local batch size {local_x.shape[0]} is not divisible by micro_step={micro_step}"
            )
            loss_value = _run_accumulated_tp_fully_shard_step(
                model,
                local_x,
                local_y,
                tp_mesh,
                x_placements,
                micro_step,
            )
        else:
            loss_value = forward_fn(dist_x, dist_y)
            loss_value.backward(Tensor(1.0 / tp_mesh.size(), ms.float32))
        grad_list = tuple(param.grad for param in optimizer.parameters)
        losses.append(_to_numpy_loss(loss_value))
        grads.append(tuple(grad.asnumpy().copy() for grad in grad_list))
        with SkipDTensorDispatch():
            optimizer(grad_list)
    return losses, grads


def _run_tp_fully_shard_compat_training(model, tp_mesh, x, y, steps):
    """Run TP + fully_shard(mesh=None) compatibility training with DTensor-backed weights."""
    model, x_placements = _wrap_tp_fully_shard_model(model, tp_mesh, mesh=None)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)
    dist_x = DTensor.from_local(x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(y, tp_mesh, x_placements)

    def forward_fn(data, label):
        y_pred = model(data)
        y_shard = label.redistribute(tp_mesh, y_pred.placements)
        loss = mse_loss_sum(y_pred, y_shard)
        if isinstance(loss, DTensor):
            loss = loss.reduce_partial()
        return loss

    losses = []
    grads = []
    for _ in range(steps):
        model.zero_grad()
        loss_value = forward_fn(dist_x, dist_y)
        loss_value.backward(Tensor(1.0 / tp_mesh.size(), ms.float32))
        grad_list = tuple(param.grad for param in optimizer.parameters)
        losses.append(_to_numpy_loss(loss_value))
        grads.append(tuple(grad.asnumpy().copy() for grad in grad_list))
        with SkipDTensorDispatch():
            optimizer(grad_list)
    return losses, grads


def _run_pure_tp_fully_shard_training(model, tp_mesh, x, y, steps):
    """Run pure-TP training where fully_shard(mesh=None) owns replicated DTensor grads."""
    model, x_placements = _wrap_pure_tp_replicate_fully_shard_model(model, tp_mesh)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)
    dist_x = DTensor.from_local(x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(y, tp_mesh, x_placements)

    losses = []
    grads = []
    for _ in range(steps):
        model.zero_grad()
        loss = mse_loss_sum(model(dist_x), dist_y)
        loss.backward(Tensor(1.0 / tp_mesh.size(), ms.float32))
        grad_list = tuple(param.grad for param in optimizer.parameters)
        losses.append(_to_numpy_loss(loss))
        grads.append(tuple(grad.asnumpy().copy() for grad in grad_list))
        with SkipDTensorDispatch():
            optimizer(grad_list)
    return model, losses, grads


def _run_tp_fully_shard_same_dim_non_dim0_training(
    model,
    dp_mesh,
    tp_mesh,
    x,
    y,
    steps,
):
    """Run same-dim TP + fully_shard training where both shard the weight on dim 1."""
    model, x_placements = _wrap_tp_fully_shard_model(
        model,
        tp_mesh,
        mesh=dp_mesh,
        shard_placement_fn=lambda param: Shard(1),  # pylint: disable=unused-argument
    )
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)
    dp_idx = _flatten_coordinate(dp_mesh.mesh_shape, dp_mesh.get_coordinate())
    local_x, local_y = _get_local_batch_slice(x, y, dp_mesh.size(), dp_idx)
    dist_x = DTensor.from_local(local_x, tp_mesh, x_placements)
    dist_y = DTensor.from_local(local_y, tp_mesh, x_placements)

    losses = []
    grads = []
    for _ in range(steps):
        model.zero_grad()
        y_pred = model(dist_x)
        y_shard = dist_y.redistribute(tp_mesh, y_pred.placements)
        loss = mse_loss_sum(y_pred, y_shard)
        if isinstance(loss, DTensor):
            loss = loss.reduce_partial()
        loss.backward(Tensor(1.0 / tp_mesh.size(), ms.float32))
        grad_list = tuple(param.grad for param in optimizer.parameters)
        losses.append(_to_numpy_loss(loss))
        grads.append(tuple(grad.asnumpy().copy() for grad in grad_list))
        with SkipDTensorDispatch():
            optimizer(grad_list)
    return losses, grads


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
    tp_size: int = 2,
    use_empty_weight: bool = False,
    accumulate_grad: bool = False,
    micro_step: int = 4,
):
    """Run one TP + fully_shard case and compare distributed vs standalone results."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()
    root_mesh, dp_mesh, tp_mesh, dp_size = _build_mesh(world_size, tp_size=tp_size)
    if root_mesh is None or dp_size <= 1:
        print(
            f"[Rank {rank}] Skip {case_name} because world_size={world_size} "
            f"cannot form TP + fully_shard mesh with tp_size={tp_size}."
        )
        return

    if batch_size % dp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because batch_size={batch_size} "
            f"is not divisible by dp_size={dp_size}."
        )
        return
    if output_size % tp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by tp_size={tp_size}."
        )
        return
    if input_size % dp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because input_size={input_size} "
            f"is not divisible by dp_size={dp_size}."
        )
        return

    input_rng = np.random.default_rng(input_seed)
    label_rng = np.random.default_rng(label_seed)
    input_data = Tensor(input_rng.standard_normal((batch_size, input_size)).astype(np.float32))
    label_data = Tensor(label_rng.standard_normal((batch_size, output_size)).astype(np.float32))
    state_dict_np = _build_model_state_dict(input_size, output_size, init_seed)

    standalone_model = TPFullyShardNet(state_dict_np)
    dist_model = TPFullyShardNet(state_dict_np)

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
        use_empty_weight=use_empty_weight,
        accumulate_grad=accumulate_grad,
        micro_step=micro_step,
    )

    for step_idx in range(steps):
        assert np.allclose(
            standalone_local_losses[step_idx],
            dist_losses[step_idx],
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected local loss "
            f"{standalone_local_losses[step_idx]}, got {dist_losses[step_idx]}"
        )

        for grad_idx, standalone_grad in enumerate(standalone_grads[step_idx]):
            expected_grad = _get_expected_grad_slice(
                standalone_grad,
                root_mesh,
                dp_size,
                tp_size,
            )
            dist_grad = dist_grads[step_idx][grad_idx]
            assert np.allclose(
                expected_grad,
                dist_grad,
                rtol=1e-5,
                atol=1e-5,
            ), (
                f"{case_name}, rank {rank}, step {step_idx}, grad {grad_idx}: expected grad slice shape "
                f"{tuple(expected_grad.shape)}, got {tuple(dist_grad.shape)}"
            )

    print(
        f"[Rank {rank}] {case_name} passed with mesh={root_mesh.mesh_shape}, "
        f"steps={steps}, local_grad_shape={tuple(dist_grads[-1][0].shape)}"
    )


def _get_expected_same_dim_non_dim0_grad_slice(
    standalone_grad: np.ndarray,
    mesh,
    dp_size: int,
    tp_size: int,
    dp_mesh_ndim: int = 1,
    tp_mesh_dim: int = 1,
) -> np.ndarray:
    """Slice standalone gradient for TP-first, then fully_shard dim-1 sharding."""
    coordinate = mesh.get_coordinate()
    dp_idx = _flatten_coordinate(mesh.mesh_shape[:dp_mesh_ndim], coordinate[:dp_mesh_ndim])
    tp_idx = coordinate[tp_mesh_dim]
    tp_chunk = standalone_grad.shape[1] // tp_size
    local_cols = standalone_grad.shape[1] // (dp_size * tp_size)
    col_start = tp_idx * tp_chunk + dp_idx * local_cols
    return standalone_grad[:, col_start: col_start + local_cols]


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
    tp_size: int = 2,
):
    """Run one same-dim TP + fully_shard(dim!=0) case and compare standalone vs distributed."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()
    root_mesh, dp_mesh, tp_mesh, dp_size = _build_mesh(world_size, tp_size=tp_size)
    if root_mesh is None or dp_size <= 1:
        print(
            f"[Rank {rank}] Skip {case_name} because world_size={world_size} "
            f"cannot form TP + fully_shard mesh with tp_size={tp_size}."
        )
        return

    if output_size % (dp_size * tp_size) != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by dp_size * tp_size = {dp_size * tp_size}."
        )
        return

    input_rng = np.random.default_rng(input_seed)
    label_rng = np.random.default_rng(label_seed)
    input_data = Tensor(input_rng.standard_normal((batch_size, input_size)).astype(np.float32))
    label_data = Tensor(label_rng.standard_normal((batch_size, output_size)).astype(np.float32))
    state_dict_np = _build_model_state_dict(input_size, output_size, init_seed)

    standalone_model = TPFullyShardNet(state_dict_np)
    dist_model = TPFullyShardNet(state_dict_np)

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
    )

    for step_idx in range(steps):
        assert np.allclose(
            standalone_local_losses[step_idx],
            dist_losses[step_idx],
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected local loss "
            f"{standalone_local_losses[step_idx]}, got {dist_losses[step_idx]}"
        )

        for grad_idx, standalone_grad in enumerate(standalone_grads[step_idx]):
            expected_grad = _get_expected_same_dim_non_dim0_grad_slice(
                standalone_grad,
                root_mesh,
                dp_size,
                tp_size,
            )
            dist_grad = dist_grads[step_idx][grad_idx]
            assert np.allclose(
                expected_grad,
                dist_grad,
                rtol=1e-5,
                atol=1e-5,
            ), (
                f"{case_name}, rank {rank}, step {step_idx}, grad {grad_idx}: expected grad slice shape "
                f"{tuple(expected_grad.shape)}, got {tuple(dist_grad.shape)}"
            )

    print(
        f"[Rank {rank}] {case_name} passed with mesh={root_mesh.mesh_shape}, "
        f"steps={steps}, local_grad_shape={tuple(dist_grads[-1][0].shape)}"
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
    """Run one TP(+EP replicate) + fully_shard case on a 3D root mesh."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()
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

    if batch_size % built_dp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because batch_size={batch_size} "
            f"is not divisible by dp_size={built_dp_size}."
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

    input_rng = np.random.default_rng(input_seed)
    label_rng = np.random.default_rng(label_seed)
    input_data = Tensor(input_rng.standard_normal((batch_size, input_size)).astype(np.float32))
    label_data = Tensor(label_rng.standard_normal((batch_size, output_size)).astype(np.float32))
    state_dict_np = _build_model_state_dict(input_size, output_size, init_seed)

    standalone_model = TPFullyShardNet(state_dict_np)
    dist_model = TPFullyShardNet(state_dict_np)

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
    )

    for step_idx in range(steps):
        assert np.allclose(
            standalone_local_losses[step_idx],
            dist_losses[step_idx],
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected local loss "
            f"{standalone_local_losses[step_idx]}, got {dist_losses[step_idx]}"
        )

        for grad_idx, standalone_grad in enumerate(standalone_grads[step_idx]):
            expected_grad = _get_expected_grad_slice(
                standalone_grad,
                root_mesh,
                built_dp_size,
                built_tp_size,
                dp_mesh_ndim=1,
                tp_mesh_dim=1,
            )
            dist_grad = dist_grads[step_idx][grad_idx]
            assert np.allclose(
                expected_grad,
                dist_grad,
                rtol=1e-5,
                atol=1e-5,
            ), (
                f"{case_name}, rank {rank}, step {step_idx}, grad {grad_idx}: expected grad slice shape "
                f"{tuple(expected_grad.shape)}, got {tuple(dist_grad.shape)}"
            )

    print(
        f"[Rank {rank}] {case_name} passed with root_mesh={root_mesh.mesh_shape}, "
        f"tp_ep_mesh={tp_ep_mesh.mesh_shape}, steps={steps}, "
        f"local_grad_shape={tuple(dist_grads[-1][0].shape)}"
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
    """Run one HSDP(2D) + TP(1D) case on a 3D root mesh."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()
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

    if batch_size % dp_domain_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because batch_size={batch_size} "
            f"is not divisible by dp_domain_size={dp_domain_size}."
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
            f"is not divisible by tp_size={built_tp_size}."
        )
        return

    input_rng = np.random.default_rng(input_seed)
    label_rng = np.random.default_rng(label_seed)
    input_data = Tensor(input_rng.standard_normal((batch_size, input_size)).astype(np.float32))
    label_data = Tensor(label_rng.standard_normal((batch_size, output_size)).astype(np.float32))
    state_dict_np = _build_model_state_dict(input_size, output_size, init_seed)

    standalone_model = TPFullyShardNet(state_dict_np)
    dist_model = TPFullyShardNet(state_dict_np)

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
    )

    for step_idx in range(steps):
        assert np.allclose(
            standalone_local_losses[step_idx],
            dist_losses[step_idx],
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected local loss "
            f"{standalone_local_losses[step_idx]}, got {dist_losses[step_idx]}"
        )

        for grad_idx, standalone_grad in enumerate(standalone_grads[step_idx]):
            expected_grad = _get_expected_hsdp_tp_grad_slice(
                standalone_grad,
                root_mesh,
                fsdp_size,
                built_tp_size,
            )
            dist_grad = dist_grads[step_idx][grad_idx]
            assert np.allclose(
                expected_grad,
                dist_grad,
                rtol=1e-5,
                atol=1e-5,
            ), (
                f"{case_name}, rank {rank}, step {step_idx}, grad {grad_idx}: expected grad slice shape "
                f"{tuple(expected_grad.shape)}, got {tuple(dist_grad.shape)}"
            )

    print(
        f"[Rank {rank}] {case_name} passed with root_mesh={root_mesh.mesh_shape}, "
        f"hsdp_mesh={hsdp_mesh.mesh_shape}, steps={steps}, "
        f"local_grad_shape={tuple(dist_grads[-1][0].shape)}"
    )


def _assert_loss_and_grad_match_compat_mesh_none(
    *,
    case_name: str,
    steps: int,
    batch_size: int,
    input_size: int,
    output_size: int,
    input_seed: int,
    label_seed: int,
    init_seed: int,
    tp_size: int = 2,
):
    """Run one TP + fully_shard(mesh=None) compatibility case and compare with standalone."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()
    tp_mesh = _build_tp_only_mesh(world_size, tp_size=tp_size)
    if tp_mesh is None:
        print(
            f"[Rank {rank}] Skip {case_name} because world_size={world_size} "
            f"does not match tp_size={tp_size} for fully_shard(mesh=None) compatibility mode."
        )
        return

    if output_size % tp_size != 0:
        print(
            f"[Rank {rank}] Skip {case_name} because output_size={output_size} "
            f"is not divisible by tp_size={tp_size}."
        )
        return

    input_rng = np.random.default_rng(input_seed)
    label_rng = np.random.default_rng(label_seed)
    input_data = Tensor(input_rng.standard_normal((batch_size, input_size)).astype(np.float32))
    label_data = Tensor(label_rng.standard_normal((batch_size, output_size)).astype(np.float32))
    state_dict_np = _build_model_state_dict(input_size, output_size, init_seed)

    standalone_model = TPFullyShardNet(state_dict_np)
    dist_model = TPFullyShardNet(state_dict_np)

    standalone_grads, standalone_losses = _run_standalone_training(
        standalone_model,
        input_data,
        label_data,
        steps,
        1,
        0,
    )
    dist_losses, dist_grads = _run_tp_fully_shard_compat_training(
        dist_model,
        tp_mesh,
        input_data,
        label_data,
        steps,
    )

    for step_idx in range(steps):
        assert np.allclose(
            standalone_losses[step_idx],
            dist_losses[step_idx],
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected loss "
            f"{standalone_losses[step_idx]}, got {dist_losses[step_idx]}"
        )

        for grad_idx, standalone_grad in enumerate(standalone_grads[step_idx]):
            expected_grad = _get_expected_tp_grad_slice(
                standalone_grad,
                tp_mesh,
                tp_size,
            )
            dist_grad = dist_grads[step_idx][grad_idx]
            assert np.allclose(
                expected_grad,
                dist_grad,
                rtol=1e-5,
                atol=1e-5,
            ), (
                f"{case_name}, rank {rank}, step {step_idx}, grad {grad_idx}: expected grad slice shape "
                f"{tuple(expected_grad.shape)}, got {tuple(dist_grad.shape)}"
            )

    print(
        f"[Rank {rank}] {case_name} passed with tp_mesh={tp_mesh.mesh_shape}, "
        f"steps={steps}, local_grad_shape={tuple(dist_grads[-1][0].shape)}"
    )


def _assert_pure_tp_replicate_grad_managed_by_fully_shard(
    *,
    case_name: str,
    steps: int,
    batch_size: int,
    input_size: int,
    output_size: int,
    input_seed: int,
    label_seed: int,
    init_seed: int,
    tp_size: int = 2,
):
    """Compare pure-TP replicated gradients managed by fully_shard(mesh=None)."""
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()
    tp_mesh = _build_tp_only_mesh(world_size, tp_size=tp_size)
    if tp_mesh is None:
        print(
            f"[Rank {rank}] Skip {case_name} because world_size={world_size} "
            f"does not match tp_size={tp_size} for pure-TP compatibility mode."
        )
        return

    input_rng = np.random.default_rng(input_seed)
    label_rng = np.random.default_rng(label_seed)
    input_data = Tensor(input_rng.standard_normal((batch_size, input_size)).astype(np.float32))
    label_data = Tensor(label_rng.standard_normal((batch_size, output_size)).astype(np.float32))
    state_dict_np = _build_model_state_dict(input_size, output_size, init_seed)

    standalone_model = TPFullyShardNet(state_dict_np)
    dist_model = TPFullyShardNet(state_dict_np)

    standalone_grads, standalone_losses = _run_standalone_training(
        standalone_model,
        input_data,
        label_data,
        steps,
        1,
        0,
    )
    dist_model, dist_losses, dist_grads = _run_pure_tp_fully_shard_training(
        dist_model,
        tp_mesh,
        input_data,
        label_data,
        steps,
    )

    hsdp_state = dist_model.hsdp_scheduler.hsdp_state
    assert hsdp_state.replicate_params == []
    assert len(hsdp_state.hsdp_params) == 3
    for managed_param in hsdp_state.hsdp_params:
        assert managed_param.param_mode == FullyShardParamMode.DTENSOR_COMPAT
        assert managed_param.enable_fsdp_shard is True
        assert managed_param.is_sharded is False
        assert managed_param.shard_size == 1
        assert managed_param.dp_size == tp_size
        # pylint: disable-next=protected-access
        assert [repr(placement) for placement in managed_param._spmd_placements] == ["Replicate()"]

    for step_idx in range(steps):
        assert np.allclose(
            standalone_losses[step_idx],
            dist_losses[step_idx],
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"{case_name}, rank {rank}, step {step_idx}: expected loss "
            f"{standalone_losses[step_idx]}, got {dist_losses[step_idx]}"
        )

        for grad_idx, standalone_grad in enumerate(standalone_grads[step_idx]):
            dist_grad = dist_grads[step_idx][grad_idx]
            assert np.allclose(
                standalone_grad,
                dist_grad,
                rtol=1e-5,
                atol=1e-5,
            ), (
                f"{case_name}, rank {rank}, step {step_idx}, grad {grad_idx}: expected full grad shape "
                f"{tuple(standalone_grad.shape)}, got {tuple(dist_grad.shape)}"
            )

    print(
        f"[Rank {rank}] {case_name} passed with tp_mesh={tp_mesh.mesh_shape}, "
        f"steps={steps}, grad_shape={tuple(dist_grads[-1][0].shape)}"
    )


def test_tp_plus_fully_shard_loss_and_grad_match_standalone():
    """TP + fully_shard: compare local loss/grad against standalone on a square case."""
    _assert_loss_and_grad_match(
        case_name="tp2_fully_shard_case_ms",
        steps=12,
        batch_size=16,
        input_size=32,
        output_size=32,
        input_seed=2,
        label_seed=3,
        init_seed=4,
        tp_size=2,
    )


def test_tp_plus_fully_shard_same_dim_non_dim0_match_standalone():
    """TP + fully_shard: compare local loss/grad when TP and fully_shard shard the same output dim."""
    _assert_same_dim_non_dim0_loss_and_grad_match(
        case_name="tp2_same_dim_non_dim0_case_ms",
        steps=4,
        batch_size=12,
        input_size=16,
        output_size=16,
        input_seed=82,
        label_seed=83,
        init_seed=84,
        tp_size=2,
    )


def test_tp_plus_fully_shard_empty_weight_match_standalone():
    """TP + fully_shard: compare empty-init load path against standalone."""
    _assert_loss_and_grad_match(
        case_name="tp2_empty_weight_case_ms",
        steps=4,
        batch_size=16,
        input_size=32,
        output_size=32,
        input_seed=52,
        label_seed=53,
        init_seed=54,
        tp_size=2,
        use_empty_weight=True,
    )


def test_hsdp_plus_tp_on_3d_root_mesh_match_standalone():
    """HSDP + TP: compare loss/grad against standalone on a (dp, fsdp, tp) root mesh."""
    _assert_loss_and_grad_match_hsdp_tp_root_mesh(
        case_name="hsdp_tp_3d_square_case_ms",
        steps=4,
        batch_size=16,
        input_size=16,
        output_size=16,
        input_seed=72,
        label_seed=73,
        init_seed=74,
    )


def test_tp_plus_fully_shard_mesh_none_compat_match_standalone():
    """TP + fully_shard(mesh=None): compare loss/grad against standalone."""
    _assert_loss_and_grad_match_compat_mesh_none(
        case_name="tp2_mesh_none_compat_case_ms",
        steps=6,
        batch_size=8,
        input_size=16,
        output_size=16,
        input_seed=32,
        label_seed=33,
        init_seed=34,
        tp_size=2,
    )


def test_pure_tp_replicate_grad_managed_by_fully_shard_matches_standalone():
    """Pure TP + fully_shard(mesh=None): replicated DTensor grads match standalone."""
    _assert_pure_tp_replicate_grad_managed_by_fully_shard(
        case_name="pure_tp_replicate_fully_shard_case_ms",
        steps=2,
        batch_size=8,
        input_size=16,
        output_size=16,
        input_seed=92,
        label_seed=93,
        init_seed=94,
        tp_size=2,
    )
