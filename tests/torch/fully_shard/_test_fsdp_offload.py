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
"""Distributed guards for FSDP CPU offload and mixed-precision storage lifecycle."""

from copy import deepcopy

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=W0611
from torch import nn, optim

from hyper_parallel import DTensor, DeviceMesh, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
from tests.torch.utils import init_dist


_HIDDEN_SIZE = 16
_LOCAL_BATCH_SIZE = 4
_TRAIN_STEPS = 2
_LEARNING_RATE = 0.001
_ATOL = 5e-3
_RTOL = 5e-3


class OffloadGuardModel(nn.Module):
    """Small model that exposes parameter dtype during mixed-precision forward."""

    def __init__(self, hidden_size: int = _HIDDEN_SIZE) -> None:
        """Initialize the two-layer model and its forward dtype observation."""
        super().__init__()
        self.input_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.forward_parameter_dtype = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Record the compute dtype and return the mean squared output loss."""
        self.forward_parameter_dtype = self.input_layer.weight.dtype
        hidden = torch.relu(self.input_layer(inputs))
        return self.output_layer(hidden).square().mean()


def _mixed_precision_policy() -> MixedPrecisionPolicy:
    """Build the mixed-precision policy used by all lifecycle guards."""
    return MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
    )


def _build_fsdp_model(model: nn.Module) -> nn.Module:
    """Build an FSDP model with CPU-offloaded parameters on the current NPU mesh."""
    mesh: DeviceMesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("dp",),
    )
    return fully_shard(
        model.npu(),
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=_mixed_precision_policy(),
        offload_policy=CPUOffloadPolicy(pin_memory=True),
    )


def _run_standalone_step(
    cpu_model: nn.Module,
    cpu_optimizer: optim.Optimizer,
    global_inputs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run NPU forward/backward and apply the resulting gradients with CPU AdamW."""
    npu_model = deepcopy(cpu_model).npu().to(torch.float16)
    npu_loss = npu_model(global_inputs.npu().to(torch.float16))
    npu_loss.backward()

    cpu_gradients = {}
    npu_parameters = dict(npu_model.named_parameters())
    for parameter_name, cpu_parameter in cpu_model.named_parameters():
        npu_gradient = npu_parameters[parameter_name].grad
        if npu_gradient is None:
            raise AssertionError(f"Standalone parameter {parameter_name} has no gradient")
        cpu_parameter.grad = npu_gradient.detach().to(device="cpu", dtype=cpu_parameter.dtype)
        cpu_gradients[parameter_name] = cpu_parameter.grad.clone()

    cpu_loss = npu_loss.detach().float().cpu()
    cpu_optimizer.step()
    cpu_optimizer.zero_grad(set_to_none=True)
    return cpu_loss, cpu_gradients


def _local_baseline_shard(
    tensor: torch.Tensor,
    shard_dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Return the baseline shard corresponding to the current data-parallel rank."""
    shards = torch.chunk(tensor, world_size, dim=shard_dim)
    if len(shards) != world_size:
        raise AssertionError(
            f"Expected {world_size} baseline shards along dim {shard_dim}, got {len(shards)}"
        )
    return shards[rank].contiguous()


def _assert_storage_is_graph_free(hsdp_param) -> None:
    """Check both DTensor storage references are detached from autograd."""
    sharded_data_grad_fn = hsdp_param._sharded_param_data.grad_fn  # pylint: disable=protected-access
    local_grad_fn = hsdp_param.sharded_param._local_tensor.grad_fn  # pylint: disable=protected-access
    local_is_leaf = hsdp_param.sharded_param._local_tensor.is_leaf  # pylint: disable=protected-access
    assert sharded_data_grad_fn is None, (
        f"Expected sharded storage grad_fn=None, got {sharded_data_grad_fn}"
    )
    assert local_grad_fn is None, f"Expected DTensor local grad_fn=None, got {local_grad_fn}"
    assert local_is_leaf, f"Expected DTensor local is_leaf=True, got {local_is_leaf}"


def _assert_optimizer_state_on_cpu(optimizer: optim.Optimizer) -> None:
    """Check that AdamW parameters, gradients, and momentum stay on CPU."""
    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group["params"]:
            assert parameter.device.type == "cpu", (
                f"Expected optimizer parameter on CPU, got {parameter.device}"
            )
            if parameter.grad is not None:
                assert parameter.grad.device.type == "cpu", (
                    f"Expected optimizer gradient on CPU, got {parameter.grad.device}"
                )
    for parameter_state in optimizer.state.values():
        for state_name, state_value in parameter_state.items():
            if isinstance(state_value, torch.Tensor):
                assert state_value.device.type == "cpu", (
                    f"Expected AdamW state {state_name} on CPU, got {state_value.device}"
                )


def _assert_distributed_step_matches_baseline(
    model: nn.Module,
    optimizer: optim.Optimizer,
    baseline_model: nn.Module,
    baseline_loss: torch.Tensor,
    baseline_gradients: dict[str, torch.Tensor],
    global_inputs: torch.Tensor,
    rank: int,
    world_size: int,
) -> None:
    """Run one distributed step and compare its loss, gradients, and parameters."""
    local_inputs = global_inputs.chunk(world_size, dim=0)[rank].npu().to(torch.float16)
    optimizer.zero_grad(set_to_none=True)
    distributed_loss = model(local_inputs)
    assert model.forward_parameter_dtype == torch.float16, (
        f"Expected forward parameter dtype=torch.float16, got {model.forward_parameter_dtype}"
    )
    assert distributed_loss.dtype == torch.float16, (
        f"Expected an uncast float16 output, got {distributed_loss.dtype}"
    )
    distributed_loss.backward()

    state = model.hsdp_scheduler.hsdp_state
    for hsdp_param in state.hsdp_params:
        parameter_name = hsdp_param._param_fqn  # pylint: disable=protected-access
        sharded_gradient = hsdp_param.sharded_param.grad
        if not isinstance(sharded_gradient, DTensor):
            raise AssertionError(
                f"Expected {parameter_name} gradient to be DTensor, got {type(sharded_gradient)}"
            )
        expected_gradient = _local_baseline_shard(
            baseline_gradients[parameter_name],
            hsdp_param.hsdp_placement.dim,
            rank,
            world_size,
        )
        torch.testing.assert_close(
            sharded_gradient._local_tensor,  # pylint: disable=protected-access
            expected_gradient,
            atol=_ATOL,
            rtol=_RTOL,
        )

    averaged_loss = distributed_loss.detach().float()
    dist.all_reduce(averaged_loss)
    averaged_loss /= world_size
    torch.testing.assert_close(averaged_loss.cpu(), baseline_loss, atol=_ATOL, rtol=_RTOL)

    optimizer.step()
    baseline_parameters = dict(baseline_model.named_parameters())
    for hsdp_param in state.hsdp_params:
        parameter_name = hsdp_param._param_fqn  # pylint: disable=protected-access
        expected_parameter = _local_baseline_shard(
            baseline_parameters[parameter_name].detach(),
            hsdp_param.hsdp_placement.dim,
            rank,
            world_size,
        )
        torch.testing.assert_close(
            hsdp_param.sharded_param._local_tensor,  # pylint: disable=protected-access
            expected_parameter,
            atol=_ATOL,
            rtol=_RTOL,
        )
        _assert_storage_is_graph_free(hsdp_param)


def _assert_apply_releases_npu_storage(model: nn.Module) -> None:
    """Check conversion releases NPU storage using internal storage references.

    No public API exposes both DTensor storage references or buffer lifetimes.
    """
    state = model.hsdp_scheduler.hsdp_state
    original_storage_ptrs = [
        hsdp_param._sharded_param_data.untyped_storage().data_ptr()  # pylint: disable=protected-access
        for hsdp_param in state.hsdp_params
    ]

    torch.npu.synchronize()
    model.unshard()
    torch.npu.synchronize()
    allocated_with_unsharded_params = torch.npu.memory_allocated()
    unsharded_storage_nbytes = sum(
        buffer.untyped_storage().nbytes()
        for hsdp_param in state.hsdp_params
        for buffer in hsdp_param.unsharded_param_buffers
    )
    assert unsharded_storage_nbytes > 0, "Expected materialized NPU parameter storage"

    model._apply(lambda tensor: tensor.to("cpu", copy=True))  # pylint: disable=protected-access
    torch.npu.synchronize()
    allocated_after_apply = torch.npu.memory_allocated()

    assert state.is_shard, f"model._apply() should reshard the state, got is_shard={state.is_shard}"
    assert allocated_with_unsharded_params - allocated_after_apply >= unsharded_storage_nbytes, (
        "Expected _apply() to release all materialized NPU parameter storage, "
        f"before={allocated_with_unsharded_params}, after={allocated_after_apply}, "
        f"unsharded_storage_nbytes={unsharded_storage_nbytes}"
    )
    for original_storage_ptr, hsdp_param in zip(original_storage_ptrs, state.hsdp_params):
        assert hsdp_param.sharded_state == ShardedState.SHARDED, (
            f"Expected sharded parameter state, got {hsdp_param.sharded_state}"
        )
        assert hsdp_param.sharded_param.device.type == "cpu", (
            f"Expected converted sharded parameter on CPU, got {hsdp_param.sharded_param.device}"
        )
        replaced_storage_ptr = (
            hsdp_param._sharded_param_data.untyped_storage().data_ptr()  # pylint: disable=protected-access
        )
        assert replaced_storage_ptr != original_storage_ptr, "Expected reset_sharded_param() to replace CPU storage"
        for buffer in hsdp_param.unsharded_param_buffers:
            assert buffer.untyped_storage().nbytes() == 0, (
                f"Expected unsharded NPU storage to be released, got {buffer.untyped_storage().nbytes()} bytes"
            )
        _assert_storage_is_graph_free(hsdp_param)


def _assert_reset_detaches_replaced_storage(model: nn.Module) -> None:
    """Replace DTensor local storage with a graph tensor and verify reset detaches it."""
    state = model.hsdp_scheduler.hsdp_state
    for hsdp_param in state.hsdp_params:
        original_storage_ptr = (
            hsdp_param._sharded_param_data.untyped_storage().data_ptr()  # pylint: disable=protected-access
        )
        source_local = (
            hsdp_param.sharded_param._local_tensor.detach().requires_grad_(True)  # pylint: disable=protected-access
        )
        graph_local = source_local + 0
        assert graph_local.grad_fn is not None, f"Expected graph-backed local tensor, got {graph_local.grad_fn}"
        hsdp_param.sharded_param._local_tensor = graph_local  # pylint: disable=protected-access
        hsdp_param.reset_sharded_param()

        replaced_storage_ptr = (
            hsdp_param._sharded_param_data.untyped_storage().data_ptr()  # pylint: disable=protected-access
        )
        assert replaced_storage_ptr != original_storage_ptr, "Expected reset_sharded_param() to replace local storage"
        _assert_storage_is_graph_free(hsdp_param)


def test_cpu_offload_mixed_precision_training():
    """Validate mixed-precision FSDP CPU offload against a CPU AdamW baseline.

    Feature: FSDP mixed precision with CPU-offloaded parameters and optimizer state.
    Description: Train an eight-card FSDP model and an equivalent standalone model with the same global batches.
        The standalone model keeps fp32 parameters and AdamW state on CPU, copies parameters to NPU for each
        forward/backward, and applies copied fp32 gradients on CPU. The distributed path also exercises `_apply()`
        and graph-backed DTensor local-storage replacement after training.
    Expectation: Losses, local gradients, and updated parameter shards match the standalone baseline; parameters,
        gradients, and AdamW state remain on CPU; `_apply()` releases materialized NPU storage; reset storage has
        no autograd history.
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    assert world_size == 8, f"Expected an eight-card test, got world_size={world_size}"

    torch.manual_seed(2026)
    baseline_model = OffloadGuardModel()
    model = _build_fsdp_model(deepcopy(baseline_model))
    baseline_optimizer = optim.AdamW(baseline_model.parameters(), lr=_LEARNING_RATE)
    optimizer = optim.AdamW(model.parameters(), lr=_LEARNING_RATE)

    input_generator = torch.Generator().manual_seed(2027)
    global_batches = [
        torch.randn(world_size * _LOCAL_BATCH_SIZE, _HIDDEN_SIZE, generator=input_generator)
        for _ in range(_TRAIN_STEPS)
    ]

    with SkipDTensorDispatch():
        for global_inputs in global_batches:
            baseline_loss, baseline_gradients = _run_standalone_step(
                baseline_model,
                baseline_optimizer,
                global_inputs,
            )
            _assert_distributed_step_matches_baseline(
                model,
                optimizer,
                baseline_model,
                baseline_loss,
                baseline_gradients,
                global_inputs,
                rank,
                world_size,
            )
            _assert_optimizer_state_on_cpu(baseline_optimizer)
            _assert_optimizer_state_on_cpu(optimizer)

        optimizer.zero_grad(set_to_none=True)
        _assert_apply_releases_npu_storage(model)
        _assert_reset_detaches_replaced_storage(model)

    model.reset_iter_state()
