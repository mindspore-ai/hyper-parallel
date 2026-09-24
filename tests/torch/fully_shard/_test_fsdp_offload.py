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

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=W0611
from torch import nn, optim

from hyper_parallel import DTensor, DeviceMesh, SkipDTensorDispatch, hsdp_sync_stream, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState
from hyper_parallel.core.fully_shard.utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from tests.torch.utils import init_dist


class OffloadGuardModel(nn.Module):
    """Small model that exposes parameter dtype during mixed-precision forward."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.input_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.forward_parameter_dtype = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_parameter_dtype = self.input_layer.weight.dtype
        hidden = torch.relu(self.input_layer(inputs))
        return self.output_layer(hidden).sum()


def _mixed_precision_policy() -> MixedPrecisionPolicy:
    """Build the mixed-precision policy used by all lifecycle guards."""
    return MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )


def _build_fsdp_model(offload_policy: OffloadPolicy) -> nn.Module:
    """Build an FSDP model on the current NPU process mesh."""
    model = OffloadGuardModel().npu()
    mesh: DeviceMesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("dp",),
    )
    return fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=_mixed_precision_policy(),
        offload_policy=offload_policy,
    )


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


def _assert_cpu_offloaded_state(model: nn.Module) -> None:
    """Check the persistent parameter and gradient storage after CPU offload."""
    state = model.hsdp_scheduler.hsdp_state
    assert state.is_shard, f"CPU offload state should be sharded, got is_shard={state.is_shard}"
    for hsdp_param in state.hsdp_params:
        assert hsdp_param.param_dtype == torch.float16, (
            f"Expected param_dtype=torch.float16, got {hsdp_param.param_dtype}"
        )
        assert hsdp_param.reduce_dtype == torch.float32, (
            f"Expected reduce_dtype=torch.float32, got {hsdp_param.reduce_dtype}"
        )
        assert hsdp_param.sharded_param.device.type == "cpu", (
            f"Expected sharded parameter on CPU, got {hsdp_param.sharded_param.device}"
        )
        sharded_data_device = hsdp_param._sharded_param_data.device  # pylint: disable=protected-access
        assert sharded_data_device.type == "cpu", (
            f"Expected communication storage on CPU, got {sharded_data_device}"
        )
        if hsdp_param.sharded_param.grad is not None:
            grad_device = hsdp_param.sharded_param.grad.device
            assert grad_device.type == "cpu", f"Expected sharded gradient on CPU, got {grad_device}"
        _assert_storage_is_graph_free(hsdp_param)


def test_cpu_offload_mixed_precision_training():
    """CPU offload keeps mixed-precision sharded parameters and gradients on CPU."""
    init_dist()
    model = _build_fsdp_model(CPUOffloadPolicy(pin_memory=True))
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    inputs = torch.randn(4, 16).npu()

    with SkipDTensorDispatch():
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(inputs)
            assert model.forward_parameter_dtype == torch.float16, (
                f"Expected forward parameter dtype=torch.float16, got {model.forward_parameter_dtype}"
            )
            assert loss.dtype == torch.float32, f"Expected output dtype=torch.float32, got {loss.dtype}"
            loss.backward()
            hsdp_sync_stream()
            _assert_cpu_offloaded_state(model)
            optimizer.step()

    model.reset_iter_state()


def test_apply_and_reset_mixed_precision_storage():
    """``_apply`` releases unsharded storage and reset detaches graph-backed local storage."""
    init_dist()
    model = _build_fsdp_model(OffloadPolicy())
    inputs = torch.randn(4, 16).npu()

    with torch.no_grad(), SkipDTensorDispatch():
        model(inputs)

    state = model.hsdp_scheduler.hsdp_state
    model.unshard()
    assert not state.is_shard, f"Expected is_shard=False after unshard, got {state.is_shard}"
    for hsdp_param in state.hsdp_params:
        for buffer in hsdp_param.unsharded_param_buffers:
            assert buffer.device.type == "npu", f"Expected unsharded buffer on NPU, got {buffer.device}"
            storage_nbytes = buffer.untyped_storage().nbytes()
            assert storage_nbytes > 0, f"Expected materialized unsharded storage, got nbytes={storage_nbytes}"

    model._apply(lambda tensor: tensor.to("cpu"))

    assert state.is_shard, f"model._apply() should reshard the state, got is_shard={state.is_shard}"
    for hsdp_param in state.hsdp_params:
        assert hsdp_param.sharded_state == ShardedState.SHARDED, (
            f"Expected sharded parameter state, got {hsdp_param.sharded_state}"
        )
        assert isinstance(hsdp_param.sharded_param, DTensor), (
            f"Expected DTensor sharded parameter, got {type(hsdp_param.sharded_param)}"
        )
        assert hsdp_param.sharded_param.device.type == "cpu", (
            f"Expected converted sharded parameter on CPU, got {hsdp_param.sharded_param.device}"
        )
        for buffer in hsdp_param.unsharded_param_buffers:
            storage_nbytes = buffer.untyped_storage().nbytes()
            assert storage_nbytes == 0, (
                f"Expected unsharded storage to be released, got nbytes={storage_nbytes}"
            )

        source_local = (
            hsdp_param.sharded_param._local_tensor.detach().requires_grad_(True)
        )  # pylint: disable=protected-access
        graph_local = source_local * 2
        assert graph_local.grad_fn is not None, f"Expected graph-backed local tensor, got {graph_local.grad_fn}"
        hsdp_param.sharded_param._local_tensor = graph_local  # pylint: disable=protected-access
        hsdp_param.reset_sharded_param()
        _assert_storage_is_graph_free(hsdp_param)

    model.reset_iter_state()
