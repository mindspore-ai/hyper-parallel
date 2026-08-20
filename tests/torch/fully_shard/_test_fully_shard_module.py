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
"""test fully_shard module api"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612,W0212
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import torch
import torch.distributed as dist
import torch_npu
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import fully_shard, HSDPModule, _UnshardHandle
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import DenseNet
from tests.torch.utils import init_dist


def test_fully_shard_module_01():
    """
    Feature: Test fully_shard module interface
    Description: Verify the HSDPModule interface methods
    Expectation: run successfully
    """
    init_dist()
    world_size = dist.get_world_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    # Create models
    model1 = DenseNet(32, 32, has_bias=False)
    model2 = DenseNet(32, 32, has_bias=False)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                     output_dtype=torch.float32, cast_forward_inputs=True)

    model1 = fully_shard(model1,
                         mesh=mesh,
                         reshard_after_forward=True,
                         mp_policy=mp_policy
                         )
    model2 = fully_shard(model2,
                         mesh=mesh,
                         reshard_after_forward=True,
                         mp_policy=mp_policy
                         )

    # Verify inheritance
    assert isinstance(model1, HSDPModule), "Model should be instance of HSDPModule"

    # Test set_requires_gradient_sync
    model1.set_requires_gradient_sync(True)
    assert model1.hsdp_scheduler.hsdp_state.reduce_grads is True

    model1.set_requires_gradient_sync(False)
    assert model1.hsdp_scheduler.hsdp_state.reduce_grads is False
    try:
        model1.set_requires_gradient_sync(1)
        assert False, "Should raise ValueError for non-bool input"
    except ValueError:
        pass

    # Test set_modules_to_forward_prefetch
    model1.set_modules_to_forward_prefetch([model2])
    assert model1.hsdp_scheduler.forward_prefetch_cells == [model2]
    try:
        model1.set_modules_to_forward_prefetch("invalid")
        assert False, "Should raise ValueError for non-list input"
    except ValueError:
        pass
    try:
        model1.set_modules_to_forward_prefetch([123])
        assert False, "Should raise ValueError for list containing non-HSDPModule"
    except ValueError:
        pass

    # Test set_modules_to_backward_prefetch
    model1.set_modules_to_backward_prefetch([model2])
    assert model1.hsdp_scheduler.backward_prefetch_cells == [model2]
    try:
        model1.set_modules_to_backward_prefetch("invalid")
        assert False, "Should raise ValueError for non-list input"
    except ValueError:
        pass

    # Test set_is_last_backward
    model1.set_is_last_backward(True)
    assert model1.hsdp_scheduler.scheduler_ctx.is_last_backward is True
    model1.set_is_last_backward(False)
    assert model1.hsdp_scheduler.scheduler_ctx.is_last_backward is False

    # Test set_requires_all_reduce
    model1.set_requires_all_reduce(True)
    assert model1.hsdp_scheduler.hsdp_state.requires_all_reduce is True
    model1.set_requires_all_reduce(False)
    assert model1.hsdp_scheduler.hsdp_state.requires_all_reduce is False
    try:
        model1.set_requires_all_reduce(1)
        assert False, "Should raise ValueError for non-bool input"
    except ValueError:
        pass

    # Test set_reshard_after_forward
    model1.set_reshard_after_forward(True)
    assert model1.hsdp_scheduler.reshard_after_forward is True
    model1.set_reshard_after_forward(False)
    assert model1.hsdp_scheduler.reshard_after_forward is False
    try:
        model1.set_reshard_after_forward(1)
        assert False, "Should raise ValueError for non-bool input"
    except ValueError:
        pass

    # Test set_reshard_after_backward
    model1.set_reshard_after_backward(True)
    assert model1.hsdp_scheduler.hsdp_state.reshard_after_backward is True
    model1.set_reshard_after_backward(False)
    assert model1.hsdp_scheduler.hsdp_state.reshard_after_backward is False
    try:
        model1.set_reshard_after_backward(1)
        assert False, "Should raise ValueError for non-bool input"
    except ValueError:
        pass

    # Test unshard and reshard
    hsdp_scheduler = model1.hsdp_scheduler
    hsdp_state = hsdp_scheduler.hsdp_state

    assert hsdp_state.is_shard is True
    expected_sharded_shape = torch.Size([32 // world_size, 32])
    param = next(model1.parameters())
    assert isinstance(param, DTensor), f"Expected DTensor, got {type(param)}"
    assert param.local_shape == expected_sharded_shape, f"Expected {expected_sharded_shape}, got {param.local_shape}"

    model1.unshard(async_op=False)
    assert hsdp_state.is_shard is False
    param = next(model1.parameters())
    assert isinstance(param, torch.nn.Parameter), f"Expected torch.nn.Parameter (not DTensor), got {type(param)}"
    assert param.shape == torch.Size([32, 32])

    model1.reshard()
    assert hsdp_state.is_shard is True
    param = next(model1.parameters())
    assert isinstance(param, DTensor), f"Expected DTensor, got {type(param)}"
    assert param.local_shape == expected_sharded_shape

    handle = model1.unshard(async_op=True)
    assert isinstance(handle, _UnshardHandle)
    assert hsdp_state.is_shard is True

    handle.wait()
    assert hsdp_state.is_shard is False
    param = next(model1.parameters())
    assert isinstance(param, torch.nn.Parameter), f"Expected torch.nn.Parameter (not DTensor), got {type(param)}"
    assert param.shape == torch.Size([32, 32])
    assert handle._hsdp_state is None

    model1.unshard(async_op=False)
    assert hsdp_state.is_shard is False

    model1.reshard()
    model1.reshard()
    assert hsdp_state.is_shard is True


class StageModel(torch.nn.Module):
    """Wraps a contiguous sequence of fully_shard-managed layers as one pipeline stage."""

    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(list(layers))

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_fully_shard_module_02():
    """
    Feature: Test set_reshard_after_backward / set_reshard_after_forward recurse on nested model
    Description: Verify that calling set_reshard_after_backward / set_reshard_after_forward on
        a StageModel whose child layers are individually wrapped with fully_shard correctly
        propagates to all child HSDPModule instances (tests the recurse=True deep traversal).
    Expectation: run successfully
    """
    init_dist()
    world_size = dist.get_world_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                     output_dtype=torch.float32, cast_forward_inputs=True)

    inner_layers = [DenseNet(32, 32, has_bias=False) for _ in range(3)]
    for layer in inner_layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    stage = StageModel(inner_layers)
    fully_shard(stage, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    child_hsdp_modules = [m for m in stage.modules() if isinstance(m, HSDPModule) and m is not stage]
    assert len(child_hsdp_modules) == len(inner_layers), (
        f"Expected {len(inner_layers)} child HSDPModules, got {len(child_hsdp_modules)}"
    )

    # set_reshard_after_backward(False) must reach all nested HSDPModule children
    stage.set_reshard_after_backward(False)
    for child in child_hsdp_modules:
        assert child.hsdp_scheduler.hsdp_state.reshard_after_backward is False, (
            f"Expected reshard_after_backward=False on child {child}, "
            f"got {child.hsdp_scheduler.hsdp_state.reshard_after_backward}"
        )

    stage.set_reshard_after_backward(True)
    for child in child_hsdp_modules:
        assert child.hsdp_scheduler.hsdp_state.reshard_after_backward is True, (
            f"Expected reshard_after_backward=True on child {child}, "
            f"got {child.hsdp_scheduler.hsdp_state.reshard_after_backward}"
        )

    # set_reshard_after_forward(False) must reach all nested HSDPModule children
    stage.set_reshard_after_forward(False)
    for child in child_hsdp_modules:
        assert child.hsdp_scheduler.reshard_after_forward is False, (
            f"Expected reshard_after_forward=False on child {child}, "
            f"got {child.hsdp_scheduler.reshard_after_forward}"
        )

    stage.set_reshard_after_forward(True)
    for child in child_hsdp_modules:
        assert child.hsdp_scheduler.reshard_after_forward is True, (
            f"Expected reshard_after_forward=True on child {child}, "
            f"got {child.hsdp_scheduler.reshard_after_forward}"
        )
