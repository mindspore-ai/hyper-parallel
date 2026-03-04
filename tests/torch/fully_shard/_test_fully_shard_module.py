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
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import torch
# pylint: disable=W0611
import torch_npu
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import fully_shard, HSDPModule, _UnshardHandle
from hyper_parallel.platform.torch.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import DenseNet
from tests.torch.utils import init_dist


def test_fully_shard_module_01():
    """
    Feature: Test fully_shard module interface
    Description: Verify the HSDPModule interface methods
    Expectation: run successfully
    """
    init_dist()
    # Assume 8 devices as per torchrun default
    world_size = 8
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
    assert model1.hsdp_scheduler.requires_grad_sync is True
    assert model1.hsdp_scheduler.hsdp_state.reduce_grads is True

    model1.set_requires_gradient_sync(False)
    assert model1.hsdp_scheduler.requires_grad_sync is False
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
    assert model1.hsdp_scheduler.config.reshard_after_forward is True
    model1.set_reshard_after_forward(False)
    assert model1.hsdp_scheduler.reshard_after_forward is False
    assert model1.hsdp_scheduler.config.reshard_after_forward is False
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
