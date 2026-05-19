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
"""test fully_shard module api (MindSpore)"""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms
from mindspore import nn
from mindspore.communication import get_group_size, get_rank, init

from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard, HSDPModule, _UnshardHandle
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

ms.set_seed(42)
ms.set_deterministic(True)
enable_mindspore_backward_compat()


class FlatDense(nn.Cell):
    """Single-layer square dense block used in module-interface tests."""

    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Dense(size, size, has_bias=False, weight_init="normal")

    def construct(self, hidden_states):
        return self.linear(hidden_states)


class MLPLayer(nn.Cell):
    """Single MLP layer used as a building block for nested model tests."""

    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = nn.Dense(d_hid, d_hid)
        self.net2 = nn.Dense(d_hid, d_hid)
        self.relu = nn.ReLU()

    def construct(self, hidden_states):
        return self.net2(self.relu(self.net1(hidden_states)))


class StageModel(nn.Cell):
    """Wraps a contiguous sequence of fully_shard-managed layers as one pipeline stage."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.CellList(list(layers))

    def construct(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_ms_fully_shard_module_01():
    """Test HSDPModule interface methods on MindSpore.

    Feature: HSDPModule interface methods (MindSpore)
    Description: Verify that set_requires_gradient_sync, set_modules_to_forward_prefetch,
        set_modules_to_backward_prefetch, set_is_last_backward, set_requires_all_reduce,
        set_reshard_after_forward, set_reshard_after_backward, unshard, and reshard
        behave correctly on MindSpore — mirroring test_fully_shard_module_01 on PyTorch.
    Expectation: run successfully
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank = get_rank()
    world_size = get_group_size()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )

    model1 = FlatDense(32)
    model2 = FlatDense(32)
    fully_shard(model1, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
    fully_shard(model2, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    assert isinstance(model1, HSDPModule), (
        f"Expected model1 to be HSDPModule, got {type(model1)}"
    )

    # set_requires_gradient_sync
    model1.set_requires_gradient_sync(True)
    assert model1.hsdp_scheduler.hsdp_state.reduce_grads is True, (
        f"Expected reduce_grads=True, got {model1.hsdp_scheduler.hsdp_state.reduce_grads}"
    )
    model1.set_requires_gradient_sync(False)
    assert model1.hsdp_scheduler.hsdp_state.reduce_grads is False, (
        f"Expected reduce_grads=False, got {model1.hsdp_scheduler.hsdp_state.reduce_grads}"
    )
    try:
        model1.set_requires_gradient_sync(1)
        assert False, "Should raise ValueError for non-bool input"
    except ValueError:
        pass

    # set_modules_to_forward_prefetch
    model1.set_modules_to_forward_prefetch([model2])
    assert model1.hsdp_scheduler.forward_prefetch_cells == [model2], (
        f"Expected forward_prefetch_cells=[model2], "
        f"got {model1.hsdp_scheduler.forward_prefetch_cells}"
    )
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

    # set_modules_to_backward_prefetch
    model1.set_modules_to_backward_prefetch([model2])
    assert model1.hsdp_scheduler.backward_prefetch_cells == [model2], (
        f"Expected backward_prefetch_cells=[model2], "
        f"got {model1.hsdp_scheduler.backward_prefetch_cells}"
    )
    try:
        model1.set_modules_to_backward_prefetch("invalid")
        assert False, "Should raise ValueError for non-list input"
    except ValueError:
        pass

    # set_is_last_backward
    model1.set_is_last_backward(True)
    assert model1.hsdp_scheduler.scheduler_ctx.is_last_backward is True, (
        f"Expected is_last_backward=True, "
        f"got {model1.hsdp_scheduler.scheduler_ctx.is_last_backward}"
    )
    model1.set_is_last_backward(False)
    assert model1.hsdp_scheduler.scheduler_ctx.is_last_backward is False, (
        f"Expected is_last_backward=False, "
        f"got {model1.hsdp_scheduler.scheduler_ctx.is_last_backward}"
    )

    # set_requires_all_reduce
    model1.set_requires_all_reduce(True)
    assert model1.hsdp_scheduler.hsdp_state.requires_all_reduce is True, (
        f"Expected requires_all_reduce=True, "
        f"got {model1.hsdp_scheduler.hsdp_state.requires_all_reduce}"
    )
    model1.set_requires_all_reduce(False)
    assert model1.hsdp_scheduler.hsdp_state.requires_all_reduce is False, (
        f"Expected requires_all_reduce=False, "
        f"got {model1.hsdp_scheduler.hsdp_state.requires_all_reduce}"
    )
    try:
        model1.set_requires_all_reduce(1)
        assert False, "Should raise ValueError for non-bool input"
    except ValueError:
        pass

    # set_reshard_after_forward
    model1.set_reshard_after_forward(True)
    assert model1.hsdp_scheduler.reshard_after_forward is True, (
        f"Expected reshard_after_forward=True, "
        f"got {model1.hsdp_scheduler.reshard_after_forward}"
    )
    assert model1.hsdp_scheduler.config.reshard_after_forward is True, (
        f"Expected config.reshard_after_forward=True, "
        f"got {model1.hsdp_scheduler.config.reshard_after_forward}"
    )
    model1.set_reshard_after_forward(False)
    assert model1.hsdp_scheduler.reshard_after_forward is False, (
        f"Expected reshard_after_forward=False, "
        f"got {model1.hsdp_scheduler.reshard_after_forward}"
    )
    assert model1.hsdp_scheduler.config.reshard_after_forward is False, (
        f"Expected config.reshard_after_forward=False, "
        f"got {model1.hsdp_scheduler.config.reshard_after_forward}"
    )
    try:
        model1.set_reshard_after_forward(1)
        assert False, "Should raise ValueError for non-bool input"
    except ValueError:
        pass

    # set_reshard_after_backward
    model1.set_reshard_after_backward(True)
    assert model1.hsdp_scheduler.hsdp_state.reshard_after_backward is True, (
        f"Expected reshard_after_backward=True, "
        f"got {model1.hsdp_scheduler.hsdp_state.reshard_after_backward}"
    )
    model1.set_reshard_after_backward(False)
    assert model1.hsdp_scheduler.hsdp_state.reshard_after_backward is False, (
        f"Expected reshard_after_backward=False, "
        f"got {model1.hsdp_scheduler.hsdp_state.reshard_after_backward}"
    )
    try:
        model1.set_reshard_after_backward(1)
        assert False, "Should raise ValueError for non-bool input"
    except ValueError:
        pass

    # unshard / reshard state transitions
    hsdp_state = model1.hsdp_scheduler.hsdp_state
    assert hsdp_state.is_shard is True, (
        f"Expected is_shard=True after init, got {hsdp_state.is_shard}"
    )
    param = next(iter(model1.trainable_params()))
    assert isinstance(param, DTensor), f"Expected DTensor after init, got {type(param)}"
    expected_local_shape = (32 // world_size, 32)
    assert param.local_shape == expected_local_shape, (
        f"Expected local_shape={expected_local_shape}, got {param.local_shape}"
    )

    model1.unshard(async_op=False)
    assert hsdp_state.is_shard is False, (
        f"Expected is_shard=False after unshard, got {hsdp_state.is_shard}"
    )
    param = next(iter(model1.trainable_params()))
    assert not isinstance(param, DTensor), (
        f"Expected plain Parameter after unshard, got {type(param)}"
    )
    assert param.shape == (32, 32), f"Expected shape=(32,32) after unshard, got {param.shape}"

    model1.reshard()
    assert hsdp_state.is_shard is True, (
        f"Expected is_shard=True after reshard, got {hsdp_state.is_shard}"
    )
    param = next(iter(model1.trainable_params()))
    assert isinstance(param, DTensor), f"Expected DTensor after reshard, got {type(param)}"
    assert param.local_shape == expected_local_shape, (
        f"Expected local_shape={expected_local_shape} after reshard, got {param.local_shape}"
    )

    handle = model1.unshard(async_op=True)
    assert isinstance(handle, _UnshardHandle), (
        f"Expected _UnshardHandle, got {type(handle)}"
    )
    assert hsdp_state.is_shard is True, (
        f"Expected is_shard=True before handle.wait(), got {hsdp_state.is_shard}"
    )

    handle.wait()
    assert hsdp_state.is_shard is False, (
        f"Expected is_shard=False after handle.wait(), got {hsdp_state.is_shard}"
    )
    param = next(iter(model1.trainable_params()))
    assert not isinstance(param, DTensor), (
        f"Expected plain Parameter after handle.wait(), got {type(param)}"
    )
    assert param.shape == (32, 32), (
        f"Expected shape=(32,32) after handle.wait(), got {param.shape}"
    )
    assert handle._hsdp_state is None, (  # pylint: disable=protected-access
        f"Expected handle._hsdp_state=None after wait(), got {handle._hsdp_state}"  # pylint: disable=protected-access
    )

    model1.unshard(async_op=False)
    assert hsdp_state.is_shard is False, (
        f"Expected is_shard=False, got {hsdp_state.is_shard}"
    )

    model1.reshard()
    model1.reshard()
    assert hsdp_state.is_shard is True, (
        f"Expected is_shard=True after double reshard, got {hsdp_state.is_shard}"
    )

    if rank == 0:
        print(f"test_ms_fully_shard_module_01 passed, world_size={world_size}")

def test_fully_shard_module_02():
    """Test set_reshard_after_backward / set_reshard_after_forward recurse on nested model (MindSpore).

    Feature: HSDPModule.set_reshard_after_backward / set_reshard_after_forward recurse
    Description: Calling set_reshard_after_backward / set_reshard_after_forward on a
        StageModel whose child layers are individually wrapped with fully_shard must
        propagate to all nested HSDPModule children via deep traversal (cells_and_names),
        not just direct children (cells). This is the MindSpore-specific regression that
        caused the PP+FSDP hacking workaround in stage.py backward_one_chunk.
    Expectation: run successfully
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    world_size = get_group_size()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )

    num_layers = 3
    inner_layers = [MLPLayer(8) for _ in range(num_layers)]
    for layer in inner_layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    stage = StageModel(inner_layers)
    fully_shard(stage, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    child_hsdp_modules = [
        mod for _, mod in stage.cells_and_names()
        if isinstance(mod, HSDPModule) and mod is not stage
    ]
    assert len(child_hsdp_modules) == num_layers, (
        f"Expected {num_layers} child HSDPModules, got {len(child_hsdp_modules)}"
    )

    # set_reshard_after_backward(False) must reach all nested HSDPModule children
    stage.set_reshard_after_backward(False)
    for child in child_hsdp_modules:
        assert child.hsdp_scheduler.hsdp_state.reshard_after_backward is False, (
            f"Expected reshard_after_backward=False on nested child, "
            f"got {child.hsdp_scheduler.hsdp_state.reshard_after_backward}"
        )

    stage.set_reshard_after_backward(True)
    for child in child_hsdp_modules:
        assert child.hsdp_scheduler.hsdp_state.reshard_after_backward is True, (
            f"Expected reshard_after_backward=True on nested child, "
            f"got {child.hsdp_scheduler.hsdp_state.reshard_after_backward}"
        )

    # set_reshard_after_forward(False) must reach all nested HSDPModule children
    stage.set_reshard_after_forward(False)
    for child in child_hsdp_modules:
        assert child.hsdp_scheduler.reshard_after_forward is False, (
            f"Expected reshard_after_forward=False on nested child, "
            f"got {child.hsdp_scheduler.reshard_after_forward}"
        )

    stage.set_reshard_after_forward(True)
    for child in child_hsdp_modules:
        assert child.hsdp_scheduler.reshard_after_forward is True, (
            f"Expected reshard_after_forward=True on nested child, "
            f"got {child.hsdp_scheduler.reshard_after_forward}"
        )

    print(f"test_ms_fully_shard_module_recurse passed, world_size={world_size}")
