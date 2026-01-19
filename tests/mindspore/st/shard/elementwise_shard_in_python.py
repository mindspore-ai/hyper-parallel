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
"""test elementwise shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor, ops
from hyper_parallel import Layout, shard
from tests.mindspore.st.shard.utils import global_to_local, local_to_global


def setup_module():
    ms.set_device("Ascend")
    D.init()


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


class MinimumNet(nn.Cell):
    """Minimum network composed of minimum operation and ReLU"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.minimum = ms.mint.minimum
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, y):
        out = self.minimum(x, y)
        out = self.relu(out)
        out = out + 1
        return out


class LessEqualNet(nn.Cell):
    """LessEqual network composed of comparison and ReLU"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.less_equal = ops.LessEqual()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, y):
        out = self.less_equal(x, y)
        out = ops.cast(out, ms.float32)
        out = self.relu(out)
        out = out + 1
        return out


class GreaterEqualNet(nn.Cell):
    """GreaterEqual network composed of comparison and ReLU"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.greater_equal = ops.GreaterEqual()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, y):
        out = self.greater_equal(x, y)
        out = ops.cast(out, ms.float32)
        out = self.relu(out)
        out = out + 1
        return out


class LogicalOrNet(nn.Cell):
    """LogicalOr network composed of logical operation and ReLU"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.logical_or = ops.LogicalOr()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, y):
        out = self.logical_or(x, y)
        out = ops.cast(out, ms.float32)
        out = self.relu(out)
        out = out + 1
        return out


class ModNet(nn.Cell):
    """Mod network composed of mod operation and ReLU"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.mod = ops.Mod()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, y):
        out = self.mod(x, y)
        out = ops.cast(out, ms.float32)
        out = self.relu(out)
        out = out + 1
        return out


def test_minimum_same_shape_parallel_1():
    """
    Feature: Minimum elementwise operation in python shard.
    Description: Test Minimum with same-shape inputs in full parallel.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = MinimumNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = MinimumNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_less_equal_same_shape_parallel_2():
    """
    Feature: LessEqual elementwise operation in python shard.
    Description: Test LessEqual with same-shape inputs, bool output cast to float.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = LessEqualNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = LessEqualNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_greater_equal_same_shape_parallel_3():
    """
    Feature: GreaterEqual elementwise operation in python shard.
    Description: Test GreaterEqual with same-shape inputs.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = GreaterEqualNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = GreaterEqualNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_logical_or_same_shape_parallel_4():
    """
    Feature: LogicalOr elementwise operation in python shard.
    Description: Test LogicalOr with same-shape bool inputs.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor((np.random.randn(d, m, k) > 0).astype(np.bool_))
    y = Tensor((np.random.randn(d, m, k) > 0).astype(np.bool_))

    # Standalone
    standalone_net = LogicalOrNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = LogicalOrNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_minimum_broadcast_dim0_parallel_5():
    """
    Feature: Minimum elementwise operation with broadcasting in python shard.
    Description: Test Minimum with broadcasting on dimension 0 (y: [1, 256, 128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(1, m, k).astype(np.float32))  # Broadcasting on dim 0

    # Standalone
    standalone_net = MinimumNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("None", "cp", "mp")  # Dimension 0 cannot be sharded
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = MinimumNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_minimum_broadcast_dim1_parallel_6():
    """
    Feature: Minimum elementwise operation with broadcasting in python shard.
    Description: Test Minimum with broadcasting on dimension 1 (y: [16, 1, 128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(d, 1, k).astype(np.float32))  # Broadcasting on dim 1

    # Standalone
    standalone_net = MinimumNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "None", "mp")  # Dimension 1 cannot be sharded
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = MinimumNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_minimum_broadcast_dim2_parallel_7():
    """
    Feature: Minimum elementwise operation with broadcasting in python shard.
    Description: Test Minimum with broadcasting on dimension 2 (y: [16, 256, 1]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(d, m, 1).astype(np.float32))  # Broadcasting on dim 2

    # Standalone
    standalone_net = MinimumNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "None")  # Dimension 2 cannot be sharded
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = MinimumNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_minimum_broadcast_rank_mismatch_parallel_8():
    """
    Feature: Minimum elementwise operation with broadcasting in python shard.
    Description: Test Minimum with rank mismatch (y: [256, 128] -> [16, 256, 128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(m, k).astype(np.float32))  # 2D -> 3D broadcasting

    # Standalone
    standalone_net = MinimumNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("cp", "mp")  # 2D layout
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = MinimumNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_minimum_broadcast_scalar_like_parallel_9():
    """
    Feature: Minimum elementwise operation with broadcasting in python shard.
    Description: Test Minimum with scalar-like broadcasting (y: [128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(k).astype(np.float32))  # 1D -> 3D broadcasting

    # Standalone
    standalone_net = MinimumNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout(
        "mp",
    )  # 1D layout
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = MinimumNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_less_equal_broadcast_multi_dim_parallel_10():
    """
    Feature: LessEqual elementwise operation with broadcasting in python shard.
    Description: Test LessEqual with multi-dimension broadcasting (y: [1, 256, 1]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(
        np.random.randn(1, m, 1).astype(np.float32)
    )  # Broadcasting on dim 0 and 2

    # Standalone
    standalone_net = LessEqualNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("None", "cp", "None")  # Dimensions 0 and 2 cannot be sharded
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = LessEqualNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_minimum_partial_shard_parallel_11():
    """
    Feature: Minimum elementwise operation in python shard.
    Description: Test Minimum with partial sharding (only one dimension sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = MinimumNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("None", "None", "mp")  # Only last dimension sharded
    y_layout = layout("None", "None", "mp")
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = MinimumNet(relu_strategy=(layout("None", "None", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_mod_same_shape_parallel_12():
    """
    Feature: Mod elementwise operation in python shard.
    Description: Test Mod with same-shape inputs in full parallel.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    # Avoid dividing by zero: shift distribution away from 0
    y = Tensor((np.random.randn(d, m, k).astype(np.float32) + 0.5))

    # Standalone
    standalone_net = ModNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = ModNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_mod_broadcast_dim0_parallel_13():
    """
    Feature: Mod elementwise operation with broadcasting in python shard.
    Description: Test Mod with broadcasting on dimension 0 (y: [1, 256, 128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor((np.random.randn(1, m, k).astype(np.float32) + 0.5))  # Broadcast on dim 0

    # Standalone
    standalone_net = ModNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("None", "cp", "mp")  # Dimension 0 cannot be sharded
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = ModNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_mod_broadcast_dim1_parallel_14():
    """
    Feature: Mod elementwise operation with broadcasting in python shard.
    Description: Test Mod with broadcasting on dimension 1 (y: [16, 1, 128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor((np.random.randn(d, 1, k).astype(np.float32) + 0.5))  # Broadcast on dim 1

    # Standalone
    standalone_net = ModNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "None", "mp")  # Dimension 1 cannot be sharded
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = ModNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_mod_broadcast_dim2_parallel_15():
    """
    Feature: Mod elementwise operation with broadcasting in python shard.
    Description: Test Mod with broadcasting on dimension 2 (y: [16, 256, 1]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor((np.random.randn(d, m, 1).astype(np.float32) + 0.5))  # Broadcast on dim 2

    # Standalone
    standalone_net = ModNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("dp", "cp", "None")  # Dimension 2 cannot be sharded
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = ModNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_mod_broadcast_rank_mismatch_parallel_16():
    """
    Feature: Mod elementwise operation with broadcasting in python shard.
    Description: Test Mod with rank mismatch (y: [256, 128] -> [16, 256, 128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor((np.random.randn(m, k).astype(np.float32) + 0.5))  # 2D -> 3D broadcasting

    # Standalone
    standalone_net = ModNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    y_layout = layout("cp", "mp")  # 2D layout
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = ModNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_mod_tensor_scalar_parallel_17():
    """
    Feature: Mod elementwise operation with scalar in python shard.
    Description: Test Mod with tensor-scalar input (scalar must be constant).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = 3.0  # constant scalar

    # Standalone
    standalone_net = ModNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = ModNet(relu_strategy=(layout("dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, y)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_mod_partial_shard_parallel_18():
    """
    Feature: Mod elementwise operation in python shard.
    Description: Test Mod with partial sharding (only one dimension sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    y = Tensor((np.random.randn(d, m, k).astype(np.float32) + 0.5))

    # Standalone
    standalone_net = ModNet()
    standalone_output = standalone_net(x, y)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("None", "None", "mp")  # Only last dimension sharded
    y_layout = layout("None", "None", "mp")
    x_local = global_to_local(x, x_layout)
    y_local = global_to_local(y, y_layout)

    parallel_net = ModNet(relu_strategy=(layout("None", "None", "mp"),))
    parallel_output = parallel_net(x_local, y_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
