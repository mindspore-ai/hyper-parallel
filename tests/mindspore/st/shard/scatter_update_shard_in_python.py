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
"""test scatter update shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor, ops
from hyper_parallel import Layout, shard
from tests.mindspore.st.shard.utils import global_to_local, local_to_global


def setup_module():
    ms.set_device("Ascend")
    D.init()


class ScatterUpdateNet(nn.Cell):
    """ScatterUpdate composed of scatter_update and ReLU"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.scatter_update = ops.ScatterUpdate()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, input_x, indices, updates):
        out = self.scatter_update(input_x, indices, updates)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_scatter_update_data_parallel_1():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate data parallel on dim 1.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n = 8

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n,)).astype(np.int32))
    updates = Tensor(np.random.randn(n, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    input_layout = layout("None", "dp", "None")
    indices_layout = layout("None",)
    updates_layout = layout("None", "dp", "None")

    input_local = global_to_local(input_x, input_layout)
    indices_local = global_to_local(indices, indices_layout)
    updates_local = global_to_local(updates, updates_layout)

    parallel_net = ScatterUpdateNet(relu_strategy=(input_layout,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_scatter_update_model_parallel_2():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate model parallel on dim 2.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n = 8

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n,)).astype(np.int32))
    updates = Tensor(np.random.randn(n, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    input_layout = layout("None", "None", "mp")
    indices_layout = layout("None",)
    updates_layout = layout("None", "None", "mp")

    input_local = global_to_local(input_x, input_layout)
    indices_local = global_to_local(indices, indices_layout)
    updates_local = global_to_local(updates, updates_layout)

    parallel_net = ScatterUpdateNet(relu_strategy=(input_layout,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_scatter_update_hybrid_parallel_3():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate hybrid parallel on dim 1 and 2.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n = 8

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n,)).astype(np.int32))
    updates = Tensor(np.random.randn(n, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    input_layout = layout("None", "dp", "mp")
    indices_layout = layout("None",)
    updates_layout = layout("None", "dp", "mp")

    input_local = global_to_local(input_x, input_layout)
    indices_local = global_to_local(indices, indices_layout)
    updates_local = global_to_local(updates, updates_layout)

    parallel_net = ScatterUpdateNet(relu_strategy=(input_layout,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_scatter_update_multi_dim_indices_4():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate with 2D indices.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n, o = 4, 2

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n, o)).astype(np.int32))
    updates = Tensor(np.random.randn(n, o, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    input_layout = layout("None", "dp", "mp")
    indices_layout = layout("None", "None")
    updates_layout = layout("None", "None", "dp", "mp")

    input_local = global_to_local(input_x, input_layout)
    indices_local = global_to_local(indices, indices_layout)
    updates_local = global_to_local(updates, updates_layout)

    parallel_net = ScatterUpdateNet(relu_strategy=(input_layout,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_scatter_update_replicate_all_5():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate with full replication.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n = 8

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n,)).astype(np.int32))
    updates = Tensor(np.random.randn(n, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    input_layout = layout("None", "None", "None")
    indices_layout = layout("None",)
    updates_layout = layout("None", "None", "None")

    input_local = global_to_local(input_x, input_layout)
    indices_local = global_to_local(indices, indices_layout)
    updates_local = global_to_local(updates, updates_layout)

    parallel_net = ScatterUpdateNet(relu_strategy=(input_layout,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
