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
"""test gathernd shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor, ops
from hyper_parallel import Layout, shard
from tests.mindspore.st.shard.utils import global_to_local, local_to_global


def setup_module():
    ms.set_device("Ascend")
    D.init()


class GatherNdNet(ms.nn.Cell):
    """GatherNd composed of GatherNd and ReLUs"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.gathernd = ops.GatherNd()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, indices):
        out = self.gathernd(x, indices)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_gathernd_partial_model_parallel_1():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with indices sharded by mp (last dim not sharded), params replicated.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m = 16, 64
    x = Tensor(np.random.randn(n, m).astype(np.float32))
    idx_np = np.stack([np.arange(n, dtype=np.int32), np.ones([n], dtype=np.int32)], axis=1)
    indices = Tensor(idx_np)

    # Standalone
    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)

    x_layout = layout("None", "None")
    indices_layout = layout("mp", "None")

    x_local = global_to_local(x, x_layout)
    indices_local = global_to_local(indices, indices_layout)

    parallel_net = GatherNdNet(relu_strategy=(layout("mp",),))

    out_layout = layout("mp",)
    net_stra = {"input": (x_layout, indices_layout), "output": (out_layout,)}
    shard(parallel_net, net_stra)

    parallel_output = parallel_net(x_local, indices_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_partial_data_parallel_2():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with indices sharded by dp (last dim not sharded), params replicated.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    # Input data
    n, m = 16, 64
    x = Tensor(np.random.randn(n, m).astype(np.float32))
    idx_np = np.stack([np.arange(n, dtype=np.int32), np.ones([n], dtype=np.int32)], axis=1)
    indices = Tensor(idx_np)

    # Standalone
    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)

    x_layout = layout("None", "None")
    indices_layout = layout("dp", "None")

    x_local = global_to_local(x, x_layout)
    indices_local = global_to_local(indices, indices_layout)

    parallel_net = GatherNdNet(relu_strategy=(layout("dp",),))

    out_layout = layout("dp",)
    net_stra = {"input": (x_layout, indices_layout), "output": (out_layout,)}
    shard(parallel_net, net_stra)

    parallel_output = parallel_net(x_local, indices_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_params_plain_tensor():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd where params is plain Tensor (not DTensor), indices sharded by mp.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m = 16, 64
    # x is plain Tensor, not wrapped with layout
    x = Tensor(np.random.randn(n, m).astype(np.float32))
    idx_np = np.stack([np.arange(n, dtype=np.int32), np.ones([n], dtype=np.int32)], axis=1)
    indices = Tensor(idx_np)

    # Standalone
    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)

    # x is plain Tensor, no layout transformation
    # indices_local is DTensor
    indices_layout = layout("mp", "None")
    indices_local = global_to_local(indices, indices_layout)

    parallel_net = GatherNdNet(relu_strategy=(layout("mp",),))

    # Only specify indices layout, x will be treated as replicated (layout=None)
    out_layout = layout("mp",)
    net_stra = {"input": (None, indices_layout), "output": (out_layout,)}
    shard(parallel_net, net_stra)

    parallel_output = parallel_net(x, indices_local)  # x is plain Tensor

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_k1_trailing_dims_shard_3():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with K=1, params sharded on trailing dims, indices sharded on indices[:-1].
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m, k = 16, 8, 32
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    indices = Tensor(np.arange(n, dtype=np.int32).reshape(n, 1))

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)

    x_layout = layout("None", "cp", "mp")
    indices_layout = layout("dp", "None")

    x_local = global_to_local(x, x_layout)
    indices_local = global_to_local(indices, indices_layout)

    parallel_net = GatherNdNet(relu_strategy=(layout("dp", "cp", "mp"),))

    out_layout = layout("dp", "cp", "mp")
    net_stra = {"input": (x_layout, indices_layout), "output": (out_layout,)}
    shard(parallel_net, net_stra)

    parallel_output = parallel_net(x_local, indices_local)

    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_k2_trailing_dim_shard_4():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with K=2, params sharded on trailing dims, indices sharded on indices[:-1].
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m, k = 16, 8, 32
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    idx_np = np.stack([np.arange(n, dtype=np.int32), np.zeros([n], dtype=np.int32)], axis=1)
    indices = Tensor(idx_np)

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)

    x_layout = layout("None", "None", "cp")
    indices_layout = layout("mp", "None")

    x_local = global_to_local(x, x_layout)
    indices_local = global_to_local(indices, indices_layout)

    parallel_net = GatherNdNet(relu_strategy=(layout("mp", "cp"),))

    out_layout = layout("mp", "cp")
    net_stra = {"input": (x_layout, indices_layout), "output": (out_layout,)}
    shard(parallel_net, net_stra)

    parallel_output = parallel_net(x_local, indices_local)

    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_k3_no_trailing_dims_5():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with K=3 and input_rank=3, params replicated, indices sharded on indices[:-1].
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m, k = 16, 8, 32
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))

    idx0 = np.arange(n, dtype=np.int32)
    idx1 = np.zeros([n], dtype=np.int32)
    idx2 = np.zeros([n], dtype=np.int32)
    indices = Tensor(np.stack([idx0, idx1, idx2], axis=1))

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    layout = Layout(base_mesh_shape, base_alias_name, base_rank_list)

    x_layout = layout("None", "None", "None")
    indices_layout = layout("mp", "None")

    x_local = global_to_local(x, x_layout)
    indices_local = global_to_local(indices, indices_layout)

    parallel_net = GatherNdNet(relu_strategy=(layout("mp",),))

    out_layout = layout("mp",)
    net_stra = {"input": (x_layout, indices_layout), "output": (out_layout,)}
    shard(parallel_net, net_stra)

    parallel_output = parallel_net(x_local, indices_local)

    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
