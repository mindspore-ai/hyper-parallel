# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test expand_dims shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor, ops
from hyper_parallel import Layout, shard
from tests.mindspore.st.tensor_parallel.utils import global_to_local, local_to_global


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


class ExpandDimsNet(nn.Cell):
    """ExpandDims composed of ExpandDims and ReLUs"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.expand_dims = ops.ExpandDims()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, axis):
        out = self.expand_dims(x, axis)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_expanddims_data_parallel_1():
    '''
    Feature: ExpandDims in python shard.
    Description: Test ExpandDims data parallel in python shard.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = ExpandDimsNet()
    standalone_output = standalone_net(x, axis=0)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = ExpandDimsNet(relu_strategy=(layout("None", "dp", "cp", "mp"),))
    parallel_output = parallel_net(x_local, axis=0)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_expanddims_model_parallel_2():
    '''
    Feature: ExpandDims in python shard.
    Description: Test ExpandDims model parallel in python shard.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = ExpandDimsNet()
    standalone_output = standalone_net(x, axis=1)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("None", "None", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = ExpandDimsNet(relu_strategy=(layout("None", "None", "None", "mp"),))
    parallel_output = parallel_net(x_local, axis=1)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_expanddims_hybrid_parallel_3():
    '''
    Feature: ExpandDims in python shard.
    Description: Test ExpandDims hybrid parallel in python shard.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = ExpandDimsNet()
    standalone_output = standalone_net(x, axis=-1)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = ExpandDimsNet(relu_strategy=(layout("dp", "cp", "mp", "None"),))
    parallel_output = parallel_net(x_local, axis=-1)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_expanddims_insert_middle_4():
    '''
    Feature: ExpandDims in python shard.
    Description: Test ExpandDims insert in middle with hybrid parallel.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = ExpandDimsNet()
    standalone_output = standalone_net(x, axis=2)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = ExpandDimsNet(relu_strategy=(layout("dp", "cp", "None", "mp"),))
    parallel_output = parallel_net(x_local, axis=2)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_expanddims_negative_axis_5():
    '''
    Feature: ExpandDims in python shard.
    Description: Test ExpandDims with negative axis.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = ExpandDimsNet()
    standalone_output = standalone_net(x, axis=-2)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = ExpandDimsNet(relu_strategy=(layout("dp", "cp", "None", "mp"),))
    parallel_output = parallel_net(x_local, axis=-2)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
