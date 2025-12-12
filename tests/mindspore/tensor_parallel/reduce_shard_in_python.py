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
"""test reduce shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import Layout, shard
from tests.mindspore.tensor_parallel.utils import global_to_local, local_to_global


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


class SumExtNet(nn.Cell):
    """SumExt composed of bmm and ReLUs"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.sum_ext = ms.mint.sum
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, dim=None, keepdim=False, dtype=None):
        out = self.sum_ext(input=x, dim=dim, keepdim=keepdim, dtype=dtype)
        out = self.relu(out)
        out = out + 1
        return out


class MeanExtNet(nn.Cell):
    """MeanExt composed of bmm and ReLUs"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.sum_ext = ms.mint.mean
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, dim=None, keepdim=False, dtype=None):
        out = self.sum_ext(input=x, dim=dim, keepdim=keepdim, dtype=dtype)
        out = self.relu(out)
        out = out + 1
        return out


base_device_matrix = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_sum_ext_dim_partial_model_parallel_1():
    '''
    Feature: SumExt in python shard.
    Description: Test SumExt model parallel in python shard.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = SumExtNet()
    standalone_output = standalone_net(x, dim=[0, 1], keepdim=True)

    # Parallel
    layout = Layout(base_device_matrix, base_alias_name)
    x_layout = layout("None", ("dp", "cp"), "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = SumExtNet(relu_strategy=(layout("None", "None", "mp"),))
    parallel_output = parallel_net(x_local, dim=[0, 1], keepdim=True)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_mean_ext_partial_model_parallel_2():
    '''
    Feature: MeanExt in python shard.
    Description: Test MeanExt model parallel in python shard.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = MeanExtNet()
    standalone_output = standalone_net(x, dim=[0, 1], keepdim=False)

    # Parallel
    layout = Layout(base_device_matrix, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = MeanExtNet(relu_strategy=(layout("mp", ),))
    parallel_output = parallel_net(x_local, dim=[0, 1], keepdim=False)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
