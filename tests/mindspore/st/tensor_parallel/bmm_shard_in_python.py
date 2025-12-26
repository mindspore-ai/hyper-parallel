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
"""bmm shard in python"""

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import Layout, shard
from tests.mindspore.st.tensor_parallel.utils import global_to_local, local_to_global


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


class BmmExtNet(nn.Cell):
    """BmmExtNet composed of bmm and ReLUs"""

    def __init__(self, relu_strategy=None):
        super().__init__()
        self.bmm = ms.mint.bmm
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, w):
        out = self.bmm(x, w)
        out = self.relu(out)
        out = out + 1
        return out


class BmmNet(nn.Cell):
    """BmmNet composed of BatchMatMul and ReLUs"""

    def __init__(self, transpose_a=False, transpose_b=False, relu_strategy=None):
        super().__init__()
        self.bmm = ms.ops.BatchMatMul(transpose_a=transpose_a, transpose_b=transpose_b)
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, w):
        out = self.bmm(x, w)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")
base_rank_list = list(range(8))


def test_bmm_ext_partial_model_parallel():
    '''
    Feature: BatchMatMulExt in python shard.
    Description: Test BatchMatMulExt model parallel in python shard.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k, n = 16, 256, 128, 64
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    w = Tensor(np.random.randn(d, k, n).astype(np.float32))

    # Standalone
    standalone_net = BmmExtNet()
    standalone_output = standalone_net(x, w)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    w_layout = layout("dp", "mp", "None")
    x_local = global_to_local(x, x_layout)
    w_local = global_to_local(w, w_layout)

    parallel_net = BmmExtNet(relu_strategy=(layout("dp", "None", "None"),))
    parallel_output = parallel_net(x_local, w_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_bmm_partial_transpose_model_parallel():
    '''
    Feature: BatchMatMul in python shard.
    Description: Test BatchMatMul model parallel in python shard with transpose.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k, n = 16, 256, 128, 64
    x = Tensor(np.random.randn(d, k, m).astype(np.float32))
    w = Tensor(np.random.randn(d, k, n).astype(np.float32))

    # Standalone
    standalone_net = BmmNet(transpose_a=True, transpose_b=False)
    standalone_output = standalone_net(x, w)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "mp", "cp")
    w_layout = layout("dp", "mp", "None")
    x_local = global_to_local(x, x_layout)
    w_local = global_to_local(w, w_layout)

    parallel_net = BmmNet(transpose_a=True, transpose_b=False, relu_strategy=(layout("dp", "None", "None"),))
    parallel_output = parallel_net(x_local, w_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
