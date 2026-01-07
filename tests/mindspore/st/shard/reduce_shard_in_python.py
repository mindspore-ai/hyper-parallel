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
from mindspore import nn, Tensor, ops
from hyper_parallel import Layout, shard, DTensor, parallelize_value_and_grad
from tests.mindspore.st.shard.utils import global_to_local, local_to_global, distribute_tensor


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


class ReduceMaxNet(nn.Cell):
    """ReduceMax composed of ReduceMax and ReLUs"""

    def __init__(self, relu_strategy=None, keep_dims=False):
        super().__init__()
        self.reduce_max = ops.ReduceMax(keep_dims=keep_dims)
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None:
            stra = {"forward": {"input": relu_strategy}}
            shard(self.relu, stra)

    def construct(self, x, axis=()):
        out = self.reduce_max(x, axis)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2, 2)
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
    layout = Layout(base_mesh_shape, base_alias_name)
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
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = MeanExtNet(relu_strategy=(layout("mp", ),))
    parallel_output = parallel_net(x_local, dim=[0, 1], keepdim=False)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_reduce_max_partial_model_parallel_3():
    '''
    Feature: ReduceMax in python shard.
    Description: Test ReduceMax model parallel in python shard.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    standalone_net = ReduceMaxNet(keep_dims=False)
    standalone_output = standalone_net(x, axis=(0, 1))

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    parallel_net = ReduceMaxNet(relu_strategy=(layout("mp",),), keep_dims=False)
    parallel_output = parallel_net(x_local, axis=(0, 1))

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_reduce_max_backward_gradient_4():
    '''
    Feature: ReduceMax backward gradient in python shard.
    Description: Test ReduceMax backward gradient with distributed training.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    np.random.seed(1)
    d, m, k = 16, 256, 128
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))

    # Standalone
    class StandaloneNet(nn.Cell):
        """Standalone network for ReduceMax testing"""
        def __init__(self, k):
            super().__init__()
            self.reduce_max = ops.ReduceMax(keep_dims=False)
            self.relu = ms.nn.ReLU()
            self.weight = ms.Parameter(Tensor(np.ones([k]).astype(np.float32)), name="weight")

        def construct(self, x):
            out = self.reduce_max(x, (0, 1))
            out = self.relu(out)
            out = out * self.weight
            loss = ms.mint.sum(out)
            return loss

    standalone_net = StandaloneNet(k)

    def forward_fn(data):
        return standalone_net(data)

    optimizer = nn.SGD(standalone_net.trainable_params(), learning_rate=0.01)
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    _, standalone_grads = grad_fn(x)

    # Parallel
    layout = Layout(base_mesh_shape, base_alias_name)
    x_layout = layout("dp", "cp", "mp")
    x_local = global_to_local(x, x_layout)

    _, _, mp = base_mesh_shape
    local_k = k // mp

    class ParallelNet(nn.Cell):
        """Parallel network for ReduceMax testing with distributed parameters"""
        def __init__(self, k, layout):
            super().__init__()
            self.reduce_max = ops.ReduceMax(keep_dims=False)
            self.relu = ms.nn.ReLU()

            weight_layout = layout("mp",)
            rank = D.get_rank()

            if rank == 0:
                full_weight = Tensor(np.ones([k]).astype(np.float32))
            else:
                full_weight = Tensor(np.zeros([k]).astype(np.float32))

            weight_dtensor = distribute_tensor(full_weight, weight_layout, src_data_rank=0)
            self.weight = ms.Parameter(weight_dtensor.to_local(), name="weight")

            # Shard relu operator
            relu_stra = {
                "forward": {
                    "input": (layout("mp",),),
                    "output": (layout("mp",),)
                }
            }
            shard(self.relu, relu_stra)

        def construct(self, x):
            out = self.reduce_max(x, (0, 1))
            out = self.relu(out)
            out = out * self.weight
            loss = ms.mint.sum(out)

            if isinstance(loss, DTensor):
                loss = loss.to_local()

            return loss

    parallel_net = ParallelNet(k, layout)

    out_layout = layout()
    net_stra = {"input": (x_layout,), "output": (out_layout,)}
    shard(parallel_net, net_stra)

    parallel_optimizer = nn.SGD(parallel_net.trainable_params(), learning_rate=0.01)

    parallel_grad_fn = parallelize_value_and_grad(parallel_net, parallel_optimizer.parameters, sens=None)

    _, parallel_grads = parallel_grad_fn(x_local)

    # Validate gradient slice
    rank = D.get_rank()
    mp_idx = rank % mp

    standalone_grad = standalone_grads[0].asnumpy()
    start_idx = mp_idx * local_k
    end_idx = start_idx + local_k
    standalone_grad_slice = standalone_grad[start_idx:end_idx]

    parallel_grad = parallel_grads[0].asnumpy()

    assert np.allclose(standalone_grad_slice, parallel_grad, 1e-3, 1e-3)
