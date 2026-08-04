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
"""test fully_shard api"""
# pylint: disable=C0413,C0412
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import torch
# pylint: disable=W0611
import torch_npu
from hyper_parallel import DeviceMesh, init_device_mesh, SkipDTensorDispatch
from hyper_parallel.platform.platform import get_torch_platform
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import FullyShardTestNet, DenseNet, BufferTestNet, MetaInitNet
from tests.torch.utils import init_dist
from tests.torch.fully_shard.fully_shard_common import train


def test_fully_shard_01():
    """
    Feature: Test fully_shard with simple network, optimization level is default ZeRO-3
    Description: The DenseNet has only one weight and no bias, verify the basic process of fully_shard
    Expectation: run successfully
    """
    batch_size = 4
    hidden_size = 32
    hidden_out = 64
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    dense_model = DenseNet(hidden_size, hidden_out, has_bias=False)
    dense_model = fully_shard(dense_model,
                              mesh=mesh,
                              reshard_after_forward=True,
                              mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                                             output_dtype=torch.float32, cast_forward_inputs=True)
                              )
    input_data = torch.rand(batch_size, hidden_size).npu()
    with SkipDTensorDispatch():
        train(dense_model, input_data, comm_async=True, train_steps=2)


def test_fully_shard_02():
    """
    Feature: Test fully_shard with multi-layer network, optimization level is default ZeRO-3
    Description: The FullyShardTestNet is a multi-layer module, verify the basic process of fully_shard
    Expectation: run successfully
    """
    batch_size = 4
    hidden_size = 32
    dense_layer_num = 2
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    multi_layer_net = FullyShardTestNet(32, dense_layer_num, has_bias=False)
    for dense_layer in multi_layer_net.dense_layers.layers:
        # Wrap each layer with fully_shard
        fully_shard(dense_layer,
                    mesh=mesh,
                    reshard_after_forward=True,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                                   output_dtype=torch.float32, cast_forward_inputs=True)
                    )
    # handle top-level Module too, manage remaining params
    fully_shard(multi_layer_net,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                               output_dtype=torch.float32, cast_forward_inputs=True)
                )
    input_data = torch.rand(batch_size, hidden_size).npu()
    with SkipDTensorDispatch():
        train(multi_layer_net, input_data, comm_async=True, train_steps=2)


def test_fully_shard_03():
    """
    Feature: Test fully_shard with network that has buffers, initialized on CPU
    Description: BufferTestNet contains BatchNorm (running_mean, running_var buffers).
    Model is on CPU at init, _move_states_to_device moves params and buffers to NPU.
    Expectation: run successfully
    """
    batch_size = 4
    hidden_size = 32
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    net = BufferTestNet(hidden_size=hidden_size)
    net = fully_shard(
        net,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
    )
    input_data = torch.rand(batch_size, hidden_size).npu()
    with SkipDTensorDispatch():
        train(net, input_data, comm_async=True, train_steps=2)


class DenseMutiLayerNet(torch.nn.Module):
    """dense net with configurable layer number"""
    def __init__(self, hidden_size, has_bias=True):
        super().__init__()
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
        layer = DenseNet(hidden_size, hidden_size, has_bias)
        self.layers1 = fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
                cast_forward_inputs=True,
            ),
        )
        layer = DenseNet(hidden_size, hidden_size, has_bias).to(torch.bfloat16)
        self.layers2 = fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
                cast_forward_inputs=True,
            ),
        )

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = torch.sum(x)
        return x


def test_fully_shard_04():
    """
    Feature: Test fully_shard with networks that have different orig_dtype, initialized on CPU
    Description: DenseMutiLayerNet contains 2 DenseNet.
    Model is on CPU at init, _move_states_to_device moves params and buffers to NPU.
    Expectation: run successfully
    """
    batch_size = 4
    hidden_size = 32
    init_dist()
    net = DenseMutiLayerNet(hidden_size, 2)
    input_data = torch.rand(batch_size, hidden_size).npu()
    with SkipDTensorDispatch():
        train(net, input_data, comm_async=True, train_steps=2)


def test_fully_shard_from_group_mesh():
    """
    Feature: When mesh created by from_group, test fully_shard with simple network, optimization level is default ZeRO-3
    Description: The DenseNet has only one weight and no bias, verify the basic process of fully_shard
    Expectation: run successfully
    """
    batch_size = 4
    hidden_size = 32
    hidden_out = 64
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=["dp"])
    dp_group = mesh.get_group()
    device_mesh = DeviceMesh.from_group(dp_group, device_type="npu", mesh_dim_names=["shard"])
    dense_model = DenseNet(hidden_size, hidden_out, has_bias=False)
    dense_model = fully_shard(dense_model,
                              mesh=device_mesh,
                              reshard_after_forward=True,
                              mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                                             output_dtype=torch.float32, cast_forward_inputs=True)
                              )
    input_data = torch.rand(batch_size, hidden_size).npu()
    with SkipDTensorDispatch():
        train(dense_model, input_data, comm_async=True, train_steps=2)


def test_fully_shard_none_mesh():
    """
    Feature: When pass none mesh, test fully_shard with simple network, optimization level is default ZeRO-3
    Description: The DenseNet has only one weight and no bias, verify the basic process of fully_shard
    Expectation: run successfully
    """
    batch_size = 4
    hidden_size = 32
    hidden_out = 64
    init_dist()
    dense_model = DenseNet(hidden_size, hidden_out, has_bias=False)
    dense_model = fully_shard(dense_model,
                              mesh=None,
                              reshard_after_forward=True,
                              mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                                             output_dtype=torch.float32, cast_forward_inputs=True)
                              )
    input_data = torch.rand(batch_size, hidden_size).npu()
    with SkipDTensorDispatch():
        train(dense_model, input_data, comm_async=True, train_steps=2)
