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
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import torch
# pylint: disable=W0611
import torch_npu
from hyper_parallel import DeviceMesh, init_device_mesh
from hyper_parallel.platform.platform import get_torch_platform
from tests.torch.common_net import FullyShardTestNet, DenseNet
from tests.torch.utils import init_dist
from tests.torch.hsdp.hsdp_test_common import train
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.platform.torch.fully_shard.utils import MixedPrecisionPolicy


def test_fully_shard_01():
    """
    Feature: Test fully_shard with simple network, optimization level is default ZeRO-3
    Description: The DenseNet only have one weight and no bias, verify the basic process of fully_shard
    Expectation: run successfully
    """
    batch_size = 4
    hidden_size = 32
    hidden_out = 64
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
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
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    multi_layer_net = FullyShardTestNet(32, dense_layer_num, has_bias=False)
    for dense_layer in multi_layer_net.dense_layers.layers:
        # Wrap each layer with fully_shard
        fully_shard(dense_layer,
                    mesh=mesh,
                    reshard_after_forward=True,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                                   output_dtype=torch.float32, cast_forward_inputs=True)
                    )
    # 对最顶层的Module也处理，管理剩下的参数。
    fully_shard(multi_layer_net,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                               output_dtype=torch.float32, cast_forward_inputs=True)
                )
    input_data = torch.rand(batch_size, hidden_size).npu()
    with SkipDTensorDispatch():
        train(multi_layer_net, input_data, comm_async=True, train_steps=2)
