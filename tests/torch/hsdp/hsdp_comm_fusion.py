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
"""enable hsdp comm fusion for performance"""
import torch
# pylint: disable=W0611
import torch_npu
from hyper_parallel import hsdp
from tests.torch.common_net import DenseMutiLayerNet
from tests.torch.utils import init_dist
from tests.torch.hsdp.hsdp_test_common import train
torch.manual_seed(0)


def test_hsdp_comm_fusion():
    """
    Feature: hsdp enable comm fusion
    Description: test hsdp comm fusion performance
    Expectation: run success
    """
    init_dist()
    hidden_size = 1024
    batch_size = 16384
    shard_size = 8
    shard_threshold = 0
    comm_async = True
    data = torch.rand(batch_size, hidden_size).npu()

    net = DenseMutiLayerNet(hidden_size, 8)
    hsdp(net, shard_size=shard_size, threshold=shard_threshold, comm_async=comm_async)
    no_fusion_time = train(net, data, comm_async=comm_async, train_steps=10)

    net2 = DenseMutiLayerNet(hidden_size, 8)
    hsdp(net2, shard_size=shard_size, threshold=shard_threshold, comm_async=comm_async, comm_fusion=True)
    fusion_time = train(net, data, comm_async=comm_async, train_steps=10)

    print(f"fusion_time {fusion_time}  vs no_fusion_time {no_fusion_time}")
