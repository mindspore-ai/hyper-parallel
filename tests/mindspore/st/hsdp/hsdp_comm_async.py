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
"""enable hsdp comm async for performance"""
import numpy as np
from mindspore import Tensor
from mindspore.communication.management import init
from hyper_parallel import hsdp
from tests.mindspore.st.common_net import DenseMutiLayerNet
from tests.mindspore.st.hsdp.hsdp_test_common import train_with_data_label


def test_hsdp_comm_async():
    """
    Feature: hsdp enable async comm
    Description: test hsdp async comm performance
    Expectation: run success
    """
    init()
    hidden_size = 16384
    batch_size = 1024
    shard_size = 8
    shard_threshod = 0

    data = Tensor(np.random.randn(batch_size, hidden_size).astype(np.float32))
    label = Tensor(np.random.randn(batch_size, hidden_size).astype(np.float32))

    net = DenseMutiLayerNet(hidden_size, 8)
    hsdp(net, shard_size=shard_size, threshold=shard_threshod)
    sync_time = train_with_data_label(net, data, label, train_steps=50)

    net2 = DenseMutiLayerNet(hidden_size, 8)
    hsdp(net2, shard_size=shard_size, threshold=shard_threshod, comm_async=True)
    async_time = train_with_data_label(net2, data, label, comm_async=True, train_steps=50)

    print(f"async_time {async_time}  vs sync_time {sync_time}")
