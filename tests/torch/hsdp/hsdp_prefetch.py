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
"""set hsdp prefetch cell for performance"""
import torch
# pylint: disable=W0611
import torch_npu
from hyper_parallel import hsdp
from tests.torch.common_net import DenseMutiLayerNet
from tests.torch.utils import init_dist
from tests.torch.hsdp.hsdp_test_common import train


def construct_net_and_data():
    hidden_size = 16384
    batch_size = 2048
    net = DenseMutiLayerNet(hidden_size, 8)
    data = torch.rand(batch_size, hidden_size).npu()
    return net, data


def test_hsdp_forward_prefetch():
    """
    Feature: hsdp prefetch
    Description: test hsdp forward prefetch
    Expectation: run success
    """
    init_dist()
    net, data = construct_net_and_data()
    shard_size = 8
    threshold = 0
    for i, layer in enumerate(net.layers):
        hsdp(layer, shard_size, threshold)
    ref_time = train(net, data)

    num_layer_to_prefetch = 1
    for i, layer in enumerate(net.layers):
        if i >= len(net.layers) - num_layer_to_prefetch:
            break
        layers_to_perfetch = [net.layers[j] for j in range(i + 1, i + 1 + num_layer_to_prefetch)]
        layer.set_forward_prefetch_cells(layers_to_perfetch)
    prefetch_time = train(net, data)

    print(f"prefetch {prefetch_time}  vs base {ref_time}")


def test_hsdp_backward_prefetch():
    """
    Feature: hsdp prefetch
    Description: test hsdp forward prefetch
    Expectation: run success
    """
    init_dist()
    net, data = construct_net_and_data()
    shard_size = 8
    threshold = 0
    for i, layer in enumerate(net.layers):
        hsdp(layer, shard_size, threshold, optimizer_level="level3")
    ref_time = train(net, data)

    num_layer_to_prefetch = 1
    for i, layer in enumerate(net.layers):
        if i < num_layer_to_prefetch:
            continue
        layers_to_perfetch = [net.layers[j] for j in range(i - num_layer_to_prefetch, i)]
        layer.set_backward_prefetch_cells(layers_to_perfetch)
    prefetch_time = train(net, data)

    print(f"pefetch {prefetch_time}  vs base {ref_time}")
