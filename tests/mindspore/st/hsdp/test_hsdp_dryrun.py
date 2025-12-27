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
"""test hsdp dryrun"""

import os
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.communication.management import init
from hyper_parallel import hsdp
from hyper_parallel import Layout
from hyper_parallel import DTensor
from hyper_parallel import parallelize_value_and_grad
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.common_net import DenseNet

os.environ["MS_SIMULATION_LEVEL"] = "1"
os.environ["RANK_SIZE"] = "32"
os.environ["RANK_ID"] = "32"
init()

loss_fn = nn.MSELoss(reduction='none')

def get_forward_fn(net):
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    return forward_fn

def run_hsdp(net, data, label, optimizer_level="level1", enable_grad_accumulation=False):
    """run hsdp"""
    shard_size = 4
    threshold = 0
    hsdp(net, shard_size, threshold, optimizer_level, enable_grad_accumulation)

    grad_fn = parallelize_value_and_grad(get_forward_fn(net), net.trainable_params())

    train_steps = 2
    for i in range(train_steps):
        if i == train_steps - 1:
            net.set_requires_grad_sync(True)
        else:
            net.set_requires_grad_sync(False)
        _, grads = grad_fn(data, label)
        print(grads)

def construct_net_and_data():
    in_channels = 256
    out_channels = 64
    hidden_size = 128
    batch_size = 4
    net = DenseNet(in_channels, out_channels, hidden_size)
    data = Tensor(np.random.randn(batch_size, in_channels).astype(np.float32))
    label = Tensor(np.random.randn(batch_size, out_channels).astype(np.float32))
    return net, data, label

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_level1():
    """
    Feature: hsdp
    Description: test hsdp level1
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    run_hsdp(net, data, label, "level1")

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_level2():
    """
    Feature: hsdp
    Description: test hsdp level2
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    run_hsdp(net, data, label, "level2")

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_level3():
    """
    Feature: hsdp
    Description: test hsdp level3
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    run_hsdp(net, data, label, "level3")

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_level1_acc_grad():
    """
    Feature: hsdp
    Description: test hsdp level1 acc grad
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    run_hsdp(net, data, label, "level1", True)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_level2_acc_grad():
    """
    Feature: hsdp
    Description: test hsdp level2 acc grad
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    run_hsdp(net, data, label, "level2", True)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_level3_acc_grad():
    """
    Feature: hsdp
    Description: test hsdp level3 acc grad
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    run_hsdp(net, data, label, "level3", True)

def run_hsdp_with_layout(w1_layout, w2_layout, data_layout, label_layout):
    """
    hsdp with layout
    """
    net, data, label = construct_net_and_data()

    global_data = DTensor.from_local(data, data_layout)
    global_label = DTensor.from_local(label, label_layout)
    net.dense1.weight = ms.Parameter(DTensor.from_local(Tensor(net.dense1.weight.asnumpy()), w1_layout))
    net.dense2.weight = ms.Parameter(DTensor.from_local(Tensor(net.dense2.weight.asnumpy()), w2_layout))

    run_hsdp(net, global_data, global_label)

def get_device_layout():
    mesh_shape = (4, 8)
    alias_name = ("dp", "mp")
    rank_list = [i + 32 for i in list(range(32))]
    layout = Layout(mesh_shape, alias_name, rank_list)
    return layout

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_with_dp_layout():
    """
    Feature: hsdp
    Description: test hsdp with data parallel layout
    Expectation: run success
    """
    layout = get_device_layout()
    w_layout = layout("None", "None")
    data_layout = layout("dp", "None")
    label_layout = layout("dp", "None")
    run_hsdp_with_layout(w_layout, w_layout, data_layout, label_layout)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_with_mp_layout():
    """
    Feature: hsdp
    Description: test hsdp with model parallel layout
    Expectation: run success
    """
    layout = get_device_layout()
    w1_layout = layout("mp", "None")
    w2_layout = layout("None", "mp")
    data_layout = layout("dp", "None")
    label_layout = layout("dp", "None")
    run_hsdp_with_layout(w1_layout, w2_layout, data_layout, label_layout)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_with_exception():
    """
    Feature: hsdp
    Description: test hsdp with exception
    Expectation: run success
    """
    result = ""
    try:
        os.environ["RANK_ID"] = "64"
        layout = get_device_layout()
        w_layout = layout("None", "None")
        data_layout = layout("dp", "None")
        label_layout = layout("dp", "None")
        run_hsdp_with_layout(w_layout, w_layout, data_layout, label_layout)
    except ValueError as e:
        result = str(e)
    assert "invalid rank" in result
    os.environ["RANK_ID"] = "32"

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_custom_shard_size():
    """
    Feature: hsdp
    Description: test self define param hsdp shard size
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    net.dense1.weight.hsdp_shard_size = 2
    net.dense2.weight.hsdp_shard_size = 8
    run_hsdp(net, data, label, "level1")
    assert net.dense1.weight.shape == (64, 256)
    assert net.dense2.weight.shape == (8, 128)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_freeze_param():
    """
    Feature: hsdp
    Description: test hsdp with weight freezed
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    net.dense1.weight.hsdp_shard_size = 2
    net.dense2.weight.hsdp_shard_size = 8
    net.dense2.weight.requires_grad = False
    run_hsdp(net, data, label, "level1")
    assert net.dense1.weight.shape == (64, 256)
    assert net.dense2.weight.shape == (8, 128)
