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
'''test initialize parameter'''
from typing import Dict
import os
import pytest
import psutil
import numpy as np
import mindspore as ms
from mindspore import context, ops, nn, Tensor, Parameter, mint
from mindspore.common import set_seed
from mindspore.communication import create_group, get_rank
from mindspore.communication.management import init
from mindspore.nn.utils import no_init_parameters
from hyper_parallel import Layout, DTensor, init_parameters
from hyper_parallel import hsdp
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.common_net import DenseL3
from tests.mindspore.st.common_net import DenseMutiLayerNet

def _check_param_init_shape_using_hsdp(
    param_names,
    before_hsdp_shape_dict: Dict,
    after_hsdp_shape_dict: Dict,
    expect_shard_size: int,
):
    """check param shape between before_shard_shape and after_shard_shape"""
    for name in param_names:
        expect_shape = list(before_hsdp_shape_dict[name])
        expect_shape[0] = expect_shape[0] // expect_shard_size
        after_shard_shape = list(after_hsdp_shape_dict[name])
        assert expect_shape == after_shard_shape, \
            f"Checking {name}, expect_shape: {expect_shape}, actual shape: {after_shard_shape}"

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize('use_hsdp', [True, False])
def test_init_parameters_with_hsdp(use_hsdp):
    '''
    Feature: init parameters.
    Description: test init parameter interface, using hyper_parallel.hsdp.
    Expectation: Run success
    '''
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "0"
    init()

    in_channels = 128
    out_channels = 32
    hidden_size = 512
    with no_init_parameters():
        net = DenseL3(in_channels, out_channels, hidden_size)
    before_shard = {name : param.shape for name, param in net.parameters_and_names()}
    if use_hsdp:
        shard_size = 4
        threshold = 0
        optimizer_level = "level1"
        hsdp(net, shard_size, threshold, optimizer_level)

    init_parameters(net)
    after_shard = {name : param.shape for name, param in net.parameters_and_names()}
    expect_shard_size = shard_size if use_hsdp else 1
    param_names = (name for name, _ in net.parameters_and_names())
    _check_param_init_shape_using_hsdp(param_names, before_shard, after_shard, expect_shard_size)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize('use_hsdp', [True, False])
def test_init_parameters_with_dtensor_and_hsdp(use_hsdp):
    '''
    Feature: init parameters.
    Description: test init parameter interface, using hyper_parallel.hsdp + DTensor.
    Expectation: Run success
    '''
    world_size = 8
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = str(world_size)
    os.environ["RANK_ID"] = "0"
    init()
    set_seed(1)
    num_layer = 1
    hidden_size = 1024
    with no_init_parameters():
        net = DenseMutiLayerNet(hidden_size, num_layer, has_bias=False)

    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(world_size))
    layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = layout("mp", "None")
    for i in range(num_layer):
        net.layers[i].dense1.weight = Parameter(DTensor.from_local(net.layers[i].dense1.weight, w_layout))
    before_shard_local_shapes = {name : param.local_shape for name, param in net.parameters_and_names()}
    if use_hsdp:
        shard_size = 2
        hsdp(net, shard_size, threshold=0, optimizer_level="level1")
    init_parameters(net)
    after_shard_local_shapes = {name : param.local_shape for name, param in net.parameters_and_names()}
    param_names = (name for name, _ in net.parameters_and_names())
    expect_shard_size = shard_size if use_hsdp else 1
    _check_param_init_shape_using_hsdp(
        param_names,
        before_shard_local_shapes,
        after_shard_local_shapes,
        expect_shard_size
    )


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_init_parameters_mem():
    '''
    Feature: init parameters.
    Description: test init parameter with device data.
    Expectation: Run success
    '''
    num_layer = 8
    hidden_size = 16384
    context.set_context(device_target="Ascend")
    process = psutil.Process(os.getpid())
    origin_memory = process.memory_info().rss

    with no_init_parameters():
        net = DenseMutiLayerNet(hidden_size, num_layer)
    init_parameters(net)
    net_init_memory = process.memory_info().rss - origin_memory

    context.set_context(device_target="CPU")

    with no_init_parameters():
        net2 = DenseMutiLayerNet(hidden_size, num_layer)
    init_parameters(net2)
    net_init_cpu_memory = process.memory_info().rss - net_init_memory
    assert net_init_cpu_memory > net_init_memory * 5

def test_param_init_with_tp_dp():
    '''
    Feature: parameter init.
    Description: parameter init with tp and dp.
    Expectation: tp slice init data is difference and dp slice init data is same
    '''
    init()
    set_seed(1)
    num_layer = 1
    hidden_size = 1024
    with no_init_parameters():
        net = DenseMutiLayerNet(hidden_size, num_layer, has_bias=False)

    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = layout("mp", "None")
    x_layout = layout("dp", "None")
    for i in range(num_layer):
        net.layers[i].dense1.weight = Parameter(DTensor.from_local(net.layers[i].dense1.weight, w_layout))

    hsdp(net, shard_size=1)
    init_parameters(net)

    rank_id = get_rank()
    dp_rank = rank_id % 4
    rank_list = [dp_rank, dp_rank + 4]
    rank_list_str = "_".join([str(i) for i in rank_list])
    dp_group = "check_dp_param_init_group_" + rank_list_str
    create_group(dp_group, rank_list)

    tp_rank = rank_id // 4 * 4
    rank_list = list(range(tp_rank, tp_rank + 4))
    rank_list_str = "_".join([str(i) for i in rank_list])
    tp_group = "check_tp_param_init_group_" + rank_list_str
    create_group(tp_group, rank_list)
    allreduce_sum_dp = ops.AllReduce(group=dp_group)
    allreduce_sum_tp = ops.AllReduce(group=tp_group)

    for i in range(num_layer):
        # dp check same value
        data = net.layers[i].dense1.weight.to_local()
        if rank_id > 3:
            data = -data
        result = allreduce_sum_dp(data)
        dp_elem_sum = result.asnumpy().sum()
        assert dp_elem_sum == 0

        # tp check difference value
        data = net.layers[i].dense1.weight.to_local()
        if rank_id % 2 != 0:
            data = -data
        result = allreduce_sum_tp(data)
        tp_elem_sum = result.asnumpy().sum()
        assert tp_elem_sum != 0

    # check train process ok
    def forward_fn(data):
        logits = net(data)
        loss = mint.sum(logits)
        return loss

    learning_rate = 1e-3
    optimizer = nn.Adam(net.trainable_params(), learning_rate)
    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=False)

    data = Tensor(np.random.randn(1, hidden_size).astype(np.float32))
    global_data = DTensor.from_local(data, x_layout)

    _, grads = grad_fn(global_data)
    optimizer(grads)
