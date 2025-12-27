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
"""test hsdp param"""
import os
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer
from mindspore.nn.utils import no_init_parameters
from mindspore.communication.management import init
from hyper_parallel import Layout, DTensor, get_platform
from hyper_parallel.platform.mindspore.hsdp.param import MindSporeHSDPParam as HSDPParam
from hyper_parallel.core.hsdp.hsdp_utils import OptimizerLevel, HSDPConfig
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.common_net import NetWithScaler
os.environ["MS_SIMULATION_LEVEL"] = "0"
os.environ["RANK_SIZE"] = "32"
os.environ["RANK_ID"] = "9"
init()

def get_hsdp_param(net, param=None):
    """get hsdp param from net"""
    shard_size = 2
    threshold = 1
    requires_acc_grad = True
    shard_level = OptimizerLevel.SHARD_OPT
    use_pynative_hook = True
    reduce_dtype = None
    if param is None:
        param = net.weight
    hsdp_config = HSDPConfig(shard_size, threshold, requires_acc_grad, shard_level, use_pynative_hook, reduce_dtype)
    hsdp_param = HSDPParam(net, param.name, param, hsdp_config, get_platform())
    return hsdp_param

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_param_to_unsharded():
    """
    Feature: hsdp param.
    Description: change hsdp param to unshared state.
    Expectation: change hsdp param to unshared state without error.
    """
    in_channels = 256
    out_channels = 64
    net = nn.Dense(in_channels, out_channels, weight_init="ones")
    hsdp_param = get_hsdp_param(net)
    hsdp_param.to_sharded()
    assert net.weight.shape == (32, 256)
    hsdp_param.to_unsharded()
    assert net.weight.shape == (64, 256)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_scaler_param():
    """
    Feature: hsdp scaler param.
    Description: scaler param.
    Expectation: get hsdp scaler param without error.
    """

    in_channels = 256
    out_channels = 64
    net = NetWithScaler(in_channels, out_channels)
    _ = get_hsdp_param(net, net.scaler)
    hsdp_param = get_hsdp_param(net, net.dense.weight)
    hsdp_param.to_sharded()
    assert net.dense.weight.shape == (32, 256)
    hsdp_param.to_unsharded()
    assert net.dense.weight.shape == (64, 256)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_no_init_param_to_unsharded():
    """
    Feature: hsdp not init param.
    Description: change hsdp param to unshared state.
    Expectation: change hsdp param to unshared state without error.
    """
    in_channels = 256
    out_channels = 64
    with no_init_parameters():
        net = nn.Dense(in_channels, out_channels, weight_init="ones")
    hsdp_param = get_hsdp_param(net)
    hsdp_param.to_sharded()
    assert net.weight.shape == (32, 256)
    hsdp_param.to_unsharded()
    assert net.weight.shape == (64, 256)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_param_with_layout():
    """
    Feature: hsdp param with layout.
    Description: construct and init hsdp param with layout.
    Expectation: construct hsdp param without error.
    """
    in_channels = 256
    out_channels = 64
    net = nn.Dense(in_channels, out_channels, weight_init="ones")

    mesh_shape = (4, 8)
    alias_name = ("dp", "mp")
    rank_list = list(range(32))
    layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = layout("mp", "None")
    net.weight = ms.Parameter(DTensor.from_local(initializer(
        "ones", [out_channels, in_channels], dtype=ms.float32), w_layout), name="weight")
    hsdp_param = get_hsdp_param(net)
    hsdp_param.to_sharded()
    assert net.weight.local_shape == (32, 256)
    hsdp_param.to_unsharded()
    assert net.weight.local_shape == (64, 256)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_no_init_param_with_layout():
    """
    Feature: hsdp no init param with layout.
    Description: construct and init hsdp param with layout.
    Expectation: construct hsdp param without error.
    """
    in_channels = 256
    out_channels = 64
    with no_init_parameters():
        net = nn.Dense(in_channels, out_channels, weight_init="ones")

    mesh_shape = (4, 8)
    alias_name = ("dp", "mp")
    rank_list = list(range(32))
    layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = layout("mp", "None")
    net.weight = ms.Parameter(DTensor.from_local(initializer(
        "ones", [out_channels, in_channels], dtype=ms.float32), w_layout), name="weight")
    hsdp_param = get_hsdp_param(net)
    hsdp_param.to_sharded()
    assert net.weight.local_shape == (32, 256)
    hsdp_param.to_unsharded()
    assert net.weight.local_shape == (64, 256)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hsdp_param_with_two_axis_unshard():
    """
    Feature: hsdp param with layout.
    Description: hsdp param mesh shape two axis not sharded.
    Expectation: construct hsdp param without error.
    """
    in_channels = 256
    out_channels = 64
    net = nn.Dense(in_channels, out_channels, weight_init="ones")

    mesh_shape = (4, 8, 1)
    alias_name = ("dp", "mp", "xp")
    rank_list = list(range(32))
    layout = Layout(mesh_shape, alias_name, rank_list)
    w_layout = layout("mp", "None")
    net.weight = ms.Parameter(DTensor.from_local(initializer(
        "ones", [out_channels, in_channels], dtype=ms.float32), w_layout), name="weight")
    hsdp_param = get_hsdp_param(net)
    hsdp_param.to_sharded()
    assert net.weight.local_shape == (32, 256)
    hsdp_param.to_unsharded()
    assert net.weight.local_shape == (64, 256)
