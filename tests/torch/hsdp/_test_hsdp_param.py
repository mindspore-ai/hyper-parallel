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
from hyper_parallel import DeviceMesh, init_device_mesh
from hyper_parallel.platform.platform import get_torch_platform
from hyper_parallel.platform.torch.hsdp.param import TorchHSDPParam
from tests.torch.common_net import DenseNet
from tests.torch.utils import init_dist
from tests.torch.hsdp.hsdp_test_common import train

def test_hsdp_param_1d_mesh_without_dtensor():
    """
    Feature: HSDPParam.
    Description: Test HSDPParam with 1D-mesh and Parameter is not DTensor
    Expectation: assertion pass.
    """
    init_dist()
    device_mesh = init_device_mesh(mesh_shape=(8,), alias_name=("dp",))
    net = DenseNet(32, 64)
    hsdp_param = TorchHSDPParam(net, net.weight.name, net.weight, config=None, mesh=device_mesh, platform=get_torch_platform())
    print(f"hsdp_param shape: {hsdp_param.param.shape}")
    print(f"shard group: {hsdp_param.sharded_group_info}")
    print(f"replicate group: {hsdp_param.unsharded_group_info}")

def test_hsdp_param_2d_mesh_without_dtensor():
    """
    Feature: HSDPParam.
    Description: Test HSDPParam with 2D-mesh and Parameter is not DTensor
    Expectation: assertion pass.
    """
    init_dist()
    device_mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "op"))
    net = DenseNet(32, 64)
    hsdp_param = TorchHSDPParam(net, net.weight.name, net.weight, config=None, mesh=device_mesh, platform=get_torch_platform())
    print(f"hsdp_param shape: {hsdp_param.param.shape}")
    print(f"shard group: {hsdp_param.sharded_group_info}")
    print(f"replicate group: {hsdp_param.unsharded_group_info}")