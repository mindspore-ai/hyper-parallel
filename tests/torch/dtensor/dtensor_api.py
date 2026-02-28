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
"""test torch dtensor api"""
import torch
# pylint: disable=W0611
import torch_npu  # 昇腾NPU核心适配
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist


def test_numel():
    """
    Feature: DTensor.numel
    Description: Verify numel() returns the global number of elements.
    Expectation: Success.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp")
    )

    # Case 1: Replicate on all dims
    global_shape = (8, 8)
    local_tensor = torch.ones(global_shape).npu()
    dtensor = DTensor.from_local(local_tensor, mesh, [Replicate(), Replicate()])
    assert dtensor.numel() == 64, f"Expected numel 64, got {dtensor.numel()}"
    assert dtensor.numel() == local_tensor.numel()

    # Case 2: Shard on dim 0
    local_tensor = torch.ones(8, 8).npu()
    dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0), Replicate()])
    # Global elements: 16 * 8 = 128
    assert dtensor.numel() == 128, f"Expected numel 128, got {dtensor.numel()}"
    assert dtensor.numel() == local_tensor.numel() * 2

    # Case 3: Shard on dim 1
    local_tensor = torch.ones(8, 8).npu()
    dtensor = DTensor.from_local(local_tensor, mesh, [Replicate(), Shard(1)])
    # Global elements: 8 * 32 = 256
    assert dtensor.numel() == 256, f"Expected numel 256, got {dtensor.numel()}"
    assert dtensor.numel() == local_tensor.numel() * 4

    # Case 4: Shard on both dims
    local_tensor = torch.ones(8, 8).npu()
    dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0), Shard(1)])
    assert dtensor.numel() == 512, f"Expected numel 512, got {dtensor.numel()}"
    assert dtensor.numel() == local_tensor.numel() * 2 * 4


def test_to():
    """
    Feature: DTensor.to
    Description: Verify to() correctly changes dtype/device and preserves distribution properties.
    Expectation: Success.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp")
    )

    local_tensor = torch.ones(8, 8, dtype=torch.float32).npu()
    placements = [Shard(0), Replicate()]
    dtensor = DTensor.from_local(local_tensor, mesh, placements)

    # Case 1: Change dtype to float16
    dtensor_fp16 = dtensor.to(torch.float16)
    assert dtensor_fp16.dtype == torch.float16
    assert isinstance(dtensor_fp16, DTensor)
    assert dtensor_fp16.device_mesh == mesh
    assert dtensor_fp16.placements == tuple(placements)
    assert dtensor_fp16.shape == dtensor.shape

    # Case 2: Change device to NPU
    current_device = torch.device(f"npu:{torch.npu.current_device()}")
    dtensor_device = dtensor.to(current_device)
    assert dtensor_device.device.type == "npu"
    assert isinstance(dtensor_device, DTensor)
    assert dtensor_device.device_mesh == mesh

    # Case 3: Change dtype to int32 and device to NPU
    dtensor_mixed = dtensor.to(device=current_device, dtype=torch.int32)
    assert dtensor_mixed.dtype == torch.int32
    assert dtensor_mixed.device.type == "npu"
    assert isinstance(dtensor_mixed, DTensor)


def test_dtensor_api():
    init_dist()
    test_numel()
    test_to()
