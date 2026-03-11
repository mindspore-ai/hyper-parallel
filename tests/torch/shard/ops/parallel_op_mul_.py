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
"""test torch dtensor with distributed inplace mul (mul_)"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input_2d_np_a = np.random.randn(8, 16).astype(np.float32)
standalone_input_2d_np_b = np.random.randn(8, 16).astype(np.float32)

standalone_input_3d_np_a = np.random.randn(4, 8, 6).astype(np.float32)
standalone_input_3d_np_b = np.random.randn(1, 8, 6).astype(np.float32)


def test_distributed_mul_inplace_basic():
    """
    Feature: dtensor + torch.Tensor.mul_ basic
    Description:
        - Inplace multiplication between two tensors with the same shape and sharding layout.
        - Input a: shape (8, 16) sharded on dim=0.
        - Input b: shape (8, 16) sharded on dim=0.
        - Output layout must preserve the original layout of input a.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference computation
    standalone_a = torch.from_numpy(standalone_input_2d_np_a).npu()
    standalone_b = torch.from_numpy(standalone_input_2d_np_b).npu()
    expected_output = standalone_a.clone()
    expected_output.mul_(standalone_b)

    # Distributed setup
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_a = distribute_tensor(standalone_a.clone(), mesh, placements)
    dist_b = distribute_tensor(standalone_b.clone(), mesh, placements)

    # Perform inplace mul
    dist_a.mul_(dist_b)

    # Layout validation: inplace operation should not change the caller's layout
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_a.layout == expected_layout, \
        f"mul_ layout mismatch: expected {expected_layout}, got {dist_a.layout}"

    # Numerical validation via gathering
    gathered_a = local_to_global(dist_a)
    assert torch.allclose(
        expected_output, gathered_a, atol=1e-5
    ), "mul_ output mismatch between standalone and distributed execution"


def test_distributed_mul_inplace_broadcast():
    """
    Feature: dtensor + torch.Tensor.mul_ with broadcasting
    Description:
        - Inplace multiplication where the second tensor requires broadcasting.
        - Input a: shape (4, 8, 6) sharded on dim=0 and dim=1.
        - Input b: shape (1, 8, 6) fully replicated or partially sharded.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_a = torch.from_numpy(standalone_input_3d_np_a).npu()
    standalone_b = torch.from_numpy(standalone_input_3d_np_b).npu()

    expected_output = standalone_a.clone()
    expected_output.mul_(standalone_b)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # a is sharded on dim 0 and 1
    a_placements = (Shard(0), Shard(1), Replicate())
    # b is sharded on dim 1, since dim 0 is size 1
    b_placements = (Replicate(), Shard(1), Replicate())

    dist_a = distribute_tensor(standalone_a.clone(), mesh, a_placements)
    dist_b = distribute_tensor(standalone_b.clone(), mesh, b_placements)

    # Perform inplace mul
    dist_a.mul_(dist_b)

    # Layout validation
    expected_layout = _build_layout(mesh, a_placements, 3)
    assert dist_a.layout == expected_layout, \
        f"mul_ broadcast layout mismatch: expected {expected_layout}, got {dist_a.layout}"

    gathered_a = local_to_global(dist_a)
    assert torch.allclose(
        expected_output, gathered_a, atol=1e-5
    ), "mul_ broadcast output mismatch between standalone and distributed execution"


def test_distributed_mul_inplace_scalar():
    """
    Feature: dtensor + torch.Tensor.mul_ with scalar
    Description:
        - Inplace multiplication of a sharded tensor by a scalar value.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_a = torch.from_numpy(standalone_input_2d_np_a).npu()
    scalar_val = 3.14

    expected_output = standalone_a.clone()
    expected_output.mul_(scalar_val)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Shard(1))

    dist_a = distribute_tensor(standalone_a.clone(), mesh, placements)
    dist_a.mul_(scalar_val)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_a.layout == expected_layout, \
        f"mul_ scalar layout mismatch: expected {expected_layout}, got {dist_a.layout}"

    gathered_a = local_to_global(dist_a)
    assert torch.allclose(
        expected_output, gathered_a, atol=1e-5
    ), "mul_ scalar output mismatch between standalone and distributed execution"
