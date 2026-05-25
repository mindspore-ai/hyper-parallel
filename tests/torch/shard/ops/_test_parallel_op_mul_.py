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
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
standalone_input_2d_np_a = np.random.randn(8, 16).astype(np.float32)
standalone_input_2d_np_b = np.random.randn(8, 16).astype(np.float32)

standalone_input_3d_np_a = np.random.randn(4, 8, 6).astype(np.float32)
standalone_input_3d_np_b = np.random.randn(1, 8, 6).astype(np.float32)


def test_mul_inplace_basic() -> None:
    """Test parallel op mul_ with identical sharding layouts."""
    init_backend(_DEVICE_TYPE)

    standalone_a = to_device(torch.from_numpy(standalone_input_2d_np_a), _DEVICE_TYPE)
    standalone_b = to_device(torch.from_numpy(standalone_input_2d_np_b), _DEVICE_TYPE)
    expected_output = standalone_a.clone()
    expected_output.mul_(standalone_b)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_a = distribute_tensor(standalone_a.clone(), mesh, placements)
    dist_b = distribute_tensor(standalone_b.clone(), mesh, placements)

    dist_a.mul_(dist_b)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_a.layout == expected_layout, \
        f"mul_ layout mismatch: expected {expected_layout}, got {dist_a.layout}"

    gathered_a = local_to_global(dist_a)
    assert torch.allclose(
        expected_output, gathered_a, atol=1e-5
    ), "mul_ output mismatch between standalone and distributed execution"


def test_mul_inplace_broadcast() -> None:
    """Test parallel op mul_ with broadcastable shapes."""
    init_backend(_DEVICE_TYPE)

    standalone_a = to_device(torch.from_numpy(standalone_input_3d_np_a), _DEVICE_TYPE)
    standalone_b = to_device(torch.from_numpy(standalone_input_3d_np_b), _DEVICE_TYPE)

    expected_output = standalone_a.clone()
    expected_output.mul_(standalone_b)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    a_placements = (Shard(0), Shard(1), Replicate())
    b_placements = (Replicate(), Shard(1), Replicate())

    dist_a = distribute_tensor(standalone_a.clone(), mesh, a_placements)
    dist_b = distribute_tensor(standalone_b.clone(), mesh, b_placements)

    dist_a.mul_(dist_b)

    expected_layout = _build_layout(mesh, a_placements, 3)
    assert dist_a.layout == expected_layout, \
        f"mul_ broadcast layout mismatch: expected {expected_layout}, got {dist_a.layout}"

    gathered_a = local_to_global(dist_a)
    assert torch.allclose(
        expected_output, gathered_a, atol=1e-5
    ), "mul_ broadcast output mismatch between standalone and distributed execution"


def test_mul_inplace_scalar() -> None:
    """Test parallel op mul_ with scalar inputs."""
    init_backend(_DEVICE_TYPE)

    standalone_a = to_device(torch.from_numpy(standalone_input_2d_np_a), _DEVICE_TYPE)
    scalar_val = 3.14

    expected_output = standalone_a.clone()
    expected_output.mul_(scalar_val)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
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
