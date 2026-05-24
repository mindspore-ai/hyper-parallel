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
"""test torch dtensor with distributed remainder"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
standalone_input_x_np = np.random.uniform(10.0, 100.0, size=(8, 4)).astype(np.float32)
standalone_input_y_np = np.random.uniform(2.0, 10.0, size=(8, 4)).astype(np.float32)
standalone_input_y_broadcast_np = np.random.uniform(2.0, 10.0, size=(1, 4)).astype(np.float32)


def test_remainder_basic() -> None:
    """Test torch.remainder basic element-wise computation."""
    init_backend(_DEVICE_TYPE)

    standalone_x = to_device(torch.from_numpy(standalone_input_x_np), _DEVICE_TYPE)
    standalone_y = to_device(torch.from_numpy(standalone_input_y_np), _DEVICE_TYPE)
    standalone_output = torch.remainder(standalone_x, standalone_y)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_x = distribute_tensor(standalone_x, mesh, placements)
    dist_y = distribute_tensor(standalone_y, mesh, placements)

    dist_output = torch.remainder(dist_x, dist_y)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Remainder output layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "Remainder output mismatch between standalone and distributed execution"


def test_remainder_scalar() -> None:
    """Test torch.remainder with scalar denominator."""
    init_backend(_DEVICE_TYPE)

    scalar_val = 3.0
    standalone_x = to_device(torch.from_numpy(standalone_input_x_np), _DEVICE_TYPE)
    standalone_output = torch.remainder(standalone_x, scalar_val)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Shard(1))

    dist_x = distribute_tensor(standalone_x, mesh, placements)
    dist_output = torch.remainder(dist_x, scalar_val)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Scalar remainder layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "Scalar remainder output mismatch"


def test_remainder_broadcast() -> None:
    """Test torch.remainder with broadcasting."""
    init_backend(_DEVICE_TYPE)

    standalone_x = to_device(torch.from_numpy(standalone_input_x_np), _DEVICE_TYPE)
    standalone_y_broadcast = to_device(torch.from_numpy(standalone_input_y_broadcast_np), _DEVICE_TYPE)
    standalone_output = torch.remainder(standalone_x, standalone_y_broadcast)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    x_placements = (Shard(0), Replicate())
    dist_x = distribute_tensor(standalone_x, mesh, x_placements)

    y_placements = (Replicate(), Replicate())
    dist_y_broadcast = distribute_tensor(standalone_y_broadcast, mesh, y_placements)

    dist_output = torch.remainder(dist_x, dist_y_broadcast)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Broadcast remainder layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "Broadcast remainder output mismatch"


def test_remainder_operator_overload() -> None:
    """Test modulo operator (%) overload."""
    init_backend(_DEVICE_TYPE)

    standalone_x = to_device(torch.from_numpy(standalone_input_x_np), _DEVICE_TYPE)
    standalone_y = to_device(torch.from_numpy(standalone_input_y_np), _DEVICE_TYPE)
    standalone_output = standalone_x % standalone_y

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_x = distribute_tensor(standalone_x, mesh, placements)
    dist_y = distribute_tensor(standalone_y, mesh, placements)

    dist_output = dist_x % dist_y

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "Operator overload (%) output mismatch"
