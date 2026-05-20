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
"""test torch dtensor with distributed flatten"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 4).astype(np.float32)
standalone_input_3d_np = np.random.randn(8, 4, 6).astype(np.float32)
standalone_input_4d_np = np.random.randn(4, 2, 4, 6).astype(np.float32)
standalone_scalar_np = np.array(3.14, dtype=np.float32)


def test_flatten_all_dims() -> None:
    """Test flatten all dimensions with dtype float32, shard dim=0 only."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.flatten(0, -1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    x_placements = (Shard(0), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(0, -1)

    expected_layout = _build_layout(mesh, (Shard(0),), 1)
    assert dist_output.layout == expected_layout, \
        (f"Flatten all dims layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_flatten_middle_dims() -> None:
    """Test flatten middle dims 1-2 of 4D tensor, shard dim0+dim1."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_4d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.flatten(1, 2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(1, 2)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        (f"Flatten middle dims layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_flatten_unsharded() -> None:
    """Test flatten replicated middle dims 1-2 of 3D tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.flatten(1, 2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(1, 2)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        (f"Flatten unsharded layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_flatten_negative_dims() -> None:
    """Test flatten with negative dims (-2, -1) on 3D tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.flatten(-2, -1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(-2, -1)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        (f"Flatten negative dims layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_flatten_scalar() -> None:
    """Test flatten a distributed scalar tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.tensor(standalone_scalar_np), _DEVICE_TYPE)
    standalone_output = standalone_input.flatten(0, -1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = ()

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(0, -1)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
    assert dist_output.layout == expected_layout, \
        (f"Flatten scalar layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_flatten_default_args() -> None:
    """Test flatten with default args (no start_dim/end_dim)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.flatten()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten()

    expected_layout = _build_layout(mesh, (Shard(0),), 1)
    assert dist_output.layout == expected_layout, \
        (f"Flatten default args layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_flatten_single_dim() -> None:
    """Test flatten a single dimension (start_dim == end_dim)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_4d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.flatten(1, 1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(1, 1)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate(), Replicate()), 4)
    assert dist_output.layout == expected_layout, \
        (f"Flatten single dim layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_flatten_2d_to_1d() -> None:
    """Test flatten 2D tensor to 1D."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.flatten(0, 1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(0, 1)

    expected_layout = _build_layout(mesh, (Shard(0),), 1)
    assert dist_output.layout == expected_layout, \
        (f"Flatten 2d to 1d layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)
