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
"""test torch dtensor with distributed repeat"""

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
standalone_input_2d_np = np.random.randn(8, 2).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 3, 6).astype(np.float32)
standalone_scalar_like_np = np.array([[3.14]], dtype=np.float32)
standalone_input_4d_np = np.random.randn(2, 3, 4, 5).astype(np.float32)


def test_repeat_basic_unsharded() -> None:
    """Test repeat basic unsharded."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.repeat(1, 3)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 3)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Repeat output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat output mismatch between standalone and distributed execution"


def test_repeat_3d() -> None:
    """Test repeat with 3D tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.repeat(1, 4, 1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 4, 1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"Repeat 3D layout failed: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat 3D output mismatch"


def test_repeat_scalar_tensor() -> None:
    """Test repeat scalar expansion."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_scalar_like_np), _DEVICE_TYPE)
    standalone_output = standalone_input.repeat(3, 4, 5)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(3, 4, 5)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Scalar repeat layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Scalar repeat output mismatch"


def test_repeat_replicated_dim() -> None:
    """Test repeat a replicated dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(np.random.randn(4, 4).astype(np.float32)), _DEVICE_TYPE)
    standalone_output = standalone_input.repeat(1, 2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 2)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Repeat replicated dim layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat replicated dim output mismatch"


def test_repeat_zero_times() -> None:
    """Test repeat zero times."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.repeat(1, 0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 0)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Repeat zero times layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    assert dist_output.shape == (8, 0), f"Expected shape (8, 0), got {dist_output.shape}"
    assert dist_output.local_shape[1] == 0, f"Expected local shape dim 1 to be 0, got {dist_output.local_shape}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat zero times output mismatch"


def test_repeat_4d_input() -> None:
    """Test repeat with 4D input."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_4d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.repeat(1, 2, 1, 3)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 2, 1, 3)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1), Replicate()), 4)
    assert dist_output.layout == expected_layout, \
        f"Repeat 4D input layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat 4D input output mismatch"


def test_repeat_sharded_dim_repeat_one() -> None:
    """Test repeat sharded dim with repeat count of 1."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.repeat(1, 3)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 3)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Repeat sharded dim repeat one layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat sharded dim repeat one output mismatch"


def test_repeat_all_dims_replicated() -> None:
    """Test repeat with all dimensions replicated."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(np.random.randn(2, 3, 4).astype(np.float32)), _DEVICE_TYPE)
    standalone_output = standalone_input.repeat(2, 3, 1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(2, 3, 1)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Repeat all dims replicated layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat all dims replicated output mismatch"
