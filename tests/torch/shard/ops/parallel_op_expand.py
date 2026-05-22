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
"""test torch dtensor with distributed expand"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global, global_to_local

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 1).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 1, 6).astype(np.float32)
standalone_scalar_like_np = np.array([[3.14]], dtype=np.float32)


def test_expand_basic_unsharded() -> None:
    """Test expand basic unsharded."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.expand(-1, 16)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.expand(-1, 16)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Expand output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Expand output mismatch between standalone and distributed execution"


def test_expand_3d() -> None:
    """Test expand with -1 preservation on 3D tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.expand(4, 10, 6)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.expand(-1, 10, -1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"Expand with -1 preservation failed: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Expand with -1 preservation output mismatch"


def test_expand_prepend_new_dimensions() -> None:
    """Test expand prepending dimensions."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.expand(2, 3, 8, 16)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.expand(2, 3, -1, 16)

    expected_layout = _build_layout(mesh, (Shard(2), Replicate()), 4)
    assert dist_output.layout == expected_layout, \
        f"Prepend expand layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Prepend expand output mismatch"


def test_expand_scalar_tensor() -> None:
    """Test expand scalar tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_scalar_like_np), _DEVICE_TYPE)
    standalone_output = standalone_input.expand(3, 4, 5)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.expand(3, 4, 5)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Scalar expand layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Scalar expand output mismatch"


def test_expand_as_basic() -> None:
    """Test expand_as basic expansion."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    target_tensor = to_device(torch.empty(8, 16), _DEVICE_TYPE)
    standalone_output = standalone_input.expand_as(target_tensor)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    target_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    dist_target = global_to_local(target_tensor, target_layout)

    dist_output = dist_input.expand_as(dist_target)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"expand_as output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "expand_as output mismatch between standalone and distributed execution"


def test_expand_as_3d_preservation() -> None:
    """Test expand_as with dimension preservation."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    target_tensor = to_device(torch.empty(4, 10, 6), _DEVICE_TYPE)
    standalone_output = standalone_input.expand_as(target_tensor)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    target_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    dist_target = global_to_local(target_tensor, target_layout)

    dist_output = dist_input.expand_as(dist_target)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"expand_as with preservation failed: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "expand_as with dimension preservation output mismatch"


def test_expand_as_prepend_dimensions() -> None:
    """Test expand_as prepending dimensions."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    target_tensor = to_device(torch.empty(2, 3, 8, 16), _DEVICE_TYPE)
    standalone_output = standalone_input.expand_as(target_tensor)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    target_layout = _build_layout(mesh, (Shard(2), Replicate(), Shard(1)), 4)
    dist_target = global_to_local(target_tensor, target_layout)

    dist_output = dist_input.expand_as(dist_target)

    expected_layout = _build_layout(mesh, (Shard(2), Replicate(), Shard(1)), 4)
    assert dist_output.layout == expected_layout, \
        f"expand_as prepend layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "expand_as prepend dimensions output mismatch"


def test_expand_as_scalar_to_tensor() -> None:
    """Test expand_as scalar expansion."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_scalar_like_np), _DEVICE_TYPE)
    target_tensor = to_device(torch.empty(3, 4, 5), _DEVICE_TYPE)
    standalone_output = standalone_input.expand_as(target_tensor)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    target_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    dist_target = global_to_local(target_tensor, target_layout)

    dist_output = dist_input.expand_as(dist_target)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Scalar expand_as layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Scalar expand_as output mismatch"
