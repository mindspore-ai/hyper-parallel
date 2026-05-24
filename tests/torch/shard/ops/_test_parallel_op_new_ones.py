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
"""test torch dtensor with distributed new_ones"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
standalone_input_np = np.random.randn(4, 4).astype(np.float32)


def test_new_ones_tuple_size() -> None:
    """Test torch.Tensor.new_ones with tuple size."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    size = (3, 5)
    standalone_output = standalone_input.new_ones(size)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Tuple size layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Tuple size output mismatch between standalone and distributed execution"


def test_new_ones_list_size() -> None:
    """Test torch.Tensor.new_ones with list size."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    size = [2, 2, 2]
    standalone_output = standalone_input.new_ones(size)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"List size layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "List size output mismatch"


def test_new_ones_int_size() -> None:
    """Test torch.Tensor.new_ones with int size (1D tensor)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    size = 8
    standalone_output = standalone_input.new_ones(size)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    expected_layout = _build_layout(mesh, (Replicate(),), 1)
    assert dist_output.layout == expected_layout, \
        f"Int size layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Int size output mismatch"


def test_new_ones_scalar() -> None:
    """Test torch.Tensor.new_ones with empty tuple (scalar tensor)."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    size = ()
    standalone_output = standalone_input.new_ones(size)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    expected_layout = _build_layout(mesh, (), 0)
    assert dist_output.layout == expected_layout, \
        f"Scalar layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Scalar output mismatch"


def test_new_ones_input_sharding_ignored() -> None:
    """Test new_ones output being Replicated regardless of input sharding."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    size = (4, 4)
    standalone_output = standalone_input.new_ones(size)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    assert dist_output.layout == expected_layout, \
        f"Sharding ignore check failed: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Output values mismatch with sharded input"
