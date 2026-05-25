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
"""test torch dtensor with distributed outer"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
vec1_np = np.random.randn(8).astype(np.float32)
vec2_np = np.random.randn(16).astype(np.float32)


def test_outer_both_replicated() -> None:
    """Test torch.outer with both inputs replicated."""
    init_backend(_DEVICE_TYPE)

    standalone_vec1 = to_device(torch.from_numpy(vec1_np), _DEVICE_TYPE)
    standalone_vec2 = to_device(torch.from_numpy(vec2_np), _DEVICE_TYPE)
    standalone_output = torch.outer(standalone_vec1, standalone_vec2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dist_vec1 = distribute_tensor(standalone_vec1, mesh, placements)
    dist_vec2 = distribute_tensor(standalone_vec2, mesh, placements)

    dist_output = torch.outer(dist_vec1, dist_vec2)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Outer output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Outer output mismatch between standalone and distributed execution"


def test_outer_both_sharded() -> None:
    """Test torch.outer with both inputs sharded."""
    init_backend(_DEVICE_TYPE)

    standalone_vec1 = to_device(torch.from_numpy(vec1_np), _DEVICE_TYPE)
    standalone_vec2 = to_device(torch.from_numpy(vec2_np), _DEVICE_TYPE)
    standalone_output = torch.outer(standalone_vec1, standalone_vec2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    v1_placements = (Shard(0), Replicate())
    v2_placements = (Replicate(), Shard(0))

    dist_vec1 = distribute_tensor(standalone_vec1, mesh, v1_placements)
    dist_vec2 = distribute_tensor(standalone_vec2, mesh, v2_placements)

    dist_output = torch.outer(dist_vec1, dist_vec2)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        f"Outer orthogonal sharding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Outer orthogonal sharding output mismatch"


def test_outer_first_sharded() -> None:
    """Test torch.outer with first input sharded."""
    init_backend(_DEVICE_TYPE)

    standalone_vec1 = to_device(torch.from_numpy(vec1_np), _DEVICE_TYPE)
    standalone_vec2 = to_device(torch.from_numpy(vec2_np), _DEVICE_TYPE)
    standalone_output = torch.outer(standalone_vec1, standalone_vec2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    v1_placements = (Shard(0), Replicate())
    v2_placements = (Replicate(), Replicate())

    dist_vec1 = distribute_tensor(standalone_vec1, mesh, v1_placements)
    dist_vec2 = distribute_tensor(standalone_vec2, mesh, v2_placements)

    dist_output = torch.outer(dist_vec1, dist_vec2)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Outer partial sharding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Outer partial sharding output mismatch"
