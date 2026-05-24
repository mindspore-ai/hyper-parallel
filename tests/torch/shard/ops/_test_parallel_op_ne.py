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
"""test torch dtensor with distributed ne (not equal) operator"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from tests.torch.utils import _DEVICE_TYPE, init_backend, init_dist, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
input_a_2d_np = np.random.randn(8, 4).astype(np.float32)
input_b_2d_np = np.random.randn(8, 4).astype(np.float32)

# Make some elements exactly equal to test 'ne' (not equal) properly
input_b_2d_np[0:4, :] = input_a_2d_np[0:4, :]

input_c_2d_np = np.random.randn(8, 1).astype(np.float32)
input_d_2d_np = np.random.randn(1, 4).astype(np.float32)


def test_ne_basic() -> None:
    """Test torch.ne basic element-wise comparison."""
    init_backend(_DEVICE_TYPE)

    standalone_a = to_device(torch.from_numpy(input_a_2d_np), _DEVICE_TYPE)
    standalone_b = to_device(torch.from_numpy(input_b_2d_np), _DEVICE_TYPE)
    standalone_output = torch.ne(standalone_a, standalone_b)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_a = distribute_tensor(standalone_a, mesh, placements)
    dist_b = distribute_tensor(standalone_b, mesh, placements)

    dist_output = torch.ne(dist_a, dist_b)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Ne output layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Ne output mismatch: standalone shape={standalone_output.shape}, "
        f"gathered shape={gathered_output.shape}"
    )


def test_ne_scalar() -> None:
    """Test torch.ne scalar comparison."""
    init_backend(_DEVICE_TYPE)

    standalone_a = to_device(torch.from_numpy(input_a_2d_np), _DEVICE_TYPE)
    scalar_val = input_a_2d_np[0, 0].item()
    standalone_output = torch.ne(standalone_a, scalar_val)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Shard(1))

    dist_a = distribute_tensor(standalone_a, mesh, placements)

    dist_output = torch.ne(dist_a, scalar_val)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Ne scalar layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"Ne scalar comparison output mismatch: "
        f"standalone shape={standalone_output.shape}, gathered shape={gathered_output.shape}"
    )

