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
"""test torch dtensor with distributed max"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 8).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 4, 4).astype(np.float32)


def test_max_element_wise() -> None:
    """Test torch.max element-wise between two tensors."""
    init_backend(_DEVICE_TYPE)

    input_b_np = np.random.randn(8, 8).astype(np.float32)
    standalone_a = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_b = to_device(torch.from_numpy(input_b_np), _DEVICE_TYPE)
    standalone_output = torch.max(standalone_a, standalone_b)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_a = distribute_tensor(standalone_a, mesh, x_placements)
    dist_b = distribute_tensor(standalone_b, mesh, x_placements)

    dist_output = torch.max(dist_a, dist_b)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Element-wise max layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Element-wise max output mismatch"


def test_max_dim_reduce_sharded() -> None:
    """Test torch.max reduction on a sharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.max(standalone_input, dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.max(dist_input, dim=0)

    expected_layout = _build_layout(mesh, (Partial(reduce_op="max"), Replicate()), 1)

    assert dist_vals.layout == expected_layout, \
        f"Max reduce sharded layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()

    assert torch.allclose(standalone_vals, gathered_vals), \
        f"Max reduce sharded values mismatch: expected {standalone_vals}, got {gathered_vals}"


def test_max_dim_reduce_replicated() -> None:
    """Test torch.max reduction on a replicated dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.max(standalone_input, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.max(dist_input, dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)

    assert dist_vals.layout == expected_layout, \
        f"Max reduce replicated layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    gathered_vals = local_to_global(dist_vals)
    assert torch.equal(
        standalone_vals, gathered_vals
    ), "Max reduce replicated dimension output mismatch"


def test_max_global_reduce() -> None:
    """Test torch.max global reduction on a fully sharded tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.max(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_output = torch.max(dist_input)

    expected_layout = _build_layout(mesh, (Partial(reduce_op="max"), Partial(reduce_op="max")), 0)

    assert dist_output.layout == expected_layout, \
        f"Global max layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    replicated_output = dist_output.redistribute(mesh, (Replicate(), Replicate()))
    gathered_output = replicated_output.to_local()

    assert torch.allclose(standalone_output, gathered_output), \
        f"Global max value mismatch: expected {standalone_output}, got {gathered_output}"


def test_max_keepdim() -> None:
    """Test torch.max with keepdim=True on sharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.max(standalone_input, dim=0, keepdim=True)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.max(dist_input, dim=0, keepdim=True)

    expected_layout = _build_layout(mesh, (Partial(reduce_op="max"), Replicate()), 2)

    assert dist_vals.layout == expected_layout, \
        f"Max keepdim layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()

    assert torch.allclose(standalone_vals, gathered_vals), \
        f"Max values mismatch: expected {standalone_vals}, got {gathered_vals}"
