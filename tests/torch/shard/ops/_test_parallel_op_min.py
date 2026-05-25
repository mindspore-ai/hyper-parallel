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
"""test torch dtensor with distributed min"""

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


def test_min_element_wise() -> None:
    """Test torch.min element-wise between two tensors."""
    init_backend(_DEVICE_TYPE)

    input_b_np = np.random.randn(8, 8).astype(np.float32)
    standalone_a = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_b = to_device(torch.from_numpy(input_b_np), _DEVICE_TYPE)
    standalone_output = torch.min(standalone_a, standalone_b)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_a = distribute_tensor(standalone_a, mesh, x_placements)
    dist_b = distribute_tensor(standalone_b, mesh, x_placements)

    dist_output = torch.min(dist_a, dist_b)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Element-wise min layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Element-wise min output mismatch"


def test_min_dim_reduce_sharded() -> None:
    """Test torch.min reduction on a sharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.min(standalone_input, dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=0)

    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"), Replicate()), 1)

    assert dist_vals.layout == expected_layout, \
        f"Min reduce sharded layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()

    assert torch.allclose(standalone_vals, gathered_vals), \
        f"Min reduce sharded values mismatch: expected {standalone_vals}, got {gathered_vals}"


def test_min_dim_reduce_replicated() -> None:
    """Test torch.min reduction on a replicated dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.min(standalone_input, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)

    assert dist_vals.layout == expected_layout, \
        f"Min reduce replicated layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    gathered_vals = local_to_global(dist_vals)
    assert torch.equal(
        standalone_vals, gathered_vals
    ), "Min reduce replicated dimension output mismatch"


def test_min_global_reduce() -> None:
    """Test torch.min global reduction on a fully sharded tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.min(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_output = torch.min(dist_input)

    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"), Partial(reduce_op="min")), 0)

    assert dist_output.layout == expected_layout, \
        f"Global min layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    replicated_output = dist_output.redistribute(mesh, (Replicate(), Replicate()))
    gathered_output = replicated_output.to_local()

    assert torch.allclose(standalone_output, gathered_output), \
        f"Global min value mismatch: expected {standalone_output}, got {gathered_output}"


def test_min_keepdim() -> None:
    """Test torch.min with keepdim=True on sharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.min(standalone_input, dim=0, keepdim=True)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=0, keepdim=True)

    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"), Replicate()), 2)

    assert dist_vals.layout == expected_layout, \
        f"Min keepdim layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()

    assert torch.allclose(standalone_vals, gathered_vals), \
        f"Min values mismatch: expected {standalone_vals}, got {gathered_vals}"


def test_min_3d_element_wise() -> None:
    """Test torch.min element-wise between two 3D tensors."""
    init_backend(_DEVICE_TYPE)

    input_b_np = np.random.randn(4, 4, 4).astype(np.float32)
    standalone_a = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_b = to_device(torch.from_numpy(input_b_np), _DEVICE_TYPE)
    standalone_output = torch.min(standalone_a, standalone_b)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_a = distribute_tensor(standalone_a, mesh, x_placements)
    dist_b = distribute_tensor(standalone_b, mesh, x_placements)

    dist_output = torch.min(dist_a, dist_b)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"3D element-wise min layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "3D element-wise min output mismatch"


def test_min_3d_reduce_negative_dim() -> None:
    """Test torch.min reduction on a negative dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.min(standalone_input, dim=-1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=-1)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_vals.layout == expected_layout, \
        f"Min reduce negative dim layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    gathered_vals = local_to_global(dist_vals)
    assert torch.allclose(standalone_vals, gathered_vals), "Min reduce negative dim values mismatch"


def test_min_3d_reduce_sharded_dim() -> None:
    """Test torch.min reduction on a 3D sharded dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.min(standalone_input, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Partial(reduce_op="min")), 2)
    assert dist_vals.layout == expected_layout, \
        f"3D Min reduce sharded layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()
    assert torch.allclose(standalone_vals, gathered_vals), "3D Min reduce sharded values mismatch"


def test_min_1d_mesh_global_reduce() -> None:
    """Test torch.min global reduction on a 1D mesh."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.min(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2,), mesh_dim_names=("dp",))
    x_placements = (Shard(0),)
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_output = torch.min(dist_input)

    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"),), 0)
    assert dist_output.layout == expected_layout, \
        f"1D mesh global min layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    replicated_output = dist_output.redistribute(mesh, (Replicate(),))
    gathered_output = replicated_output.to_local()
    assert torch.allclose(standalone_output, gathered_output), "1D mesh global min value mismatch"


def test_min_1d_mesh_element_wise() -> None:
    """Test torch.min element-wise between two tensors on a 1D mesh."""
    init_backend(_DEVICE_TYPE)

    input_b_np = np.random.randn(8, 8).astype(np.float32)
    standalone_a = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_b = to_device(torch.from_numpy(input_b_np), _DEVICE_TYPE)
    standalone_output = torch.min(standalone_a, standalone_b)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2,), mesh_dim_names=("dp",))
    x_placements = (Shard(0),)

    dist_a = distribute_tensor(standalone_a, mesh, x_placements)
    dist_b = distribute_tensor(standalone_b, mesh, x_placements)

    dist_output = torch.min(dist_a, dist_b)

    expected_layout = _build_layout(mesh, (Shard(0),), 2)
    assert dist_output.layout == expected_layout, \
        f"1D mesh element-wise layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "1D mesh element-wise mismatch"


def test_min_keepdim_negative_dim() -> None:
    """Test torch.min with keepdim=True and negative dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.min(standalone_input, dim=-2, keepdim=True)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=-2, keepdim=True)

    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"), Replicate()), 2)
    assert dist_vals.layout == expected_layout, \
        f"Min keepdim negative dim layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()
    assert torch.allclose(standalone_vals, gathered_vals), "Min keepdim negative dim values mismatch"


def test_min_4_cards() -> None:
    """Test torch.min reduction on a 2x2 mesh with 4 cards."""
    init_backend(_DEVICE_TYPE)

    input_np = np.random.randn(4, 4).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_vals, _ = torch.min(standalone_input, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Partial(reduce_op="min")), 1)
    assert dist_vals.layout == expected_layout, \
        f"4-cards Min reduce layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()
    assert torch.allclose(standalone_vals, gathered_vals), "4-cards Min reduce values mismatch"
