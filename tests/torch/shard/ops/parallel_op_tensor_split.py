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
"""test torch dtensor with distributed tensor_split"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 6).astype(np.float32)


def test_distributed_tensor_split_by_sections_unsharded():
    """
    Feature: dtensor + torch.tensor_split by sections (int)
    Description:
        - Split a tensor into specified number of sections on an unsharded dimension.
        - Input: shape (8, 6) sharded on dim=0, split dim=1 into 3 sections.
        - Output layout should be preserved for all resulting chunks.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 6)
    standalone_outputs = standalone_input.tensor_split(3, dim=1)  # splits dim 1 into 3 chunks

    # Distributed setup: shard dim=0 ("dp"), keep dim=1 unsharded (to split)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(3, dim=1)

    # Layout validation
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert len(dist_outputs) == 3, f"Expected 3 outputs, got {len(dist_outputs)}"

    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout, \
            f"Output {i} layout mismatch: expected {expected_layout}, got {out.layout}"

        # Numerical validation via gathering
        gathered_output = local_to_global(out)
        assert torch.equal(
            standalone_outputs[i], gathered_output
        ), f"Split output {i} mismatch between standalone and distributed execution"


def test_distributed_tensor_split_by_indices_unsharded():
    """
    Feature: dtensor + torch.tensor_split by indices (tuple/list)
    Description:
        - Split a tensor using specific indices on an unsharded dimension.
        - Input: shape (8, 6) sharded on dim=0, split dim=1 at indices [1, 4].
        - Output layout should be preserved.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_outputs = standalone_input.tensor_split((1, 4), dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split((1, 4), dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert len(dist_outputs) == 3, f"Expected 3 outputs (indices len + 1), got {len(dist_outputs)}"

    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout, \
            f"Output {i} layout mismatch: expected {expected_layout}, got {out.layout}"

        gathered_output = local_to_global(out)
        assert torch.equal(
            standalone_outputs[i], gathered_output
        ), f"Indices split output {i} mismatch"


def test_distributed_tensor_split_default_dim():
    """
    Feature: dtensor + torch.tensor_split default dim (0)
    Description:
        - Call tensor_split without dim parameter (defaults to dim 0).
        - Input: shape (8, 6) sharded on dim=1.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_outputs = standalone_input.tensor_split(2)  # defaults to dim=0

    # Dim 1 has size 6, which is evenly divisible by 2.
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(2)

    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_distributed_tensor_split_negative_dim():
    """
    Feature: dtensor + torch.tensor_split with negative dimension
    Description:
        - Use negative dimension index for splitting.
        - Input: shape (8, 6) sharded on dim=0, split dim=-1 (which is 1).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_outputs = standalone_input.tensor_split(2, dim=-1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(2, dim=-1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)



def test_distributed_tensor_split_3d_sections():
    """
    Feature: dtensor + torch.tensor_split on 3D tensor
    Description:
        - Split a 3D tensor into specified number of sections on an unsharded dimension.
        - Input: shape (8, 6, 8) sharded on dim=0 and dim=2, split dim=1 into 2 sections.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    input_np = np.random.randn(8, 6, 8).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    standalone_outputs = standalone_input.tensor_split(2, dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Mesh dim 0 shards tensor dim 0; Mesh dim 1 shards tensor dim 2
    x_placements = (Shard(0), Shard(2))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(2, dim=1)

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert len(dist_outputs) == 2

    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_distributed_tensor_split_1d_tensor_indices():
    """
    Feature: dtensor + torch.tensor_split by 1D tensor indices
    Description:
        - Split a tensor using a 1D tensor of indices on an unsharded dimension.
        - Input: shape (8, 6), split dim=1 at indices [1, 4] given as torch.tensor.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    split_indices = torch.tensor([1, 4])
    standalone_outputs = standalone_input.tensor_split(split_indices, dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(split_indices, dim=1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_distributed_tensor_split_uneven_sections():
    """
    Feature: dtensor + torch.tensor_split with uneven sections
    Description:
        - Split a tensor into sections that don't divide evenly.
        - Input: shape (8, 7), split dim=1 into 3 sections (results in sizes 3, 2, 2).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    input_np = np.random.randn(8, 7).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    standalone_outputs = standalone_input.tensor_split(3, dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(3, dim=1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_distributed_tensor_split_out_of_bounds_indices():
    """
    Feature: dtensor + torch.tensor_split with out-of-bounds indices
    Description:
        - Split a tensor using indices where some exceed the dimension size.
        - Input: shape (8, 6), split dim=1 at indices (2, 10).
    Expectation: Success with correct layout and numerical equivalence (produces empty tensors for out-of-bounds).
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_outputs = standalone_input.tensor_split((2, 10), dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split((2, 10), dim=1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_distributed_tensor_split_4d_multi_shard():
    """
    Feature: dtensor + torch.tensor_split on 4D tensor with multi-sharding
    Description:
        - Input: shape (8, 4, 6, 8) sharded on dim=0 and dim=3.
        - Split dim=2 into 2 sections.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    input_np = np.random.randn(8, 4, 6, 8).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    standalone_outputs = standalone_input.tensor_split(2, dim=2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(3))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(2, dim=2)

    expected_layout = _build_layout(mesh, x_placements, 4)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_distributed_tensor_split_list_indices():
    """
    Feature: dtensor + torch.tensor_split by list of indices
    Description:
        - Split a tensor using a list of indices instead of a tuple.
        - Input: shape (8, 6), split dim=1 at indices [2, 5].
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    indices_list = [2, 5]
    standalone_outputs = standalone_input.tensor_split(indices_list, dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(indices_list, dim=1)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)


def test_distributed_tensor_split_replicated():
    """
    Feature: dtensor + torch.tensor_split on fully replicated tensor
    Description:
        - Split a tensor that is fully replicated across all devices.
        - Input: shape (8, 6), Replicate on both mesh dims. Split dim=0 into 4 sections.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_outputs = standalone_input.tensor_split(4, dim=0)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = dist_input.tensor_split(4, dim=0)

    expected_layout = _build_layout(mesh, x_placements, 2)
    for i, out in enumerate(dist_outputs):
        assert out.layout == expected_layout
        gathered_output = local_to_global(out)
        assert torch.equal(standalone_outputs[i], gathered_output)
