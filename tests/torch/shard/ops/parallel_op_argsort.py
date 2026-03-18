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
"""test torch dtensor with distributed argsort"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

# Use distinct random values to avoid ambiguous sort orders
np.random.seed(42)
standalone_input_2d_np = np.random.rand(8, 4).astype(np.float32)
standalone_input_3d_np = np.random.rand(4, 8, 6).astype(np.float32)


def test_distributed_argsort_basic_unsharded():
    """
    Feature: dtensor + torch.argsort basic execution
    Description:
        - Perform argsort on an unsharded dimension (dim=-1).
        - Input: shape (8, 4) sharded on dim=0, dim=1 is unsharded.
        - Output layout should be identical to input layout.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference computation
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 4)
    standalone_output = torch.argsort(standalone_input, dim=-1)        # shape (8, 4)

    # Distributed setup: shard dim=0 ("dp"), keep dim=1 unsharded (safe to sort)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.argsort(dist_input, dim=-1)

    # Layout validation: Output layout must perfectly match input layout
    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, \
        f"Argsort output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering local parts to global
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Argsort output mismatch between standalone and distributed execution"


def test_distributed_argsort_specific_dim():
    """
    Feature: dtensor + torch.argsort on a specific dimension
    Description:
        - Perform argsort on dim=0.
        - Input: shape (8, 4) sharded on dim=1, dim=0 is unsharded.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 4)
    standalone_output = torch.argsort(standalone_input, dim=0)         # shape (8, 4)

    # Distributed setup: keep dim=0 unsharded (to sort), shard dim=1 ("tp")
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.argsort(dist_input, dim=0)

    # Layout validation
    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, \
        f"Argsort layout mismatch on specific dim: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Argsort output mismatch on specific dim"


def test_distributed_argsort_descending():
    """
    Feature: dtensor + torch.argsort with descending flag
    Description:
        - Perform argsort with descending=True.
        - Verifies that extra kwargs are correctly passed and executed.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()  # shape (4, 8, 6)
    standalone_output = torch.argsort(standalone_input, dim=1, descending=True)

    # Shard dim=0 ("dp") and dim=2 ("tp"), keep dim=1 unsharded (to sort)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.argsort(dist_input, dim=1, descending=True)

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert dist_output.layout == expected_layout, \
        f"Argsort descending layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Argsort descending output mismatch"


def test_distributed_argsort_sharded_dim_error():
    """
    Feature: dtensor + torch.argsort error on sharded dimension
    Description:
        - Attempt to sort along a dimension that is currently sharded.
        - Input: shape (8, 4) sharded on dim=1.
        - Sort on dim=1. This should trigger the distributed constraint.
    Expectation: Raise ValueError with a descriptive message.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()

    # INVALID: shard dim=1 (the dimension we want to sort along)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    try:
        # Attempt to argsort the sharded dim=1
        torch.argsort(dist_input, dim=1)
        assert False, "Expected ValueError when performing argsort on a sharded dimension"
    except ValueError as e:
        assert "Cannot perform argsort along dimension 1 because it is currently sharded" in str(e), \
            f"Unexpected error message: {str(e)}"
