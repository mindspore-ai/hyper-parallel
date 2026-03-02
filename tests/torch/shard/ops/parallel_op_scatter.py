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
"""test torch dtensor with distributed scatter"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

# Fixed numpy random seed
np.random.seed(42)

def test_distributed_scatter_basic():
    """
    Feature: dtensor + torch.Tensor.scatter
    Description:
        - Verify scatter works correctly when the scatter dimension is NOT sharded.
        - Input: sharded on dim 0, scatter along dim 1.
        - Index/Src: must have same layout as input.
    Expectation: Distributed result matches local execution.
    """
    init_dist()
    # FIX: Ensure deterministic random generation across all ranks
    torch.manual_seed(1234)

    # 1. Standalone Execution
    # Shape: (8, 8)
    # Scatter along dim 1
    input_shape = (8, 8)
    standalone_input = torch.zeros(input_shape).npu()

    # Construct index and src
    # Scatter some random values into random indices along dim 1
    # For dim 1 (size 8), indices must be in [0, 8)
    standalone_src = torch.randn(input_shape).npu()
    standalone_index = torch.randint(0, 8, input_shape).npu()

    standalone_output = standalone_input.scatter(1, standalone_index, standalone_src)

    # 2. Distributed Execution
    # Mesh: (2, 4) -> "dp", "tp"
    # Shard input on dim 0 ("dp"), Replicate on dim 1
    # Scatter dimension is 1 (Replicated), so this is allowed.
    mesh = init_device_mesh(device_type="npu",mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # All tensors (input, index, src) must share the same layout for consistency
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_index = distribute_tensor(standalone_index, mesh, placements)
    dist_src = distribute_tensor(standalone_src, mesh, placements)

    # Perform distributed scatter
    dist_output = dist_input.scatter(1, dist_index, dist_src)

    # 3. Validation
    # Layout Check: Output layout should match input layout
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, \
        f"Scatter output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Check
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Distributed scatter output does not match standalone output"


def test_distributed_scatter_scalar_src():
    """
    Feature: dtensor + torch.Tensor.scatter with scalar src
    Description:
        - Verify scatter works with a scalar 'value' instead of a src tensor.
    Expectation: Success with correct values filled.
    """
    init_dist()
    # FIX: Ensure deterministic random generation across all ranks
    torch.manual_seed(2345)

    # 1. Standalone
    input_shape = (8, 8)
    standalone_input = torch.zeros(input_shape).npu()
    standalone_index = torch.randint(0, 8, (8, 4)).npu() # Index can be smaller on other dims
    scalar_val = 3.14159

    standalone_output = standalone_input.scatter(1, standalone_index, scalar_val)

    # 2. Distributed
    mesh = init_device_mesh(device_type="npu",mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_index = distribute_tensor(standalone_index, mesh, placements)

    # Perform distributed scatter with scalar value
    dist_output = dist_input.scatter(1, dist_index, scalar_val)

    # 3. Validation
    gathered_output = local_to_global(dist_output)

    # Allow small float error due to distributed gathering potential precision diffs (unlikely here but safe)
    assert torch.allclose(standalone_output, gathered_output), \
        "Distributed scatter with scalar src output mismatch"


def test_distributed_scatter_sharded_dim_error():
    """
    Feature: dtensor + torch.Tensor.scatter error checking
    Description:
        - Attempt to scatter along a dimension that is sharded.
        - This should fail because indices might point to cross-device locations.
    Expectation: Raise ValueError.
    """
    init_dist()
    torch.manual_seed(3456)

    input_shape = (8, 8)
    standalone_input = torch.zeros(input_shape).npu()
    standalone_index = torch.zeros(input_shape, dtype=torch.long).npu()
    standalone_src = torch.randn(input_shape).npu()

    mesh = init_device_mesh(device_type="npu",mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Shard on dim 0
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_index = distribute_tensor(standalone_index, mesh, placements)
    dist_src = distribute_tensor(standalone_src, mesh, placements)

    try:
        # ATTEMPT to scatter along dim 0 (which is Sharded)
        dist_input.scatter(0, dist_index, dist_src)
        assert False, "Expected ValueError when scattering along a sharded dimension"
    except ValueError as e:
        assert "is sharded" in str(e) or "supported" in str(e), \
            f"Unexpected error message: {str(e)}"


def test_distributed_scatter_layout_mismatch_index():
    """
    Feature: dtensor + torch.Tensor.scatter layout consistency
    Description:
        - Verify that input and index must have the same layout.
    Expectation: Raise ValueError.
    """
    init_dist()
    torch.manual_seed(4567)

    input_shape = (8, 8)
    tensor_data = torch.randn(input_shape).npu()
    index_data = torch.zeros(input_shape, dtype=torch.long).npu()
    src_data = torch.randn(input_shape).npu()

    mesh = init_device_mesh(device_type="npu",mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Input is Sharded on dim 0
    input_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(tensor_data, mesh, input_placements)
    dist_src = distribute_tensor(src_data, mesh, input_placements)

    # Index is Sharded on dim 1 (Mismatch!)
    index_placements = (Replicate(), Shard(1))
    dist_index = distribute_tensor(index_data, mesh, index_placements)

    try:
        # Scatter along dim 1 (Replicate in input, so valid dim choice, but index layout mismatch)
        dist_input.scatter(1, dist_index, dist_src)
        assert False, "Expected ValueError due to index layout mismatch"
    except ValueError as e:
        assert "Index tensor layout" in str(e) or "must match input" in str(e), \
            f"Unexpected error message: {str(e)}"


def test_distributed_scatter_layout_mismatch_src():
    """
    Feature: dtensor + torch.Tensor.scatter layout consistency
    Description:
        - Verify that input and src tensor must have the same layout.
    Expectation: Raise ValueError.
    """
    init_dist()
    torch.manual_seed(5678)

    input_shape = (8, 8)
    tensor_data = torch.randn(input_shape).npu()
    index_data = torch.zeros(input_shape, dtype=torch.long).npu()
    src_data = torch.randn(input_shape).npu()

    mesh = init_device_mesh(device_type="npu",mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Input/Index correct
    placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(tensor_data, mesh, placements)
    dist_index = distribute_tensor(index_data, mesh, placements)

    # Src has mismatch layout (Replicate, Replicate)
    src_placements = (Replicate(), Replicate())
    dist_src = distribute_tensor(src_data, mesh, src_placements)

    try:
        dist_input.scatter(1, dist_index, dist_src)
        assert False, "Expected ValueError due to src layout mismatch"
    except ValueError as e:
        assert "Src tensor layout" in str(e) or "must match input" in str(e), \
            f"Unexpected error message: {str(e)}"
