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
"""test torch dtensor with distributed squeeze"""

import numpy as np
import torch
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)

def test_distributed_squeeze_basic():
    """
    Feature: dtensor + torch.Tensor.squeeze basic
    Description:
    - Squeeze an unsharded singleton dimension.
    - Input: shape (8, 1) sharded on dim=0, squeeze dim=1.
    - Output: shape (8,) sharded on dim=0.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()
    # Standalone reference
    input_np = np.random.randn(8, 1).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    standalone_output = standalone_input.squeeze(1)

    # Distributed setup: shard dim=0 ("dp"), dim=1 is unsharded (Replicate)
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.squeeze(1)

    # Layout validation: dim0 preserved (sharded on dp), dim1 removed
    # Output dim 0 corresponds to Input dim 0, so it should still be sharded on Mesh 0
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)
    assert dist_output.layout == expected_layout, \
        f"Squeeze output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Squeeze output mismatch between standalone and distributed execution"

def test_distributed_squeeze_no_args_all_dims():
    """
    Feature: dtensor + torch.Tensor.squeeze with no args (squeeze all)
    Description:
    - Squeeze all singleton dimensions.
    - Input: shape (1, 4, 1, 8) sharded on dim=1 and dim=3.
    - Output: shape (4, 8) sharded on dim=0 and dim=1.
    Expectation: Success with correct layout propagation.
    """
    init_dist()
    input_np = np.random.randn(1, 4, 1, 8).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    standalone_output = standalone_input.squeeze()

    # Mesh (2, 2). Map: Tensor dim 1 -> "dp" (Mesh 0), Tensor dim 3 -> "tp" (Mesh 1)
    mesh = init_device_mesh(mesh_shape=(2, 2), alias_name=("dp", "tp"))
    x_placements = (Shard(1), Shard(3))

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.squeeze()

    # Layout validation:
    # Input T1 (sharded on Mesh 0) becomes Output T0.
    # Input T3 (sharded on Mesh 1) becomes Output T1.
    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        f"Squeeze all dims layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Squeeze all dims output mismatch"

def test_distributed_squeeze_specific_axis_negative():
    """
    Feature: dtensor + torch.Tensor.squeeze with negative axis
    Description:
    - Squeeze specific dimension using negative index.
    - Input: shape (4, 1, 8) sharded on dim=0 and dim=2. Squeeze dim=-2 (dim 1).
    - Output: shape (4, 8).
    Expectation: Success with correct layout.
    """
    init_dist()
    input_np = np.random.randn(4, 1, 8).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    standalone_output = standalone_input.squeeze(-2)

    # Mesh (2, 2). Map: Tensor dim 0 -> Mesh 0, Tensor dim 2 -> Mesh 1
    mesh = init_device_mesh(mesh_shape=(2, 2), alias_name=("dp", "tp"))
    x_placements = (Shard(0), Shard(2))

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.squeeze(-2)

    # Layout validation:
    # Input T0 (Mesh 0) -> Output T0
    # Input T2 (Mesh 1) -> Output T1
    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        f"Squeeze negative axis layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Squeeze negative axis output mismatch"

def test_distributed_squeeze_error_non_singleton():
    """
    Feature: dtensor + torch.Tensor.squeeze validation error
    Description:
    - Attempt to squeeze a dimension that is not size 1.
    - Note: Unlike PyTorch which ignores this, DistributedOp is implemented to raise ValueError.
    Expectation: Raise ValueError.
    """
    init_dist()
    input_np = np.random.randn(8, 2).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()

    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    try:
        # Try to squeeze dim 1 which has size 2
        dist_input.squeeze(1)
        assert False, "Expected ValueError when squeezing dimension of size != 1"
    except ValueError as e:
        assert "dimension should have size 1" in str(e), \
            f"Unexpected error message: {str(e)}"

def test_distributed_squeeze_scalar_like():
    """
    Feature: dtensor + torch.Tensor.squeeze on scalar-like tensor
    Description:
    - Squeeze (1, 1) tensor to ().
    Expectation: Success.
    """
    init_dist()
    input_np = np.array([[3.14]], dtype=np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    standalone_output = standalone_input.squeeze()

    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.squeeze()

    # Layout validation: 0-dim tensor, fully replicated
    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 0)
    assert dist_output.layout == expected_layout, \
        f"Squeeze scalar layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Squeeze scalar output mismatch"
