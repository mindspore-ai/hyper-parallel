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
"""test torch dtensor with distributed multinomial"""

import numpy as np
import torch
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

# Set random seed for reproducibility
SEED = 42

def _set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Ensure CUDA/NPU determinism if applicable, though manual_seed suffices for logic checks
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_distributed_multinomial_1d_replicated():
    """
    Feature: dtensor + torch.multinomial 1D Input
    Description:
        - Input: 1D probability tensor (C,), fully replicated.
        - Operation: multinomial sampling.
        - Expectation: Output layout is 1D fully replicated.
          Results should match standalone if seeded identically (since it's replicated).
    """
    init_dist()

    # 1. Standalone Execution
    _set_seed()
    weights_np = np.abs(np.random.randn(10)).astype(np.float32)
    standalone_input = torch.from_numpy(weights_np).npu()
    # Normalize isn't strictly necessary for multinomial but good practice
    standalone_output = torch.multinomial(standalone_input, num_samples=5, replacement=True)

    # 2. Distributed Execution
    _set_seed() # Reset seed to ensure same random sequence on all ranks
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(),) # 1D Replicated

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.multinomial(num_samples=5, replacement=True)

    # 3. Layout Validation
    # Input: (-1,) -> Output: (-1,) ("None")
    expected_layout = _build_layout(mesh, (Replicate(),), 1)
    assert dist_output.layout == expected_layout, \
        f"1D Replicated layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # 4. Numerical/Shape Validation
    gathered_output = local_to_global(dist_output)

    # Since it is fully replicated and seeded, outputs should identical on all ranks and match standalone
    assert torch.equal(standalone_output, gathered_output), \
        "1D Replicated output mismatch between standalone and distributed execution"


def test_distributed_multinomial_2d_batch_sharded():
    """
    Feature: dtensor + torch.multinomial 2D Input (Batch Sharded)
    Description:
        - Input: 2D (N, C) sharded on Batch dimension (dim 0).
        - Operation: multinomial sampling.
        - Expectation:
            - Output layout preserves sharding on dim 0.
            - New sample dimension (dim 1) is unsharded (Replicated).
            - Note: Strict equality with standalone is skipped here because splitting
              batches across ranks causes RNG state divergence compared to sequential execution.
    """
    init_dist()

    n, c = 8, 10
    num_samples = 5

    weights_np = np.abs(np.random.randn(n, c)).astype(np.float32)
    standalone_input = torch.from_numpy(weights_np).npu()

    # Distributed Setup: Shard dim 0 ("dp"), Replicate dim 1
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.multinomial(num_samples=num_samples, replacement=True)

    # 1. Layout Validation
    # Input: (Shard(0), Replicate()) -> Output: (Shard(0), Replicate())
    # The new dimension is created locally and is not sharded.
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"2D Batch Sharded layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # 2. Shape and Value Sanity Check
    gathered_output = local_to_global(dist_output)

    assert gathered_output.shape == (n, num_samples), \
        f"Output shape mismatch: expected {(n, num_samples)}, got {gathered_output.shape}"

    # Check values are valid indices [0, C)
    assert gathered_output.min() >= 0 and gathered_output.max() < c, \
        "Output contains invalid indices"


def test_distributed_multinomial_2d_fully_replicated():
    """
    Feature: dtensor + torch.multinomial 2D Input (Fully Replicated)
    Description:
        - Input: 2D (N, C) fully replicated.
        - Operation: multinomial sampling.
        - Expectation: Output fully replicated. Matches standalone with same seed.
    """
    init_dist()

    _set_seed()
    weights_np = np.abs(np.random.randn(4, 5)).astype(np.float32)
    standalone_input = torch.from_numpy(weights_np).npu()
    standalone_output = torch.multinomial(standalone_input, num_samples=3, replacement=True)

    _set_seed() # Reset seed
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.multinomial(num_samples=3, replacement=True)

    # Layout Validation
    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"2D Fully Replicated layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Value Validation
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "2D Fully Replicated output mismatch"


def test_distributed_multinomial_error_sharded_prob():
    """
    Feature: dtensor + torch.multinomial Error Handling
    Description:
        - Attempt to run multinomial when probability dimension (last dim) is sharded.
        - Expectation: Raise ValueError.
    """
    init_dist()

    weights_np = np.abs(np.random.randn(4, 8)).astype(np.float32)
    standalone_input = torch.from_numpy(weights_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # INVALID: Sharding the probability dimension (dim 1)
    x_placements = (Replicate(), Shard(1))

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    try:
        dist_input.multinomial(num_samples=2, replacement=True)
        assert False, "Expected ValueError when probability dimension is sharded"
    except ValueError as e:
        assert "must not be sharded" in str(e), \
            f"Unexpected error message: {str(e)}"
