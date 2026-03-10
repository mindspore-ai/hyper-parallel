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
"""test torch dtensor with distributed div_ (in-place division)"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

# Set seed and use range [1.0, 10.0] to avoid division by zero
np.random.seed(42)
standalone_input_2d_np = np.random.uniform(1.0, 10.0, (8, 4)).astype(np.float32)
standalone_other_2d_np = np.random.uniform(1.0, 10.0, (8, 4)).astype(np.float32)
standalone_other_broadcast_np = np.random.uniform(1.0, 10.0, (1, 4)).astype(np.float32)


def test_distributed_div__identical_sharding():
    """
    Feature: dtensor + torch.Tensor.div_ basic execution
    Description:
        - Perform in-place division on two tensors with identical sharding layouts.
        - Input: Both shape (8, 4) sharded on dim=0 ("dp").
        - Output layout should be identical to the input layout.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_other = torch.from_numpy(standalone_other_2d_np).npu()

    # Clone to preserve original for comparison if needed, though we just check the output
    standalone_output = standalone_input.clone()
    standalone_output.div_(standalone_other)

    # Distributed setup: shard dim=0 ("dp"), replicate dim=1 ("tp")
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(torch.from_numpy(standalone_input_2d_np).npu(), mesh, placements)
    dist_other = distribute_tensor(torch.from_numpy(standalone_other_2d_np).npu(), mesh, placements)

    # In-place division
    dist_input.div_(dist_other)

    # Layout validation: must remain unchanged for in-place ops
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_input.layout == expected_layout, \
        f"div_ output layout mismatch: expected {expected_layout}, got {dist_input.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_input)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "div_ output mismatch between standalone and distributed execution"



def test_distributed_div__broadcast():
    """
    Feature: dtensor + torch.Tensor.div_ with broadcasting
    Description:
        - In-place division where the divisor requires broadcasting.
        - Input1: shape (8, 4) sharded on dim=0 and dim=1.
        - Input2: shape (1, 4) replicated on dim=0, sharded on dim=1.
    Expectation: Success with broadcasting and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_other = torch.from_numpy(standalone_other_broadcast_np).npu()

    standalone_output = standalone_input.clone()
    standalone_output.div_(standalone_other)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Input1: split across both dp and tp
    input_placements = (Shard(0), Shard(1))

    # Input2: shape is (1, 4), so dim 0 must be Replicate, dim 1 can be Shard(1) to match Input1
    other_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(torch.from_numpy(standalone_input_2d_np).npu(), mesh, input_placements)
    dist_other = distribute_tensor(standalone_other, mesh, other_placements)

    dist_input.div_(dist_other)

    expected_layout = _build_layout(mesh, input_placements, 2)
    assert dist_input.layout == expected_layout, "In-place div_ with broadcast altered layout."

    gathered_output = local_to_global(dist_input)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "div_ output mismatch during broadcast division"


def test_distributed_div__scalar():
    """
    Feature: dtensor + torch.Tensor.div_ with scalar
    Description:
        - Divide a sharded tensor by a Python scalar.
        - Input1: shape (8, 4) sharded on dim=0.
        - Scalar: 2.0
    Expectation: Success, mathematically equivalent to standalone execution.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    scalar_val = 2.0

    standalone_output = standalone_input.clone()
    standalone_output.div_(scalar_val)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(torch.from_numpy(standalone_input_2d_np).npu(), mesh, placements)
    dist_input.div_(scalar_val)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_input.layout == expected_layout, "In-place scalar div_ altered layout."

    gathered_output = local_to_global(dist_input)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "div_ scalar output mismatch"
