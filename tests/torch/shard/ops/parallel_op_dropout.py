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
"""test torch dtensor with distributed dropout"""

import numpy as np
import torch
import torch.nn.functional as F
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 8).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 4, 6).astype(np.float32)


def test_distributed_dropout_basic_sharded():
    """
    Feature: dtensor + torch.nn.functional.dropout basic execution
    Description:
        - Apply dropout to a 2D tensor sharded on dim=0.
        - Input: shape (8, 8) sharded on dim=0, p=0.5.
        - Output layout should be exactly the same as input layout, as dropout
          operates element-wise and does not alter the tensor's shape or distribution.
    Expectation: Success with correct layout and shape.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()

    # Distributed setup: shard dim=0 ("dp"), keep dim=1 unsharded
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.dropout(dist_input, p=0.5, training=True)

    # Layout validation: Output layout must match input layout
    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, \
        f"Dropout output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Shape validation via gathering
    gathered_output = local_to_global(dist_output)
    assert gathered_output.shape == standalone_input.shape, "Dropout output shape mismatch"


def test_distributed_dropout_p0_exact_match():
    """
    Feature: dtensor + torch.nn.functional.dropout exact match (p=0.0)
    Description:
        - Apply dropout with p=0.0 to check exact numerical equivalence between
          distributed execution and standalone execution.
    Expectation: Success with correct layout and exact numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = F.dropout(standalone_input, p=0.0, training=True)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.dropout(dist_input, p=0.0, training=True)

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Dropout (p=0.0) output mismatch between standalone and distributed execution"


def test_distributed_dropout_3d():
    """
    Feature: dtensor + torch.nn.functional.dropout on 3D tensor
    Description:
        - Apply dropout to a 3D tensor sharded on multiple dimensions.
        - Input: shape (4, 4, 6) sharded on dim=0 and dim=2.
    Expectation: Success with correct layout propagation.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.dropout(dist_input, p=0.3, training=True)

    # Layout validation
    expected_layout = _build_layout(mesh, x_placements, 3)
    assert dist_output.layout == expected_layout, \
        f"Dropout 3D output layout mismatch: expected {expected_layout}, got {dist_output.layout}"


def test_distributed_dropout_replicate():
    """
    Feature: dtensor + torch.nn.functional.dropout fully replicated
    Description:
        - Apply dropout to a fully replicated tensor.
    Expectation: Success with correct layout.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.dropout(dist_input, p=0.5, training=True)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, \
        f"Dropout replicate output layout mismatch: expected {expected_layout}, got {dist_output.layout}"
