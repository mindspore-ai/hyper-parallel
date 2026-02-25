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
"""test torch dtensor with distributed pad"""

import numpy as np
import torch
import torch.nn.functional as F
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
# Shape: (Batch=4, Channel=4, Height=8, Width=8)
input_4d_np = np.random.randn(4, 4, 8, 8).astype(np.float32)


def test_distributed_pad_basic_unsharded():
    """
    Feature: dtensor + torch.nn.functional.pad basic
    Description:
        - Pad dimensions that are Replicated.
        - Input: (4, 4, 8, 8), Shard dim=0 (Batch), Replicate others.
        - Pad: (1, 1, 2, 2) -> Pads Width (last dim) and Height (2nd last dim).
        - Since H and W are Replicated, this should succeed.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(input_4d_np).npu()
    # Pad last dim (W) by (1,1) and 2nd last dim (H) by (2,2)
    # Output shape: (4, 4, 12, 10)
    standalone_output = F.pad(standalone_input, (1, 1, 2, 2), mode='constant', value=0.5)

    # Distributed setup
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Shard dim 0 (Batch), Replicate others
    x_placements = (Shard(0), Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.pad(dist_input, (1, 1, 2, 2), mode='constant', value=0.5)

    # Layout validation: Output layout should match input layout (preserves sharding)
    expected_layout = _build_layout(mesh, x_placements, 4)
    assert dist_output.layout == expected_layout, \
        f"Pad output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Pad output mismatch between standalone and distributed execution"


def test_distributed_pad_zero_on_sharded_dim():
    """
    Feature: dtensor + torch.nn.functional.pad with zero padding on sharded dim
    Description:
        - It is valid to provide (0, 0) padding for a Sharded dimension.
        - Input: (4, 4, 8, 8), Shard dim=0.
        - Pad: (1, 1, 0, 0, 0, 0, 0, 0) -> Pads W (Repl) by (1,1), others by 0.
        - Specifically explicitly passing 0 padding for sharded N and C dimensions.
    Expectation: Success.
    """
    init_dist()

    standalone_input = torch.from_numpy(input_4d_np).npu()
    # Pad tuple len 8 covers all 4 dims: (W_l, W_r, H_l, H_r, C_l, C_r, N_l, N_r)
    pad_args = (1, 1, 0, 0, 0, 0, 0, 0)
    standalone_output = F.pad(standalone_input, pad_args, mode='constant', value=0)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.pad(dist_input, pad_args, mode='constant', value=0)

    # Layout validation
    expected_layout = _build_layout(mesh, x_placements, 4)
    assert dist_output.layout == expected_layout, \
        f"Pad output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Pad zero-on-sharded output mismatch"


def test_distributed_pad_sharded_dim_error():
    """
    Feature: dtensor + torch.nn.functional.pad error on sharded dim
    Description:
        - Attempt to pad a Sharded dimension with non-zero values.
        - Input: (4, 4, 8, 8), Shard dim=0.
        - Pad: (0, 0, 0, 0, 0, 0, 1, 1) -> Pads N (dim 0) by (1,1).
    Expectation: Raise ValueError.
    """
    init_dist()

    standalone_input = torch.from_numpy(input_4d_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Shard dim 0
    x_placements = (Shard(0), Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    # Attempt to pad dim 0 (N) which is sharded
    pad_args = (0, 0, 0, 0, 0, 0, 1, 1)

    try:
        F.pad(dist_input, pad_args)
        assert False, "Expected ValueError when padding sharded dimension"
    except ValueError as e:
        assert "does not support padding on a sharded dimension" in str(e), \
            f"Unexpected error message: {str(e)}"
