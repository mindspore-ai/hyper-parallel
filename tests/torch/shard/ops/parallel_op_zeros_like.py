# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""test torch dtensor with distributed zeros_like"""

import numpy as np
import torch
from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
# Shape (4, 8): dim 0 divisible by dp=2, dim 1 divisible by tp=2
standalone_input_2d_np = np.random.randn(4, 8).astype(np.float32)


def test_distributed_zeros_like_data_parallel():
    """
    Feature: dtensor + torch.zeros_like with data parallel
    Description:
        - Input: (4, 8) sharded on dim 0 (batch dimension) across 2 dp ranks.
        - zeros_like preserves the input layout; all output values must be zero.
    Expectation: Success with correct layout and all-zero numerical output.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = torch.zeros_like(standalone_input)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.zeros_like(dist_input)

    # Layout validation: output layout must equal input layout
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"zeros_like data parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    # Numerical validation: all output values must be zero
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"zeros_like data parallel output mismatch: "
        f"standalone={standalone_output}, gathered={gathered_output}"
    )


def test_distributed_zeros_like_model_parallel():
    """
    Feature: dtensor + torch.zeros_like with model parallel
    Description:
        - Input: (4, 8) sharded on dim 1 (feature dimension) across 2 tp ranks.
        - zeros_like preserves the input layout; all output values must be zero.
    Expectation: Success with correct layout and all-zero numerical output.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = torch.zeros_like(standalone_input)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.zeros_like(dist_input)

    # Layout validation: output layout must equal input layout
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"zeros_like model parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    # Numerical validation: all output values must be zero
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"zeros_like model parallel output mismatch: "
        f"standalone={standalone_output}, gathered={gathered_output}"
    )


def test_distributed_zeros_like_no_skip():
    """
    Feature: dtensor + torch.zeros_like inside SkipDTensorDispatch with no_skip
    Description:
        - Wrap execution in SkipDTensorDispatch(no_skip={torch.zeros_like}).
        - zeros_like is exempt from the skip, so it still dispatches through
          DTensor and must return a DTensor with the same layout as the input.
        - All output values must be zero.
    Expectation: Output is a DTensor with correct layout and all-zero values.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = torch.zeros_like(standalone_input)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)

    with SkipDTensorDispatch(no_skip={torch.zeros_like}):
        dist_output = torch.zeros_like(dist_input)

    # zeros_like is in no_skip, so the result must still be a DTensor
    assert isinstance(dist_output, DTensor), (
        f"zeros_like with no_skip should return DTensor, got {type(dist_output)}"
    )

    # Layout validation: layout must match the input layout
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"zeros_like no_skip layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    # Numerical validation: all output values must be zero
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        f"zeros_like no_skip output mismatch: "
        f"standalone={standalone_output}, gathered={gathered_output}"
    )
