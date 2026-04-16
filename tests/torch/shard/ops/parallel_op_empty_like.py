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
"""test torch dtensor with distributed empty_like"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist

np.random.seed(42)
# Shape (4, 8): dim 0 divisible by dp=2, dim 1 divisible by tp=2
standalone_input_2d_np = np.random.randn(4, 8).astype(np.float32)


def test_distributed_empty_like_data_parallel():
    """
    Feature: dtensor + torch.empty_like with data parallel
    Description:
        - Input: (4, 8) sharded on dim 0 (batch dimension) across 2 dp ranks.
        - empty_like preserves the input layout; output shape and dtype must
          match the input.
    Expectation: Success with correct layout, shape, and dtype.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.empty_like(dist_input)

    # Layout validation: output layout must equal input layout
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"empty_like data parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    # Shape and dtype validation: empty_like preserves shape and dtype
    assert dist_output.shape == standalone_input.shape, (
        f"empty_like data parallel shape mismatch: "
        f"expected={standalone_input.shape}, got={dist_output.shape}"
    )
    assert dist_output.dtype == standalone_input.dtype, (
        f"empty_like data parallel dtype mismatch: "
        f"expected={standalone_input.dtype}, got={dist_output.dtype}"
    )


def test_distributed_empty_like_model_parallel():
    """
    Feature: dtensor + torch.empty_like with model parallel
    Description:
        - Input: (4, 8) sharded on dim 1 (feature dimension) across 2 tp ranks.
        - empty_like preserves the input layout; output shape and dtype must
          match the input.
    Expectation: Success with correct layout, shape, and dtype.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.empty_like(dist_input)

    # Layout validation: output layout must equal input layout
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"empty_like model parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    # Shape and dtype validation
    assert dist_output.shape == standalone_input.shape, (
        f"empty_like model parallel shape mismatch: "
        f"expected={standalone_input.shape}, got={dist_output.shape}"
    )
    assert dist_output.dtype == standalone_input.dtype, (
        f"empty_like model parallel dtype mismatch: "
        f"expected={standalone_input.dtype}, got={dist_output.dtype}"
    )
