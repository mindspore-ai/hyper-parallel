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
"""test torch dtensor with distributed atleast_1d"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
standalone_0d_np = np.array(3.14, dtype=np.float32)
standalone_1d_np = np.random.randn(8).astype(np.float32)
standalone_2d_np = np.random.randn(8, 4).astype(np.float32)


def test_atleast_1d_0d() -> None:
    """Test torch.atleast_1d with 0D input."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.tensor(standalone_0d_np), _DEVICE_TYPE)
    standalone_output = torch.atleast_1d(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.atleast_1d(dist_input)

    expected_layout = _build_layout(mesh, x_placements, 1)
    assert dist_output.layout == expected_layout, \
        f"0D atleast_1d output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "0D atleast_1d output mismatch between standalone and distributed execution"


def test_atleast_1d_1d() -> None:
    """Test torch.atleast_1d with 1D input."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_1d_np), _DEVICE_TYPE)
    standalone_output = torch.atleast_1d(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.atleast_1d(dist_input)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)
    assert dist_output.layout == expected_layout, \
        f"1D atleast_1d layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "1D atleast_1d output mismatch"


def test_atleast_1d_2d() -> None:
    """Test torch.atleast_1d with 2D input."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_2d_np), _DEVICE_TYPE)
    standalone_output = torch.atleast_1d(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.atleast_1d(dist_input)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        f"2D atleast_1d layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "2D atleast_1d output mismatch"


def test_atleast_1d_multiple_tensors() -> None:
    """Test torch.atleast_1d with multiple tensors."""
    init_backend(_DEVICE_TYPE)

    standalone_in_0 = to_device(torch.tensor(standalone_0d_np), _DEVICE_TYPE)
    standalone_in_1 = to_device(torch.from_numpy(standalone_1d_np), _DEVICE_TYPE)
    standalone_out_0, standalone_out_1 = torch.atleast_1d(standalone_in_0, standalone_in_1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    dist_in_0 = distribute_tensor(standalone_in_0, mesh, (Replicate(), Replicate()))
    dist_in_1 = distribute_tensor(standalone_in_1, mesh, (Shard(0), Replicate()))

    dist_out_0, dist_out_1 = torch.atleast_1d(dist_in_0, dist_in_1)

    expected_layout_0 = _build_layout(mesh, (Replicate(), Replicate()), 1)
    expected_layout_1 = _build_layout(mesh, (Shard(0), Replicate()), 1)

    assert dist_out_0.layout == expected_layout_0, "0D item layout mismatch in multiple inputs test"
    assert dist_out_1.layout == expected_layout_1, "1D item layout mismatch in multiple inputs test"

    gathered_out_0 = local_to_global(dist_out_0)
    gathered_out_1 = local_to_global(dist_out_1)

    assert torch.equal(standalone_out_0, gathered_out_0), "Output 0 mismatch in multiple inputs test"
    assert torch.equal(standalone_out_1, gathered_out_1), "Output 1 mismatch in multiple inputs test"
