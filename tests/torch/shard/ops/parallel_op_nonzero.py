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
"""test torch dtensor with distributed nonzero"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
# Create a 2D array with specific zero and non-zero elements
standalone_input_2d_np = np.array([
    [1.0, 0.0, 2.0],
    [0.0, 3.0, 0.0],
    [4.0, 5.0, 0.0]
], dtype=np.float32)

# Create a 3D array and randomly set some elements to zero
standalone_input_3d_np = np.random.randn(4, 2, 6).astype(np.float32)
standalone_input_3d_np[standalone_input_3d_np < 0] = 0.0


def test_nonzero_basic() -> None:
    """Test nonzero() basic."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.nonzero()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.nonzero()

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, (
        f"Nonzero output layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        "Nonzero output mismatch between standalone and distributed execution"
    )


def test_nonzero_as_tuple() -> None:
    """Test nonzero() with as_tuple=True."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = standalone_input.nonzero(as_tuple=True)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.nonzero(as_tuple=True)

    assert isinstance(dist_output, tuple), "Nonzero with as_tuple=True should return a tuple"
    assert len(dist_output) == standalone_input.ndim, "Returned tuple length should match input ndim"

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
    for i, out_tensor in enumerate(dist_output):
        assert out_tensor.layout == expected_layout, (
            f"Nonzero tuple output layout mismatch at index {i}: "
            f"expected {expected_layout}, got {out_tensor.layout}"
        )

    for i, (expected_tensor, local_tensor) in enumerate(zip(standalone_output, dist_output)):
        gathered_output = local_to_global(local_tensor)
        assert torch.equal(expected_tensor, gathered_output), (
            f"Nonzero tuple output mismatch at index {i}"
        )
