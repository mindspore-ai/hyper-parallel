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
"""test torch dtensor with distributed unbind"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
input_2d_shape = (4, 8)
input_3d_shape = (2, 4, 8)

input_2d_np = np.random.randn(*input_2d_shape).astype(np.float32)
input_3d_np = np.random.randn(*input_3d_shape).astype(np.float32)


def test_unbind_dim0() -> None:
    """Test torch.unbind on dimension 0."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.unbind(standalone_input, dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.unbind(dist_input, dim=0)

    assert isinstance(dist_output, tuple)
    assert len(dist_output) == input_2d_shape[0]

    expected_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)

    for i, dist_tensor in enumerate(dist_output):
        assert dist_tensor.layout == expected_layout, (
            f"Unbind output[{i}] layout mismatch: expected={expected_layout}, got={dist_tensor.layout}"
        )
        gathered_tensor = local_to_global(dist_tensor)
        assert torch.equal(standalone_output[i], gathered_tensor), (
            f"Unbind output[{i}] mismatch between standalone and distributed execution"
        )


def test_unbind_dim1() -> None:
    """Test torch.unbind on dimension 1."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.unbind(standalone_input, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.unbind(dist_input, dim=1)

    assert len(dist_output) == input_2d_shape[1]

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)

    for i, dist_tensor in enumerate(dist_output):
        assert dist_tensor.layout == expected_layout, (
            f"Unbind output[{i}] layout mismatch: expected={expected_layout}, got={dist_tensor.layout}"
        )
        gathered_tensor = local_to_global(dist_tensor)
        assert torch.equal(standalone_output[i], gathered_tensor), (
            f"Unbind output[{i}] numerical mismatch"
        )


def test_unbind_negative_dim() -> None:
    """Test torch.unbind with negative dimension."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(input_3d_np), _DEVICE_TYPE)
    standalone_output = torch.unbind(standalone_input, dim=-1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.unbind(dist_input, dim=-1)

    assert len(dist_output) == input_3d_shape[2]

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

    for i, dist_tensor in enumerate(dist_output):
        assert dist_tensor.layout == expected_layout, (
            f"Unbind output[{i}] layout mismatch"
        )
        gathered_tensor = local_to_global(dist_tensor)
        assert torch.equal(standalone_output[i], gathered_tensor), (
            f"Unbind output[{i}] numerical mismatch"
        )
