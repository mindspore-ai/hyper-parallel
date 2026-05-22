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
"""test torch dtensor with distributed isinf"""

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

standalone_input_1d_np = np.random.randn(8).astype(np.float32)
standalone_input_1d_np[2] = np.inf
standalone_input_1d_np[5] = -np.inf

standalone_input_2d_np = np.random.randn(8, 8).astype(np.float32)
standalone_input_2d_np[1, 3] = np.inf
standalone_input_2d_np[6, 7] = -np.inf

standalone_input_3d_np = np.random.randn(4, 4, 8).astype(np.float32)
standalone_input_3d_np[0, 1, 2] = np.inf
standalone_input_3d_np[3, 3, 5] = -np.inf


def test_isinf_replicate() -> None:
    """Test torch.isinf on fully replicated tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_1d_np), _DEVICE_TYPE)
    standalone_output = torch.isinf(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = [Replicate()]

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.isinf(dist_input)

    expected_layout = _build_layout(mesh, placements, 1)
    assert dist_output.layout == expected_layout, \
        f"isinf output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "isinf output mismatch between standalone and distributed execution"


def test_isinf_1d_shard() -> None:
    """Test torch.isinf on 1D sharded tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_1d_np), _DEVICE_TYPE)
    standalone_output = torch.isinf(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = [Shard(0)]

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.isinf(dist_input)

    expected_layout = _build_layout(mesh, placements, 1)
    assert dist_output.layout == expected_layout, \
        f"isinf output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "isinf output mismatch between standalone and distributed execution"


def test_isinf_2d_shard_dim0() -> None:
    """Test torch.isinf on 2D tensor sharded on dim 0."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.isinf(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = [Shard(0), Replicate()]

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.isinf(dist_input)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, \
        f"isinf output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "isinf output mismatch between standalone and distributed execution"


def test_isinf_2d_shard_dim1() -> None:
    """Test torch.isinf on 2D tensor sharded on dim 1."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.isinf(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = [Replicate(), Shard(1)]

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.isinf(dist_input)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, \
        f"isinf output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "isinf output mismatch between standalone and distributed execution"


def test_isinf_3d_multi_shard() -> None:
    """Test torch.isinf on 3D tensor with multiple sharding."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = torch.isinf(standalone_input)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = [Shard(0), Shard(2)]

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.isinf(dist_input)

    expected_layout = _build_layout(mesh, placements, 3)
    assert dist_output.layout == expected_layout, \
        f"isinf output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "isinf output mismatch between standalone and distributed execution"
