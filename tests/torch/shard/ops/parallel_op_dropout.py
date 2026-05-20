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
standalone_input_2d_np = np.random.randn(8, 8).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 4, 6).astype(np.float32)


def test_dropout_basic_sharded() -> None:
    """Test dropout basic execution with sharded input."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.dropout(dist_input, p=0.5, training=True)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Dropout output layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert gathered_output.shape == standalone_input.shape, "Dropout output shape mismatch"


def test_dropout_p0_exact_match() -> None:
    """Test dropout with p=0.0 for exact numerical equivalence."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = F.dropout(standalone_input, p=0.0, training=True)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.dropout(dist_input, p=0.0, training=True)

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Dropout (p=0.0) output mismatch between standalone and distributed execution"


def test_dropout_3d() -> None:
    """Test dropout on 3D tensor."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.dropout(dist_input, p=0.3, training=True)

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert dist_output.layout == expected_layout, (
        f"Dropout 3D output layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )


def test_dropout_replicate() -> None:
    """Test dropout fully replicated."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.dropout(dist_input, p=0.5, training=True)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Dropout replicate output layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )
