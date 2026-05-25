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
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
# Shape: (Batch=4, Channel=4, Height=8, Width=8)
input_4d_np = np.random.randn(4, 4, 8, 8).astype(np.float32)


def test_pad_basic_unsharded() -> None:
    """Test F.pad basic on unsharded dims."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(input_4d_np), _DEVICE_TYPE)
    standalone_output = F.pad(standalone_input, (1, 1, 2, 2), mode='constant', value=0.5)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.pad(dist_input, (1, 1, 2, 2), mode='constant', value=0.5)

    expected_layout = _build_layout(mesh, x_placements, 4)
    assert dist_output.layout == expected_layout, (
        f"Pad output layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        "Pad output mismatch between standalone and distributed execution"
    )


def test_pad_zero_on_sharded_dim() -> None:
    """Test F.pad with zero padding on sharded dim."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(input_4d_np), _DEVICE_TYPE)
    pad_args = (1, 1, 0, 0, 0, 0, 0, 0)
    standalone_output = F.pad(standalone_input, pad_args, mode='constant', value=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = F.pad(dist_input, pad_args, mode='constant', value=0)

    expected_layout = _build_layout(mesh, x_placements, 4)
    assert dist_output.layout == expected_layout, (
        f"Pad output layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        "Pad zero-on-sharded output mismatch"
    )
