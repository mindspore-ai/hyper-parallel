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
"""test torch dtensor with distributed div_ (in-place division)"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


# Set seed and use range [1.0, 10.0] to avoid division by zero
np.random.seed(42)
standalone_input_2d_np = np.random.uniform(1.0, 10.0, (8, 4)).astype(np.float32)
standalone_other_2d_np = np.random.uniform(1.0, 10.0, (8, 4)).astype(np.float32)
standalone_other_broadcast_np = np.random.uniform(1.0, 10.0, (1, 4)).astype(np.float32)


def test_div__identical_sharding() -> None:
    """Test torch.Tensor.div_ with identical sharding."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_other = to_device(torch.from_numpy(standalone_other_2d_np), _DEVICE_TYPE)

    standalone_output = standalone_input.clone()
    standalone_output.div_(standalone_other)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(
        to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE), mesh, placements
    )
    dist_other = distribute_tensor(
        to_device(torch.from_numpy(standalone_other_2d_np), _DEVICE_TYPE), mesh, placements
    )

    dist_input.div_(dist_other)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_input.layout == expected_layout, (
        f"div_ output layout mismatch: expected={expected_layout}, got={dist_input.layout}"
    )

    gathered_output = local_to_global(dist_input)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "div_ output mismatch between standalone and distributed execution"


def test_div__broadcast() -> None:
    """Test torch.Tensor.div_ with broadcasting."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_other = to_device(torch.from_numpy(standalone_other_broadcast_np), _DEVICE_TYPE)

    standalone_output = standalone_input.clone()
    standalone_output.div_(standalone_other)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    input_placements = (Shard(0), Shard(1))
    other_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(
        to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE), mesh, input_placements
    )
    dist_other = distribute_tensor(standalone_other, mesh, other_placements)

    dist_input.div_(dist_other)

    expected_layout = _build_layout(mesh, input_placements, 2)
    assert dist_input.layout == expected_layout, "In-place div_ with broadcast altered layout."

    gathered_output = local_to_global(dist_input)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "div_ output mismatch during broadcast division"


def test_div__scalar() -> None:
    """Test torch.Tensor.div_ with scalar."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    scalar_val = 2.0

    standalone_output = standalone_input.clone()
    standalone_output.div_(scalar_val)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(
        to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE), mesh, placements
    )
    dist_input.div_(scalar_val)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_input.layout == expected_layout, "In-place scalar div_ altered layout."

    gathered_output = local_to_global(dist_input)
    assert torch.allclose(
        standalone_output, gathered_output, rtol=1e-5, atol=1e-5
    ), "div_ scalar output mismatch"
