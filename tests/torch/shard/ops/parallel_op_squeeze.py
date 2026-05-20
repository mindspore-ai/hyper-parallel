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
"""test torch dtensor with distributed squeeze"""

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


def test_squeeze_basic() -> None:
    """Test squeeze an unsharded singleton dimension."""
    init_backend(_DEVICE_TYPE)

    input_np = np.random.randn(8, 1).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_output = standalone_input.squeeze(1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.squeeze(1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)
    assert dist_output.layout == expected_layout, \
        f"Squeeze output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Squeeze output mismatch between standalone and distributed execution"


def test_squeeze_no_args_all_dims() -> None:
    """Test squeeze with no args (squeeze all singleton dimensions)."""
    init_backend(_DEVICE_TYPE)

    input_np = np.random.randn(1, 4, 1, 8).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_output = standalone_input.squeeze()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(1), Shard(3))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.squeeze()

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        f"Squeeze all dims layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Squeeze all dims output mismatch"


def test_squeeze_specific_axis_negative() -> None:
    """Test squeeze specific dimension using negative index."""
    init_backend(_DEVICE_TYPE)

    input_np = np.random.randn(4, 1, 8).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_output = standalone_input.squeeze(-2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(2))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.squeeze(-2)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        f"Squeeze negative axis layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Squeeze negative axis output mismatch"


def test_squeeze_scalar_like() -> None:
    """Test squeeze (1, 1) tensor to scalar."""
    init_backend(_DEVICE_TYPE)

    input_np = np.array([[3.14]], dtype=np.float32)
    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_output = standalone_input.squeeze()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.squeeze()

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 0)
    assert dist_output.layout == expected_layout, \
        f"Squeeze scalar layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Squeeze scalar output mismatch"
