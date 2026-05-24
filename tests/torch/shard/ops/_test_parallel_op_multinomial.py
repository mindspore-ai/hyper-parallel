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
"""test torch dtensor with distributed multinomial"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


# Set random seed for reproducibility
SEED = 42


def _set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_multinomial_1d_replicated() -> None:
    """Test torch.multinomial 1D replicated."""
    init_backend(_DEVICE_TYPE)

    _set_seed()
    weights_np = np.abs(np.random.randn(10)).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(weights_np), _DEVICE_TYPE)
    standalone_output = torch.multinomial(standalone_input, num_samples=5, replacement=True)

    _set_seed()
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(),)

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.multinomial(num_samples=5, replacement=True)

    expected_layout = _build_layout(mesh, (Replicate(),), 1)
    assert dist_output.layout == expected_layout, (
        f"1D Replicated layout mismatch: expected {expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)

    assert torch.equal(standalone_output, gathered_output), (
        "1D Replicated output mismatch between standalone and distributed execution"
        f"standalone_output: {standalone_output}, "
        f"gathered_output: {gathered_output}"
    )


def test_multinomial_2d_data_parallel() -> None:
    """Test torch.multinomial 2D data parallel."""
    init_backend(_DEVICE_TYPE)

    n, c = 8, 10
    num_samples = 5

    weights_np = np.abs(np.random.randn(n, c)).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(weights_np), _DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.multinomial(num_samples=num_samples, replacement=True)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, (
        f"2D Data Parallel layout mismatch: expected {expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)

    assert gathered_output.shape == (n, num_samples), (
        f"Output shape mismatch: expected {(n, num_samples)}, got={gathered_output.shape}"
    )

    assert gathered_output.min() >= 0 and gathered_output.max() < c, (
        f"Output contains invalid indices: {gathered_output.min()} to {gathered_output.max()}"
    )


def test_multinomial_2d_fully_replicated() -> None:
    """Test torch.multinomial 2D fully replicated."""
    init_backend(_DEVICE_TYPE)

    _set_seed()
    weights_np = np.abs(np.random.randn(4, 5)).astype(np.float32)
    standalone_input = to_device(torch.from_numpy(weights_np), _DEVICE_TYPE)
    standalone_output = torch.multinomial(standalone_input, num_samples=3, replacement=True)

    _set_seed()
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.multinomial(num_samples=3, replacement=True)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, (
        f"2D Fully Replicated layout mismatch: expected {expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        "2D Fully Replicated output mismatch between standalone and distributed execution"
        f"standalone_output: {standalone_output}, "
        f"gathered_output: {gathered_output}"
    )
