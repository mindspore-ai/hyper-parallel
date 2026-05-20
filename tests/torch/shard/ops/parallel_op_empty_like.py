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
from tests.torch.utils import init_backend, to_device

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
# Shape (4, 8): dim 0 divisible by dp=2, dim 1 divisible by tp=2
standalone_input_2d_np = np.random.randn(4, 8).astype(np.float32)


def test_empty_like_data_parallel() -> None:
    """Test torch.empty_like with data parallel."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.empty_like(dist_input)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"empty_like data parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    assert dist_output.shape == standalone_input.shape, (
        f"empty_like data parallel shape mismatch: "
        f"expected={standalone_input.shape}, got={dist_output.shape}"
    )
    assert dist_output.dtype == standalone_input.dtype, (
        f"empty_like data parallel dtype mismatch: "
        f"expected={standalone_input.dtype}, got={dist_output.dtype}"
    )


def test_empty_like_model_parallel() -> None:
    """Test torch.empty_like with model parallel."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_output = torch.empty_like(dist_input)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"empty_like model parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    assert dist_output.shape == standalone_input.shape, (
        f"empty_like model parallel shape mismatch: "
        f"expected={standalone_input.shape}, got={dist_output.shape}"
    )
    assert dist_output.dtype == standalone_input.dtype, (
        f"empty_like model parallel dtype mismatch: "
        f"expected={standalone_input.dtype}, got={dist_output.dtype}"
    )
