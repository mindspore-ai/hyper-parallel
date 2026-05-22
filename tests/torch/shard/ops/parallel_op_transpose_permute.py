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
"""test torch dtensor with distributed transpose and permute"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
standalone_input_np = np.random.randn(8, 16, 4).astype(np.float32)


def test_permute_layout_inference() -> None:
    """Test torch.permute layout inference."""
    init_backend(_DEVICE_TYPE)

    dims = (2, 0, 1)

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_output = torch.permute(standalone_input, dims)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.permute(dist_input, dims)

    expected_layout = _build_layout(mesh, (Shard(1), Shard(2)), 3)
    assert dist_output.layout == expected_layout, (
        f"Permute layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), (
        "Permute values mismatch between standalone and distributed"
    )
    assert gathered_output.shape == standalone_output.shape, (
        "Permute shape mismatch between standalone and distributed"
    )


def test_transpose_layout_inference() -> None:
    """Test torch.transpose layout inference."""
    init_backend(_DEVICE_TYPE)

    dim0, dim1 = 1, 2

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_output = torch.transpose(standalone_input, dim0, dim1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.transpose(dist_input, dim0, dim1)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
    assert dist_output.layout == expected_layout, (
        f"Transpose layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), (
        "Transpose values mismatch between standalone and distributed"
    )


def test_transpose_negative_dim() -> None:
    """Test torch.transpose with negative dimensions."""
    init_backend(_DEVICE_TYPE)

    dim0, dim1 = 0, -1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_output = torch.transpose(standalone_input, dim0, dim1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.transpose(dist_input, dim0, dim1)

    expected_layout = _build_layout(mesh, (Shard(2), Shard(1)), 3)
    assert dist_output.layout == expected_layout, (
        f"Transpose neg dim layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), (
        "Transpose (neg dim) values mismatch"
    )
