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
"""test torch dtensor with distributed repeat_interleave"""

import numpy as np
import torch
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global, global_to_local

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
standalone_input_np = np.random.randn(8, 16).astype(np.float32)


def test_repeat_interleave_layout_inference() -> None:
    """Test torch.repeat_interleave layout inference."""
    init_backend(_DEVICE_TYPE)
    repeats = 3
    dim = -1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_output = torch.repeat_interleave(standalone_input, repeats, dim=dim)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    dist_input = global_to_local(standalone_input, x_layout)
    dist_output = torch.repeat_interleave(dist_input, repeats, dim=dim)

    assert dist_output.layout == x_layout, "Torch repeat_interleave: output layout mismatch input"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "repeat_interleave output mismatch between standalone and distributed"


def test_repeat_interleave_with_tensor() -> None:
    """Test torch.repeat_interleave with repeats_tensor."""
    init_backend(_DEVICE_TYPE)

    repeats_tensor_np = np.random.randint(1, 4, size=(16,)).astype(np.int64)
    repeats_tensor = to_device(torch.from_numpy(repeats_tensor_np), _DEVICE_TYPE)
    dim = 1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_output = torch.repeat_interleave(standalone_input, repeats_tensor, dim=dim)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    dist_input = global_to_local(standalone_input, x_layout)
    dist_output = torch.repeat_interleave(dist_input, repeats_tensor, dim=dim)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "repeat_interleave_with_tensor output mismatch between standalone and distributed"


def test_repeat_interleave_dim_none() -> None:
    """Test torch.repeat_interleave with dim=None."""
    init_backend(_DEVICE_TYPE)
    repeats = 3

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_output = torch.repeat_interleave(standalone_input, repeats)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    dist_input = global_to_local(standalone_input, x_layout)
    dist_output = torch.repeat_interleave(dist_input, repeats)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "repeat_interleave_dim_None output mismatch between standalone and distributed"
