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
"""test torch dtensor with distributed scatter"""

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

# Fixed numpy random seed
np.random.seed(42)

def test_scatter_basic() -> None:
    """Test torch.Tensor.scatter."""
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(1234)

    input_shape = (8, 8)
    standalone_input = to_device(torch.zeros(input_shape), _DEVICE_TYPE)
    standalone_src = to_device(torch.randn(input_shape), _DEVICE_TYPE)
    standalone_index = to_device(torch.randint(0, 8, input_shape), _DEVICE_TYPE)

    standalone_output = standalone_input.scatter(1, standalone_index, standalone_src)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_index = distribute_tensor(standalone_index, mesh, placements)
    dist_src = distribute_tensor(standalone_src, mesh, placements)

    dist_output = dist_input.scatter(1, dist_index, dist_src)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"Scatter output layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), (
        "Distributed scatter output does not match standalone output"
    )


def test_scatter_scalar_src() -> None:
    """Test torch.Tensor.scatter with scalar src."""
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(2345)

    input_shape = (8, 8)
    standalone_input = to_device(torch.zeros(input_shape), _DEVICE_TYPE)
    standalone_index = to_device(torch.randint(0, 8, (8, 4)), _DEVICE_TYPE)
    scalar_val = 3.14159

    standalone_output = standalone_input.scatter(1, standalone_index, scalar_val)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_index = distribute_tensor(standalone_index, mesh, placements)

    dist_output = dist_input.scatter(1, dist_index, scalar_val)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output), (
        "Distributed scatter with scalar src output mismatch"
    )
