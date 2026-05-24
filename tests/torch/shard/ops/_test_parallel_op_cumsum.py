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
"""test torch dtensor with distributed cumsum"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


# Generate input data using numpy at file header
np.random.seed(42)
standalone_input_np = np.random.randn(8, 16).astype(np.float32)

def test_cumsum_layout_inference() -> None:
    """Test torch.cumsum layout inference."""
    init_backend(_DEVICE_TYPE)
    cumsum_dim = 1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_output = torch.cumsum(standalone_input, dim=cumsum_dim)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.cumsum(dist_input, dim=cumsum_dim)

    assert dist_output.layout == dist_input.layout, (
        f"Cumsum: output layout {dist_output.layout} mismatch input layout {dist_input.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "Cumsum output mismatch between standalone and distributed execution"


def test_cumsum_negative_dim() -> None:
    """Test torch.cumsum with negative dimension indexing."""
    init_backend(_DEVICE_TYPE)
    cumsum_dim = -1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_output = torch.cumsum(standalone_input, dim=cumsum_dim)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.cumsum(dist_input, dim=cumsum_dim)

    assert dist_output.layout == dist_input.layout, "Cumsum with negative dim: layout mismatch"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "Cumsum with negative dim output mismatch"
