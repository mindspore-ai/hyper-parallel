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
"""test torch dtensor with distributed topk"""

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

# Generate input data using numpy at file header
np.random.seed(42)
standalone_input_np = np.random.randn(8, 16).astype(np.float32)

def test_topk_layout_inference() -> None:
    """Test torch.topk layout inference."""
    init_backend(_DEVICE_TYPE)
    k = 5
    dim = -1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_values, standalone_indices = torch.topk(standalone_input, k, dim=dim)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_values, dist_indices = torch.topk(dist_input, k, dim=dim)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_values.layout == expected_layout, (
        f"TopK values layout mismatch: expected={expected_layout}, got={dist_values.layout}"
    )
    assert dist_indices.layout == expected_layout, (
        f"TopK indices layout mismatch: expected={expected_layout}, got={dist_indices.layout}"
    )

    gathered_values = local_to_global(dist_values)
    gathered_indices = local_to_global(dist_indices)

    assert torch.allclose(standalone_values, gathered_values, atol=1e-5), (
        "Topk values mismatch between standalone and distributed"
    )
    assert torch.equal(standalone_indices, gathered_indices), (
        "Topk indices mismatch between standalone and distributed"
    )
