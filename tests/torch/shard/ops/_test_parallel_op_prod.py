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
"""test torch dtensor with distributed prod"""
import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


# Generate input data using numpy at file header
# Use uniform distribution [0.5, 1.5] to avoid 0 and maintain numerical stability for product
np.random.seed(42)
standalone_input_np = np.random.uniform(0.5, 1.5, (8, 16)).astype(np.float32)

def test_prod_unsharded_dim() -> None:
    """Test torch.prod with unsharded dim reduction."""
    init_backend(_DEVICE_TYPE)
    dim = 1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_result = torch.prod(standalone_input, dim=dim)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_result = torch.prod(dist_input, dim, False)

    expected_layout = _build_layout(mesh, (Shard(0),), 1)
    assert dist_result.layout == expected_layout, (
        f"Layout mismatch: expected={expected_layout}, got={dist_result.layout}"
    )

    assert not dist_result.layout.is_partial(), (
        "Result should not be partial for unsharded dim reduction"
    )

    gathered_result = local_to_global(dist_result)
    assert torch.allclose(
        standalone_result, gathered_result, atol=1e-5
    ), "Prod values mismatch between standalone and distributed (unsharded dim)"


def test_prod_sharded_dim() -> None:
    """Test torch.prod with sharded dim reduction."""
    init_backend(_DEVICE_TYPE)
    dim = 1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_result = torch.prod(standalone_input, dim=dim)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_result = torch.prod(dist_input, dim, False)

    assert dist_result.layout.is_partial(), "Result should be partial when reducing sharded dim"
    assert dist_result.layout.get_partial_by_dev_id("tp") == "prod", (
        "Partial operator should be 'prod' for tp axis"
    )

    final_result = dist_result.reduce_partial()
    gathered_result = local_to_global(final_result)

    assert torch.allclose(
        standalone_result, gathered_result, atol=1e-5
    ), "Prod values mismatch between standalone and distributed (sharded dim)"
