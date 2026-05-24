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
"""test torch dtensor with distributed isin"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
standalone_elements_np = np.random.randint(0, 50, size=(8, 16)).astype(np.int32)
standalone_test_elements_np = np.random.choice(np.arange(50), size=20, replace=False).astype(np.int32)


def test_isin_layout_inference() -> None:
    """Test torch.isin layout inference."""
    init_backend(_DEVICE_TYPE)

    standalone_elements = to_device(torch.from_numpy(standalone_elements_np), _DEVICE_TYPE)
    standalone_test_elements = to_device(torch.from_numpy(standalone_test_elements_np), _DEVICE_TYPE)
    standalone_output = torch.isin(standalone_elements, standalone_test_elements)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    elements_placements = (Shard(0), Replicate())
    test_elements_placements = (Replicate(), Replicate())

    dist_elements = distribute_tensor(standalone_elements, mesh, elements_placements)
    dist_test_elements = distribute_tensor(standalone_test_elements, mesh, test_elements_placements)

    dist_output = torch.isin(dist_elements, dist_test_elements)

    assert dist_output.layout == dist_elements.layout, (
        f"Isin: output layout {dist_output.layout} mismatch elements layout {dist_elements.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Isin output mismatch between standalone and distributed execution"


def test_isin_invert_and_assume_unique() -> None:
    """Test torch.isin with invert/assume_unique parameters."""
    init_backend(_DEVICE_TYPE)

    elements_unique_np = np.arange(128, dtype=np.int32).reshape(8, 16)
    test_elements_unique_np = np.arange(20, dtype=np.int32)

    standalone_elements = to_device(torch.from_numpy(elements_unique_np), _DEVICE_TYPE)
    standalone_test_elements = to_device(torch.from_numpy(test_elements_unique_np), _DEVICE_TYPE)

    standalone_invert = torch.isin(standalone_elements, standalone_test_elements, invert=True)
    standalone_unique = torch.isin(
        standalone_elements, standalone_test_elements, assume_unique=True
    )

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    elements_placements = (Shard(0), Replicate())
    test_elements_placements = (Replicate(), Replicate())

    dist_elements = distribute_tensor(standalone_elements, mesh, elements_placements)
    dist_test_elements = distribute_tensor(standalone_test_elements, mesh, test_elements_placements)

    dist_invert = torch.isin(dist_elements, dist_test_elements, invert=True)
    dist_unique = torch.isin(dist_elements, dist_test_elements, assume_unique=True)

    gathered_invert = local_to_global(dist_invert)
    assert torch.equal(
        standalone_invert, gathered_invert
    ), "Isin with invert=True mismatch between standalone and distributed"

    gathered_unique = local_to_global(dist_unique)
    assert torch.equal(
        standalone_unique, gathered_unique
    ), "Isin with assume_unique=True mismatch between standalone and distributed"


def test_isin_mixed_parallel_3d() -> None:
    """Test torch.isin with 3D mixed parallelism."""
    init_backend(_DEVICE_TYPE)

    np.random.seed(43)
    standalone_elements_3d = to_device(
        torch.from_numpy(np.random.randint(0, 50, size=(4, 6, 8)).astype(np.int32)), _DEVICE_TYPE
    )
    standalone_test_elements = to_device(torch.from_numpy(standalone_test_elements_np), _DEVICE_TYPE)
    standalone_output = torch.isin(standalone_elements_3d, standalone_test_elements)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "mp"))
    elements_placements = (Shard(0), Shard(2))
    test_elements_placements = (Replicate(), Replicate())

    dist_elements = distribute_tensor(standalone_elements_3d, mesh, elements_placements)
    dist_test_elements = distribute_tensor(standalone_test_elements, mesh, test_elements_placements)
    dist_output = torch.isin(dist_elements, dist_test_elements)

    assert dist_output.layout == dist_elements.layout, "3D mixed parallel isin: layout mismatch"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "3D mixed parallel isin output mismatch"
