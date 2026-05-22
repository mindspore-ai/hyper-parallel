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
"""test torch dtensor with distributed stack"""

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
standalone_input1_np = np.random.randn(8, 6).astype(np.float32)
standalone_input2_np = np.random.randn(8, 6).astype(np.float32)
standalone_input3_np = np.random.randn(8, 6).astype(np.float32)
standalone_input_3d_1_np = np.random.randn(4, 2, 6).astype(np.float32)
standalone_input_3d_2_np = np.random.randn(4, 2, 6).astype(np.float32)
standalone_scalar_1_np = np.array(3.14, dtype=np.float32)
standalone_scalar_2_np = np.array(2.71, dtype=np.float32)


def test_stack_basic_dim0() -> None:
    """Test stack tensors along dim=0 with shard index shift."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(standalone_input1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(standalone_input2_np), _DEVICE_TYPE)
    standalone_output = torch.stack((t1, t2), dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=0)

    expected_layout = _build_layout(mesh, (Shard(1), Replicate()), 3)
    assert dist_output.layout == expected_layout, (
        f"Stack dim0 layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "Stack dim0 output mismatch"


def test_stack_dim1() -> None:
    """Test stack tensors along dim=1 with shard index unchanged."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(standalone_input1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(standalone_input2_np), _DEVICE_TYPE)
    standalone_output = torch.stack((t1, t2), dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
    assert dist_output.layout == expected_layout, (
        f"Stack dim1 layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "Stack dim1 output mismatch"


def test_stack_negative_dim() -> None:
    """Test stack tensors along dim=-1 with negative dimension handling."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(standalone_input1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(standalone_input2_np), _DEVICE_TYPE)
    standalone_output = torch.stack((t1, t2), dim=-1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Shard(1))

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=-1)

    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, (
        f"Stack negative dim layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "Stack negative dim output mismatch"


def test_stack_multiple_tensors() -> None:
    """Test stack 3 tensors along dim=1."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(standalone_input1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(standalone_input2_np), _DEVICE_TYPE)
    t3 = to_device(torch.from_numpy(standalone_input3_np), _DEVICE_TYPE)
    standalone_output = torch.stack((t1, t2, t3), dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)
    dt3 = distribute_tensor(t3, mesh, placements)

    dist_output = torch.stack((dt1, dt2, dt3), dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
    assert dist_output.layout == expected_layout, (
        f"Stack multiple layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "Stack multiple output mismatch"


def test_stack_3d_tensors() -> None:
    """Test stack 3D tensors along dim=2 with shard shifting to dim=3."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(standalone_input_3d_1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(standalone_input_3d_2_np), _DEVICE_TYPE)
    standalone_output = torch.stack((t1, t2), dim=2)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Shard(1))

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=2)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 4)
    assert dist_output.layout == expected_layout, (
        f"Stack 3D layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "Stack 3D output mismatch"


def test_stack_scalars() -> None:
    """Test stack 0-D scalar tensors to create 1-D tensor."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(standalone_scalar_1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(standalone_scalar_2_np), _DEVICE_TYPE)
    standalone_output = torch.stack((t1, t2), dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=0)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
    assert dist_output.layout == expected_layout, (
        f"Stack scalars layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "Stack scalars output mismatch"


def test_stack_fully_replicated() -> None:
    """Test stack fully replicated tensors."""
    init_backend(_DEVICE_TYPE)

    t1 = to_device(torch.from_numpy(standalone_input1_np), _DEVICE_TYPE)
    t2 = to_device(torch.from_numpy(standalone_input2_np), _DEVICE_TYPE)
    standalone_output = torch.stack((t1, t2), dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=0)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, (
        f"Stack replicated layout mismatch: expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "Stack replicated output mismatch"
