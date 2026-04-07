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
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input1_np = np.random.randn(8, 6).astype(np.float32)
standalone_input2_np = np.random.randn(8, 6).astype(np.float32)
standalone_input3_np = np.random.randn(8, 6).astype(np.float32)
standalone_input_3d_1_np = np.random.randn(4, 2, 6).astype(np.float32)
standalone_input_3d_2_np = np.random.randn(4, 2, 6).astype(np.float32)
standalone_scalar_1_np = np.array(3.14, dtype=np.float32)
standalone_scalar_2_np = np.array(2.71, dtype=np.float32)


def test_distributed_stack_basic_dim0():
    """
    Feature: dtensor + torch.stack basic operation
    Description:
        - Stack tensors along dim=0.
        - Inputs: shape (8, 6) sharded on dim=0.
        - The old dim=0 shifts to dim=1, so the output layout should map the shard to dim=1.
        - The newly inserted dim=0 must be unsharded.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    t1 = torch.from_numpy(standalone_input1_np).npu()  # shape (8, 6)
    t2 = torch.from_numpy(standalone_input2_np).npu()  # shape (8, 6)
    standalone_output = torch.stack((t1, t2), dim=0)   # shape (2, 8, 6)

    # Distributed setup: shard dim=0 ("dp")
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=0)

    # Layout validation: new dim=0 is unsharded, original dim=0 shifted to dim=1
    expected_layout = _build_layout(mesh, (Shard(1), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Stack output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Stack output mismatch between standalone and distributed execution"


def test_distributed_stack_dim1():
    """
    Feature: dtensor + torch.stack on middle dimension
    Description:
        - Stack tensors along dim=1.
        - Inputs: shape (8, 6) sharded on dim=0.
        - The old dim=0 remains dim=0, so the shard mapping doesn't change for it.
        - The newly inserted dim=1 must be unsharded.
    Expectation: Success with correct layout propagation.
    """
    init_dist()

    t1 = torch.from_numpy(standalone_input1_np).npu()
    t2 = torch.from_numpy(standalone_input2_np).npu()
    standalone_output = torch.stack((t1, t2), dim=1)  # shape (8, 2, 6)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=1)

    # Layout validation: dim=0 stays sharded on dp, new dim=1 unsharded
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Stack dim1 layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Stack dim1 output mismatch"


def test_distributed_stack_negative_dim():
    """
    Feature: dtensor + torch.stack with negative dimension
    Description:
        - Stack tensors along dim=-1 (which resolves to the last dimension).
        - Inputs: shape (8, 6) sharded on dim=1.
        - The old dim=1 remains dim=1, the new dimension is added at the end (dim=2).
    Expectation: Success with correct layout handling of negative indices.
    """
    init_dist()

    t1 = torch.from_numpy(standalone_input1_np).npu()
    t2 = torch.from_numpy(standalone_input2_np).npu()
    standalone_output = torch.stack((t1, t2), dim=-1)  # shape (8, 6, 2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Shard(1))

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=-1)

    # Layout validation: old dim=1 stays sharded on tp, new dim=2 unsharded
    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"Stack negative dim layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Stack negative dim output mismatch"



def test_distributed_stack_multiple_tensors():
    """
    Feature: dtensor + torch.stack with > 2 tensors
    Description:
        - Stack 3 tensors along dim=1.
        - Inputs: shape (8, 6) sharded on dim=0.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    t1 = torch.from_numpy(standalone_input1_np).npu()
    t2 = torch.from_numpy(standalone_input2_np).npu()
    t3 = torch.from_numpy(standalone_input3_np).npu()
    standalone_output = torch.stack((t1, t2, t3), dim=1)  # shape (8, 3, 6)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)
    dt3 = distribute_tensor(t3, mesh, placements)

    dist_output = torch.stack((dt1, dt2, dt3), dim=1)

    # Layout validation: dim=0 stays sharded on dp, new dim=1 unsharded
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Stack multiple layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Stack multiple output mismatch"


def test_distributed_stack_3d_tensors():
    """
    Feature: dtensor + torch.stack with 3D tensors
    Description:
        - Stack 3D tensors along dim=2.
        - Inputs: shape (4, 2, 6) sharded on dim=0 and dim=2.
    Expectation: Output becomes 4D, with the new dim=2 unsharded, and the old dim=2 shifting to dim=3.
    """
    init_dist()

    t1 = torch.from_numpy(standalone_input_3d_1_np).npu()
    t2 = torch.from_numpy(standalone_input_3d_2_np).npu()
    standalone_output = torch.stack((t1, t2), dim=2)  # shape (4, 2, 2, 6)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Shard(1))

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=2)

    # Layout validation: new dim=2 is unsharded, original dim=2 shifts to dim=3
    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 4)
    assert dist_output.layout == expected_layout, \
        f"Stack 3D layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Stack 3D output mismatch"


def test_distributed_stack_scalars():
    """
    Feature: dtensor + torch.stack with scalar (0-D) tensors
    Description:
        - Stack 0-D tensors to create a 1-D tensor.
        - Scalars have no placements/sharding.
    Expectation: Result is a 1D unsharded tensor.
    """
    init_dist()

    t1 = torch.from_numpy(standalone_scalar_1_np).npu()
    t2 = torch.from_numpy(standalone_scalar_2_np).npu()
    standalone_output = torch.stack((t1, t2), dim=0)  # shape (2,)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=0)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
    assert dist_output.layout == expected_layout, \
        f"Stack scalars layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Stack scalars output mismatch"


def test_distributed_stack_fully_replicated():
    """
    Feature: dtensor + torch.stack with fully replicated tensors
    Description:
        - Stack tensors where no dimension is sharded (all Replicate).
    Expectation: The output tensor should also be fully replicated.
    """
    init_dist()

    t1 = torch.from_numpy(standalone_input1_np).npu()
    t2 = torch.from_numpy(standalone_input2_np).npu()
    standalone_output = torch.stack((t1, t2), dim=0)  # shape (2, 8, 6)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dt1 = distribute_tensor(t1, mesh, placements)
    dt2 = distribute_tensor(t2, mesh, placements)

    dist_output = torch.stack((dt1, dt2), dim=0)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Stack replicated layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Stack replicated output mismatch"
