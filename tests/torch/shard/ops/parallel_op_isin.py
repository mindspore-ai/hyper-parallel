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
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_elements_np = np.random.randint(0, 50, size=(8, 16)).astype(np.int32)
standalone_test_elements_np = np.random.choice(np.arange(50), size=20, replace=False).astype(np.int32)


def test_distributed_isin_layout_inference():
    """
    Feature: dtensor + torch.isin layout inference
    Description:
        - Test layout inference and numerical correctness for isin in distributed setting.
        - Ensure that:
            a) Output layout matches elements layout (same shape semantics).
            b) Results are numerically consistent between standalone and distributed execution.
            c) test_elements MUST be fully replicated for correctness.
    Expectation: Success with identical results to standalone execution.
    """
    init_dist()

    # Standalone execution (single device)
    standalone_elements = torch.from_numpy(standalone_elements_np).npu()
    standalone_test_elements = torch.from_numpy(standalone_test_elements_np).npu()
    standalone_output = torch.isin(standalone_elements, standalone_test_elements)

    # Distributed setup:
    # elements: sharded on dim0 ("dp")
    # test_elements: MUST be fully replicated (critical constraint)
    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    elements_placements = (Shard(0), Replicate()) # tensor_map = (1, -1)
    test_elements_placements = (Replicate(), Replicate()) # tensor_map = (-1,) - fully replicated 1D tensor

    # Convert to distributed tensors
    dist_elements = DTensor.distribute_tensor(standalone_elements, mesh, elements_placements)
    dist_test_elements = DTensor.distribute_tensor(standalone_test_elements, mesh, test_elements_placements)

    # Distributed execution
    dist_output = torch.isin(dist_elements, dist_test_elements)

    assert dist_output.layout == dist_elements.layout, (
        f"Isin: output layout {dist_output.layout} mismatch elements layout {dist_elements.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Isin output mismatch between standalone and distributed execution"


def test_distributed_isin_sharded_test_elements_error():
    """
    Feature: dtensor + torch.isin error on sharded test_elements
    Description:
        - Attempt to run isin where test_elements is sharded.
        - Should raise ValueError during layout inference due to global view requirement.
        - This is a CRITICAL constraint: sharded test_elements causes silent correctness errors.
    Expectation: Raise ValueError with descriptive message about unsharded requirement.
    """
    init_dist()

    # Standalone inputs
    standalone_elements = torch.from_numpy(standalone_elements_np).npu()
    standalone_test_elements = torch.from_numpy(standalone_test_elements_np).npu()

    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    elements_placements = (Shard(0), Replicate()) # tensor_map = (1, -1)
    test_elements_placements = (Shard(0), Replicate())

    # Convert to distributed tensors (test_elements will be partial on each device)
    dist_elements = DTensor.distribute_tensor(standalone_elements, mesh, elements_placements)
    dist_test_elements = DTensor.distribute_tensor(standalone_test_elements, mesh, test_elements_placements)

    # Attempt distributed execution → should fail at layout inference stage
    try:
        torch.isin(dist_elements, dist_test_elements)
        assert False, (
            "Expected ValueError when test_elements is sharded (violates global view requirement)"
        )
    except ValueError as e:
        error_msg = str(e)
        assert "'test_elements' must be unsharded" in error_msg, (
            f"Expected unsharded constraint error, got: {error_msg}"
        )


def test_distributed_isin_invert_and_assume_unique():
    """
    Feature: dtensor + torch.isin with invert/assume_unique parameters
    Description:
        - Test that invert=True and assume_unique=True parameters work correctly in distributed setting.
        - These parameters affect computation but NOT layout derivation.
        - Validates numerical correctness with special parameters.
    Expectation: Success with identical results to standalone execution.
    """
    init_dist()

    elements_unique_np = np.arange(128, dtype=np.int32).reshape(8, 16)
    test_elements_unique_np = np.arange(20, dtype=np.int32)

    standalone_elements = torch.from_numpy(elements_unique_np).npu()
    standalone_test_elements = torch.from_numpy(test_elements_unique_np).npu()

    # Case 1: invert=True (logical NOT of membership)
    standalone_invert = torch.isin(standalone_elements, standalone_test_elements, invert=True)

    # Case 2: assume_unique=True (optimization hint, should not change result)
    standalone_unique = torch.isin(
        standalone_elements,
        standalone_test_elements,
        assume_unique=True
    )

    # Distributed setup (same as normal case)
    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    elements_placements = (Shard(0), Replicate()) # tensor_map = (1, -1)
    test_elements_placements = (Replicate(), Replicate())

    dist_elements = DTensor.distribute_tensor(standalone_elements, mesh, elements_placements)
    dist_test_elements = DTensor.distribute_tensor(standalone_test_elements, mesh, test_elements_placements)

    # Distributed execution with parameters
    dist_invert = torch.isin(dist_elements, dist_test_elements, invert=True)
    dist_unique = torch.isin(dist_elements, dist_test_elements, assume_unique=True)

    # Verify invert parameter
    gathered_invert = local_to_global(dist_invert)
    assert torch.equal(
        standalone_invert, gathered_invert
    ), "Isin with invert=True mismatch between standalone and distributed"

    # Verify assume_unique parameter
    gathered_unique = local_to_global(dist_unique)
    assert torch.equal(
        standalone_unique, gathered_unique
    ), "Isin with assume_unique=True mismatch between standalone and distributed"


def test_distributed_isin_mixed_parallel_3d():
    """
    Feature: dtensor + torch.isin with 3D mixed parallelism
    Description:
        - Test isin with 3D elements tensor using mixed parallelism (dp on dim0, mp on dim2).
        - test_elements remains fully replicated (1D tensor).
        - Validates layout propagation for higher-dimensional tensors.
    Expectation: Success with identical results to standalone execution.
    """
    init_dist()

    # 3D standalone inputs
    np.random.seed(43)  # Different seed for 3D data
    standalone_elements_3d = torch.from_numpy(
        np.random.randint(0, 50, size=(4, 6, 8)).astype(np.int32)
    ).npu()
    standalone_test_elements = torch.from_numpy(standalone_test_elements_np).npu()
    standalone_output = torch.isin(standalone_elements_3d, standalone_test_elements)

    # Distributed setup: mixed parallelism on 3D tensor
    mesh = init_device_mesh(mesh_shape = (2, 2, 2), alias_name = ("dp", "tp", "mp"))
    elements_placements = (Shard(0), Replicate(), Shard(2)) # tensor_map = (2, -1, 0)
    test_elements_placements = (Replicate(), Replicate(), Replicate())

    # Convert and execute
    dist_elements = DTensor.distribute_tensor(standalone_elements_3d, mesh, elements_placements)
    dist_test_elements = DTensor.distribute_tensor(standalone_test_elements, mesh, test_elements_placements)
    dist_output = torch.isin(dist_elements, dist_test_elements)

    # Verify layout correctness
    assert dist_output.layout == dist_elements.layout, "3D mixed parallel isin: layout mismatch"

    # Verify numerical correctness
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "3D mixed parallel isin output mismatch"
