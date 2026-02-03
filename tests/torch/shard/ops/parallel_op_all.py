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
"""test torch dtensor with distributed all"""
import numpy as np
import torch
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global


# Generate input data using numpy at file header
np.random.seed(42)
standalone_input_np = np.random.randn(8, 16).astype(np.float32)
standalone_input_np[0][0] = 0.0

def test_distributed_all_unsharded_dim():
    """
    Feature: dtensor + torch.all layout inference (unsharded dim)
    Description:
        ▪ Test layout inference for torch.all when reducing along an unsharded dimension.
        ▪ Ensure that:
            a) The output layout is correctly inferred (rank reduction).
            b) Results are consistent between standalone and distributed.
    Expectation: Success.
    """
    init_dist()
    dim = 1

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_result = torch.all(standalone_input, dim=dim)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_result = torch.all(dist_input, dim, keepdim=False)

    expected_layout = _build_layout(mesh, (Shard(0),), 1)
    assert dist_result.layout == expected_layout, \
        f"Layout mismatch. Got {dist_result.layout}, expected {expected_layout}"

    # Ensure no partial state since we reduced a replicated dimension
    assert not dist_result.layout.is_partial(), "Result should not be partial for unsharded dim reduction"

    # Gather distributed results back to global view
    gathered_result = local_to_global(dist_result)

    assert (standalone_result == gathered_result).all(), \
        "All values mismatch between standalone and distributed (unsharded dim)"


def test_distributed_all_sharded_dim():
    """
    Feature: dtensor + torch.all layout inference (sharded dim)
    Description:
        ▪ Test layout inference for torch.all when reducing along a sharded dimension.
        ▪ Ensure that:
            a) The output layout is marked as PARTIAL (with op 'all').
            b) After resolving partial (reduce_partial), results match standalone.
    Expectation: Success.
    """
    init_dist()
    dim = 1

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_result = torch.all(standalone_input, dim=dim)

    # Distributed setup
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    # FIX: Use positional arguments to ensure correct layout inference
    dist_result = torch.all(dist_input, dim, keepdim=False)

    # Layout correctness check
    # We reduced dim 1 which corresponds to 'tp'.
    # The result should be partial on 'tp'.
    assert dist_result.layout.is_partial(), "Result should be partial when reducing sharded dim"
    assert dist_result.layout.get_partial_by_dev_id("tp") == "all", \
        "Partial operator should be 'all' for tp axis"

    # Resolve partial state (triggers communication/allreduce)
    final_result = dist_result.reduce_partial()

    # Gather distributed results back to global view
    gathered_result = local_to_global(final_result)

    assert (standalone_result == gathered_result).all(), \
        "All values mismatch between standalone and distributed (sharded dim)"
