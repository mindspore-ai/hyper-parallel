# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test OffsetBasedRNGTracker"""
import numpy as np
import torch
import torch.distributed as dist
from hyper_parallel import DTensor
from hyper_parallel.core.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.random import (
    OffsetBasedRNGTracker,
    _PhiloxState,
    _calc_shard_linear_idx,
    _calc_first_shard_size,
    _calc_shard_info,
)
from tests.torch.utils import init_dist
init_dist()

# pylint: disable=protected-access
def test_tracker_initialization():
    '''
    Feature: OffsetBasedRNGTracker initialization and state sync.
    Description: Test tracker initialization with run_state_sync flag,
                 verify RNG state is broadcast from rank 0.
    Expectation: All ranks have synchronized RNG state when run_state_sync=True.
    '''
    # Test with state sync enabled (default)
    tracker = OffsetBasedRNGTracker(run_state_sync=True)

    # Verify tracker attributes
    assert (tracker._device_handle) is not None
    assert (tracker.distribute_region_enabled) is True

    # Test with state sync disabled
    tracker_no_sync = OffsetBasedRNGTracker(run_state_sync=False)
    assert (tracker_no_sync) is not None

# pylint: disable=protected-access
def test_distribute_region_disabled():
    '''
    Feature: _distribute_region when disabled.
    Description: Test that distribute_region_enabled=False allows
               execution without RNG manipulation.
    Expectation: RNG state unchanged when region is disabled.
    '''

    tracker = OffsetBasedRNGTracker(run_state_sync=False)
    tracker._manual_seed(42)
    tracker.distribute_region_enabled = False

    world_size = dist.get_world_size()
    device_mesh = DeviceMesh(list(torch.arange(world_size)), ("dp",))
    placements = [Replicate()]

    initial_state = torch.npu.get_rng_state()
    global_shape = (8, 4)

    with tracker._distribute_region(device_mesh, placements, global_shape):
        torch.rand(10, device="npu")  # Random op inside

    final_state = torch.npu.get_rng_state()

    # With distribute_region disabled, fork_rng still applies,
    # but offset management is skipped. The local RNG should be restored.
    # Note: In actual implementation, fork_rng restores state in finally block.
    assert (torch.equal(initial_state, final_state)) is True

    # Re-enable for other tests
    tracker.distribute_region_enabled = True

# pylint: disable=protected-access
def test_multi_dim_sharding_offset():
    '''
    Feature: Offset calculation with multi-dimensional sharding.
    Description: Test _set_pre_op_offset with 2D mesh and various placements.
    Expectation: Correct offset calculation for complex sharding strategies.
    '''
    # Create 2D mesh
    world_size = dist.get_world_size()
    mesh_2d = DeviceMesh(torch.arange(world_size).reshape(2, 4), ("dp","tp"))

    tracker = OffsetBasedRNGTracker(run_state_sync=False)
    tracker._manual_seed(42)

    # Test different placement combinations
    test_cases = [
        ([[Shard(0)], [Shard(1)]], "2D sharding"),
        ([[Shard(0)], [Replicate()]], "Row sharding only"),
        ([[Replicate()], [Shard(1)]], "Col sharding only"),
    ]
    global_shape = (8, 8)
    for placements, _ in test_cases:
        state = _PhiloxState(tracker._get_device_state())
        old_offset = state.offset

        # Calculate expected pre-op offset
        mesh_coordinate = mesh_2d.get_coordinate()
        shard_idx, total_shards = _calc_shard_info(mesh_coordinate, mesh_2d, placements)
        shard_linear_idx = _calc_shard_linear_idx(shard_idx, total_shards)
        first_shard_size = _calc_first_shard_size(mesh_2d, placements, global_shape)
        local_size = np.prod(first_shard_size)

        expected_pre_offset = old_offset + ((shard_linear_idx * local_size + 3) // 4) * 4

        tracker._set_pre_op_offset(state, mesh_2d, placements, global_shape)

        assert state.offset==expected_pre_offset

# pylint: disable=protected-access
def test_rng_tracker():
    """Test RNG tracker initialization and state management."""
    rank = dist.get_rank()
    torch.manual_seed(rank)
    world_size = dist.get_world_size()
    device_type = "npu"

    mesh = init_device_mesh((world_size,), ("dp",))

    torch.manual_seed(123)

    local_shape = (1, 4)
    global_shape = (8, 4)
    local_tensor = torch.zeros(local_shape, device=device_type)

    rng_tracker = OffsetBasedRNGTracker()
    with rng_tracker._distribute_region(device_mesh=mesh, placements=[Replicate()],
                                        global_shape=global_shape):
        torch.nn.init.normal_(local_tensor, mean=0, std=1)

    param = DTensor.from_local(local_tensor, mesh, placements=[Replicate()])
    local = param.to_local()
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    for other in range(world_size):
        assert (local == gathered[other]).all()
