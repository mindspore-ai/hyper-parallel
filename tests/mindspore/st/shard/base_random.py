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
"""test OffsetBasedRNGTracker for MindSpore"""
import numpy as np
import mindspore as ms
from mindspore import context, set_seed
import mindspore.communication.management as D
from mindspore.communication import get_rank, get_group_size
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

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
D.init()

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
    assert tracker._device_handle is not None
    assert tracker.distribute_region_enabled is True

    # Test with state sync disabled
    tracker_no_sync = OffsetBasedRNGTracker(run_state_sync=False)
    assert tracker_no_sync is not None

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

    world_size = get_group_size()
    device_mesh = DeviceMesh(list(ms.mint.arange(world_size)), ("dp",))
    placements = [Replicate()]
    global_shape = (8,10)

    initial_state = _PhiloxState(tracker._get_device_state())
    initial_offset = initial_state.offset

    with tracker._distribute_region(device_mesh, placements, global_shape):
        # Random op simulation
        set_seed(42)
        _ = ms.mint.rand((1,10))

    final_state = _PhiloxState(tracker._get_device_state())
    final_offset = final_state.offset

    # With distribute_region disabled, offset should not change
    # Note: fork_rng still applies in actual implementation
    assert final_offset == initial_offset, \
        "Offset should remain unchanged when distribute_region is disabled"

    # Re-enable for other tests
    tracker.distribute_region_enabled = True

# pylint: disable=protected-access
def test_multi_dim_sharding_offset():
    '''
    Feature: Offset calculation with multi-dimensional sharding.
    Description: Test _set_pre_op_offset with 2D mesh and various placements.
    Expectation: Correct offset calculation for complex sharding strategies.
    '''
    world_size = get_group_size()
    if world_size < 4:
        print("Requires at least 4 devices for 2D mesh test, skipping")
        return

    # Create 2D mesh: 2x4
    mesh_2d = DeviceMesh(
        ms.mint.arange(world_size).reshape(2, 4).tolist(),
        ("dp", "tp")
    )

    tracker = OffsetBasedRNGTracker(run_state_sync=False)
    tracker._manual_seed(42)

    # Test different placement combinations
    test_cases = [
        ([[Shard(0)], [Shard(1)]], "2D sharding"),
        ([[Shard(0)], [Replicate()]], "Row sharding only"),
        ([[Replicate()], [Shard(1)]], "Col sharding only"),
    ]
    global_shape = (8,8)
    for placements, _ in test_cases:
        state = _PhiloxState(tracker._get_device_state())
        old_offset = state.offset

        # Calculate expected pre-op offset
        mesh_coordinate = mesh_2d.get_coordinate()
        if mesh_coordinate is None:
            continue

        shard_idx, total_shards = _calc_shard_info(mesh_coordinate, mesh_2d, placements)
        shard_linear_idx = _calc_shard_linear_idx(shard_idx, total_shards)
        first_shard_size = _calc_first_shard_size(mesh_2d, placements, global_shape)
        local_size = np.prod(first_shard_size)

        expected_pre_offset = old_offset + ((shard_linear_idx * local_size + 3) // 4) * 4

        tracker._set_pre_op_offset(state, mesh_2d, placements, global_shape)

        assert state.offset == expected_pre_offset

# pylint: disable=protected-access
def test_rng_tracker():
    """Test RNG tracker initialization and state management."""
    rank = get_rank()
    ms.manual_seed(rank)
    world_size = get_group_size()

    mesh = init_device_mesh((world_size,), ("dp",))

    ms.manual_seed(123)

    local_shape = (1, 4)
    global_shape = (8, 4)

    rng_tracker = OffsetBasedRNGTracker()
    with rng_tracker._distribute_region(device_mesh=mesh, placements=[Replicate()],
                                        global_shape=global_shape):
        local_tensor = ms.mint.rand(local_shape)

    param = DTensor.from_local(local_tensor, mesh, placements=[Replicate()])
    local = param.to_local()
    gathered = [ms.mint.empty_like(local) for _ in range(world_size)]
    ms.mint.distributed.all_gather(gathered, local)

    for other in range(world_size):
        assert (local == gathered[other]).all()
