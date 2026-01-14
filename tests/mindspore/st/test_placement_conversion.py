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
"""test placement and tensor_map conversion"""
import pytest
from tests.common.mark_utils import arg_mark
from hyper_parallel.core.layout import Layout, _DeviceMesh
from hyper_parallel.core.placement_types import Shard, Replicate, Partial


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_basic_conversion():
    """
    Feature: Layout conversion.
    Description: Test basic conversion between placements and tensor_map.
    Expectation: Conversion results match expected tuple values.
    """
    mesh_shape = (2, 2)
    alias_name = ("dp", "tp")
    layout = Layout(mesh_shape, alias_name)

    # Test placement_to_tensor_map
    layout.set_placements([Shard(0), Shard(1)])
    t_map = layout.placement_to_tensor_map(dim=2)
    assert tuple(t_map) == (1, 0), f"Expected (1, 0), got {tuple(t_map)}"

    layout.set_placements([Replicate(), Shard(0)])
    t_map = layout.placement_to_tensor_map(dim=1)
    assert tuple(t_map) == (0,), f"Expected (0,), got {tuple(t_map)}"

    # Test tensor_map_to_placement
    layout.set_tensor_map((1, 0))
    layout.set_placements(None)
    placements = layout.tensor_map_to_placement()
    assert placements[0] == Shard(0)
    assert placements[1] == Shard(1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_shard_combinations():
    """
    Feature: Shard permutations.
    Description: Test various Shard combinations and permutations.
    Expectation: Round-trip conversion preserves Shard logic.
    """
    mesh_shape = (4, 2)
    alias_name = ("dp", "mp")
    layout = Layout(mesh_shape, alias_name)

    # Case 1: Reverse sharding
    layout.set_placements([Shard(1), Shard(0)])
    t_map = layout.placement_to_tensor_map(dim=2)
    assert tuple(t_map) == (0, 1), f"Expected (0, 1), got {tuple(t_map)}"

    # Verify round trip
    placements = layout.tensor_map_to_placement()
    assert placements[0] == Shard(1)
    assert placements[1] == Shard(0)

    # Case 2: Skip dimension
    layout.set_placements([Shard(0), Replicate()])
    t_map = layout.placement_to_tensor_map(dim=3)
    assert tuple(t_map) == (1, -1, -1), f"Expected (1, -1, -1), got {tuple(t_map)}"

    # Verify round trip
    placements = layout.tensor_map_to_placement()
    assert placements[0] == Shard(0)
    assert isinstance(placements[1], Replicate)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_full_replication():
    """
    Feature: Full replication.
    Description: Test scenarios with full replication across all mesh dimensions.
    Expectation: All dimensions mapped to -1 and placements are Replicate.
    """
    mesh_shape = (2, 2, 2)
    alias_name = ("dp", "sp", "mp")
    layout = Layout(mesh_shape, alias_name)

    layout.set_placements([Replicate(), Replicate(), Replicate()])
    t_map = layout.placement_to_tensor_map(dim=4)
    assert tuple(t_map) == (-1, -1, -1, -1)

    # Round trip
    placements = layout.tensor_map_to_placement()
    assert all(isinstance(p, Replicate) for p in placements)
    assert len(placements) == 3


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_partial_shard_mixed():
    """
    Feature: Mixed placements.
    Description: Test mixed placements of Partial and Shard.
    Expectation: Partial ops and Shard dims are correctly converted.
    """
    mesh_shape = (2, 2)
    alias_name = ("dp", "mp")
    layout = Layout(mesh_shape, alias_name)

    # Mesh 0 -> Partial(sum), Mesh 1 -> Shard(0)
    layout.set_placements([Partial("sum"), Shard(0)])
    t_map = layout.placement_to_tensor_map(dim=2)

    assert tuple(t_map) == (0, -1)
    assert layout.partial[0] == "sum"
    assert layout.partial[1] is None

    # Round trip
    placements = layout.tensor_map_to_placement()
    assert isinstance(placements[0], Partial)
    assert placements[0].reduce_op == "sum"
    assert placements[1] == Shard(0)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_partial_reduce_ops():
    """
    Feature: Partial operations.
    Description: Test different partial reduction operations (min, max, avg).
    Expectation: Reduction ops are correctly preserved in conversion.
    """
    mesh_shape = (2,)
    alias_name = ("dp",)
    layout = Layout(mesh_shape, alias_name)

    for op in ["min", "max", "avg"]:
        layout.set_placements([Partial(op)])
        t_map = layout.placement_to_tensor_map(dim=1)
        assert tuple(t_map) == (-1,)
        assert layout.partial[0] == op

        placements = layout.tensor_map_to_placement()
        assert isinstance(placements[0], Partial)
        assert placements[0].reduce_op == op


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_conversion_errors():
    """
    Feature: Error handling.
    Description: Test error handling and boundary conditions for conversion.
    Expectation: ValueError raised for invalid inputs.
    """
    mesh_shape = (2, 2)
    alias_name = ("dp", "mp")
    layout = Layout(mesh_shape, alias_name)

    # 1. Shard dimension out of bounds
    layout.set_placements([Shard(5), Replicate()])
    with pytest.raises(ValueError, match="out of bounds"):
        layout.placement_to_tensor_map(dim=2)

    # 2. Duplicate sharding on same tensor dimension
    layout.set_placements([Shard(0), Shard(0)])
    with pytest.raises(ValueError, match="has been sharded"):
        layout.placement_to_tensor_map(dim=2)

    # 3. Invalid tensor dimension
    with pytest.raises(ValueError, match="positive"):
        layout.placement_to_tensor_map(dim=-1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_from_device_mesh():
    """
    Feature: Layout creation.
    Description: Test Layout creation from device_mesh.
    Expectation: Layout initialized with correct mesh properties and empty state.
    """
    mesh_shape = (4, 2)
    alias_name = ("dp", "mp")
    rank_list = tuple(range(8))
    device_mesh = _DeviceMesh(mesh_shape, alias_name, rank_list)

    layout = Layout.from_device_mesh(device_mesh)

    assert layout.mesh_shape == mesh_shape
    assert layout.alias_name == alias_name
    assert layout.rank_list == rank_list
    assert layout.tensor_map is None
    assert layout.placements is None
    assert layout.partial == [None, None]

    layout.set_placements([Shard(0), Replicate()])
    t_map = layout.placement_to_tensor_map(dim=2)
    assert tuple(t_map) == (1, -1)
    assert layout.mesh.mesh_shape == mesh_shape
