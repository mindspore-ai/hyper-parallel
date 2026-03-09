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
"""Unit tests for DeviceMesh class."""
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from hyper_parallel.core.device_mesh import (
    init_device_mesh, DeviceMesh, _group_map, _DEVICE_MESH_MAP
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(name="mock_platform")
def fixture_mock_platform():
    """Mock platform to avoid dependency on real distributed environment."""
    with patch("hyper_parallel.core.device_mesh.platform") as platform_mock:
        platform_mock.get_rank.return_value = 0
        platform_mock.get_world_size.return_value = 8
        platform_mock.Tensor = torch.Tensor

        mock_group = Mock()
        mock_group.group_name = "mock_group"
        platform_mock.split_group.return_value = mock_group
        platform_mock.get_process_group_ranks.return_value = [0, 1, 2, 3, 4, 5, 6, 7]

        def mock_tensor_to_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
            return tensor

        platform_mock.tensor_to_numpy.side_effect = mock_tensor_to_numpy

        with patch("hyper_parallel.core.device_mesh.Tensor", torch.Tensor):
            yield platform_mock


@pytest.fixture(autouse=True)
def fixture_clear_global_state():
    """Clear global caches before and after each test to prevent pollution."""
    _group_map.clear()
    _DEVICE_MESH_MAP.clear()
    yield
    _group_map.clear()
    _DEVICE_MESH_MAP.clear()


@pytest.fixture(name="mesh_2d")
def fixture_mesh_2d(mock_platform):
    """2D mesh: shape=(2, 4), dim_names=("dp", "tp"), ranks=[0..7]."""
    _ = mock_platform
    return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))


@pytest.fixture(name="mesh_3d")
def fixture_mesh_3d(mock_platform):
    """3D mesh: shape=(2, 2, 2), dim_names=("dp", "cp", "tp"), ranks=[0..7]."""
    _ = mock_platform
    return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))


# ===========================================================================
# Test: Construction
# ===========================================================================

class TestDeviceMeshConstruction:
    """Tests for DeviceMesh construction via various input types."""

    def test_construct_with_list(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [[0, 1, 2, 3], [4, 5, 6, 7]], mesh_dim_names=("dp", "tp"))

        assert dm.mesh_shape == (2, 4)
        assert dm.mesh_dim_names == ("dp", "tp")
        assert dm.rank_list == (0, 1, 2, 3, 4, 5, 6, 7)
        assert dm.ndim == 2

    def test_construct_with_tensor(self, mock_platform):
        _ = mock_platform
        t = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        dm = DeviceMesh("npu", t, mesh_dim_names=("dp", "tp"))

        assert dm.mesh_shape == (2, 2)
        assert dm.rank_list == (0, 1, 2, 3)

    def test_construct_with_numpy(self, mock_platform):
        _ = mock_platform
        arr = np.array([[0, 1], [2, 3]], dtype=np.int64)
        dm = DeviceMesh("npu", arr, mesh_dim_names=("dp", "tp"))

        assert dm.mesh_shape == (2, 2)
        assert dm.rank_list == (0, 1, 2, 3)

    def test_construct_with_tuple(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", (0, 1, 2, 3))

        assert dm.mesh_shape == (4,)
        assert dm.rank_list == (0, 1, 2, 3)
        assert dm.ndim == 1

    def test_construct_mesh_none_auto_1d(self, mock_platform):
        """mesh=None should auto-generate a 1D mesh of [0..world_size-1]."""
        _ = mock_platform
        dm = DeviceMesh("npu")

        assert dm.mesh_shape == (8,)
        assert dm.rank_list == (0, 1, 2, 3, 4, 5, 6, 7)
        assert dm.ndim == 1

    def test_construct_mesh_none_world_size_4(self, mock_platform):
        """mesh=None respects the current world_size."""
        mock_platform.get_world_size.return_value = 4
        dm = DeviceMesh("npu")

        assert dm.mesh_shape == (4,)
        assert dm.rank_list == (0, 1, 2, 3)

    def test_construct_preserves_device_type(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("cuda", [0, 1, 2, 3])
        assert dm.device_type() == "cuda"

    def test_construct_no_mesh_dim_names(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3])
        assert dm.mesh_dim_names is None

    def test_construct_custom_layout(self, mock_platform):
        """Non-sequential rank layout is correctly preserved."""
        _ = mock_platform
        dm = DeviceMesh("npu", [[0, 2], [1, 3]], mesh_dim_names=("dp", "tp"))

        assert dm.rank_list == (0, 2, 1, 3)
        assert dm.mesh_shape == (2, 2)


# ===========================================================================
# Test: Construction validation
# ===========================================================================

class TestDeviceMeshConstructionValidation:
    """Tests for invalid construction parameters."""

    def test_0d_mesh_raises(self, mock_platform):
        _ = mock_platform
        with pytest.raises(ValueError, match="mesh must be at least 1-dimensional"):
            DeviceMesh("npu", 42)

    def test_mesh_dim_names_length_mismatch(self, mock_platform):
        _ = mock_platform
        with pytest.raises(ValueError, match="mesh dimensions.*should be equal to.*mesh_dim_names length"):
            DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp",))

    def test_mesh_dim_names_duplicate(self, mock_platform):
        _ = mock_platform
        with pytest.raises(ValueError, match="Each element of mesh_dim_names.*should be different"):
            DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "dp"))

    def test_interleaved_parallel_must_be_last(self, mock_platform):
        _ = mock_platform
        with pytest.raises(ValueError, match="interleaved_parallel.*should be at the last dim"):
            DeviceMesh(
                "npu",
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                mesh_dim_names=("interleaved_parallel", "dp", "tp"),
            )

    def test_invalid_mesh_type(self, mock_platform):
        _ = mock_platform
        with pytest.raises(TypeError, match="mesh must be Tensor, list, tuple or numpy array"):
            DeviceMesh("npu", "invalid")


# ===========================================================================
# Test: init_device_mesh factory
# ===========================================================================

class TestInitDeviceMesh:
    """Tests for the init_device_mesh factory function."""

    def test_basic(self, mock_platform):
        _ = mock_platform
        dm = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))

        assert dm.mesh_shape == (2, 4)
        assert dm.mesh_dim_names == ("dp", "tp")
        assert dm.rank_list == (0, 1, 2, 3, 4, 5, 6, 7)

    def test_cache_returns_same_instance(self, mock_platform):
        _ = mock_platform
        dm1 = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))
        dm2 = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))
        assert dm1 is dm2

    def test_explicit_rank_list(self, mock_platform):
        _ = mock_platform
        dm = init_device_mesh("npu", (4,), mesh_dim_names=("dp",), rank_list=(4, 5, 6, 7))

        assert dm.rank_list == (4, 5, 6, 7)
        assert dm.mesh_shape == (4,)

    def test_rank_list_length_mismatch(self, mock_platform):
        _ = mock_platform
        with pytest.raises(ValueError, match="rank_list length.*must equal mesh size"):
            init_device_mesh("npu", (2, 4), rank_list=(0, 1))

    def test_no_mesh_dim_names(self, mock_platform):
        _ = mock_platform
        dm = init_device_mesh("npu", (2, 4))
        assert dm.mesh_dim_names is None
        assert dm.mesh_shape == (2, 4)


# ===========================================================================
# Test: Properties
# ===========================================================================

class TestDeviceMeshProperties:
    """Tests for DeviceMesh property accessors."""

    def test_ndim(self, mesh_2d, mesh_3d):
        assert mesh_2d.ndim == 2
        assert mesh_3d.ndim == 3

    def test_mesh_shape(self, mesh_2d, mesh_3d):
        assert mesh_2d.mesh_shape == (2, 4)
        assert mesh_3d.mesh_shape == (2, 2, 2)

    def test_rank_list(self, mesh_2d):
        assert mesh_2d.rank_list == (0, 1, 2, 3, 4, 5, 6, 7)

    def test_rank(self, mesh_2d):
        assert mesh_2d.rank == 0

    def test_device_type(self, mesh_2d):
        assert mesh_2d.device_type() == "npu"

    def test_root_mesh_is_none_for_top_level(self, mesh_2d):
        assert mesh_2d.root_mesh is None

    def test_sub_mesh_initially_empty(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3])
        assert not dm.sub_mesh

    def test_mesh_tensor_shape(self, mesh_2d):
        assert mesh_2d.mesh.shape == (2, 4)

    def test_size_no_dim(self, mesh_2d):
        assert mesh_2d.size() == 8

    def test_size_with_dim(self, mesh_2d):
        assert mesh_2d.size(0) == 2
        assert mesh_2d.size(1) == 4

    def test_shape_property(self, mesh_2d, mesh_3d):
        assert mesh_2d.shape == (2, 4)
        assert mesh_3d.shape == (2, 2, 2)

    def test_shape_equals_mesh_shape(self, mesh_2d, mesh_3d):
        assert mesh_2d.shape == mesh_2d.mesh_shape
        assert mesh_3d.shape == mesh_3d.mesh_shape

    def test_shape_1d(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3])
        assert dm.shape == (4,)

    def test_get_coordinate(self, mesh_2d):
        coord = mesh_2d.get_coordinate()
        assert coord == (0, 0)


# ===========================================================================
# Test: __getitem__ (sub-mesh slicing)
# ===========================================================================

class TestDeviceMeshGetItem:
    """Tests for sub-mesh creation via __getitem__."""

    def test_single_dim_by_string(self, mesh_2d):
        dp = mesh_2d["dp"]
        assert dp.mesh_shape == (2,)
        assert dp.mesh_dim_names == ("dp",)
        assert dp.rank_list == (0, 4)
        assert dp.root_mesh is mesh_2d

        tp = mesh_2d["tp"]
        assert tp.mesh_shape == (4,)
        assert tp.rank_list == (0, 1, 2, 3)

    def test_multi_dim_by_tuple(self, mesh_3d):
        dp_cp = mesh_3d[("dp", "cp")]
        assert dp_cp.mesh_shape == (2, 2)
        assert dp_cp.mesh_dim_names == ("dp", "cp")
        assert dp_cp.rank_list == (0, 2, 4, 6)

    def test_all_dims_returns_self(self, mesh_2d):
        result = mesh_2d[("dp", "tp")]
        assert result is mesh_2d

    def test_cache_returns_same_sub_mesh(self, mesh_2d):
        dp1 = mesh_2d["dp"]
        dp2 = mesh_2d["dp"]
        assert dp1 is dp2

    def test_sub_mesh_tracked_in_parent(self, mesh_2d):
        dp = mesh_2d["dp"]
        assert dp in mesh_2d.sub_mesh

    def test_sub_mesh_inherits_device_type(self, mesh_2d):
        dp = mesh_2d["dp"]
        assert dp.device_type() == "npu"

    def test_no_mesh_dim_names_raises(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3])
        with pytest.raises(RuntimeError, match="Cannot slice a DeviceMesh without mesh_dim_names"):
            _ = dm["dp"]

    def test_invalid_dim_name_raises(self, mesh_2d):
        with pytest.raises(KeyError, match="Dimension name 'invalid'.*not found"):
            _ = mesh_2d["invalid"]

    def test_wrong_order_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="must follow the order"):
            _ = mesh_2d[("tp", "dp")]

    def test_empty_dim_names_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="sub_mesh_dim_names cannot be empty"):
            _ = mesh_2d[()]


# ===========================================================================
# Test: get_local_rank
# ===========================================================================

class TestDeviceMeshGetLocalRank:
    """Tests for get_local_rank method."""

    def test_by_name(self, mesh_2d):
        assert mesh_2d.get_local_rank("dp") == 0
        assert mesh_2d.get_local_rank("tp") == 0

    def test_by_index(self, mesh_2d):
        assert mesh_2d.get_local_rank(0) == 0
        assert mesh_2d.get_local_rank(1) == 0

    def test_different_rank(self, mesh_2d):
        with patch.object(mesh_2d, "_rank", 5):
            assert mesh_2d.get_local_rank("dp") == 1
            assert mesh_2d.get_local_rank("tp") == 1

    def test_1d_mesh_none_defaults_to_0(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3], mesh_dim_names=("dp",))
        assert dm.get_local_rank() == 0

    def test_nd_mesh_none_raises(self, mesh_2d):
        with pytest.raises(RuntimeError, match="mesh_dim.*needs to be specified"):
            mesh_2d.get_local_rank()

    def test_invalid_dim_name_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="mesh_dim.*not found"):
            mesh_2d.get_local_rank("invalid")

    def test_invalid_dim_index_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="mesh_dim must be an integer"):
            mesh_2d.get_local_rank(10)

    def test_rank_not_in_list_raises(self, mesh_2d):
        with patch.object(mesh_2d, "_rank", 99):
            with pytest.raises(ValueError, match="Current rank.*not found in rank_list"):
                mesh_2d.get_local_rank("dp")


# ===========================================================================
# Test: flatten
# ===========================================================================

class TestDeviceMeshFlatten:
    """Tests for the flatten method."""

    def test_flatten_default_name(self, mesh_3d, mock_platform):
        _ = mock_platform
        dp_cp = mesh_3d[("dp", "cp")]
        flat = dp_cp.flatten()

        assert flat.mesh_shape == (4,)
        assert flat.mesh_dim_names == ("dp_cp",)
        assert flat.rank_list == (0, 2, 4, 6)
        assert flat.root_mesh is mesh_3d

    def test_flatten_custom_name(self, mesh_3d, mock_platform):
        _ = mock_platform
        dp_cp = mesh_3d[("dp", "cp")]
        flat = dp_cp.flatten(mesh_dim_name="my_flat")

        assert flat.mesh_dim_names == ("my_flat",)

    def test_flatten_accessible_from_root(self, mesh_3d, mock_platform):
        _ = mock_platform
        dp_cp = mesh_3d[("dp", "cp")]
        dp_cp.flatten()

        flat_via_root = mesh_3d["dp_cp"]
        assert flat_via_root.mesh_shape == (4,)

    def test_flatten_1d_returns_self(self, mesh_2d):
        dp = mesh_2d["dp"]
        result = dp.flatten()
        assert result is dp

    def test_flatten_cached(self, mesh_3d, mock_platform):
        _ = mock_platform
        dp_cp = mesh_3d[("dp", "cp")]
        f1 = dp_cp.flatten()
        f2 = dp_cp.flatten()
        assert f1 is f2

    def test_flatten_name_conflict_raises(self, mesh_2d, mock_platform):
        _ = mock_platform
        with pytest.raises(ValueError, match="already exists in the root mesh"):
            mesh_2d.flatten(mesh_dim_name="dp")

    def test_flatten_inherits_device_type(self, mesh_3d, mock_platform):
        _ = mock_platform
        dp_cp = mesh_3d[("dp", "cp")]
        flat = dp_cp.flatten()
        assert flat.device_type() == "npu"


# ===========================================================================
# Test: get_group
# ===========================================================================

class TestDeviceMeshGetGroup:
    """Tests for get_group method."""

    def test_by_name_and_index_return_same(self, mesh_2d):
        g_name = mesh_2d.get_group("dp")
        g_idx = mesh_2d.get_group(0)
        assert g_name is g_idx

    def test_1d_mesh_none_returns_group(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3], mesh_dim_names=("dp",))
        group = dm.get_group()
        assert group is not None

    def test_nd_mesh_none_raises(self, mesh_2d):
        with pytest.raises(RuntimeError, match="mesh_dim.*needs to be specified"):
            mesh_2d.get_group()

    def test_no_init_backend_raises(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3], _init_backend=False)
        with pytest.raises(RuntimeError, match="process groups not initialized"):
            dm.get_group()

    def test_flatten_group_via_root(self, mesh_3d, mock_platform):
        _ = mock_platform
        dp_cp = mesh_3d[("dp", "cp")]
        dp_cp.flatten()

        group = mesh_3d.get_group("dp_cp")
        assert group is not None


# ===========================================================================
# Test: get_all_groups
# ===========================================================================

class TestDeviceMeshGetAllGroups:
    """Tests for get_all_groups method."""

    def test_returns_list_of_correct_length(self, mesh_2d):
        groups = mesh_2d.get_all_groups()
        assert isinstance(groups, list)
        assert len(groups) == 2

    def test_1d_mesh(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3], mesh_dim_names=("dp",))
        groups = dm.get_all_groups()
        assert len(groups) == 1

    def test_3d_mesh(self, mesh_3d):
        groups = mesh_3d.get_all_groups()
        assert len(groups) == 3

    def test_consistent_with_get_group(self, mesh_2d):
        all_groups = mesh_2d.get_all_groups()
        for i in range(mesh_2d.ndim):
            assert all_groups[i] is mesh_2d.get_group(i)

    def test_no_init_backend_raises(self, mock_platform):
        _ = mock_platform
        dm = DeviceMesh("npu", [0, 1, 2, 3], _init_backend=False)
        with pytest.raises(RuntimeError, match="process groups not initialized"):
            dm.get_all_groups()


# ===========================================================================
# Test: from_group
# ===========================================================================

class TestDeviceMeshFromGroup:
    """Tests for the from_group static method."""

    def test_1d_from_group(self, mock_platform):  # pylint: disable=C0116
        _ = mock_platform
        group = Mock()
        group.group_name = "test_pg"
        mock_platform.get_process_group_ranks.return_value = [0, 1, 2, 3]

        dm = DeviceMesh.from_group(group, "npu", mesh_dim_names=("tp",))

        assert dm.mesh_shape == (4,)
        assert dm.rank_list == (0, 1, 2, 3)
        assert dm.mesh_dim_names == ("tp",)

    def test_nd_from_group(self, mock_platform):  # pylint: disable=C0116
        _ = mock_platform
        g1 = Mock()
        g1.group_name = "dp_pg"
        g2 = Mock()
        g2.group_name = "tp_pg"

        dm = DeviceMesh.from_group(
            [g1, g2],
            "npu",
            mesh=[[0, 1, 2, 3], [4, 5, 6, 7]],
            mesh_dim_names=("dp", "tp"),
        )

        assert dm.mesh_shape == (2, 4)
        assert dm.mesh_dim_names == ("dp", "tp")

    def test_1d_mesh_mismatch_raises(self, mock_platform):
        _ = mock_platform
        group = Mock()
        group.group_name = "test_pg"
        mock_platform.get_process_group_ranks.return_value = [0, 1, 2, 3]

        with pytest.raises(ValueError, match="Invalid mesh_shape"):
            DeviceMesh.from_group(group, "npu", mesh=[10, 20, 30, 40])

    def test_nd_no_mesh_raises(self, mock_platform):
        _ = mock_platform
        g1 = Mock()
        g1.group_name = "g1"
        with pytest.raises(ValueError, match="mesh_shape is must specified"):
            DeviceMesh.from_group([g1], "npu")

    def test_nd_dim_mismatch_raises(self, mock_platform):
        _ = mock_platform
        g1 = Mock()
        g1.group_name = "g1"
        with pytest.raises(ValueError, match="mesh dimensions must match group dimensions"):
            DeviceMesh.from_group([g1], "npu", mesh=[[0, 1], [2, 3]])

    def test_1d_string_group(self, mock_platform):
        """MindSpore-style string group names."""
        _ = mock_platform
        mock_platform.get_process_group_ranks.return_value = [0, 1]

        dm = DeviceMesh.from_group("my_group_name", "npu", mesh_dim_names=("dp",))

        assert dm.mesh_shape == (2,)
        assert dm.rank_list == (0, 1)


# ===========================================================================
# Test: axis helpers
# ===========================================================================

class TestDeviceMeshAxisHelpers:
    """Tests for axis_id, axis_index, and get_device_num_along_axis."""

    def test_axis_id(self, mesh_2d):
        assert mesh_2d.axis_id("tp") == 0
        assert mesh_2d.axis_id("dp") == 1

    def test_axis_id_none_string(self, mesh_2d):
        assert mesh_2d.axis_id("None") == -1

    def test_axis_id_invalid_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="axis name must be one of"):
            mesh_2d.axis_id("invalid")

    def test_axis_index(self, mesh_2d):
        assert mesh_2d.axis_index("dp") == 0
        assert mesh_2d.axis_index("tp") == 1

    def test_get_device_num_along_axis(self, mesh_2d):
        assert mesh_2d.get_device_num_along_axis("dp") == 2
        assert mesh_2d.get_device_num_along_axis("tp") == 4


# ===========================================================================
# Test: get_rank_list_along_axis / get_devices_for_axis
# ===========================================================================

class TestDeviceMeshRankListAlongAxis:
    """Tests for get_rank_list_along_axis and get_devices_for_axis."""

    def test_rank_list_along_dp(self, mesh_2d):
        result = mesh_2d.get_rank_list_along_axis("dp")
        assert result == [0, 4]

    def test_rank_list_along_tp(self, mesh_2d):
        result = mesh_2d.get_rank_list_along_axis("tp")
        assert result == [0, 1, 2, 3]

    def test_rank_list_is_cached(self, mesh_2d):
        r1 = mesh_2d.get_rank_list_along_axis("dp")
        r2 = mesh_2d.get_rank_list_along_axis("dp")
        assert r1 is r2

    def test_get_devices_for_axis_by_name(self, mesh_2d):
        result = mesh_2d.get_devices_for_axis("dp", rank=0)
        assert result == [0, 4]

    def test_get_devices_for_axis_by_index(self, mesh_2d):
        result = mesh_2d.get_devices_for_axis(0, rank=0)
        assert result == [0, 4]

    def test_get_devices_for_axis_different_rank(self, mesh_2d):
        result = mesh_2d.get_devices_for_axis("tp", rank=5)
        assert result == [4, 5, 6, 7]

    def test_invalid_axis_name_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="not found in mesh_dim_names"):
            mesh_2d.get_rank_list_along_axis("invalid")

    def test_rank_not_in_list_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="Rank.*not found in rank_list"):
            mesh_2d.get_devices_for_axis("dp", rank=99)


# ===========================================================================
# Test: get_global_shape
# ===========================================================================

class TestDeviceMeshGetGlobalShape:
    """Tests for get_global_shape method."""

    def test_basic(self, mesh_2d):
        global_shape = mesh_2d.get_global_shape((16, 32), (1, 0))
        assert global_shape == (16 * 4, 32 * 2)

    def test_replicate_dim(self, mesh_2d):
        global_shape = mesh_2d.get_global_shape((16, 32), (-1, 0))
        assert global_shape == (16, 32 * 2)

    def test_cached(self, mesh_2d):
        s1 = mesh_2d.get_global_shape((16, 32), (1, 0))
        s2 = mesh_2d.get_global_shape((16, 32), (1, 0))
        assert s1 == s2

    def test_none_tensor_map_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="tensor_map is not set"):
            mesh_2d.get_global_shape((16, 32), None)

    def test_shape_map_length_mismatch_raises(self, mesh_2d):
        with pytest.raises(ValueError, match="Length of slice_shape.*must match"):
            mesh_2d.get_global_shape((16,), (1, 0))


# ===========================================================================
# Test: repr / str
# ===========================================================================

class TestDeviceMeshRepr:
    """Tests for __repr__ and __str__."""

    def test_repr_contains_info(self, mesh_2d):
        r = repr(mesh_2d)
        assert "DeviceMesh" in r
        assert "(2, 4)" in r
        assert "dp" in r
        assert "tp" in r

    def test_str_equals_repr(self, mesh_2d):
        assert str(mesh_2d) == repr(mesh_2d)


# ===========================================================================
# Test: to_hash
# ===========================================================================

class TestDeviceMeshHash:
    """Tests for to_hash method."""

    def test_to_hash_deterministic(self, mesh_2d):
        h1 = mesh_2d.to_hash()
        h2 = mesh_2d.to_hash()
        assert h1 == h2

    def test_to_hash_contains_expected_fields(self, mesh_2d):
        h = mesh_2d.to_hash()
        assert h == ((2, 4), ("dp", "tp"), (0, 7))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
