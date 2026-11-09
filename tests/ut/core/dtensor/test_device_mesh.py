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
"""Coverage supplement tests for hyper_parallel.core.dtensor.device_mesh."""

import os


import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import torch

from hyper_parallel.core.dtensor.device_mesh import (
    DeviceMesh,
    _MeshEnv,
    _DEVICE_MESH_MAP,
    _normalize_backend_value,
    _normalize_backend_override,
    _should_defer_group_init,
    _get_sub_rank_list,
    init_device_mesh,
)
from hyper_parallel.core.dtensor._mesh_layout import _MeshLayout, _contiguous_strides
from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS


def _setup_mock_platform(platform_mock):
    """Configure a mock platform for DeviceMesh construction."""
    platform_mock.get_rank.return_value = 0
    platform_mock.get_world_size.return_value = 8
    platform_mock.Tensor = torch.Tensor
    mock_group = Mock()
    mock_group.group_name = "mock_group"

    def _split_group_side_effect(split_ranks, parent_pg=None, timeout=None, pg_options=None, group_desc=None):
        """ split group side effect."""
        for sr in split_ranks:
            key = str(tuple(sorted(sr)))
            EXISTING_COMM_GROUPS[key] = mock_group
        return mock_group

    platform_mock.split_group.side_effect = _split_group_side_effect
    platform_mock.get_created_group.return_value = None
    platform_mock.get_process_group_ranks.return_value = list(range(8))

    def mock_tensor_to_numpy(tensor):
        """Mock tensor to numpy."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    platform_mock.tensor_to_numpy.side_effect = mock_tensor_to_numpy


class _MockedDeviceMeshTestCase(unittest.TestCase):
    """Base class that patches the device_mesh distributed backend."""

    def setUp(self):
        """Set up test fixtures."""
        patcher_dm = patch("hyper_parallel.core.dtensor.device_mesh.dist", MagicMock())
        patcher_tensor = patch("hyper_parallel.core.dtensor.device_mesh.Tensor", torch.Tensor)
        self.mock_platform = patcher_dm.start()
        patcher_tensor.start()
        _setup_mock_platform(self.mock_platform)
        self.addCleanup(patcher_dm.stop)
        self.addCleanup(patcher_tensor.stop)
        self.addCleanup(_DEVICE_MESH_MAP.clear)
        self.addCleanup(EXISTING_COMM_GROUPS.clear)
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()


# ===========================================================================
# _MeshEnv tests
# ===========================================================================

class TestMeshEnv(unittest.TestCase):
    """Tests for MeshEnv."""
    def test_empty_stack_raises(self):
        """Test empty stack raises."""
        env = _MeshEnv()
        with self.assertRaises(RuntimeError):
            env.get_current_mesh()

    def test_push_pop(self):
        """Test push pop."""
        env = _MeshEnv()
        sentinel = object()
        env.mesh_stack.append(sentinel)
        self.assertIs(env.get_current_mesh(), sentinel)
        env.mesh_stack.pop()
        with self.assertRaises(RuntimeError):
            env.get_current_mesh()


# ===========================================================================
# DeviceMesh construction tests
# ===========================================================================

class TestDeviceMeshConstruction(_MockedDeviceMeshTestCase):
    """Tests for DeviceMeshConstruction."""
    def test_from_list(self):
        """Test from list."""
        dm = DeviceMesh("npu", [[0, 1, 2, 3], [4, 5, 6, 7]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.mesh_shape, (2, 4))
        self.assertEqual(dm.ndim, 2)
        self.assertEqual(dm.rank_list, (0, 1, 2, 3, 4, 5, 6, 7))

    def test_from_tuple(self):
        """Test from tuple."""
        dm = DeviceMesh("npu", ((0, 1), (2, 3)), mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.mesh_shape, (2, 2))

    def test_from_ndarray(self):
        """Test from ndarray."""
        arr = np.array([[0, 1], [2, 3]])
        dm = DeviceMesh("npu", arr, mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.mesh_shape, (2, 2))

    def test_from_tensor(self):
        """Test from tensor."""
        t = torch.tensor([[0, 1], [2, 3]])
        dm = DeviceMesh("npu", t, mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.mesh_shape, (2, 2))

    def test_mesh_dim_names(self):
        """Test mesh dim names."""
        dm = DeviceMesh("npu", [0, 1, 2, 3], mesh_dim_names=("x",), _init_backend=False)
        self.assertEqual(dm.mesh_dim_names, ("x",))

    def test_rank_list_property(self):
        """Test rank list property."""
        dm = DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.rank_list, (0, 1, 2, 3))


# ===========================================================================
# Context manager tests
# ===========================================================================

class TestDeviceMeshContextManager(_MockedDeviceMeshTestCase):
    """Tests for DeviceMeshContextManager."""
    def test_enter_exit(self):
        """Test enter exit."""
        dm = DeviceMesh("npu", [0, 1], mesh_dim_names=("dp",), _init_backend=False)
        from hyper_parallel.core.dtensor.device_mesh import _mesh_resources
        initial_len = len(_mesh_resources.mesh_stack)
        with dm:
            self.assertEqual(len(_mesh_resources.mesh_stack), initial_len + 1)
            self.assertIs(_mesh_resources.get_current_mesh(), dm)
        self.assertEqual(len(_mesh_resources.mesh_stack), initial_len)

    def test_nested_with(self):
        """Test nested with."""
        dm1 = DeviceMesh("npu", [0, 1], mesh_dim_names=("dp",), _init_backend=False)
        dm2 = DeviceMesh("npu", [0, 1], mesh_dim_names=("tp",), _init_backend=False)
        from hyper_parallel.core.dtensor.device_mesh import _mesh_resources
        with dm1:
            self.assertIs(_mesh_resources.get_current_mesh(), dm1)
            with dm2:
                self.assertIs(_mesh_resources.get_current_mesh(), dm2)
            self.assertIs(_mesh_resources.get_current_mesh(), dm1)


# ===========================================================================
# __getitem__ sub-mesh slicing tests
# ===========================================================================

class TestDeviceMeshGetItem(_MockedDeviceMeshTestCase):
    """Tests for DeviceMeshGetItem."""
    def test_by_name(self):
        """Test by name."""
        dm = DeviceMesh("npu", [[0, 1, 2, 3], [4, 5, 6, 7]],
                        mesh_dim_names=("dp", "tp"), _init_backend=False)
        sub = dm["dp"]
        self.assertIsInstance(sub, DeviceMesh)
        self.assertEqual(sub.ndim, 1)

    def test_multi_name_slice(self):
        """Test multi name slice."""
        dm = DeviceMesh("npu", np.arange(8).reshape(2, 2, 2),
                        mesh_dim_names=("dp", "cp", "tp"), _init_backend=False)
        sub = dm["dp", "tp"]
        self.assertIsInstance(sub, DeviceMesh)

    def test_cache(self):
        """Test cache."""
        dm = DeviceMesh("npu", [[0, 1, 2, 3], [4, 5, 6, 7]],
                        mesh_dim_names=("dp", "tp"), _init_backend=False)
        sub1 = dm["dp"]
        sub2 = dm["dp"]
        self.assertIs(sub1, sub2)


# ===========================================================================
# get_coordinate tests
# ===========================================================================

class TestGetCoordinate(_MockedDeviceMeshTestCase):
    """Tests for GetCoordinate."""
    def test_rank_in_mesh(self):
        """Test rank in mesh."""
        self.mock_platform.get_rank.return_value = 0
        dm = DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        coord = dm.get_coordinate()
        self.assertIsNotNone(coord)
        self.assertEqual(len(coord), 2)

    def test_rank_not_in_mesh(self):
        """Test rank not in mesh."""
        self.mock_platform.get_rank.return_value = 99
        dm = DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        coord = dm.get_coordinate()
        self.assertIsNone(coord)


# ===========================================================================
# get_local_rank tests
# ===========================================================================

class TestGetLocalRank(_MockedDeviceMeshTestCase):
    """Tests for GetLocalRank."""
    def test_by_name(self):
        """Test by name."""
        self.mock_platform.get_rank.return_value = 0
        dm = DeviceMesh("npu", [[0, 1, 2, 3], [4, 5, 6, 7]],
                        mesh_dim_names=("dp", "tp"), _init_backend=False)
        rank = dm.get_local_rank("dp")
        self.assertIsInstance(rank, (int, np.integer))

    def test_1d_mesh_none(self):
        """Test 1d mesh none."""
        self.mock_platform.get_rank.return_value = 0
        dm = DeviceMesh("npu", [0, 1, 2, 3], mesh_dim_names=("dp",), _init_backend=False)
        rank = dm.get_local_rank()
        self.assertIsInstance(rank, (int, np.integer))


# ===========================================================================
# flatten tests
# ===========================================================================

class TestFlatten(_MockedDeviceMeshTestCase):
    """Tests for Flatten."""
    def test_flatten_2d(self):
        """Test flatten 2d."""
        dm = DeviceMesh("npu", [[0, 1, 2, 3], [4, 5, 6, 7]],
                        mesh_dim_names=("dp", "tp"), _init_backend=False)
        flat = dm.flatten("dp_tp")
        self.assertEqual(flat.ndim, 1)

    def test_flatten_1d_returns_self(self):
        """Test flatten 1d returns self."""
        dm = DeviceMesh("npu", [0, 1, 2, 3], mesh_dim_names=("dp",), _init_backend=False)
        flat = dm.flatten()
        self.assertIs(flat, dm)


# ===========================================================================
# Standalone function tests
# ===========================================================================

class TestNormalizeBackendValue(unittest.TestCase):
    """Tests for NormalizeBackendValue."""
    def test_none(self):
        """Test none."""
        self.assertIsNone(_normalize_backend_value(None))

    def test_string(self):
        """Test string."""
        self.assertEqual(_normalize_backend_value("nccl"), "nccl")

    def test_tuple_with_string(self):
        """Test tuple with string."""
        self.assertEqual(_normalize_backend_value(("gloo",)), "gloo")

    def test_tuple_with_none(self):
        """Test tuple with none."""
        self.assertIsNone(_normalize_backend_value((None,)))

    def test_other(self):
        """Test other."""
        self.assertIsNone(_normalize_backend_value(42))

    def test_empty_tuple(self):
        """Test empty tuple."""
        self.assertIsNone(_normalize_backend_value(()))


class TestNormalizeBackendOverride(unittest.TestCase):
    """Tests for NormalizeBackendOverride."""
    def test_by_index(self):
        """Test by index."""
        result = _normalize_backend_override({0: "nccl"}, 2)
        self.assertEqual(result, ("nccl", None))

    def test_by_name(self):
        """Test by name."""
        result = _normalize_backend_override({"dp": "nccl"}, 2, ("dp", "tp"))
        self.assertEqual(result, ("nccl", None))

    def test_redundant_raises(self):
        """Test redundant raises."""
        with self.assertRaises(RuntimeError):
            _normalize_backend_override({0: "nccl", "dp": "gloo"}, 2, ("dp", "tp"))

    def test_invalid_key_raises(self):
        """Test invalid key raises."""
        with self.assertRaises(RuntimeError):
            _normalize_backend_override({"invalid": "nccl"}, 2, ("dp", "tp"))


class TestShouldDeferGroupInit(unittest.TestCase):
    """Tests for ShouldDeferGroupInit."""
    def test_fake_backend(self):
        """Test fake backend."""
        ml = _MeshLayout((2, 4), (4, 1))
        self.assertTrue(_should_defer_group_init(ml, "fake"))

    def test_numel_1(self):
        """Test numel 1."""
        ml = _MeshLayout(1, 1)
        self.assertTrue(_should_defer_group_init(ml, None))

    def test_normal(self):
        """Test normal."""
        ml = _MeshLayout((2, 4), (4, 1))
        self.assertFalse(_should_defer_group_init(ml, None))


class TestGetSubRankList(unittest.TestCase):
    """Tests for GetSubRankList."""
    def test_basic(self):
        """Test basic."""
        mesh_shape = (2, 4)
        mesh_dim_names = ("dp", "tp")
        rank_list = tuple(range(8))
        sub = _get_sub_rank_list(mesh_shape, mesh_dim_names, rank_list, ("tp",), 0)
        self.assertEqual(len(sub), 4)
        self.assertIn(0, sub)

    def test_single_dim(self):
        """Test single dim."""
        mesh_shape = (2, 4)
        mesh_dim_names = ("dp", "tp")
        rank_list = tuple(range(8))
        sub = _get_sub_rank_list(mesh_shape, mesh_dim_names, rank_list, ("dp",), 0)
        self.assertEqual(len(sub), 2)
        self.assertIn(0, sub)


# ===========================================================================
# DeviceMesh validate_device_type tests
# ===========================================================================

class TestValidateDeviceType(_MockedDeviceMeshTestCase):
    """Tests for ValidateDeviceType."""
    def test_valid_device_type(self):
        """Test valid device type."""
        # Should not raise
        dm = DeviceMesh("npu", [0, 1], mesh_dim_names=("dp",), _init_backend=False)
        self.assertEqual(dm.device_type, "npu")

    def test_cpu_device_type(self):
        """Test cpu device type."""
        dm = DeviceMesh("cpu", [0, 1], mesh_dim_names=("dp",), _init_backend=False)
        self.assertEqual(dm.device_type, "cpu")


# ===========================================================================
# DeviceMesh convert_mesh_to_tensor tests
# ===========================================================================

class TestConvertMeshToTensor(_MockedDeviceMeshTestCase):
    """Tests for ConvertMeshToTensor."""
    def test_list_input(self):
        """Test list input."""
        dm = DeviceMesh("npu", [0, 1, 2, 3], mesh_dim_names=("dp",), _init_backend=False)
        self.assertEqual(dm.mesh_shape, (4,))

    def test_nested_list(self):
        """Test nested list."""
        dm = DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.mesh_shape, (2, 2))

    def test_numpy_input(self):
        """Test numpy input."""
        arr = np.array([[0, 1], [2, 3]])
        dm = DeviceMesh("npu", arr, mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.mesh_shape, (2, 2))


# ===========================================================================
# DeviceMesh repr/hash tests
# ===========================================================================

class TestDeviceMeshRepr(_MockedDeviceMeshTestCase):
    """Tests for DeviceMeshRepr."""
    def test_repr(self):
        """Test repr."""
        dm = DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        r = repr(dm)
        self.assertIn("DeviceMesh", r)
        self.assertIn("npu", r)

    def test_str_equals_repr(self):
        """Test str equals repr."""
        dm = DeviceMesh("npu", [0, 1], mesh_dim_names=("dp",), _init_backend=False)
        self.assertEqual(str(dm), repr(dm))

    def test_to_hash_deterministic(self):
        """Test to hash deterministic."""
        dm = DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        h1 = dm.to_hash()
        h2 = dm.to_hash()
        self.assertEqual(h1, h2)


# ===========================================================================
# DeviceMesh size / shape tests
# ===========================================================================

class TestDeviceMeshSize(_MockedDeviceMeshTestCase):
    """Tests for DeviceMeshSize."""
    def test_size_no_dim(self):
        """Test size no dim."""
        dm = DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.size(), 4)

    def test_size_with_dim(self):
        """Test size with dim."""
        dm = DeviceMesh("npu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.size(0), 2)
        self.assertEqual(dm.size(1), 2)

    def test_shape_property(self):
        """Test shape property."""
        dm = DeviceMesh("npu", [[0, 1, 2, 3], [4, 5, 6, 7]],
                        mesh_dim_names=("dp", "tp"), _init_backend=False)
        self.assertEqual(dm.shape, (2, 4))


# ===========================================================================
# init_device_mesh tests
# ===========================================================================

class TestInitDeviceMesh(_MockedDeviceMeshTestCase):
    """Tests for InitDeviceMesh."""
    def test_basic(self):
        """Test basic."""
        dm = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
        self.assertEqual(dm.mesh_shape, (2, 4))

    def test_cache(self):
        """Test cache."""
        dm1 = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
        dm2 = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
        self.assertIs(dm1, dm2)

    def test_rank_list_mismatch_raises(self):
        """Test rank list mismatch raises."""
        with self.assertRaises(ValueError):
            init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"),
                             rank_list=(0, 1, 2), init_backend=False)


# ===========================================================================
# Process-group split derivation tests
# ===========================================================================

def _brute_force_split_ranks(mesh_shape, mesh_dim_names, dim_name, rank_list):
    """Reference subgroup list derived from the per-rank helper."""
    dim_names = (dim_name,) if not isinstance(dim_name, tuple) else dim_name
    groups = {
        tuple(sorted(_get_sub_rank_list(mesh_shape, mesh_dim_names, rank_list, dim_names, rank)))
        for rank in rank_list
    }
    return sorted(list(group) for group in groups)


class TestInitOneProcessGroup(_MockedDeviceMeshTestCase):
    """Tests for DeviceMesh._init_one_process_group."""

    def _run(self, mesh_shape, mesh_dim_names, dim_name, rank_list):
        captured = {}

        def _split_group_side_effect(*, split_ranks):
            captured["split_ranks"] = split_ranks

        with patch("hyper_parallel.core.dtensor.device_mesh._utils.split_group",
                   side_effect=_split_group_side_effect):
            group_key = DeviceMesh._init_one_process_group(
                mesh_shape, mesh_dim_names, dim_name, rank_list
            )
        return captured["split_ranks"], group_key

    def test_matches_per_rank_reference(self):
        """Vectorized derivation must match the per-rank helper for every axis."""
        cases = [
            ((4,), ("dp",), "dp", tuple(range(4))),
            ((2, 2), ("dp", "tp"), "dp", tuple(range(4))),
            ((2, 2), ("dp", "tp"), "tp", tuple(range(4))),
            ((2, 3), ("dp", "tp"), "tp", tuple(range(6))),
            ((2, 2, 3), ("dp", "cp", "tp"), "cp", tuple(range(12))),
            ((2, 2, 3), ("dp", "cp", "tp"), ("dp", "tp"), tuple(range(12))),
        ]
        for mesh_shape, names, dim_name, rank_list in cases:
            with self.subTest(mesh_shape=mesh_shape, dim_name=dim_name):
                split_ranks, group_key = self._run(mesh_shape, names, dim_name, rank_list)
                expected = _brute_force_split_ranks(mesh_shape, names, dim_name, rank_list)
                self.assertEqual(split_ranks, expected)

                dim_names = (dim_name,) if not isinstance(dim_name, tuple) else dim_name
                expected_key = str(tuple(sorted(
                    _get_sub_rank_list(mesh_shape, names, rank_list, dim_names, 0)
                )))
                self.assertEqual(group_key, expected_key)

                # Subgroups must form a partition of the mesh ranks.
                self.assertEqual(
                    sorted(rank for group in split_ranks for rank in group),
                    sorted(rank_list),
                )

    def test_current_rank_outside_mesh_returns_none(self):
        """A process that is not part of the mesh must not report a group key."""
        self.mock_platform.get_rank.return_value = 99
        split_ranks, group_key = self._run((2, 2), ("dp", "tp"), "dp", tuple(range(4)))
        self.assertEqual(split_ranks, [[0, 2], [1, 3]])
        self.assertIsNone(group_key)


class TestBuildDimSplitRanks(_MockedDeviceMeshTestCase):
    """Tests for DeviceMesh._build_dim_split_ranks."""

    @staticmethod
    def _rank_map(ranks):
        return DeviceMesh._convert_rank_map_to_tensor(tuple(ranks))

    def test_contiguous_layout_matches_per_rank_reference(self):
        """Every contiguous axis must match the per-rank helper."""
        cases = [
            ((4,), ("dp",)),
            ((2, 4), ("dp", "tp")),
            ((2, 2, 2), ("dp", "cp", "tp")),
        ]
        for mesh_shape, names in cases:
            numel = int(np.prod(mesh_shape))
            rank_list = tuple(range(numel))
            rank_map = self._rank_map(rank_list)
            layout = _MeshLayout(mesh_shape, _contiguous_strides(mesh_shape))
            for dim, sub_layout in enumerate(layout):
                with self.subTest(mesh_shape=mesh_shape, dim=dim):
                    split_ranks, group_key = DeviceMesh._build_dim_split_ranks(
                        sub_layout, rank_map
                    )
                    expected = _brute_force_split_ranks(
                        mesh_shape, names, names[dim], rank_list
                    )
                    self.assertEqual(split_ranks, expected)

                    expected_key = str(tuple(sorted(_get_sub_rank_list(
                        mesh_shape, names, rank_list, (names[dim],), 0
                    ))))
                    self.assertEqual(group_key, expected_key)

    def test_non_contiguous_layout(self):
        """Interleaved strides must still yield the strided rank groups."""
        sub_layout = _MeshLayout((2, 2), (4, 1))
        rank_map = self._rank_map(range(8))
        split_ranks, group_key = DeviceMesh._build_dim_split_ranks(sub_layout, rank_map)
        self.assertEqual(split_ranks, [[0, 1, 4, 5], [2, 3, 6, 7]])
        self.assertEqual(group_key, "(0, 1, 4, 5)")

    def test_non_contiguous_layout_respects_rank_map_order(self):
        """Group membership follows the rank map, not the physical rank id."""
        sub_layout = _MeshLayout((2, 2), (4, 1))
        rank_map = self._rank_map([4, 5, 6, 7, 0, 1, 2, 3])
        split_ranks, group_key = DeviceMesh._build_dim_split_ranks(sub_layout, rank_map)
        self.assertEqual(split_ranks, [[0, 1, 4, 5], [2, 3, 6, 7]])
        self.assertEqual(group_key, "(0, 1, 4, 5)")

    def test_current_rank_outside_mesh_returns_none(self):
        """A process outside the rank map must not report a group key."""
        self.mock_platform.get_rank.return_value = 99
        sub_layout = _MeshLayout((2, 2), (4, 1))
        split_ranks, group_key = DeviceMesh._build_dim_split_ranks(
            sub_layout, self._rank_map(range(8))
        )
        self.assertEqual(split_ranks, [[0, 1, 4, 5], [2, 3, 6, 7]])
        self.assertIsNone(group_key)

    def test_duplicate_membership_raises(self):
        """A rank appearing in two subgroups is a layout error."""
        sub_layout = _MeshLayout((2, 2), (4, 1))
        rank_map = self._rank_map([0, 1, 0, 1, 0, 1, 0, 1])
        with self.assertRaises(RuntimeError):
            DeviceMesh._build_dim_split_ranks(sub_layout, rank_map)


if __name__ == "__main__":
    unittest.main()
