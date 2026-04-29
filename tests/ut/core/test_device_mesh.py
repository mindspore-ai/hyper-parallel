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
"""Unit tests for DeviceMesh functionality.

This module provides comprehensive unit tests for the DeviceMesh class and
related functions in the hyper_parallel framework. All tests use mocking to avoid
dependencies on actual hardware or distributed communication.

Note:
    All test methods decorated with ``@patch("hyper_parallel.core.dtensor.device_mesh.platform")``
    receive a ``mock_platform`` argument injected by the patch decorator.
"""
import copy
import os
import threading
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# Set platform to torch for testing

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    DeviceMesh,
    _DEVICE_MESH_MAP,
    _get_sub_rank_list,
    _mesh_resources,
    _create_device_mesh,
    init_device_mesh,
)
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


@unittest.skip("Skipped: all TestDeviceMesh cases (full UT session may hit MindSpore/Ascend init on some hosts).")
class TestDeviceMesh(unittest.TestCase):
    """Unit tests for DeviceMesh class and related functions."""

    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _mesh_resources.mesh_stack.clear()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests.

        Args:
            mock_platform: The MagicMock object injected by @patch.
            platform_type: Optional PlatformType to set on the mock.
            world_size: Value returned by mock_platform.get_world_size().
        """
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2×4 (dp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2×2×2 (dp, cp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))

    def _make_2x2_mesh_no_backend(self, mock_platform, mesh_dim_names=("dp", "tp")):
        """Set up mock and return a 2×2 DeviceMesh created without backend init."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return DeviceMesh("npu", mesh=[[0, 1], [2, 3]], mesh_dim_names=mesh_dim_names, _init_backend=False)

    def _build_torchtitan_like_meshes_8card_cp2(self, mock_platform):
        """Build TorchTitan-like meshes for an 8-card cp=2 topology."""
        self._setup_mock_platform(mock_platform, world_size=8)
        world_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("world",))

        dataloading_mesh = world_mesh._unflatten(
            0,
            (1, 4, 2, 1),
            ("pp", "batch", "cp", "tp"),
            backend_override={"pp": "fake", "tp": "fake"},
        )
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
        dense_mesh = world_mesh._unflatten(
            0,
            (1, 1, 8, 1),
            ("pp", "dp_replicate", "fsdp", "tp"),
            backend_override={"pp": "fake", "tp": "fake"},
        )
        return world_mesh, dataloading_mesh, loss_mesh, dense_mesh

    # ------------------------------------------------------------------
    # Construction tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_init_device_mesh_basic(self, mock_platform):
        """Test basic DeviceMesh construction with explicit mesh.

        Scenario: Create a 2x2 DeviceMesh directly with explicit mesh tensor.
        Expected behavior: DeviceMesh should be created with correct shape,
        dimension names, and rank list.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mock_group = MagicMock()
        mock_platform.split_group.return_value = mock_group

        mesh = DeviceMesh(
            device_type="npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )

        self.assertEqual(mesh.mesh_shape, (2, 2))
        self.assertEqual(mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(mesh.rank_list, (0, 1, 2, 3))
        self.assertEqual(mesh.ndim, 2)
        self.assertEqual(mesh.rank, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_init_device_mesh_caching(self, mock_platform):
        """Test that init_device_mesh caches results correctly.

        Scenario: Call init_device_mesh twice with same parameters.
        Expected behavior: Second call should return the same cached instance.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mock_platform.split_group.return_value = MagicMock()

        mesh1 = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"))
        mesh2 = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"))

        self.assertIs(mesh1, mesh2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_with_tensor(self, mock_platform):
        """Test DeviceMesh construction with custom mesh tensor.

        Scenario: Create DeviceMesh with a 2x2 tensor mesh.
        Expected behavior: DeviceMesh should have correct shape (2, 2).
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh_tensor = self.platform.tensor([[0, 2], [1, 3]])

        device_mesh = DeviceMesh("npu", mesh_tensor, mesh_dim_names=("dp", "tp"))

        self.assertEqual(device_mesh.mesh_shape, (2, 2))
        self.assertEqual(device_mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(device_mesh.rank_list, (0, 2, 1, 3))
        self.assertEqual(device_mesh.ndim, 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_with_list(self, mock_platform):
        """Test DeviceMesh construction with list input."""
        self._setup_mock_platform(mock_platform, world_size=4)

        device_mesh = DeviceMesh("npu", [[0, 2], [1, 3]], mesh_dim_names=("dp", "tp"))

        self.assertEqual(device_mesh.mesh_shape, (2, 2))
        self.assertEqual(device_mesh.rank_list, (0, 2, 1, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_with_numpy(self, mock_platform):
        """Test DeviceMesh construction with numpy array mesh.

        Scenario: Create DeviceMesh with numpy array mesh.
        Expected behavior: DeviceMesh should be created successfully.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh = np.array([[0, 2], [1, 3]], dtype=np.int64)

        device_mesh = DeviceMesh("npu", mesh, mesh_dim_names=("dp", "tp"))

        self.assertEqual(device_mesh.mesh_shape, (2, 2))
        self.assertEqual(device_mesh.rank_list, (0, 2, 1, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_with_none_mesh(self, mock_platform):
        """Test DeviceMesh with mesh=None (auto 1D mesh).

        Scenario: Create DeviceMesh without explicit mesh (mesh=None).
        Expected behavior: DeviceMesh should auto-generate 1D mesh based on world_size.
        """
        self._setup_mock_platform(mock_platform, world_size=4)

        device_mesh = DeviceMesh("npu", mesh=None, _init_backend=False)

        self.assertEqual(device_mesh.mesh_shape, (4,))
        self.assertEqual(device_mesh.ndim, 1)
        self.assertEqual(device_mesh.rank_list, (0, 1, 2, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_without_dim_names(self, mock_platform):
        """Test DeviceMesh without mesh_dim_names.

        Scenario: Create DeviceMesh without mesh_dim_names.
        Expected behavior: DeviceMesh should work with default behavior.
        """
        self._setup_mock_platform(mock_platform, world_size=4)

        mesh = DeviceMesh("npu", mesh=[0, 1, 2, 3], _init_backend=False)

        self.assertIsNone(mesh.mesh_dim_names)
        self.assertEqual(mesh.mesh_shape, (4,))
        self.assertEqual(mesh.rank_list, (0, 1, 2, 3))
        self.assertEqual(mesh.axis_id("None"), -1)

    # ------------------------------------------------------------------
    # __getitem__ / slicing tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_getitem_single_dim(self, mock_platform):
        """Test DeviceMesh __getitem__ with single dimension."""
        mesh = self._make_2x4_mesh(mock_platform)

        dp_mesh = mesh["dp"]
        tp_mesh = mesh["tp"]

        self.assertEqual(dp_mesh.mesh_shape, (2,))
        self.assertEqual(dp_mesh.mesh_dim_names, ("dp",))
        self.assertEqual(dp_mesh.root_mesh, mesh)
        self.assertEqual(dp_mesh.rank_list, (0, 4))

        self.assertEqual(tp_mesh.mesh_shape, (4,))
        self.assertEqual(tp_mesh.mesh_dim_names, ("tp",))
        self.assertEqual(tp_mesh.root_mesh, mesh)
        self.assertEqual(tp_mesh.rank_list, (0, 1, 2, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_getitem_multiple_dims(self, mock_platform):
        """Test DeviceMesh __getitem__ with multiple dimensions."""
        mesh = self._make_2x2x2_mesh(mock_platform)

        dp_cp_mesh = mesh[("dp", "cp")]

        self.assertEqual(dp_cp_mesh.mesh_shape, (2, 2))
        self.assertEqual(dp_cp_mesh.mesh_dim_names, ("dp", "cp"))
        self.assertEqual(dp_cp_mesh.rank_list, (0, 2, 4, 6))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_getitem_no_dim_names_raises(self, mock_platform):
        """Test DeviceMesh __getitem__ without mesh_dim_names raises RuntimeError.

        Scenario: Access submesh without setting mesh_dim_names.
        Expected behavior: Should raise RuntimeError with appropriate message.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh = DeviceMesh("npu", mesh=[0, 1, 2, 3], _init_backend=False)

        with self.assertRaises(RuntimeError) as context:
            _ = mesh["dp"]
        self.assertIn("Cannot slice a DeviceMesh without mesh_dim_names", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_getitem_invalid_dim_name(self, mock_platform):
        """Test DeviceMesh __getitem__ with invalid dimension name raises KeyError.

        Scenario: Access submesh with invalid dimension name.
        Expected behavior: Should raise KeyError with appropriate message.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

        with self.assertRaises(KeyError) as context:
            _ = mesh["invalid_dim"]
        self.assertIn("invalid_dim", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_getitem_out_of_order(self, mock_platform):
        """Test DeviceMesh __getitem__ with out-of-order dimensions raises ValueError.

        Scenario: Access submesh with out-of-order dimension names.
        Expected behavior: Should raise ValueError with appropriate message.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)

        with self.assertRaises(ValueError) as context:
            _ = mesh[("cp", "dp")]  # Wrong order, should be ("dp", "cp")
        self.assertIn("must follow the order", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_getitem_supports_contiguous_flatten_and_original_dim(self, mock_platform):
        """Test root mesh slicing can mix a contiguous flattened dim with an original dim."""
        mesh = self._make_2x2x2_mesh(mock_platform)

        flat_dp_cp = mesh[("dp", "cp")].flatten()
        mixed_mesh = mesh[("dp_cp", "tp")]

        self.assertIs(mesh["dp_cp"], flat_dp_cp)
        self.assertEqual(mixed_mesh.mesh_shape, (4, 2))
        self.assertEqual(mixed_mesh.mesh_dim_names, ("dp_cp", "tp"))
        self.assertEqual(mixed_mesh.rank_list, (0, 1, 2, 3, 4, 5, 6, 7))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_getitem_rejects_non_contiguous_flatten_mix(self, mock_platform):
        """Test root mesh slicing still rejects non-contiguous flattened dims mixed with originals."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        _ = mesh[("dp", "tp")].flatten()

        with self.assertRaisesRegex(NotImplementedError, "contiguous flattened dim"):
            _ = mesh[("dp_tp", "cp")]

    # ------------------------------------------------------------------
    # concatenate tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_returns_unified_mesh_in_root_order(self, mock_platform):
        """Test concatenate combines explicit submeshes in root order."""
        mesh = self._make_2x2_mesh_no_backend(mock_platform)
        dp_mesh = mesh["dp"]
        tp_mesh = mesh["tp"]

        unified_mesh = DeviceMesh.concatenate([dp_mesh, tp_mesh])

        self.assertEqual(unified_mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(unified_mesh.mesh_shape, (2, 2))
        self.assertEqual(unified_mesh.to_hash(), mesh.to_hash())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_single_mesh_returns_input(self, mock_platform):
        """Test concatenate returns the original mesh when only one mesh is provided."""
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh = DeviceMesh(
            "npu",
            np.array([0, 1, 2, 3]),
            mesh_dim_names=("dp",),
            _init_backend=False,
        )

        result = DeviceMesh.concatenate([mesh])

        self.assertIs(result, mesh)

    def test_concatenate_rejects_empty_input(self):
        """Test concatenate raises ValueError for empty input."""
        with self.assertRaisesRegex(ValueError, "at least one mesh"):
            DeviceMesh.concatenate([])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_rejects_mismatched_root_meshes(self, mock_platform):
        """Test concatenate rejects meshes from different root meshes."""
        self._setup_mock_platform(mock_platform, world_size=8)

        dp_mesh = DeviceMesh(
            "npu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("dp", "fsdp"),
            _init_backend=False,
        )
        tp_mesh = DeviceMesh(
            "npu",
            np.array([[4, 5], [6, 7]]),
            mesh_dim_names=("tp", "ep"),
            _init_backend=False,
        )

        with self.assertRaisesRegex(ValueError, "share the same root mesh"):
            DeviceMesh.concatenate([dp_mesh, tp_mesh])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_rejects_duplicate_mesh_dims(self, mock_platform):
        """Test concatenate rejects duplicated mesh dimension names."""
        mesh = self._make_2x2_mesh_no_backend(mock_platform)
        dp_mesh = mesh["dp"]

        with self.assertRaisesRegex(ValueError, "disjoint mesh dims"):
            DeviceMesh.concatenate([dp_mesh, dp_mesh])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_rejects_meshes_out_of_root_order(self, mock_platform):
        """Test concatenate rejects meshes that do not follow root mesh order."""
        mesh = self._make_2x2_mesh_no_backend(mock_platform, mesh_dim_names=("tp", "dp"))
        dp_mesh = mesh["dp"]
        tp_mesh = mesh["tp"]

        with self.assertRaisesRegex(ValueError, "follow the root mesh order"):
            DeviceMesh.concatenate([dp_mesh, tp_mesh])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_supports_contiguous_flattened_dim(self, mock_platform):
        """Test concatenate can stitch a contiguous flattened dim with an original dim."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        flat_dp_cp = mesh[("dp", "cp")].flatten()
        tp_mesh = mesh["tp"]

        unified_mesh = DeviceMesh.concatenate([flat_dp_cp, tp_mesh])

        self.assertEqual(unified_mesh.mesh_dim_names, ("dp_cp", "tp"))
        self.assertEqual(unified_mesh.mesh_shape, (4, 2))
        self.assertEqual(unified_mesh.rank_list, (0, 1, 2, 3, 4, 5, 6, 7))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_preserves_root_mesh_after_layout_deepcopy(self, mock_platform):
        """Test concatenate still works after deepcopying a layout built from a submesh."""
        mesh = self._make_2x2_mesh_no_backend(mock_platform)
        dp_mesh = mesh["dp"]
        tp_mesh = mesh["tp"]

        copied_layout = copy.deepcopy(Layout.from_device_mesh(tp_mesh))
        unified_mesh = DeviceMesh.concatenate([dp_mesh, copied_layout.mesh])

        self.assertEqual(copied_layout.mesh._get_root_mesh().to_hash(), mesh.to_hash())  # pylint: disable=W0212
        self.assertEqual(unified_mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(unified_mesh.to_hash(), mesh.to_hash())

    # ------------------------------------------------------------------
    # TorchTitan compatibility tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_supports_torchtitan_build_mesh_primitives(self, mock_platform):
        """Test world->_unflatten->slice->_flatten matches TorchTitan's common mesh construction flow."""
        self._setup_mock_platform(mock_platform, world_size=16)
        world_mesh = init_device_mesh("npu", (16,), mesh_dim_names=("world",))

        dataloading_mesh = world_mesh._unflatten(
            0,
            (1, 4, 2, 2),
            ("pp", "batch", "cp", "tp"),
            backend_override={"pp": "fake", "batch": "fake"},
        )
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
        dense_mesh = world_mesh._unflatten(
            0,
            (1, 2, 4, 2),
            ("pp", "dp_replicate", "fsdp", "tp"),
            backend_override={"pp": "fake"},
        )
        sparse_mesh = world_mesh._unflatten(
            0,
            (1, 2, 4, 2, 1),
            ("pp", "dp_replicate", "efsdp", "ep", "etp"),
            backend_override={"pp": "fake", "etp": "fake"},
        )

        self.assertEqual(dataloading_mesh.mesh_shape, (1, 4, 2, 2))
        self.assertEqual(dataloading_mesh.mesh_dim_names, ("pp", "batch", "cp", "tp"))
        self.assertEqual(dataloading_mesh._dim_group_backends, ("fake", "fake", None, None))  # pylint: disable=W0212
        self.assertEqual(dataloading_mesh["batch"].rank_list, (0, 4, 8, 12))
        self.assertEqual(dataloading_mesh["batch"]._dim_group_backends, ("fake",))  # pylint: disable=W0212
        self.assertEqual(dataloading_mesh["batch", "cp"].rank_list, (0, 2, 4, 6, 8, 10, 12, 14))

        self.assertEqual(loss_mesh.mesh_dim_names, ("loss_mesh",))
        self.assertEqual(loss_mesh.mesh_shape, (8,))
        self.assertEqual(loss_mesh.rank_list, (0, 2, 4, 6, 8, 10, 12, 14))
        self.assertEqual(loss_mesh._dim_group_backends, (None,))  # pylint: disable=W0212
        self.assertIs(world_mesh["loss_mesh"], loss_mesh)

        self.assertEqual(dense_mesh["tp"].rank_list, (0, 1))
        self.assertEqual(dense_mesh["fsdp"].size(), 4)
        self.assertEqual(dense_mesh[("dp_replicate", "fsdp")].mesh_shape, (2, 4))

        self.assertEqual(sparse_mesh["ep"].size(), 2)
        self.assertEqual(sparse_mesh["etp"].size(), 1)
        self.assertEqual(sparse_mesh["etp"]._dim_group_backends, ("fake",))  # pylint: disable=W0212
        self.assertEqual(sparse_mesh[("ep", "etp")].mesh_shape, (2, 1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_supports_torchtitan_8card_cp2_fsdp_semantics(self, mock_platform):
        """Test TorchTitan-like 8-card cp=2 keeps batch sharding and fsdp sharding semantics separate."""
        world_mesh, dataloading_mesh, loss_mesh, dense_mesh = self._build_torchtitan_like_meshes_8card_cp2(
            mock_platform
        )

        self.assertEqual(dataloading_mesh["batch"].size(), 4)
        self.assertEqual(dataloading_mesh["batch"].rank_list, (0, 2, 4, 6))
        self.assertEqual(dataloading_mesh["cp"].size(), 2)
        self.assertEqual(dataloading_mesh["cp"].rank_list, (0, 1))

        self.assertEqual(loss_mesh.size(), 8)
        self.assertEqual(loss_mesh.rank_list, (0, 1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(dense_mesh["fsdp"].size(), 8)
        self.assertEqual(dense_mesh["fsdp"].rank_list, (0, 1, 2, 3, 4, 5, 6, 7))

        self.assertIs(world_mesh["loss_mesh"], loss_mesh)
        self.assertEqual(loss_mesh.rank_list, dense_mesh["fsdp"].rank_list)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_supports_torchtitan_8card_cp2_concatenate_views(self, mock_platform):
        """Test concatenate rebuilds TorchTitan-like batch/cp and fsdp views on 8 cards."""
        _, dataloading_mesh, _, dense_mesh = self._build_torchtitan_like_meshes_8card_cp2(mock_platform)

        batch_cp_mesh = DeviceMesh.concatenate([dataloading_mesh["batch"], dataloading_mesh["cp"]])
        fsdp_tp_mesh = DeviceMesh.concatenate([dense_mesh["fsdp"], dense_mesh["tp"]])

        self.assertEqual(batch_cp_mesh.mesh_dim_names, ("batch", "cp"))
        self.assertEqual(batch_cp_mesh.mesh_shape, (4, 2))
        self.assertEqual(batch_cp_mesh.rank_list, dataloading_mesh["batch", "cp"].rank_list)

        self.assertEqual(fsdp_tp_mesh.mesh_dim_names, ("fsdp", "tp"))
        self.assertEqual(fsdp_tp_mesh.mesh_shape, (8, 1))
        self.assertEqual(fsdp_tp_mesh.rank_list, dense_mesh["fsdp", "tp"].rank_list)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_supports_torchtitan_8card_cp2_axis_queries(self, mock_platform):
        """Test axis helpers return the expected TorchTitan-like ranks on 8-card cp=2 meshes."""
        _, dataloading_mesh, _, dense_mesh = self._build_torchtitan_like_meshes_8card_cp2(mock_platform)

        self.assertEqual(dataloading_mesh.get_rank_list_along_axis("batch"), [0, 2, 4, 6])
        self.assertEqual(dataloading_mesh.get_rank_list_along_axis("cp"), [0, 1])
        self.assertEqual(dense_mesh.get_rank_list_along_axis("fsdp"), [0, 1, 2, 3, 4, 5, 6, 7])

        with patch.object(dataloading_mesh, "_rank", 5):
            self.assertEqual(dataloading_mesh.get_local_rank("batch"), 2)
            self.assertEqual(dataloading_mesh.get_local_rank("cp"), 1)

        with patch.object(dense_mesh, "_rank", 5):
            self.assertEqual(dense_mesh.get_local_rank("fsdp"), 5)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_unflatten_rejects_invalid_backend_override_keys(self, mock_platform):
        """Test _unflatten validates backend_override keys like Torch."""
        self._setup_mock_platform(mock_platform, world_size=8)
        world_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("world",))

        with self.assertRaisesRegex(RuntimeError, "invalid keys"):
            world_mesh._unflatten(
                "world",
                (2, 4),
                ("dp", "tp"),
                backend_override={"invalid": "fake"},
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_size_one_dims_skip_eager_group_creation(self, mock_platform):
        """Test singleton mesh dimensions do not eagerly create communication groups."""
        self._setup_mock_platform(mock_platform, world_size=1)
        mock_group = MagicMock()
        mock_platform.split_group.return_value = mock_group

        mesh = init_device_mesh("npu", (1, 1, 1), mesh_dim_names=("pp", "dp", "tp"))

        mock_platform.split_group.assert_not_called()
        self.assertEqual(mesh._dim_group_names, [None, None, None])  # pylint: disable=W0212

        group = mesh.get_group("pp")

        self.assertIs(group, mock_group)
        mock_platform.split_group.assert_called_once_with(split_ranks=[[0]])

        group_again = mesh.get_group("pp")
        self.assertIs(group_again, mock_group)
        mock_platform.split_group.assert_called_once()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_fake_backend_dims_materialize_groups_lazily(self, mock_platform):
        """Test fake backend dimensions skip eager group init and only create groups on demand."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mock_group = MagicMock()
        mock_platform.split_group.return_value = mock_group

        world_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("world",))
        mock_platform.split_group.reset_mock()

        mesh = world_mesh._unflatten(
            0,
            (2, 2, 2),
            ("dp", "cp", "tp"),
            backend_override={"dp": "fake"},
        )

        self.assertEqual(mock_platform.split_group.call_count, 2)
        self.assertEqual(mesh._dim_group_names[0], None)  # pylint: disable=W0212
        self.assertEqual(mesh._dim_group_backends[0], "fake")  # pylint: disable=W0212

        dp_group = mesh["dp"].get_group()

        self.assertIs(dp_group, mock_group)
        self.assertEqual(mock_platform.split_group.call_count, 3)

    # ------------------------------------------------------------------
    # Properties and methods
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_properties(self, mock_platform):
        """Test DeviceMesh basic properties.

        Scenario: Access various properties of DeviceMesh instance.
        Expected behavior: Properties should return expected values for device_type,
        shape, mesh_shape, ndim, and rank.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        self.assertEqual(mesh.device_type, "npu")
        self.assertEqual(mesh.shape, (2, 4))
        self.assertEqual(mesh.mesh_shape, (2, 4))
        self.assertEqual(mesh.ndim, 2)
        self.assertEqual(mesh.rank, 0)
        self.assertEqual(len(mesh.rank_list), 8)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_repr(self, mock_platform):
        """Test DeviceMesh __repr__ method.

        Scenario: Get string representation of DeviceMesh instance.
        Expected behavior: __repr__ should contain DeviceMesh class name and key properties.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

        repr_str = repr(mesh)

        self.assertIn("DeviceMesh", repr_str)
        self.assertIn("device_type='npu'", repr_str)
        self.assertIn("mesh_shape=(2, 2)", repr_str)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_size(self, mock_platform):
        """Test DeviceMesh size method.

        Scenario: Get total size and size along specific dimensions.
        Expected behavior: size() should return total elements, size(dim) should return
        elements along that dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        self.assertEqual(mesh.size(), 8)
        self.assertEqual(mesh.size(0), 2)
        self.assertEqual(mesh.size(1), 4)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_axis_methods(self, mock_platform):
        """Test DeviceMesh axis-related methods.

        Scenario: Access axis-related information using axis_id, axis_index,
        and get_device_num_along_axis methods.
        Expected behavior: Methods should return correct dimension indices and device counts.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        self.assertEqual(mesh.axis_id("dp"), 1)
        self.assertEqual(mesh.axis_id("tp"), 0)
        self.assertEqual(mesh.axis_index("dp"), 0)
        self.assertEqual(mesh.axis_index("tp"), 1)
        self.assertEqual(mesh.get_device_num_along_axis("dp"), 2)
        self.assertEqual(mesh.get_device_num_along_axis("tp"), 4)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_local_rank(self, mock_platform):
        """Test DeviceMesh get_local_rank method.

        Scenario: Get local rank within a specific mesh dimension.
        Expected behavior: Should return the local rank within the specified dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        self.assertEqual(mesh.get_local_rank("dp"), 0)
        self.assertEqual(mesh.get_local_rank("tp"), 0)
        self.assertEqual(mesh.get_local_rank(0), 0)
        self.assertEqual(mesh.get_local_rank(1), 0)

        with patch.object(mesh, "_rank", 5):
            self.assertEqual(mesh.get_local_rank("dp"), 1)
            self.assertEqual(mesh.get_local_rank(1), 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_rank_list_along_axis(self, mock_platform):
        """Test DeviceMesh get_rank_list_along_axis method.

        Scenario: Get the list of ranks along a specific axis for the current rank.
        Expected behavior: Should return list of ranks in the same slice along the specified axis.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        self.assertEqual(mesh.get_rank_list_along_axis("dp"), [0, 4])
        self.assertEqual(mesh.get_rank_list_along_axis("tp"), [0, 1, 2, 3])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_global_shape(self, mock_platform):
        """Test DeviceMesh get_global_shape method.

        Scenario: Calculate global tensor shape from local slice shape and tensor mapping.
        Expected behavior: Should return global shape accounting for sharding across mesh dimensions.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        # Mapping: dim 0 sharded on dp (mesh idx 1), dim 1 sharded on tp (mesh idx 0)
        global_shape = mesh.get_global_shape(slice_shape=(4, 8), tensor_map=(1, 0))

        self.assertEqual(global_shape, (8, 32))  # (4*2, 8*4)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_coordinate(self, mock_platform):
        """Test DeviceMesh get_coordinate method.

        Scenario: Get coordinate of current rank in the mesh.
        Expected behavior: Should return coordinate tuple for current rank.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        self.assertEqual(mesh.get_coordinate(), (0, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_root_mesh_property(self, mock_platform):
        """Test DeviceMesh root_mesh property.

        Scenario: Access root_mesh property of parent and child meshes.
        Expected behavior: Parent mesh should have None, child should reference parent.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        dp_mesh = mesh["dp"]

        self.assertIsNone(mesh.root_mesh)
        self.assertEqual(dp_mesh.root_mesh, mesh)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_sub_mesh_property(self, mock_platform):
        """Test DeviceMesh sub_mesh property.

        Scenario: Access sub_mesh property after creating submeshes.
        Expected behavior: Should contain all created submeshes.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        self.assertEqual(len(mesh.sub_mesh), 0)

        dp_mesh = mesh["dp"]

        self.assertEqual(len(mesh.sub_mesh), 1)
        self.assertEqual(dp_mesh, mesh.sub_mesh[0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_flatten(self, mock_platform):
        """Test DeviceMesh flatten method.

        Scenario: Flatten a multi-dimensional mesh into a 1D mesh.
        Expected behavior: Should return a new DeviceMesh with flattened dimensions.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        dp_cp_mesh = mesh[("dp", "cp")]

        flat_mesh = dp_cp_mesh.flatten()

        self.assertEqual(flat_mesh.mesh_shape, (4,))
        self.assertEqual(flat_mesh.mesh_dim_names, ("dp_cp",))
        self.assertEqual(flat_mesh.rank_list, (0, 2, 4, 6))
        self.assertEqual(flat_mesh.root_mesh, mesh)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_flatten_mapping(self, mock_platform):
        """Test DeviceMesh flatten mapping methods.

        Scenario: Create flattened mesh and check mapping.
        Expected behavior: get_flatten_mapping should contain the flattened mesh.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        self.assertEqual(mesh.get_flatten_mapping(), {})

        flat_dp_tp = mesh.flatten()

        self.assertIn("dp_tp", mesh.get_flatten_mapping())
        self.assertEqual(mesh.get_flatten_mapping()["dp_tp"], flat_dp_tp)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_to_hash(self, mock_platform):
        """Test DeviceMesh to_hash method.

        Scenario: Generate hash key from DeviceMesh.
        Expected behavior: to_hash should return tuple of shape, names, and rank list.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

        hash_key = mesh.to_hash()

        self.assertEqual(hash_key, ((2, 2), ("dp", "tp"), (0, 1, 2, 3)))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_assert_axis(self, mock_platform):
        """Test DeviceMesh assert_axis method.

        Scenario: Validate axis name exists in mesh.
        Expected behavior: Should raise ValueError for invalid axis names.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        mesh.assert_axis("dp", "test_op")
        mesh.assert_axis("tp", "test_op")

        with self.assertRaises(ValueError) as context:
            mesh.assert_axis("invalid", "test_op")
        self.assertIn("invalid", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_all_groups(self, mock_platform):
        """Test DeviceMesh get_all_groups method.

        Scenario: Get all communication groups from DeviceMesh.
        Expected behavior: Should return list of all groups used in the mesh.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mock_group = MagicMock()
        mock_platform.split_group.return_value = mock_group

        mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

        # Manually set up groups to avoid assertion errors with mocked platform
        mesh._dim_group_names = ['(0, 1)', '(0, 1, 2, 3)']
        EXISTING_COMM_GROUPS['(0, 1)'] = mock_group
        EXISTING_COMM_GROUPS['(0, 1, 2, 3)'] = mock_group

        all_groups = mesh.get_all_groups()

        self.assertEqual(len(all_groups), 2)

    # ------------------------------------------------------------------
    # Validation / error-path tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_invalid_mesh_type(self, mock_platform):
        """Test DeviceMesh with invalid mesh type raises TypeError.

        Scenario: Create DeviceMesh with invalid mesh type (integer).
        Expected behavior: Should raise TypeError with appropriate message.
        """
        self._setup_mock_platform(mock_platform, world_size=1)

        with self.assertRaises(TypeError) as context:
            DeviceMesh("npu", mesh=0, _init_backend=False)
        self.assertIn("mesh must be Tensor, list, tuple or numpy array", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_invalid_mesh_dim_names_length(self, mock_platform):
        """Test DeviceMesh with mismatched mesh_dim_names length raises ValueError.

        Scenario: Create DeviceMesh with 2D mesh but 3 mesh_dim_names.
        Expected behavior: Should raise ValueError with appropriate message.
        """
        self._setup_mock_platform(mock_platform, world_size=4)

        with self.assertRaises(ValueError) as context:
            DeviceMesh("npu", mesh=[[0, 1], [2, 3]], mesh_dim_names=("dp", "tp", "cp"), _init_backend=False)
        self.assertIn("mesh dimensions", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_invalid_mesh_dim_names_duplicate(self, mock_platform):
        """Test DeviceMesh with duplicate mesh_dim_names raises ValueError.

        Scenario: Create DeviceMesh with duplicate dimension names.
        Expected behavior: Should raise ValueError with appropriate message.
        """
        self._setup_mock_platform(mock_platform, world_size=4)

        with self.assertRaises(ValueError) as context:
            DeviceMesh("npu", mesh=[[0, 1], [2, 3]], mesh_dim_names=("dp", "dp"), _init_backend=False)
        self.assertIn("Each element of mesh_dim_names", str(context.exception))

    # ------------------------------------------------------------------
    # Helper function tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_sub_rank_list(self, mock_platform):
        """Test _get_sub_rank_list helper function.

        Scenario: Extract sub-rank list for a sub-mesh from original mesh.
        Expected behavior: Should return list of ranks in the sub-mesh slice containing current rank.
        """
        mock_platform.get_rank.return_value = 0

        sub_rank_list = _get_sub_rank_list(
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "tp"),
            rank_list=(0, 1, 2, 3, 4, 5, 6, 7),
            sub_mesh_dim_names=("dp",),
            current_rank=0,
        )

        self.assertEqual(sub_rank_list, [0, 4])

    # ------------------------------------------------------------------
    # from_group tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_from_group_with_1d_group(self, mock_platform):
        """Test DeviceMesh.from_group with 1D group.

        Scenario: Create DeviceMesh from a 1D communication group.
        Expected behavior: DeviceMesh should be created with ranks from the group.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mock_group = MagicMock()
        mock_platform.get_process_group_ranks.return_value = [0, 1, 2, 3]
        mock_platform.get_created_group.return_value = False

        device_mesh = DeviceMesh.from_group(
            group=mock_group,
            device_type="npu",
            mesh_dim_names=("dp",),
        )

        self.assertEqual(device_mesh.mesh_shape, (4,))
        self.assertEqual(device_mesh.rank_list, (0, 1, 2, 3))
        self.assertEqual(device_mesh.mesh_dim_names, ("dp",))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_from_group_with_nd_groups(self, mock_platform):
        """Test DeviceMesh.from_group with nD groups.

        Scenario: Create DeviceMesh from a list of communication groups for 2D mesh.
        Expected behavior: DeviceMesh should be created with appropriate rank layout.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mock_group_dp = MagicMock()
        mock_group_tp = MagicMock()
        mock_platform.get_process_group_ranks.side_effect = [
            [0, 1],  # dp group
            [0, 2],  # tp group
        ]
        mock_platform.get_created_group.return_value = False

        device_mesh = DeviceMesh.from_group(
            group=[mock_group_dp, mock_group_tp],
            device_type="npu",
            mesh=[[0, 1], [2, 3]],
            mesh_dim_names=("dp", "tp"),
        )

        self.assertEqual(device_mesh.mesh_shape, (2, 2))
        self.assertEqual(device_mesh.rank_list, (0, 1, 2, 3))
        self.assertEqual(device_mesh.mesh_dim_names, ("dp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_from_group_invalid_mesh(self, mock_platform):
        """Test DeviceMesh.from_group with invalid mesh raises ValueError.

        Scenario: Create DeviceMesh with 1D group but mismatched mesh.
        Expected behavior: Should raise ValueError when mesh doesn't match group ranks.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mock_group = MagicMock()
        mock_platform.get_process_group_ranks.return_value = [0, 1]
        mock_platform.get_created_group.return_value = False

        with self.assertRaises(ValueError) as context:
            DeviceMesh.from_group(
                group=mock_group,
                device_type="npu",
                mesh=[0, 1, 2, 3],  # 4 ranks, but group only has 2
                mesh_dim_names=("dp",),
            )
        self.assertIn("Invalid mesh", str(context.exception))

    # ------------------------------------------------------------------
    # get_devices_for_axis tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_with_str(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with string dimension name.

        Scenario: Get devices along a specific axis by dimension name.
        Expected behavior: Should return list of ranks along that dimension.
        """
        mesh = self._make_2x2_mesh_no_backend(mock_platform)

        devices = mesh.get_devices_for_axis("dp", 0)

        self.assertEqual(sorted(devices), [0, 2])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_with_int(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with integer dimension index.

        Scenario: Get devices along a specific axis by dimension index.
        Expected behavior: Should return list of ranks along that dimension.
        """
        mesh = self._make_2x2_mesh_no_backend(mock_platform)

        devices = mesh.get_devices_for_axis(0, 0)

        self.assertEqual(sorted(devices), [0, 2])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_invalid_dim(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with invalid dimension raises ValueError.

        Scenario: Get devices with invalid dimension name.
        Expected behavior: Should raise ValueError with appropriate message.
        """
        mesh = self._make_2x2_mesh_no_backend(mock_platform)

        with self.assertRaises(ValueError) as context:
            mesh.get_devices_for_axis("invalid", 0)
        self.assertIn("not found", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_no_dim_names(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis without mesh_dim_names raises ValueError.

        Scenario: Get devices when mesh_dim_names is not set.
        Expected behavior: Should raise ValueError when using string dimension.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mesh = DeviceMesh("npu", mesh=[[0, 1], [2, 3]], _init_backend=False)

        with self.assertRaises(ValueError) as context:
            mesh.get_devices_for_axis("dp", 0)
        self.assertIn("not set", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_invalid_rank(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with invalid rank raises ValueError.

        Scenario: Get devices with rank not in mesh.
        Expected behavior: Should raise ValueError with appropriate message.
        """
        mesh = self._make_2x2_mesh_no_backend(mock_platform)

        with self.assertRaises(ValueError) as context:
            mesh.get_devices_for_axis("dp", 100)
        self.assertIn("not found", str(context.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_get_devices_for_axis_out_of_range(self, mock_platform):
        """Test DeviceMesh.get_devices_for_axis with out-of-range dimension index raises ValueError.

        Scenario: Get devices with dimension index out of valid range.
        Expected behavior: Should raise ValueError with appropriate message.
        """
        mesh = self._make_2x2_mesh_no_backend(mock_platform)

        with self.assertRaises(ValueError) as context:
            mesh.get_devices_for_axis(5, 0)
        self.assertIn("out of range", str(context.exception))

    # ------------------------------------------------------------------
    # init_process_group ordering tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_init_backend_calls_init_process_group(self, mock_platform):
        """Test that DeviceMesh calls init_process_group before mesh setup when _init_backend=True.

        Scenario: Construct DeviceMesh with _init_backend=True.
        Expected behavior: init_process_group is called exactly once before get_world_size
        (i.e., before the mesh is constructed).
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mock_platform.split_group.return_value = MagicMock()

        call_order = []
        mock_platform.init_process_group.side_effect = lambda *a, **kw: call_order.append("init_process_group")
        mock_platform.get_world_size.side_effect = lambda *a, **kw: call_order.append("get_world_size") or 4

        DeviceMesh("npu", mesh=[[0, 1], [2, 3]], _init_backend=True)

        mock_platform.init_process_group.assert_called_once()
        # init_process_group must be called before get_world_size (mesh construction)
        if "get_world_size" in call_order:
            self.assertLess(
                call_order.index("init_process_group"),
                call_order.index("get_world_size"),
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_mesh_no_init_process_group_when_init_backend_false(self, mock_platform):
        """Test that DeviceMesh does not call init_process_group when _init_backend=False.

        Scenario: Construct DeviceMesh with _init_backend=False.
        Expected behavior: init_process_group is never called.
        """
        self._setup_mock_platform(mock_platform, world_size=4)

        DeviceMesh("npu", mesh=[[0, 1], [2, 3]], _init_backend=False)

        mock_platform.init_process_group.assert_not_called()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_init_device_mesh_calls_init_process_group_when_init_backend_true(self, mock_platform):
        """Test that init_device_mesh calls init_process_group when init_backend=True and rank_list is None.

        Scenario: Call init_device_mesh with init_backend=True and no explicit rank_list.
        Expected behavior: init_process_group is called before get_rank.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mock_platform.split_group.return_value = MagicMock()

        call_order = []
        mock_platform.init_process_group.side_effect = lambda *a, **kw: call_order.append("init_process_group")
        mock_platform.get_rank.side_effect = lambda *a, **kw: call_order.append("get_rank") or 0

        init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"), init_backend=True)

        mock_platform.init_process_group.assert_called()
        self.assertIn("init_process_group", call_order)
        self.assertIn("get_rank", call_order)
        self.assertLess(
            call_order.index("init_process_group"),
            call_order.index("get_rank"),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_init_device_mesh_no_init_process_group_when_rank_list_provided(self, mock_platform):
        """Test that init_device_mesh does not call init_process_group from its own body
        when an explicit rank_list is provided (regardless of init_backend).

        Scenario: Call init_device_mesh with explicit rank_list and init_backend=True.
        Expected behavior: init_process_group is NOT called in the rank_list branch of
        init_device_mesh (the DeviceMesh constructor may still call it via _init_backend).
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mock_platform.split_group.return_value = MagicMock()

        # Patch init_process_group to track calls
        init_calls = []
        mock_platform.init_process_group.side_effect = lambda *a, **kw: init_calls.append(1)

        init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"),
                         rank_list=(0, 1, 2, 3), init_backend=False)

        mock_platform.init_process_group.assert_not_called()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_init_device_mesh_get_rank_failure_raises_runtime_error(self, mock_platform):
        """Test that init_device_mesh wraps get_rank failures in RuntimeError with guidance.

        Scenario: init_device_mesh is called without rank_list and get_rank raises an exception.
        Expected behavior: RuntimeError is raised with a message guiding the user to either
        pass rank_list explicitly or use init_backend=True.
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        mock_platform.get_rank.side_effect = RuntimeError("process group not initialized")

        with self.assertRaises(RuntimeError) as ctx:
            init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"), init_backend=False)

        msg = str(ctx.exception)
        self.assertIn("init_device_mesh", msg)
        self.assertIn("rank_list", msg)
        self.assertIn("init_backend=True", msg)

    # ------------------------------------------------------------------
    # device_type validation tests
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_type_valid_torch_platform(self, mock_platform):
        """Test that all valid device_type values are accepted on PyTorch platform.

        Scenario: Create DeviceMesh with each of "cpu", "cuda", "npu" when the
        platform is PyTorch.
        Expected behavior: No exception is raised and device_type is stored as-is.
        """
        self._setup_mock_platform(mock_platform, PlatformType.PYTORCH, world_size=4)

        for dtype in ("cpu", "cuda", "npu"):
            with self.subTest(device_type=dtype):
                mesh = DeviceMesh(dtype, mesh=[[0, 1], [2, 3]], _init_backend=False)
                self.assertEqual(mesh.device_type, dtype)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_type_invalid_torch_platform(self, mock_platform):
        """Test that invalid device_type values raise ValueError on PyTorch platform.

        Scenario: Create DeviceMesh with device_type values that are not in
        {"cpu", "cuda", "npu"} when the platform is PyTorch.
        Expected behavior: ValueError is raised and its message contains the
        invalid value and the platform name.
        """
        self._setup_mock_platform(mock_platform, PlatformType.PYTORCH, world_size=4)

        for dtype in ("gpu", "xla", "mlu", "unknown"):
            with self.subTest(device_type=dtype):
                with self.assertRaises(ValueError) as ctx:
                    DeviceMesh(dtype, mesh=[[0, 1], [2, 3]], _init_backend=False)
                self.assertIn(dtype, str(ctx.exception))
                self.assertIn("PYTORCH", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_type_valid_mindspore_platform(self, mock_platform):
        """Test that all valid device_type values are accepted on MindSpore platform.

        Scenario: Create DeviceMesh with each of "cpu", "gpu", "npu" when the
        platform is MindSpore.
        Expected behavior: No exception is raised and device_type is stored as-is.
        """
        self._setup_mock_platform(mock_platform, PlatformType.MINDSPORE, world_size=4)

        for dtype in ("cpu", "gpu", "npu"):
            with self.subTest(device_type=dtype):
                mesh = DeviceMesh(dtype, mesh=[[0, 1], [2, 3]], _init_backend=False)
                self.assertEqual(mesh.device_type, dtype)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_device_type_invalid_mindspore_platform(self, mock_platform):
        """Test that invalid device_type values raise ValueError on MindSpore platform.

        Scenario: Create DeviceMesh with device_type values that are not in
        {"cpu", "gpu", "npu"} when the platform is MindSpore.
        Expected behavior: ValueError is raised and its message contains the
        invalid value and the platform name.
        """
        self._setup_mock_platform(mock_platform, PlatformType.MINDSPORE, world_size=4)

        for dtype in ("cuda", "xla", "mlu", "unknown"):
            with self.subTest(device_type=dtype):
                with self.assertRaises(ValueError) as ctx:
                    DeviceMesh(dtype, mesh=[[0, 1], [2, 3]], _init_backend=False)
                self.assertIn(dtype, str(ctx.exception))
                self.assertIn("MINDSPORE", str(ctx.exception))

    # ------------------------------------------------------------------
    # Active mesh context: ``with mesh:`` and ``get_current_mesh``
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_current_mesh_raises_when_no_context(self, mock_platform):
        """
        Feature: get_current_mesh validation with empty thread-local stack
        Description: call get_current_mesh without entering ``with device_mesh``
        Expectation: RuntimeError whose message indicates no active mesh
        """
        self._setup_mock_platform(mock_platform, world_size=2)
        self.assertEqual(len(_mesh_resources.mesh_stack), 0)
        with self.assertRaises(RuntimeError) as ctx:
            _mesh_resources.get_current_mesh()
        self.assertIn("device mesh", str(ctx.exception).lower())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_with_mesh_sets_get_current_mesh(self, mock_platform):
        """
        Feature: DeviceMesh context manager sets current mesh for get_current_mesh
        Description: enter ``with`` using a 1-D DeviceMesh with mocked platform
        Expectation: inside block get_current_mesh() is that mesh and stack length is 1;
            after exit stack is empty
        """
        self._setup_mock_platform(mock_platform, world_size=2)
        mesh = DeviceMesh(
            "npu",
            mesh=[0, 1],
            mesh_dim_names=("tp",),
            _init_backend=False,
        )
        with mesh:
            self.assertIs(_mesh_resources.get_current_mesh(), mesh)
            self.assertEqual(len(_mesh_resources.mesh_stack), 1)
        self.assertEqual(len(_mesh_resources.mesh_stack), 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_with_mesh_as_returns_self(self, mock_platform):
        """
        Feature: DeviceMesh __enter__ return value for ``with mesh as m``
        Description: bind context manager target to a variable
        Expectation: ``m is mesh`` (same object identity)
        """
        self._setup_mock_platform(mock_platform, world_size=2)
        mesh = DeviceMesh(
            "npu",
            mesh=[0, 1],
            mesh_dim_names=("tp",),
            _init_backend=False,
        )
        with mesh as m:
            self.assertIs(m, mesh)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nested_with_inner_mesh_is_current(self, mock_platform):
        """
        Feature: nested DeviceMesh context managers stack correctly
        Description: enter outer then inner 1-D meshes with mocked platform
        Expectation: inner block sees inner mesh and stack depth 2; after inner exits
            current is outer and depth 1; after outer exits stack is empty
        """
        self._setup_mock_platform(mock_platform, world_size=4)
        outer = DeviceMesh(
            "npu",
            mesh=[0, 1, 2, 3],
            mesh_dim_names=("tp",),
            _init_backend=False,
        )
        inner = DeviceMesh(
            "npu",
            mesh=[0, 1],
            mesh_dim_names=("tp",),
            _init_backend=False,
        )
        with outer:
            self.assertIs(_mesh_resources.get_current_mesh(), outer)
            with inner:
                self.assertIs(_mesh_resources.get_current_mesh(), inner)
                self.assertEqual(len(_mesh_resources.mesh_stack), 2)
            self.assertIs(_mesh_resources.get_current_mesh(), outer)
            self.assertEqual(len(_mesh_resources.mesh_stack), 1)
        self.assertEqual(len(_mesh_resources.mesh_stack), 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_with_mesh_pops_stack_on_exception(self, mock_platform):
        """
        Feature: DeviceMesh __exit__ clears stack when the body raises
        Description: ``with mesh:`` block raises ValueError after pushing mesh
        Expectation: exception propagates and thread-local mesh stack is empty afterward
        """
        self._setup_mock_platform(mock_platform, world_size=2)
        mesh = DeviceMesh(
            "npu",
            mesh=[0, 1],
            mesh_dim_names=("tp",),
            _init_backend=False,
        )
        with self.assertRaises(ValueError):
            with mesh:
                self.assertEqual(len(_mesh_resources.mesh_stack), 1)
                raise ValueError("deliberate")
        self.assertEqual(len(_mesh_resources.mesh_stack), 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_current_mesh_thread_local_isolation(self, mock_platform):
        """
        Feature: per-thread isolation of active DeviceMesh stack
        Description: two threads each ``with`` a different DeviceMesh, synchronized with Barrier
        Expectation: get_current_mesh() in each thread is that thread's mesh object
        """
        self._setup_mock_platform(mock_platform, world_size=2)
        mesh_a = DeviceMesh(
            "npu",
            mesh=[0, 1],
            mesh_dim_names=("tp",),
            _init_backend=False,
        )
        mesh_b = DeviceMesh(
            "npu",
            mesh=[0, 1],
            mesh_dim_names=("tp",),
            _init_backend=False,
        )
        barrier = threading.Barrier(2)
        results: dict[int, bool] = {}

        def worker(key: int, mesh: DeviceMesh) -> None:
            with mesh:
                barrier.wait()
                results[key] = _mesh_resources.get_current_mesh() is mesh

        t1 = threading.Thread(target=worker, args=(1, mesh_a))
        t2 = threading.Thread(target=worker, args=(2, mesh_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertTrue(results.get(1), "thread 1 should see mesh_a")
        self.assertTrue(results.get(2), "thread 2 should see mesh_b")


if __name__ == "__main__":
    unittest.main()
