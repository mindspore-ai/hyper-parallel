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
"""Unit tests for DeviceMesh.concatenate."""
import copy
import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestDeviceMeshConcatenate(unittest.TestCase):
    """Unit tests for DeviceMesh.concatenate."""

    def setUp(self):
        """Clear global caches before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        """Clear global caches after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_returns_unified_mesh_in_root_order(self, mock_platform):
        """Test concatenate combines explicit submeshes in root order."""
        mock_platform.get_rank.return_value = 0
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)

        root_mesh = DeviceMesh(
            "npu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )
        dp_mesh = root_mesh["dp"]
        tp_mesh = root_mesh["tp"]

        unified_mesh = DeviceMesh.concatenate([dp_mesh, tp_mesh])

        self.assertEqual(unified_mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(unified_mesh.mesh_shape, (2, 2))
        self.assertEqual(unified_mesh.to_hash(), root_mesh.to_hash())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_single_mesh_returns_input(self, mock_platform):
        """Test concatenate returns the original mesh when only one mesh is provided."""
        mock_platform.get_rank.return_value = 0
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)

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
        mock_platform.get_rank.return_value = 0
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)

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
        mock_platform.get_rank.return_value = 0
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)

        root_mesh = DeviceMesh(
            "npu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )
        dp_mesh = root_mesh["dp"]

        with self.assertRaisesRegex(ValueError, "disjoint mesh dims"):
            DeviceMesh.concatenate([dp_mesh, dp_mesh])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_rejects_meshes_out_of_root_order(self, mock_platform):
        """Test concatenate rejects meshes that do not follow root mesh order."""
        mock_platform.get_rank.return_value = 0
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)

        root_mesh = DeviceMesh(
            "npu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("tp", "dp"),
            _init_backend=False,
        )
        dp_mesh = root_mesh["dp"]
        tp_mesh = root_mesh["tp"]

        with self.assertRaisesRegex(ValueError, "follow the root mesh order"):
            DeviceMesh.concatenate([dp_mesh, tp_mesh])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concatenate_preserves_root_mesh_after_layout_deepcopy(self, mock_platform):
        """Test concatenate still works after deepcopying a layout built from a submesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.tensor_to_numpy.side_effect = lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)

        root_mesh = DeviceMesh(
            "npu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )
        dp_mesh = root_mesh["dp"]
        tp_mesh = root_mesh["tp"]

        copied_layout = copy.deepcopy(Layout.from_device_mesh(tp_mesh))
        unified_mesh = DeviceMesh.concatenate([dp_mesh, copied_layout.mesh])

        self.assertEqual(copied_layout.mesh._get_root_mesh().to_hash(), root_mesh.to_hash())  # pylint: disable=W0212
        self.assertEqual(unified_mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(unified_mesh.to_hash(), root_mesh.to_hash())


if __name__ == "__main__":
    unittest.main()
