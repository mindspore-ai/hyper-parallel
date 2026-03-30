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
"""parallel_gatherd test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_gather import GatherDDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelGatherD(unittest.TestCase):
    """Unit tests for GatherDDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()
        self.op = GatherDDistributedOp("GatherD")

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "cp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_data_parallel_dim0(self, mock_platform):
        """
        Feature: Data Parallel for GatherD
        Description: Input is sharded on the gather dimension, while index is replicated.
        Expectation: Output layout becomes fully replicated and enters partial sum state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        output_layout = self.op.infer_layout((input_layout, None, index_layout), [dim])
        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel dim0 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
        assert output_layout.partial == ['sum', None], (
            f"Expected partial ['sum', None], got {output_layout.partial}"
        )
        impl = self.op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
        assert impl is not None, "Data parallel dim0 should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_data_parallel_dim1(self, mock_platform):
        """
        Feature: GatherD dim-axis sharding inference
        Description: Input is sharded on the gather dimension, while index is replicated.
        Expectation: Output layout becomes fully replicated and enters partial sum state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Replicate(), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 1
        output_layout = self.op.infer_layout((input_layout, None, index_layout), [dim])
        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel dim1 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
        assert output_layout.partial == [None, 'sum'], (
            f"Expected partial [None, 'sum'], got {output_layout.partial}"
        )
        impl = self.op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
        assert impl is not None, "Data parallel dim1 should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_input_both_shard_replicate_index(self, mock_platform):
        """
        Feature: GatherD multi-dim sharding validation
        Description: Input is sharded on both batch and sequence dimensions, while index is replicated.
        Expectation: infer_layout should reject mismatched non-dim sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        with self.assertRaisesRegex(ValueError, "same sharding on non-dim axis 1"):
            self.op.infer_layout((input_layout, None, index_layout), [dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_column_parallel(self, mock_platform):
        """
        Feature: GatherD cross-axis sharding validation
        Description: Input is sharded on the gather dimension, and index is sharded on another dimension.
        Expectation: infer_layout should reject mismatched non-dim sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        with self.assertRaisesRegex(ValueError, "same sharding on non-dim axis 1"):
            self.op.infer_layout((input_layout, None, index_layout), [dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_row_parallel(self, mock_platform):
        """
        Feature: Enhanced Model Parallel for GatherD
        Description: Both input and index sharded identically on dim axis.
        Expectation: Output follows the sharded batch mapping and has Partial Sum state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Shard(0))
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        output_layout = self.op.infer_layout((input_layout, None, index_layout), [dim])
        expected_map = (0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Row parallel expand failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "Row parallel should generate partial state"
        assert output_layout.partial == ['sum', None], (
            f"Expected partial ['sum', None], got {output_layout.partial}"
        )
        impl = self.op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
        assert impl is not None, "Row parallel should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_input_shard_dp_index_both_shard_conflict(self, mock_platform):
        """
        Feature: GatherD conflicting shard and partial inference
        Description: Input is sharded on batch with dp, while index is sharded on both batch and
                     sequence. This makes the output require dp for both sharding and partial reduction.
        Expectation: infer_layout should raise ValueError: "Partial dim must be replicate."
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        with self.assertRaisesRegex(ValueError, "Partial dim must be replicate."):
            self.op.infer_layout((input_layout, None, index_layout), [dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_input_multi_shard_with_matched_index_non_dim_shard(self, mock_platform):
        """
        Feature: GatherD multi-shard inference with matched non-dim axis
        Description: Input is sharded on both batch and sequence, while index matches the input sharding
                     on the non-gather axis and stays replicated on the gather axis.
        Expectation: Output follows the index layout and enters partial sum state.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)
        index_placements = (Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 0
        output_layout = self.op.infer_layout((input_layout, None, index_layout), [dim])
        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Matched non-dim shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
        assert output_layout.partial == ['sum', None], (
            f"Expected partial ['sum', None], got {output_layout.partial}"
        )
        impl = self.op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
        assert impl is not None, "Matched non-dim shard case should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_3d_mesh_input_multi_shard(self, mock_platform):
        """
        Feature: GatherD 3D mesh sharding inference
        Description: Mesh shape (2,2,2) with dims ("dp","tp","cp").
                     Input is sharded on the gather axis by tp, and index is sharded on the same
                     gather axis by cp, while non-dim axes remain replicated.
        Expectation: Output follows the index layout and enters partial sum state.
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("dp", "tp", "cp"))
        input_placements = (Replicate(), Shard(1), Replicate())
        input_layout = _build_layout(mesh, input_placements, 3)
        index_placements = (Replicate(), Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 3)
        dim = 1
        output_layout = self.op.infer_layout((input_layout, None, index_layout), [dim])
        expected_map = (-1, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"3D mesh multi-shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "3D mesh with sharded index should generate partial state"
        assert output_layout.partial == [None, 'sum', None], (
            f"Expected partial [None, 'sum', None], got {output_layout.partial}"
        )
        impl = self.op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
        assert impl is not None, "3D mesh with sharded index should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_3d_mesh_matched_non_dim_shard(self, mock_platform):
        """
        Feature: GatherD 3D mesh sharding inference
        Description: Mesh shape (2,2,2) with dims ("dp","tp","cp").
                     Input and index match on non-gather axes, while the gather axis is sharded on
                     different mesh axes.
        Expectation: Output follows the index layout and enters partial sum state.
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("dp", "tp", "cp"))
        input_placements = (Shard(0), Shard(1), Replicate())
        input_layout = _build_layout(mesh, input_placements, 3)
        index_placements = (Shard(0), Replicate(), Shard(1))
        index_layout = _build_layout(mesh, index_placements, 3)
        dim = 1
        output_layout = self.op.infer_layout((input_layout, None, index_layout), [dim])
        expected_map = (2, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"3D mesh matched non-dim shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert output_layout.is_partial(), "3D mesh with sharded gather axis should generate partial state"
        assert output_layout.partial == [None, 'sum', None], (
            f"Expected partial [None, 'sum', None], got {output_layout.partial}"
        )
        impl = self.op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
        assert impl is not None, "3D mesh matched non-dim shard case should have expand implementation"
        assert callable(impl), "Returned impl should be a callable function"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_rank_mismatch_error(self, mock_platform):
        """
        Feature: GatherD layout inference error handling
        Description: Input and index have different ranks.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate(), Replicate())
        input_layout = _build_layout(mesh, input_placements, 3)
        index_placements = (Shard(0), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)
        dim = 1
        with self.assertRaisesRegex(ValueError, "same number of dimensions"):
            self.op.infer_layout((input_layout, None, index_layout), [dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_invalid_layouts_error(self, mock_platform):
        """
        Feature: GatherD input layout validation
        Description: Fewer than the required three layouts are provided.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        dim = 0
        with self.assertRaisesRegex(ValueError, "requires 3 layouts"):
            self.op.infer_layout((input_layout, None,), [dim])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_missing_dim_arg_error(self, mock_platform):
        """
        Feature: GatherD extra argument validation
        Description: The required dim argument is not provided.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        index_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        with self.assertRaisesRegex(ValueError, "requires 1 extra arg"):
            self.op.infer_layout((input_layout, None, index_layout), [])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gatherd_invalid_dim_error(self, mock_platform):
        """
        Feature: GatherD layout inference error handling
        Description: Dim value is out of valid range.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_layout = _build_layout(mesh, input_placements, 2)
        dim = 5
        with self.assertRaisesRegex(ValueError, "out of valid range"):
            self.op.infer_layout((input_layout, None, index_layout), [dim])


if __name__ == "__main__":
    unittest.main()
