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
"""parallel_expand_dims test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_expand_dims import ExpandDimsDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelExpandDims(unittest.TestCase):
    """Unit tests for ExpandDimsDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()
        self.op = ExpandDimsDistributedOp("ExpandDims")

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

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "cp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    def _run_scenario(self, x_layout, expected_map, extra_args):
        """Infer layout of ExpandDims operator"""
        output_layout = self.op.infer_layout((x_layout,), extra_args)
        assert output_layout.tensor_map == expected_map, (
            f"ExpandDims failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )
        assert self.op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
            f"ExpandDims get_expand_impl failed. Expected None, "
            f"got {self.op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_data_parallel_1(self, mock_platform):
        """
        Feature: Data parallel.
        Description: insert dimension at beginning, axis=0.
        Expectation: new dimension inserted at position 0, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        self._run_scenario(x_layout, expected_map=(-1, 2, -1, -1), extra_args=[0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_model_parallel_2(self, mock_platform):
        """
        Feature: Model parallel.
        Description: insert dimension before mp axis, axis=2.
        Expectation: new dimension at position 2, mp shifted to position 3, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        self._run_scenario(x_layout, expected_map=(-1, -1, -1, 0), extra_args=[2])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_hybrid_parallel_3(self, mock_platform):
        """
        Feature: Hybrid parallel.
        Description: insert dimension in middle, axis=1.
        Expectation: new dimension at position 1, others shifted, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        self._run_scenario(x_layout, expected_map=(2, -1, 1, 0), extra_args=[1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_insert_at_end_4(self, mock_platform):
        """
        Feature: Insert at end.
        Description: insert dimension at the end, axis=-1.
        Expectation: new dimension appended at the end, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        self._run_scenario(x_layout, expected_map=(2, 1, 0, -1), extra_args=[-1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_negative_axis_5(self, mock_platform):
        """
        Feature: Negative axis indexing.
        Description: axis=-2 for rank-3 input, equivalent to axis=2.
        Expectation: new dimension at position 2, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        self._run_scenario(x_layout, expected_map=(2, 1, -1, 0), extra_args=[-2])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_all_replicated_6(self, mock_platform):
        """
        Feature: All replicated.
        Description: insert dimension with all replicated input.
        Expectation: new dimension inserted with None, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        self._run_scenario(x_layout, expected_map=(-1, -1, -1, -1), extra_args=[1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_2d_tensor_7(self, mock_platform):
        """
        Feature: 2D tensor.
        Description: insert dimension in 2D tensor.
        Expectation: correct tensor_map for 3D output, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        self._run_scenario(x_layout, expected_map=(2, -1, 1), extra_args=[1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_scalar_to_1d_8(self, mock_platform):
        """
        Feature: Expand scalar.
        Description: expand scalar (rank-0) to 1D, axis=0.
        Expectation: output has one dimension with None, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 0)

        self._run_scenario(x_layout, expected_map=(-1,), extra_args=[0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_extreme_negative_axis_9(self, mock_platform):
        """
        Feature: Extreme negative axis.
        Description: axis=-(rank+1), equivalent to axis=0.
        Expectation: insert at beginning, get_expand_impl returns None.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        self._run_scenario(x_layout, expected_map=(-1, 2, 1, 0), extra_args=[-4])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expanddims_invalid_axis_10(self, mock_platform):
        """
        Feature: Invalid axis.
        Description: axis out of valid range.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        with self.assertRaisesRegex(ValueError, "out of range for input rank"):
            self.op.infer_layout((x_layout,), [5])


if __name__ == "__main__":
    unittest.main()
