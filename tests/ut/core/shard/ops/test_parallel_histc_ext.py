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
"""
Unit tests for HistcExtDistributedOp.
"""

import os
import unittest
from unittest.mock import patch
import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_histc_ext import HistcExtDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = HistcExtDistributedOp("HistcExt")


class TestHistcExt(unittest.TestCase):
    """Unit tests for HistcExtDistributedOp."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=8):
        """Configure common mock-platform attributes used across tests."""
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    def _make_8_mesh(self, mock_platform, mesh_dim_name="dp"):
        """Set up mock and return a standard 8-element mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(8,),
            mesh_dim_names=(mesh_dim_name,),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_data_parallel_success(self, mock_platform):
        """Test HistcExt with data parallel sharding."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (100, 0, 3)
        output_layout = op.infer_layout((x_layout,), extra_args)

        assert output_layout.tensor_map == (-1,), (
            f"Data Parallel test failed. Expected (-1,), "
            f"got {output_layout.tensor_map}"
        )

        assert output_layout.is_partial(), (
            "Output should have partial state when input is sharded" 
        )
        assert output_layout.partial == ["sum", None], (
            f"Partial state mismatch. Expected ['sum', None], "
            f"got {output_layout.partial}"
        )

        # impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
        # assert callable(impl), (
        #     f"get_expand_impl should return callable for sharded input, "
        #     f"got {type(impl)}"
        # )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_model_parallel_success(self, mock_platform):
        """Test HistcExt with model parallel sharding."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (50, 0, 10)
        output_layout = op.infer_layout((x_layout,), extra_args)

        assert output_layout.tensor_map == (-1,), (
            f"Model Parallel test failed. Expected (-1,), "
            f"got {output_layout.tensor_map}"
        )

        assert output_layout.is_partial(), (
            "Output should have partial state when input is sharded"
        )
        assert output_layout.partial == [None, "sum"], (
            f"Partial state mismatch. Expected [None, 'sum'], "
            f"got {output_layout.partial}"
        )

        # impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
        # assert callable(impl), (
        #     f"get_expand_impl should return callable for sharded input, "
        #     f"got {type(impl)}"
        # )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_hybrid_parallel_success(self, mock_platform):
        """Test HistcExt with hybrid parallel sharding."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (200, -5, 5)
        output_layout = op.infer_layout((x_layout,), extra_args)

        assert output_layout.tensor_map == (-1,), (
            f"Hybrid Parallel test failed. Expected (-1,), "
            f"got {output_layout.tensor_map}"
        )

        assert output_layout.is_partial(), (
            "Output should have partial state when input is sharded"
        )
        assert output_layout.partial == ["sum", "sum"], (
            f"Partial state mismatch. Expected ['sum', 'sum'], "
            f"got {output_layout.partial}"
        )

        # impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
        # assert callable(impl), (
        #     f"get_expand_impl should return callable for sharded input, "
        #     f"got {type(impl)}"
        # )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_all_replicated(self, mock_platform):
        """Test HistcExt with all replicated input."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (100, 0, 3)
        output_layout = op.infer_layout((x_layout,), extra_args)

        assert output_layout.tensor_map == (-1,), (
            f"All Replicated test failed. Expected (-1,), "
            f"got {output_layout.tensor_map}"
        )

        assert not output_layout.is_partial(), (
            "Output should not have partial state when input is replicated"
        )

        # assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
        #     f"get_expand_impl should return None for replicated input, "
        #     f"got {op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
        # )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_3d_tensor(self, mock_platform):
        """Test HistcExt with 3D tensor input."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        extra_args = (100, 0, 3)
        output_layout = op.infer_layout((x_layout,), extra_args)

        assert output_layout.tensor_map == (-1,), (
            f"3D tensor test failed. Expected (-1,), "
            f"got {output_layout.tensor_map}"
        )

        assert output_layout.is_partial(), (
            "Output should have partial state for 3D sharded input"
        )

        # impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
        # assert callable(impl), (
        #     f"get_expand_impl should return callable for 3D sharded input, "
        #     f"got {type(impl)}"
        # )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_single_mesh_dim(self, mock_platform):
        """Test HistcExt with single mesh dimension."""
        mesh = self._make_8_mesh(mock_platform, "dp")
        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (100, 0, 3)
        output_layout = op.infer_layout((x_layout,), extra_args)

        assert output_layout.tensor_map == (-1,), (
            f"Single mesh dimension test failed. Expected (-1,), "
            f"got {output_layout.tensor_map}"
        )

        assert output_layout.is_partial(), (
            "Output should have partial state for sharded input"
        )
        assert output_layout.partial == ["sum"], (
            f"Partial state mismatch. Expected ['sum'], "
            f"got {output_layout.partial}"
        )

        # impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
        # assert callable(impl), (
        #     f"get_expand_impl should return callable for sharded input, "
        #     f"got {type(impl)}"
        # )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_default_parameters(self, mock_platform):
        """Test HistcExt with default parameters."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = ()
        output_layout = op.infer_layout((x_layout,), extra_args)

        assert output_layout.tensor_map == (-1,), (
            f"Default parameters test failed. Expected (-1,), "
            f"got {output_layout.tensor_map}"
        )

        assert output_layout.is_partial(), (
            "Output should have partial state with default parameters"
        )

        # impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
        # assert callable(impl), (
        #     f"get_expand_impl should return callable with default parameters, "
        #     f"got {type(impl)}"
        # )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_partial_bins_parameter(self, mock_platform):
        """Test HistcExt with only bins parameter."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (50,)
        output_layout = op.infer_layout((x_layout,), extra_args)

        assert output_layout.tensor_map == (-1,), (
            f"Partial bins parameter test failed. Expected (-1,), "
            f"got {output_layout.tensor_map}"
        )

        assert output_layout.is_partial(), (
            "Output should have partial state with partial parameters"
        )

        # impl = op.get_expand_impl(None, output_layout, (x_layout,), extra_args)
        # assert callable(impl), (
        #     f"get_expand_impl should return callable with partial parameters, "
        #     f"got {type(impl)}"
        # )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_invalid_input_layout_failure(self, mock_platform):
        """Test error when input layout is None."""
        self._setup_mock_platform(mock_platform, world_size=8)

        with self.assertRaisesRegex(ValueError, "Input layout cannot be None"):
            op.infer_layout((None,), (100, 0, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_empty_layouts_failure(self, mock_platform):
        """Test error when layouts is empty."""
        self._setup_mock_platform(mock_platform, world_size=8)

        with self.assertRaisesRegex(ValueError, "requires at least one input layout"):
            op.infer_layout((), (100, 0, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_none_layouts_failure(self, mock_platform):
        """Test error when layouts is None."""
        self._setup_mock_platform(mock_platform, world_size=8)

        with self.assertRaisesRegex(ValueError, "requires at least one input layout"):
            op.infer_layout(None, (100, 0, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_invalid_bins_type_failure(self, mock_platform):
        """Test error when bins is not an integer."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "bins must be an integer"):
            op.infer_layout((x_layout,), ("100", 0, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_invalid_bins_value_failure(self, mock_platform):
        """Test error when bins is not positive."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "bins must be a positive integer"):
            op.infer_layout((x_layout,), (0, 0, 3))

        with self.assertRaisesRegex(ValueError, "bins must be a positive integer"):
            op.infer_layout((x_layout,), (-10, 0, 3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_histc_ext_invalid_min_max_failure(self, mock_platform):
        """Test error when min > max."""
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "min must be less than or equal to max"):
            op.infer_layout((x_layout,), (100, 5, 3))

if __name__ == '__main__':
    unittest.main()
