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
"""parallel_gather test for index_select"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_gather import IndexSelectDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = IndexSelectDistributedOp("index_select")


class TestParallelIndexSelect(unittest.TestCase):
    """Unit tests for IndexSelectDistributedOp."""
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

    def _make_2x4_mesh(self, mock_platform, mesh_dim_names=("dp", "tp")):
        """Set up mock and return a standard 2x4 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=mesh_dim_names)

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "cp", "tp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    def _make_1d_mesh(self, mock_platform, world_size=8):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_valid_axis_0(self, mock_platform):
        """
        Feature: Valid index_select on axis 0
        Description: Param is unsharded on axis 0 and sharded on axis 1. Index is 1D and sharded.
        Expectation: Output layout correctly splices the index alias map and the param alias map.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_placements = (Replicate(), Shard(1))
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = (Shard(0), Replicate())
        i_layout = _build_layout(mesh, i_placements, 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (0,)

        output_layout = op.infer_layout(layouts, extra_args)

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Axis 0 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_valid_axis_1(self, mock_platform):
        """
        Feature: Valid index_select on axis 1
        Description: Param is sharded on axis 0 and unsharded on axis 1. Index is 1D and sharded.
        Expectation: Output layout combines the correct sharding strategies.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_placements = (Shard(0), Replicate())
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = (Replicate(), Shard(0))
        i_layout = _build_layout(mesh, i_placements, 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (1,)

        output_layout = op.infer_layout(layouts, extra_args)

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Axis 1 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_valid_negative_axis(self, mock_platform):
        """
        Feature: Valid index_select with negative axis
        Description: Pass a negative axis (-1) for a 2D parameter tensor.
        Expectation: Operation calculates the correct positive axis and proceeds without errors.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_placements = (Shard(0), Replicate())
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = (Replicate(), Shard(0))
        i_layout = _build_layout(mesh, i_placements, 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (-1,)

        output_layout = op.infer_layout(layouts, extra_args)

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Negative axis inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_index_ndim(self, mock_platform):
        """
        Feature: Invalid multi-dimensional index tensor
        Description: Provide an index tensor that is 2D instead of the required 1D.
        Expectation: Raises ValueError regarding index dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        layouts = (p_layout, None, i_layout)
        extra_args = (0,)

        with self.assertRaisesRegex(ValueError, "index is not a one-dimensional Tensor"):
            op.infer_layout(layouts, extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_axis_positive(self, mock_platform):
        """
        Feature: Invalid positive axis
        Description: Pass a positive axis value that exceeds the dimensions of the parameter tensor.
        Expectation: Raises ValueError for index out of bounds.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (2,)

        with self.assertRaisesRegex(ValueError, "is out of valid range"):
            op.infer_layout(layouts, extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_axis_negative(self, mock_platform):
        """
        Feature: Invalid negative axis
        Description: Pass a negative axis value that exceeds the negative dimension range.
        Expectation: Raises ValueError for index out of bounds.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

        layouts = (p_layout, None, i_layout)
        extra_args = (-3,)

        with self.assertRaisesRegex(ValueError, "is out of valid range"):
            op.infer_layout(layouts, extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_layouts_length(self, mock_platform):
        """
        Feature: Invalid layouts length
        Description: Pass an incomplete layouts tuple to infer_layout.
        Expectation: Raises ValueError about missing required layouts length.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

        layouts = (p_layout, i_layout)
        extra_args = (0,)

        with self.assertRaisesRegex(ValueError, "Gather ops requires 3 layouts"):
            op.infer_layout(layouts, extra_args)

        layouts_valid = (p_layout, None, i_layout)
        extra_args_invalid = ()

        with self.assertRaisesRegex(ValueError, "Gather ops requires 1 extra args"):
            op.infer_layout(layouts_valid, extra_args_invalid)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_unsharded_axis_unsharded_index(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Unsharded axis with an unsharded index tensor.
        Expectation: Output layout preserves sharding on non-axis dims, axis dim remains unsharded.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(1,))

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Unsharded axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_sharded_axis_unsharded_index(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Sharded axis with an unsharded index tensor.
        Expectation: Output layout drops sharding on the axis, replacing it with the unsharded index layout.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Shard(1)]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(1,))

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Sharded axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_negative_axis(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Using a negative axis value.
        Expectation: Output layout processes the negative axis correctly as a positive index internally.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Shard(1)]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(-1,))

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Negative axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_axis_out_of_bounds_positive(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Axis value exceeds the input tensor dimensions limits.
        Expectation: ValueError is raised with clear range context.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)
        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "dim value 2 is out of valid range"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(2,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_axis_out_of_bounds_negative(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Negative axis value is smaller than -ndim.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)
        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "dim value -3 is out of valid range"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(-3,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_index_ndim_2(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Index tensor is provided as a 2D tensor instead of 1D.
        Expectation: ValueError is raised enforcing 1D index requirement.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 2)

        with self.assertRaisesRegex(ValueError, "index is not a one-dimensional Tensor"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_layouts_length_2(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: An invalid number of layouts are passed to the operator.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)
        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "Gather ops requires 3 layouts"):
            op.infer_layout((p_layout, i_layout), extra_args=(0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_extra_args_length(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: An invalid number of extra arguments are passed.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))
        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)
        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "Gather ops requires 1 extra args"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(0, 1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_invalid_partial_input(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Input layout has a Partial status, which is not supported for this op.
        Expectation: ValueError is raised via _check_partial_inputs blocking.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Partial("sum"), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)
        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "Partial status which is not allowed"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_3d_input_axis_0(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: 3D input tensor, selecting on axis 0, with an unsharded index.
        Expectation: The output layout correctly replaces the first dimension's sharding with the index's sharding.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Shard(1), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 3)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

        expected_map = (-1, 0, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"3D input axis 0 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_1d_input_axis_0(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: 1D input tensor and 1D unsharded index tensor.
        Expectation: Output layout is 1D and reflects the index tensor's unsharded layout.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 1)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

        expected_map = (-1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D input axis 0 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_scalar_index_invalid(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: The index tensor is a 0D scalar (invalid for PyTorch index_select).
        Expectation: ValueError is raised ensuring the index is strictly 1-dimensional.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 0)

        with self.assertRaisesRegex(ValueError, "index is not a one-dimensional Tensor"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_get_expand_impl_unsharded(self, mock_platform):
        """
        Feature: Index select expand implementation
        Description: The selected axis is not sharded ("None").
        Expectation: The implementation falls back and returns the original standard function.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Replicate(), Shard(1)]
        p_layout = _build_layout(mesh, p_placements, 2)

        def dummy_func():
            pass

        impl = op.get_expand_impl(dummy_func, None, [p_layout], [0])

        assert impl is dummy_func, "Should return original function when axis is unsharded"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_get_expand_impl_sharded(self, mock_platform):
        """
        Feature: Index select expand implementation
        Description: The selected axis is sharded across a device mesh dimension.
        Expectation: The implementation returns a custom wrapper function for distributed execution.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        def dummy_func():
            pass

        impl = op.get_expand_impl(dummy_func, None, [p_layout], [0])

        assert impl is not dummy_func, "Should return custom wrapper when axis is sharded"
        assert impl.__name__ == "expand_impl", "Wrapper function should be named 'expand_impl'"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_negative_axis_on_3d_tensor(self, mock_platform):
        """
        Feature: Index select layout inference
        Description: Use a negative axis (-2) on a 3D tensor to verify robust boundary and translation mapping.
        Expectation: Axis -2 correctly translates to positive axis 1, yielding correct layout mapping.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        p_placements = [Shard(0), Shard(1), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 3)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(-2,))

        expected_map = (1, -1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Negative axis on 3D tensor failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_4d_param_axis_2(self, mock_platform):
        """
        Feature: Index select on 4D tensor
        Description: 4D parameter tensor sharded on dim 0 and dim 3. Index select on unsharded dim 2.
        Expectation: The output layout correctly drops the index mapping for dim 2 and inserts the index tensor's map.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_placements = [Shard(0), Shard(3)]
        p_layout = _build_layout(mesh, p_placements, 4)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(2,))

        expected_map = (1, -1, -1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"4D tensor axis 2 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_3d_mesh_fully_sharded_axis_1(self, mock_platform):
        """
        Feature: Index select with a 3D DeviceMesh
        Description: Use a 3D device mesh ("dp", "cp", "tp"). 3D Param is sharded across all 3 mesh dimensions.
        Expectation: Selecting axis 1 replaces its sharding with the index tensor's replicated layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)

        p_placements = [Shard(0), Shard(1), Shard(2)]
        p_layout = _build_layout(mesh, p_placements, 3)

        i_placements = [Replicate(), Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(1,))

        expected_map = (2, -1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"3D mesh fully sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_1d_mesh_param_sharded(self, mock_platform):
        """
        Feature: Index select with 1D DeviceMesh
        Description: 1D device mesh ("dp"). 2D Param is sharded on dim 0. Index is replicated.
        Expectation: Output layout processes 1D mesh correctly without crashing.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_placements = [Shard(0)]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D mesh param sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_1d_mesh_index_sharded(self, mock_platform):
        """
        Feature: Index select with 1D DeviceMesh and sharded index
        Description: 1D device mesh ("dp"). 2D Param is replicated. Index is sharded on dim 0.
        Expectation: The output inherits the index's sharding on the selected axis.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_placements = [Replicate()]
        p_layout = _build_layout(mesh, p_placements, 2)

        i_placements = [Shard(0)]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(1,))

        expected_map = (-1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D mesh index sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_1d_param_negative_axis(self, mock_platform):
        """
        Feature: Index select on 1D tensor with negative axis
        Description: 1D Param tensor, axis is -1.
        Expectation: Correctly resolves -1 to 0 and computes layout correctly.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_placements = [Shard(0), Replicate()]
        p_layout = _build_layout(mesh, p_placements, 1)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(-1,))

        expected_map = (-1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"1D param negative axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_5d_param_last_axis(self, mock_platform):
        """
        Feature: Index select on 5D tensor
        Description: 5D parameter tensor, selecting the last axis (axis=4).
        Expectation: The layout maps the first 4 dimensions strictly and replaces the 5th.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_placements = [Shard(0), Shard(4)]
        p_layout = _build_layout(mesh, p_placements, 5)

        i_placements = [Replicate(), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(4,))

        expected_map = (1, -1, -1, -1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"5D param last axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_output_mesh_shape_preservation(self, mock_platform):
        """
        Feature: Output layout properties
        Description: Verify the output layout correctly preserves the input mesh shape.
        Expectation: Output layout mesh shape is identical to input layout mesh shape.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

        assert output_layout.mesh_shape == p_layout.mesh_shape, (
            "Output mesh shape does not match input mesh shape."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_output_alias_name_preservation(self, mock_platform):
        """
        Feature: Output layout properties
        Description: Verify the output layout correctly preserves the input alias names.
        Expectation: Output layout alias name tuple is identical to input alias name tuple.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

        assert output_layout.alias_name == p_layout.alias_name, (
            "Output alias name does not match input alias name."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_output_rank_list_preservation(self, mock_platform):
        """
        Feature: Output layout properties
        Description: Verify the output layout correctly preserves the process rank list.
        Expectation: Output layout rank list is identical to input layout rank list.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

        assert output_layout.rank_list == p_layout.rank_list, (
            "Output rank list does not match input rank list."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_expand_impl_negative_sharded_axis(self, mock_platform):
        """
        Feature: get_expand_impl with negative axis
        Description: Check if get_expand_impl correctly identifies a sharded axis when axis is negative.
        Expectation: Returns a custom `expand_impl` closure rather than the original function.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Replicate(), Shard(1)], 2)

        def dummy_func():
            pass

        impl = op.get_expand_impl(dummy_func, None, [p_layout], [-1])

        assert impl is not dummy_func, "Should return custom wrapper for negative sharded axis."
        assert impl.__name__ == "expand_impl", "Wrapper function should be named 'expand_impl'."

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_expand_impl_other_dims_sharded_only(self, mock_platform):
        """
        Feature: get_expand_impl with unsharded axis but other sharded dims
        Description: The target axis is unsharded, but a different dimension of the tensor is sharded.
        Expectation: Returns the original function because the selected axis itself requires no cross-device sync.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)

        def dummy_func():
            pass

        impl = op.get_expand_impl(dummy_func, None, [p_layout], [1])

        assert impl is dummy_func, (
            "Should return original function when target axis is unsharded, "
            "regardless of other dims."
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_partial_index_layout_invalid(self, mock_platform):
        """
        Feature: Partial layout rejection
        Description: The index layout contains a Partial placement.
        Expectation: Raises ValueError blocking Partial inputs.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Replicate(), Replicate()], 2)

        i_placements = [Partial("sum"), Replicate()]
        i_layout = _build_layout(mesh, i_placements, 1)

        with self.assertRaisesRegex(ValueError, "Partial status which is not allowed"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_empty_layouts_invalid(self, mock_platform):
        """
        Feature: Layout validation
        Description: Pass an empty tuple for layouts.
        Expectation: Raises ValueError regarding the required layout length.
        """
        with self.assertRaisesRegex(ValueError, "Gather ops requires 3 layouts"):
            op.infer_layout((), extra_args=(0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_0d_param_invalid(self, mock_platform):
        """
        Feature: 0D scalar parameter layout
        Description: The parameter tensor is a 0D scalar (which has no dimensions to index).
        Expectation: Raises ValueError regarding out-of-bounds axis.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_layout = _build_layout(mesh, [Replicate(), Replicate()], 0)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        with self.assertRaisesRegex(ValueError, "dim value 0 is out of valid range"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_index_select_two_extra_args_invalid(self, mock_platform):
        """
        Feature: Extra arguments validation
        Description: Pass an extra argument tuple with length greater than 1.
        Expectation: Raises ValueError regarding required extra_args length.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        p_layout = _build_layout(mesh, [Replicate(), Replicate()], 2)
        i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

        with self.assertRaisesRegex(ValueError, "Gather ops requires 1 extra args"):
            op.infer_layout((p_layout, None, i_layout), extra_args=(0, 1))


if __name__ == "__main__":
    unittest.main()
