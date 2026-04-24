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
"""parallel_squeeze test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_squeeze import SqueezeDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelSqueeze(unittest.TestCase):
    """Unit tests for SqueezeDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()
        self.op = SqueezeDistributedOp("Squeeze")

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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "mp"),
            init_backend=False
        )

    def _run_scenario(self, x_layout, expected_map, axis, input_shape):
        """Infer layout of Squeeze operator"""
        if axis is not None:
            extra_args = [axis, [input_shape]]
        else:
            extra_args = [[input_shape]]

        output_layout = self.op.infer_layout((x_layout,), extra_args)
        actual_map = output_layout.tensor_map
        assert actual_map == expected_map, (
            f"Squeeze failed. Expected {expected_map}, got {actual_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert self.op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {self.op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_data_parallel_1(self, mock_platform):
        """
        Feature: Data parallel.
        Description: remove singleton dimension at position 1, axis=1.
        Expectation: singleton dimension removed, DP dimension at position 0 remains.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        input_shape = [8, 1, 4]

        self._run_scenario(x_layout, expected_map=(2, -1), axis=1, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_model_parallel_2(self, mock_platform):
        """
        Feature: Model parallel.
        Description: remove singleton dimension before mp axis, axis=0.
        Expectation: singleton dimension removed, MP dimension at position 0.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)
        input_shape = [1, 1, 8]

        self._run_scenario(x_layout, expected_map=(-1, 0), axis=0, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_hybrid_parallel_3(self, mock_platform):
        """
        Feature: Hybrid parallel.
        Description: remove singleton dimension in middle, axis=1.
        Expectation: singleton dimension removed, DP at position 0, CP at position 1, MP at position 2.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(3)), 4)
        input_shape = [8, 1, 4, 2]

        self._run_scenario(x_layout, expected_map=(2, 1, 0), axis=1, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_remove_at_end_4(self, mock_platform):
        """
        Feature: Remove singleton at end.
        Description: remove singleton dimension at the end, axis=-1.
        Expectation: last singleton dimension removed.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2), Replicate()), 4)
        input_shape = [8, 4, 2, 1]

        self._run_scenario(x_layout, expected_map=(2, 1, 0), axis=-1, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_negative_axis_5(self, mock_platform):
        """
        Feature: Negative axis indexing.
        Description: axis=-2 for rank-3 input with singleton at position 1, equivalent to axis=1.
        Expectation: singleton dimension at position 1 removed.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        input_shape = [8, 1, 2]

        self._run_scenario(x_layout, expected_map=(2, 0), axis=-2, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_all_singletons_6(self, mock_platform):
        """
        Feature: Remove all singleton dimensions.
        Description: no axis specified, squeeze all singleton dimensions.
        Expectation: all shape=1 dimensions removed.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(1), Shard(3), Shard(5)), 7)
        input_shape = [1, 8, 1, 4, 1, 2, 1]

        self._run_scenario(x_layout, expected_map=(2, 1, 0), axis=None, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_multiple_axes_7(self, mock_platform):
        """
        Feature: Remove multiple singleton dimensions.
        Description: specify multiple axes to squeeze.
        Expectation: specified singleton dimensions removed.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(1), Shard(3), Shard(5)), 6)
        input_shape = [1, 8, 1, 4, 1, 2]

        self._run_scenario(x_layout, expected_map=(2, 1, 0), axis=[0, 2, 4], input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_no_singletons_8(self, mock_platform):
        """
        Feature: No singleton dimensions to remove.
        Description: try to squeeze distributed dimensions.
        Expectation: ValueError should be raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        input_shape = [8, 4, 2]

        with self.assertRaises(ValueError):
            self._run_scenario(x_layout, expected_map=(), axis=1, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_scalar_9(self, mock_platform):
        """
        Feature: Squeeze scalar.
        Description: squeeze all dimensions from scalar (rank-0).
        Expectation: scalar remains scalar (empty tensor map).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (), 0)
        input_shape = []

        self._run_scenario(x_layout, expected_map=(), axis=None, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_invalid_axis_10(self, mock_platform):
        """
        Feature: Invalid axis.
        Description: axis out of valid range.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        input_shape = [8, 1, 2]

        with self.assertRaises(ValueError):
            self._run_scenario(x_layout, expected_map=(), axis=5, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_negative_axes_list_11(self, mock_platform):
        """
        Feature: Multiple negative axes.
        Description: specify multiple negative axes to squeeze.
        Expectation: specified singleton dimensions removed.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(1), Shard(3), Shard(5)), 6)
        input_shape = [1, 8, 1, 4, 1, 2]

        self._run_scenario(x_layout, expected_map=(2, 1, 0), axis=[-6, -4, -2], input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_mixed_axes_12(self, mock_platform):
        """
        Feature: Mixed positive and negative axes.
        Description: specify mixed positive and negative axes.
        Expectation: specified singleton dimensions removed.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(1), Shard(3), Shard(5)), 6)
        input_shape = [1, 8, 1, 4, 1, 2]

        self._run_scenario(x_layout, expected_map=(2, 1, 0), axis=[0, -4, 4], input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_partial_axes_13(self, mock_platform):
        """
        Feature: Partial axes removal.
        Description: squeeze only some singleton dimensions.
        Expectation: only specified singleton dimensions removed.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(1), Shard(3), Shard(5)), 7)
        input_shape = [1, 8, 1, 4, 1, 2, 1]

        self._run_scenario(x_layout, expected_map=(2, 1, 0, -1), axis=[0, 2, 4], input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_shape_not_one_14(self, mock_platform):
        """
        Feature: Try to squeeze dimension with shape not equal to 1.
        Description: axis specified but shape is not 1.
        Expectation: ValueError should be raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        input_shape = [8, 4, 2]

        with self.assertRaises(ValueError):
            self._run_scenario(x_layout, expected_map=(), axis=0, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_distributed_dimension_15(self, mock_platform):
        """
        Feature: Try to squeeze distributed dimension.
        Description: axis specified for a dimension that is distributed.
        Expectation: ValueError should be raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        input_shape = [8, 1, 2]

        with self.assertRaises(ValueError):
            self._run_scenario(x_layout, expected_map=(), axis=1, input_shape=input_shape)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_no_input_shapes_16(self, mock_platform):
        """
        Feature: No input_shapes provided.
        Description: extra_args doesn't contain input_shapes.
        Expectation: ValueError should be raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        with self.assertRaises(ValueError):
            self.op.infer_layout((x_layout,), [])

        with self.assertRaises(ValueError):
            self.op.infer_layout((x_layout,), [1])

        with self.assertRaises(ValueError):
            self.op.infer_layout((x_layout,), None)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_squeeze_scalar_with_axis_17(self, mock_platform):
        """
        Feature: Try to specify axis for scalar.
        Description: axis specified for scalar input.
        Expectation: ValueError should be raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (), 0)
        input_shape = []

        with self.assertRaises(ValueError):
            self.op.infer_layout((x_layout,), [0, [input_shape]])


if __name__ == "__main__":
    unittest.main()
