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
"""parallel_new_ones test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_new_ones import NewOnesDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = NewOnesDistributedOp("new_ones")


class TestParallelNewOnes(unittest.TestCase):
    """Unit tests for NewOnesDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2 (dp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_new_ones_infer_layout_tuple_size(self, mock_platform):
        """
        Feature: Infer layout for new_ones with tuple size
        Description: Create a new tensor of ones with a specific tuple shape from a sharded input tensor.
        Expectation: Output layout should be fully replicated (all dimensions -1) on the same device mesh.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=((3, 4),))

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Tuple size inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )
        assert output_layout.mesh_shape == mesh.mesh_shape

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args=((3, 4),)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout,), extra_args=((3, 4),))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_new_ones_infer_layout_list_size(self, mock_platform):
        """
        Feature: Infer layout for new_ones with list size
        Description: Create a new tensor of ones with a specific list shape.
        Expectation: Output layout should be fully replicated.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=([5, 2, 2],))

        expected_map = (-1, -1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"List size inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_new_ones_infer_layout_int_size(self, mock_platform):
        """
        Feature: Infer layout for new_ones with int size
        Description: Create a 1D new tensor of ones with an integer size.
        Expectation: Output layout should be a 1D Replicated layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=(10,))

        expected_map = (-1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Int size inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_new_ones_ignores_input_sharding(self, mock_platform):
        """
        Feature: Ignore input sharding
        Description: Even if input is fully sharded, the new tensor should be Replicated.
        Expectation: Output layout is strictly Replicated (-1, ...).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=((4, 4),))

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Should ignore input sharding. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_new_ones_scalar_shape(self, mock_platform):
        """
        Feature: Infer layout for scalar new_ones
        Description: Create a scalar tensor (empty tuple size).
        Expectation: Output layout tensor_map should be empty tuple.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op.infer_layout((x_layout,), extra_args=((),))

        expected_map = ()
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Scalar inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_new_ones_missing_args(self, mock_platform):
        """
        Feature: Error handling for missing args
        Description: Call infer_layout without size argument.
        Expectation: ValueError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "expected 'size' in extra_args"):
            op.infer_layout((x_layout,), extra_args=())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_new_ones_invalid_size_type(self, mock_platform):
        """
        Feature: Error handling for invalid size type
        Description: Call infer_layout with a string as size.
        Expectation: TypeError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        with self.assertRaisesRegex(TypeError, "must be int, tuple or list"):
            op.infer_layout((x_layout,), extra_args=("invalid_size",))


if __name__ == "__main__":
    unittest.main()
