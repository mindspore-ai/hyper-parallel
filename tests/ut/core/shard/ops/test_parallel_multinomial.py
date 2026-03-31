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
"""parallel_multinomial test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_multinomial import MultinomialDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = MultinomialDistributedOp("multinomial")


class TestParallelMultinomial(unittest.TestCase):
    """Unit tests for MultinomialDistributedOp."""
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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multinomial_infer_layout_1d_replicated(self, mock_platform):
        """
        Feature: Multinomial layout inference for 1D input
        Description: Input is a 1D probability vector (replicated).
        Expectation: Output layout should be 1D and Replicated ("None").
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 1)
        extra_args = (10, True, None)

        output_layout = op.infer_layout((x_layout,), extra_args)

        expected_map = (-1,)
        assert output_layout.tensor_map == expected_map, (
            f"1D inference failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
            f"get_expand_impl should return None, "
            f"but got {output_layout}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multinomial_infer_layout_2d_data_parallel(self, mock_platform):
        """
        Feature: Multinomial layout inference for 2D input (Data Parallel)
        Description: Input (N, C) is sharded on Batch dim (dim 0), C dim (dim 1) is replicated.
        Expectation: Output (N, num_samples) preserves sharding on dim 0, dim 1 is Replicated.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (10, True, None)
        output_layout = op.infer_layout((x_layout,), extra_args=extra_args)

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"2D Data Parallel inference failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multinomial_infer_layout_2d_fully_replicated(self, mock_platform):
        """
        Feature: Multinomial layout inference for 2D input (Fully Replicated)
        Description: Input (N, C) is fully replicated.
        Expectation: Output (N, num_samples) is fully replicated.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (10, True, None)
        output_layout = op.infer_layout((x_layout,), extra_args=extra_args)

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"2D Replicated inference failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multinomial_error_sharded_prob_dim(self, mock_platform):
        """
        Feature: Error handling for sharded probability dimension
        Description: Attempt to run multinomial when the last dimension (probability classes) is sharded.
        Expectation: ValueError raised requiring redistribution.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        assert x_layout.alias_tensor_map == ("None", "mp")

        with self.assertRaisesRegex(ValueError, "must not be sharded"):
            op.infer_layout((x_layout,), extra_args=(10, True, None))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multinomial_error_invalid_ndim_3d(self, mock_platform):
        """
        Feature: Error handling for invalid input dimensions
        Description: Attempt to run multinomial on a 3D tensor.
        Expectation: ValueError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        with self.assertRaisesRegex(ValueError, "input dimension must be 1 or 2"):
            op.infer_layout((x_layout,), extra_args=(10, True, None))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multinomial_error_invalid_ndim_0d(self, mock_platform):
        """
        Feature: Error handling for 0D input
        Description: Attempt to run multinomial on a 0D tensor (scalar).
        Expectation: ValueError raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 0)

        with self.assertRaisesRegex(ValueError, "input dimension must be 1 or 2"):
            op.infer_layout((x_layout,), extra_args=(10, True, None))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multinomial_2d_with_replacement_false(self, mock_platform):
        """
        Feature: Multinomial with replacement=False
        Description: Test 2D input with replacement=False.
        Expectation: Output layout follows same sharding rules.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        extra_args = (5, False, None)
        output_layout = op.infer_layout((x_layout,), extra_args=extra_args)

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"2D with replacement=False inference failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multinomial_partial_input(self, mock_platform):
        """
        Feature: Multinomial with partial input
        Description: Input with partial state
        Expectation: Raise ValueError since _allow_partial_inputs is False
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            op.infer_layout((x_layout,), extra_args=(10, True, None))


if __name__ == "__main__":
    unittest.main()
