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
"""parallel_norm test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_norm import NormDistributedOp, LayerNormDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

rmsnorm_op = NormDistributedOp("RmsNorm")
layernorm_op = LayerNormDistributedOp("layernorm")


class TestRmsNorm(unittest.TestCase):
    """Unit tests for RmsNorm distributed operator."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "mp", "tp"))

    def _make_8_mesh(self, mock_platform, mesh_dim_name="dp"):
        """Set up mock and return a standard 8-element mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=(mesh_dim_name,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_data_parallel_success(self, mock_platform):
        """
        Feature: RmsNorm data parallel
        Description: Data parallel scenario with no splitting on normalization axis
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        gamma_placements = (Replicate(), Replicate())
        gamma_layout = _build_layout(mesh, gamma_placements, 1)
        x_layout._partial = [None] * len(x_layout._partial)

        input_layouts = (x_layout, gamma_layout, None)
        _, output_layout = rmsnorm_op.infer_layout(input_layouts)

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"RmsNorm data parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert rmsnorm_op.get_expand_impl(None, output_layout, input_layouts, None) is None, (
            f"get_expand_impl should return None"
            f"got {rmsnorm_op.get_expand_impl(None, output_layout, input_layouts, None)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_model_parallel_success(self, mock_platform):
        """
        Feature: RmsNorm model parallel
        Description: Model parallel scenario with multiple dimensions sharded
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)
        gamma_placements = (Replicate(), Replicate(), Replicate())
        gamma_layout = _build_layout(mesh, gamma_placements, 1)
        x_layout._partial = [None] * len(x_layout._partial)

        input_layouts = (x_layout, gamma_layout, None)
        _, output_layout = rmsnorm_op.infer_layout(input_layouts)

        expected_map = (-1, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"RmsNorm model parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_hybrid_parallel_success(self, mock_platform):
        """
        Feature: RmsNorm hybrid parallel
        Description: Hybrid parallel scenario with multiple dimensions sharded
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)
        gamma_placements = (Replicate(), Replicate(), Replicate())
        gamma_layout = _build_layout(mesh, gamma_placements, 1)

        input_layouts = (x_layout, gamma_layout, None)
        _, output_layout = rmsnorm_op.infer_layout(input_layouts)

        expected_map = (1, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"RmsNorm hybrid parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_all_replicated(self, mock_platform):
        """
        Feature: RmsNorm all replicated
        Description: All replicated scenario with no sharding
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        gamma_placements = (Replicate(), Replicate())
        gamma_layout = _build_layout(mesh, gamma_placements, 1)
        x_layout._partial = [None] * len(x_layout._partial)

        input_layouts = (x_layout, gamma_layout, None)
        _, output_layout = rmsnorm_op.infer_layout(input_layouts)

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"RmsNorm all replicated test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_3d_tensor(self, mock_platform):
        """
        Feature: RmsNorm 3D tensor
        Description: Test with 3D tensor input
        Expectation: Success
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)
        gamma_placements = (Replicate(), Replicate())
        gamma_layout = _build_layout(mesh, gamma_placements, 1)

        input_layouts = (x_layout, gamma_layout, None)
        _, output_layout = rmsnorm_op.infer_layout(input_layouts)

        expected_map = (2, 1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"RmsNorm 3D tensor test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_sharded_dim_failure(self, mock_platform):
        """
        Feature: RmsNorm sharded dimension error
        Description: Test error when normalization dimension is sharded
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        gamma_placements = (Replicate(), Replicate())
        gamma_layout = _build_layout(mesh, gamma_placements, 1)

        input_layouts = (x_layout, gamma_layout, None)

        with self.assertRaisesRegex(ValueError, "RmsNorm is disabled to support the splitting"):
            rmsnorm_op.infer_layout(input_layouts)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_insufficient_inputs_failure(self, mock_platform):
        """
        Feature: RmsNorm insufficient inputs error
        Description: Test error when input layouts size is less than 3
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        input_layouts = (x_layout,)

        with self.assertRaisesRegex(ValueError, "input layouts size .* is less than 3"):
            rmsnorm_op.infer_layout(input_layouts)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_inconsistent_inputs_failure(self, mock_platform):
        """
        Feature: RmsNorm inconsistent inputs error
        Description: Test error when input layouts are inconsistent
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        x2_placements = (Replicate(), Shard(1))
        x2_layout = _build_layout(mesh, x2_placements, 2)
        gamma_placements = (Replicate(), Replicate())
        gamma_layout = _build_layout(mesh, gamma_placements, 1)
        x_layout._partial = [None] * len(x_layout._partial)
        x2_layout._partial = [None] * len(x2_layout._partial)

        input_layouts = (x_layout, x2_layout, gamma_layout, None)

        with self.assertRaisesRegex(ValueError, "RmsNorm inputs must have same layout"):
            rmsnorm_op.infer_layout(input_layouts)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_mismatched_mesh_shape_failure(self, mock_platform):
        """
        Feature: RmsNorm mismatched mesh shape error
        Description: Test error when input layouts have different mesh shapes
        Expectation: Raise ValueError
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh1 = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
        mesh2 = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0),)
        x_layout = _build_layout(mesh1, x_placements, 2)
        gamma_placements = (Replicate(), Replicate())
        gamma_layout = _build_layout(mesh2, gamma_placements, 1)

        input_layouts = (x_layout, gamma_layout, None)

        with self.assertRaisesRegex(ValueError, "inputs must have same mesh_shape"):
            rmsnorm_op.infer_layout(input_layouts)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rmsnorm_partial_input_failure(self, mock_platform):
        """
        Feature: RmsNorm partial input error
        Description: Test error when input has partial status
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        x_layout.set_partial_by_dev_axis("mp", "sum")
        gamma_placements = (Replicate(), Replicate())
        gamma_layout = _build_layout(mesh, gamma_placements, 1)

        input_layouts = (x_layout, gamma_layout, None)

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            rmsnorm_op.infer_layout(input_layouts)


class TestLayerNorm(unittest.TestCase):
    """Unit tests for LayerNorm distributed operator."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "mp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_data_parallel_success(self, mock_platform):
        """
        Feature: LayerNorm data parallel
        Description: Data parallel scenario with no splitting on normalization axis
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = layernorm_op.infer_layout((x_layout,), extra_args=((64,),))

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"LayerNorm data parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert layernorm_op.get_expand_impl(None, output_layout, (x_layout,), ((64,),)) is None, (
            f"get_expand_impl should return None"
            f"got {layernorm_op.get_expand_impl(None, output_layout, (x_layout,), ((64,),))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_model_parallel_success(self, mock_platform):
        """
        Feature: LayerNorm model parallel
        Description: Model parallel scenario with multiple dimensions sharded
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = layernorm_op.infer_layout((x_layout,), extra_args=((64,),))

        expected_map = (-1, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"LayerNorm model parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_hybrid_parallel_success(self, mock_platform):
        """
        Feature: LayerNorm hybrid parallel
        Description: Hybrid parallel scenario with multiple dimensions sharded
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = layernorm_op.infer_layout((x_layout,), extra_args=((64,),))

        expected_map = (1, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"LayerNorm hybrid parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_all_replicated(self, mock_platform):
        """
        Feature: LayerNorm all replicated
        Description: All replicated scenario with no sharding
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = layernorm_op.infer_layout((x_layout,), extra_args=((64,),))

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"LayerNorm all replicated test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_3d_tensor(self, mock_platform):
        """
        Feature: LayerNorm 3D tensor
        Description: Test with 3D tensor input
        Expectation: Success
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = layernorm_op.infer_layout((x_layout,), extra_args=((64,),))

        expected_map = (2, 1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"LayerNorm 3D tensor test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_sharded_dim_failure(self, mock_platform):
        """
        Feature: LayerNorm sharded dimension error
        Description: Test error when normalized dimension is sharded
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "Cannot perform sharding on normalized dimension"):
            layernorm_op.infer_layout((x_layout,), extra_args=((64,),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_normalized_shape_too_large_failure(self, mock_platform):
        """
        Feature: LayerNorm normalized_shape too large error
        Description: Test error when normalized_shape is larger than input ndim
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "larger than input ndim"):
            layernorm_op.infer_layout((x_layout,), extra_args=((128, 64, 56),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_none_input_failure(self, mock_platform):
        """
        Feature: LayerNorm None input error
        Description: Test error when input layout is None
        Expectation: Raise ValueError
        """
        with self.assertRaisesRegex(ValueError, "requires a valid input tensor layout"):
            layernorm_op.infer_layout((None,), extra_args=((64,),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_missing_extra_args_failure(self, mock_platform):
        """
        Feature: LayerNorm missing extra_args error
        Description: Test error when extra_args is missing
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "requires normalized_shape in extra_args"):
            layernorm_op.infer_layout((x_layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layernorm_invalid_normalized_shape_type_failure(self, mock_platform):
        """
        Feature: LayerNorm invalid normalized_shape type error
        Description: Test error when normalized_shape has invalid type
        Expectation: Raise ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "normalized_shape must be int, list, or tuple"):
            layernorm_op.infer_layout((x_layout,), extra_args=("invalid",))


if __name__ == "__main__":
    unittest.main()
