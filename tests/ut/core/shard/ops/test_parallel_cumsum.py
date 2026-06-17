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
"""parallel_cumsum test"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_cumsum import CumsumDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = CumsumDistributedOp("cumsum")
op_ms = CumsumDistributedOp("CumsumExt")


class TestParallelCumsum(unittest.TestCase):
    """Unit tests for CumsumDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

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

    def _make_2x3x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x3x4 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=24)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 3, 4), mesh_dim_names=("dp", "tp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_data_parallel(self, mock_platform):
        """
        Feature: Cumsum data parallel
        Description: Data parallel on non-cumsum dimension (dim=-1 unsharded)
        Expectation: Output layout identical to input layout
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, -1]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel cumsum failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert output_layout is not x_layout, "Cumsum output layout should be a deep copy"
        assert extra_info is None, f"Cumsum extra_info should be None, got {extra_info}"
        
        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_tensor_parallel(self, mock_platform):
        """
        Feature: Cumsum tensor parallel
        Description: Tensor parallel on non-cumsum dimension (dim=0 unsharded)
        Expectation: Output layout identical to input layout
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Tensor parallel cumsum failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"Cumsum extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_mixed_parallel(self, mock_platform):
        """
        Feature: Cumsum mixed parallel
        Description: Mixed parallel with cumsum on unsharded middle dimension
        Expectation: Output layout identical to input layout
        """
        mesh = self._make_2x3x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = (2, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Mixed parallel cumsum failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"Cumsum extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_negative_dim(self, mock_platform):
        """
        Feature: Cumsum with negative dimension
        Description: Test negative dimension indexing (dim=-2) on 3D tensor
        Expectation: Correctly normalized and validated
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, -2]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Negative dimension cumsum failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        assert extra_info is None, f"Cumsum extra_info should be None, got {extra_info}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_invalid_sharding_on_cumsum_dim(self, mock_platform):
        """
        Feature: Cumsum on sharded dimension
        Description: Attempt cumsum on a sharded dimension should fail
        Expectation: ValueError raised with clear message
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            op.infer_layout([x_layout, -1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_dim_out_of_range_positive(self, mock_platform):
        """
        Feature: Cumsum with invalid positive dimension
        Description: Dimension index exceeds tensor rank
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "dimension out of range"):
            op.infer_layout([x_layout, 2])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_missing_dim_parameter(self, mock_platform):
        """
        Feature: Cumsum without dim parameter
        Description: extra_args missing required 'dim' parameter
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "dimension should be int"):
            op.infer_layout([x_layout, None])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_layout_invalid_dim_type(self, mock_platform):
        """
        Feature: Cumsum with non-integer dim
        Description: dim parameter must be integer
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "dimension should be int"):
            op.infer_layout([x_layout, "invalid"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_partial_input_raises_error(self, mock_platform):
        """
        Feature: Cumsum with Partial input
        Description: Input layout has Partial status
        Expectation: ValueError raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout([x_layout, -1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_preprocess_torch_dtype_in_kwargs(self, mock_platform):
        """
        Feature: Cumsum preprocess for PyTorch
        Description: dtype is keyword-only parameter
        Expectation: local tensor and dim are positional; dtype is in kwargs
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor,),
            {'dim': -1, 'dtype': 'float32'}
        )

        assert len(local_args) == 2, (
            f"For PyTorch 'cumsum', local_args should be (tensor, dim), got {local_args}"
        )
        assert local_args[1] == -1, f"dim should be preserved as -1, got {local_args[1]}"
        assert local_kwargs == {'dtype': 'float32'}, (
            f"For PyTorch 'cumsum', dtype should be kwargs, got {local_kwargs}"
        )
        assert cache_values == [x_layout, -1], f"Unexpected cache_values: {cache_values}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cumsum_preprocess_mindspore_primitive_in_args(self, mock_platform):
        """
        Feature: Cumsum preprocess for MindSpore Primitive
        Description: CumsumExt does not accept kwargs
        Expectation: dim and dtype are routed to positional args, including dtype=None
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op_ms.preprocess(
            (mock_tensor,),
            {'dim': -1}
        )

        assert not local_kwargs, (
            f"For MindSpore 'CumsumExt', local_kwargs should be empty, got {local_kwargs}"
        )
        assert len(local_args) == 3, (
            f"For MindSpore 'CumsumExt', local_args should be (tensor, dim, dtype), got {local_args}"
        )
        assert local_args[1:] == (-1, None), (
            f"MindSpore positional dim/dtype mismatch, got {local_args[1:]}"
        )
        assert cache_values == [x_layout, -1], f"Unexpected cache_values: {cache_values}"


if __name__ == "__main__":
    unittest.main()
