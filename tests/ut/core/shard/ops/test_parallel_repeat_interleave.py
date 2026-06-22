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
"""parallel_repeat_interleave test"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Partial, Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_repeat_interleave import (
    RepeatInterleaveDistributedOp,
    _normalize_repeat_interleave_args,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = RepeatInterleaveDistributedOp("repeat_interleave")


class TestParallelRepeatInterleave(unittest.TestCase):
    """Unit tests for RepeatInterleaveDistributedOp."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

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
        """Set up mock and return a standard 2x4 (dp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_data_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave data parallel
        Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        dim = 1
        cache_values = [x_layout, dim]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Data Parallel with torch repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, result, cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, result, cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_tensor_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave tensor parallel
        Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(0))
        x_layout = _build_layout(mesh, x_placements, 2)
        dim = 1
        cache_values = [x_layout, dim]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]
        expected_map = (0, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Tensor Parallel with torch repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_with_tensor_layout_data_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave data parallel
        Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        dim = 1
        cache_values = [x_layout, dim]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]
        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Data Parallel with torch repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_with_tensor_layout_tensor_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave tensor parallel
        Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(0))
        x_layout = _build_layout(mesh, x_placements, 2)
        dim = 1
        cache_values = [x_layout, dim]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]
        expected_map = (0, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Tensor Parallel with torch repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_dim_none_layout_data_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave data parallel
        Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        cache_values = [x_layout, None]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]
        expected_map = (1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Data Parallel with dim None repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_dim_none_layout_tensor_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave tensor parallel
        Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(0))
        x_layout = _build_layout(mesh, x_placements, 2)
        cache_values = [x_layout, None]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]
        expected_map = (0,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Tensor Parallel dim None repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_hybrid_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave hybrid parallel
        Description: Hybrid scenario with multiple input dimensions sharded and repeated dim replicated
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, ("dp", "None", "tp"), 3)
        dim = 1
        cache_values = [x_layout, dim]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]
        expected_map = (1, -1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Hybrid parallel repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_all_replicated(self, mock_platform):
        """
        Feature: RepeatInterleave all replicated
        Description: All-replicated input keeps replicated layout after repeat_interleave
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        dim = 1
        cache_values = [x_layout, dim]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]
        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"All replicated repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_negative_dim(self, mock_platform):
        """
        Feature: RepeatInterleave negative dim
        Description: Negative repeat dim is normalized before validating sharding
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        dim = -1
        cache_values = [x_layout, dim]
        result = op.infer_layout(cache_values)
        output_layout = result[0][0]
        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Negative dim repeat_interleave test failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_partial_input_error(self, mock_platform):
        """
        Feature: RepeatInterleave partial input validation
        Description: Partial input layout should be rejected before layout inference
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Partial(), Replicate()), 2)
        dim = 1
        cache_values = [x_layout, dim]
        with self.assertRaisesRegex(ValueError, "status which is not allowed"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_strided_shard_repeat_dim_error(self, mock_platform):
        """
        Feature: RepeatInterleave StridedShard repeat dim validation
        Description: Repeat dim using tuple alias map should be treated as sharded
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (("dp", "tp"), "None"), 2)
        dim = 0
        cache_values = [x_layout, dim]
        with self.assertRaisesRegex(ValueError, "repeat dimension should be replicated"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_sharded_dim_error(self, mock_platform):
        """
        Feature: RepeatInterleave on sharded dimension
        Description: Repeat on a sharded dimension should raise error
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        dim = 1
        cache_values = [x_layout, dim]
        with self.assertRaisesRegex(ValueError, "repeat dimension should be replicated"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_error_dim_out_of_range(self, mock_platform):
        """
        Feature: Test indicating a invalid dim
        Description: Test indicating a invalid dim.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        dim = 5
        cache_values = [x_layout, dim]
        with self.assertRaisesRegex(ValueError, "dimension should be in"):
            op.infer_layout(cache_values)

    def test_normalize_repeat_interleave_args_basic(self):
        """
        Feature: Normalize repeat_interleave args
        Description: Verify normalize function returns correct structure
        Expectation: Success
        """
        mock_input = MagicMock()
        args, kwargs = _normalize_repeat_interleave_args(mock_input, 3, 1)
        assert args == (mock_input, 3, 1), (
            f"Normalize args mismatch: expected (mock_input, 3, 1), got {args}"
        )
        assert not kwargs, (
            f"Normalize kwargs mismatch: expected {{}}, got {kwargs}"
        )

    def test_normalize_repeat_interleave_args_with_output_size(self):
        """
        Feature: Normalize repeat_interleave args with output_size
        Description: Verify output_size goes into kwargs
        Expectation: Success
        """
        mock_input = MagicMock()
        args, kwargs = _normalize_repeat_interleave_args(mock_input, 3, 1, output_size=10)
        assert args == (mock_input, 3, 1), (
            f"Normalize args mismatch: expected (mock_input, 3, 1), got {args}"
        )
        assert kwargs == {'output_size': 10}, (
            f"Normalize kwargs mismatch: expected {{'output_size': 10}}, got {kwargs}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_with_int_repeats(self, mock_platform):
        """
        Feature: Preprocess with int repeats
        Description: Verify preprocess correctly handles int repeats
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_input = MagicMock()
        mock_input.layout = x_layout
        mock_local = MagicMock()
        mock_input.to_local.return_value = mock_local

        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_input, 3, 1), {}
        )

        assert local_args[0] is mock_local, (
            f"local_args[0] should be local input: expected={mock_local}, got={local_args[0]}"
        )
        assert local_args[1] == 3, f"local_args[1] should be 3, got {local_args[1]}"
        assert local_args[2] == 1, f"local_args[2] should be 1, got {local_args[2]}"
        assert not local_kwargs, f"local_kwargs should be empty, got {local_kwargs}"
        assert cache_values[0] is x_layout, (
            f"cache_values[0] should be input layout: expected={x_layout}, got={cache_values[0]}"
        )
        assert cache_values[1] == 1, f"cache_values[1] should be 1, got {cache_values[1]}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_dim_none(self, mock_platform):
        """
        Feature: Preprocess with dim=None
        Description: Verify preprocess correctly handles dim=None (flatten mode)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_input = MagicMock()
        mock_input.layout = x_layout
        mock_local = MagicMock()
        mock_input.to_local.return_value = mock_local

        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_input, 3, None), {}
        )

        assert local_args[0] is mock_local, (
            f"local_args[0] should be local input: expected={mock_local}, got={local_args[0]}"
        )
        assert local_args[1] == 3, f"local_args[1] should be 3, got {local_args[1]}"
        assert local_args[2] is None, f"local_args[2] should be None, got {local_args[2]}"
        assert cache_values[1] is None, f"cache_values[1] should be None, got {cache_values[1]}"


if __name__ == "__main__":
    unittest.main()
