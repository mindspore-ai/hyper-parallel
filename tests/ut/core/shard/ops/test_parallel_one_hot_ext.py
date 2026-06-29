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
"""parallel_one_hot_ext test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_one_hot_ext import OneHotExtDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelOneHotExt(unittest.TestCase):
    """Unit tests for OneHotExtDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    def _run_scenario(self, cache_values, expected_map):
        """Infer layout of OneHotExt operator"""
        op = OneHotExtDistributedOp("OneHotExt")
        infer_result = op.infer_layout(cache_values)
        output_layout = infer_result[0][0]
        assert output_layout.tensor_map == expected_map, (
            f"OneHotExt failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_1d_axis_minus1_data_parallel_1(self, mock_platform):
        """
        Feature: 1D indices, axis=-1.
        Description: Indices sharded on dp.
        Expectation: Output keeps dp sharding and inserts replicated one-hot dim at end.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0),), 1)

        self._run_scenario(
            cache_values=[indices_layout, None, None, 32, -1],
            expected_map=(2, -1),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_1d_axis_0_data_parallel_2(self, mock_platform):
        """
        Feature: 1D indices, axis=0.
        Description: Indices sharded on dp.
        Expectation: Insert replicated one-hot dim at position 0.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0),), 1)

        self._run_scenario(
            cache_values=[indices_layout, None, None, 32, 0],
            expected_map=(-1, 2),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_1d_axis_1_data_parallel_3(self, mock_platform):
        """
        Feature: 1D indices, axis=1.
        Description: axis=1 equals append for rank-1 tensor.
        Expectation: Same as axis=-1.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0),), 1)

        self._run_scenario(
            cache_values=[indices_layout, None, None, 32, 1],
            expected_map=(2, -1),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_1d_replicate_all_4(self, mock_platform):
        """
        Feature: 1D indices fully replicated.
        Description: tensor_map is (-1,).
        Expectation: Output fully replicated with inserted one-hot dim replicated.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)

        self._run_scenario(
            cache_values=[indices_layout, None, None, 32, -1],
            expected_map=(-1, -1),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_2d_axis_minus1_data_parallel_5(self, mock_platform):
        """
        Feature: 2D indices, axis=-1.
        Description: Only data-parallel (dim0 sharded), dim1 replicated.
        Expectation: Output keeps dp on dim0, inserts replicated one-hot dim at end.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        self._run_scenario(
            cache_values=[indices_layout, None, None, 64, -1],
            expected_map=(2, -1, -1),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_3d_axis_minus1_data_parallel_6(self, mock_platform):
        """
        Feature: 3D indices, axis=-1.
        Description: Only data-parallel (dim0 sharded), others replicated.
        Expectation: Output keeps dp on dim0, inserts replicated one-hot dim at end.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        self._run_scenario(
            cache_values=[indices_layout, None, None, 16, -1],
            expected_map=(2, -1, -1, -1),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_2d_axis_not_minus1_should_raise_7(self, mock_platform):
        """
        Feature: Multi-dimensional restriction.
        Description: For rank>1, axis must be -1.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        op = OneHotExtDistributedOp("OneHotExt")
        with self.assertRaises(ValueError):
            op.infer_layout([indices_layout, None, None, 64, 0])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_2d_not_data_parallel_should_raise_8(self, mock_platform):
        """
        Feature: Multi-dimensional restriction.
        Description: For rank>1, only dim0 can be sharded (DP only). Sharding dim1 is illegal.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        op = OneHotExtDistributedOp("OneHotExt")
        with self.assertRaises(ValueError):
            op.infer_layout([indices_layout, None, None, 64, -1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_axis_out_of_range_should_raise_9(self, mock_platform):
        """
        Feature: Axis validation.
        Description: axis must be in [-1, 1] for this implementation.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0),), 1)

        op = OneHotExtDistributedOp("OneHotExt")
        with self.assertRaises(ValueError):
            op.infer_layout([indices_layout, None, None, 32, 2])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_one_hot_ext_num_classes_invalid_should_raise_10(self, mock_platform):
        """
        Feature: num_classes validation.
        Description: num_classes must be int and >= -1.
        Expectation: Raise ValueError when num_classes < -1.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        indices_layout = _build_layout(mesh, (Shard(0),), 1)

        op = OneHotExtDistributedOp("OneHotExt")
        with self.assertRaises(ValueError):
            op.infer_layout([indices_layout, None, None, -2, -1])


class TestOneHotExtValidationPaths(unittest.TestCase):
    """
    Feature: OneHotExtDistributedOp early-validation and helper method coverage
    Description: Cover infer_layout early-exit paths and helper validation methods.
    Expectation: Correct return values or exceptions for each validation path.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def test_infer_layout_none_indices_layout(self):
        """None as indices layout in cache_values raises ValueError."""
        op = OneHotExtDistributedOp("OneHotExt_none")
        with self.assertRaises(ValueError):
            op.infer_layout([None, None, None, 32, -1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_partial_index_layout_raises(self, mock_platform):
        """Partial indices layout raises ValueError."""
        self._setup_mock_platform(mock_platform)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "mp"), init_backend=False,
        )
        indices_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 1)
        indices_layout.set_partial_by_dev_axis("dp", "sum")

        op = OneHotExtDistributedOp("OneHotExt_partial")
        cache_values = [indices_layout, None, None, 32, -1]
        with self.assertRaises(ValueError):
            op.infer_layout(cache_values)

    def test_validate_num_classes_non_int_raises_type_error(self):
        """_validate_num_classes with non-int raises TypeError."""
        op = OneHotExtDistributedOp("OneHotExt_validate")
        with self.assertRaises(TypeError):
            op._validate_num_classes("not_an_int")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_expand_impl_no_sharding_returns_none(self, mock_platform):
        """get_expand_impl with fully replicated layout returns None."""
        self._setup_mock_platform(mock_platform)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "mp"), init_backend=False,
        )
        indices_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 1)
        op = OneHotExtDistributedOp("OneHotExt_no_shard")
        cache_values = [indices_layout, None, None, -1, -1]
        result = op.get_expand_impl(None, None, cache_values)
        self.assertIsNone(result)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_expand_impl_sharded_returns_callable(self, mock_platform):
        """get_expand_impl with sharded layout returns a callable closure."""
        self._setup_mock_platform(mock_platform)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "mp"), init_backend=False,
        )
        indices_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 1)
        op = OneHotExtDistributedOp("OneHotExt_sharded")
        cache_values = [indices_layout, None, None, -1, -1]
        result = op.get_expand_impl(None, None, cache_values)
        self.assertTrue(callable(result))

    def test_validate_indices_dtype_non_int64_raises(self):
        """_validate_indices_dtype with non-int64 tensor raises TypeError."""
        import mindspore as ms
        from mindspore import Tensor as MSTensor
        op = OneHotExtDistributedOp("OneHotExt_dtype")
        float_indices = MSTensor([1, 2, 3], ms.float32)
        with self.assertRaises(TypeError):
            op._validate_indices_dtype(float_indices)


if __name__ == "__main__":
    unittest.main()
