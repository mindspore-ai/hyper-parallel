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
"""parallel_transpose_ext_view test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.shard.ops.parallel_transpose import TransposeDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelTransposeExtView(unittest.TestCase):
    """Unit tests for TransposeDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    def _run_scenario(self, x_layout, expected_map, extra_args):
        """Infer layout of TransposeExtView operator and validate tensor_map."""
        op = TransposeDistributedOp("TransposeExtView")
        output_layout = op.infer_layout((x_layout,), extra_args)
        assert output_layout.tensor_map == expected_map, (
            f"TransposeExtView failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

        assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_transpose_ext_view_basic_swap_3d_1(self, mock_platform):
        """
        Feature: Basic swap.
        Description: swap dim0=0 and dim1=2 on 3D tensor map.
        Expectation: tensor_map dims swapped.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        self._run_scenario(x_layout, expected_map=(0, 1, 2), extra_args=(0, 2))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_transpose_ext_view_negative_dims_2(self, mock_platform):
        """
        Feature: Negative dims.
        Description: swap dim0=-1 and dim1=-3 on 3D tensor map.
        Expectation: normalized dims swapped.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        self._run_scenario(
            x_layout,
            expected_map=(0, 1, 2),
            extra_args=(-1, -3),
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_transpose_ext_view_noop_same_dims_3(self, mock_platform):
        """
        Feature: No-op.
        Description: dim0 == dim1.
        Expectation: output tensor_map unchanged.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        self._run_scenario(x_layout, expected_map=(2, 1, 0), extra_args=(1, 1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_transpose_ext_view_dim_out_of_range_4(self, mock_platform):
        """
        Feature: Error handling.
        Description: dim0 or dim1 out of range [-ndim, ndim-1].
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        with self.assertRaises(ValueError):
            self._run_scenario(x_layout, expected_map=(2, 1, 0), extra_args=(3, 0))

        with self.assertRaises(ValueError):
            self._run_scenario(
                x_layout,
                expected_map=(2, 1, 0),
                extra_args=(-4, 0),
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_transpose_ext_view_dim_type_error_5(self, mock_platform):
        """
        Feature: Error handling.
        Description: dim0 or dim1 is not int.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        with self.assertRaises(ValueError):
            self._run_scenario(x_layout, expected_map=(2, 1, 0), extra_args=("0", 1))

        with self.assertRaises(ValueError):
            self._run_scenario(x_layout, expected_map=(2, 1, 0), extra_args=(0, None))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_transpose_ext_view_extra_args_invalid_6(self, mock_platform):
        """
        Feature: Error handling.
        Description: extra_args is not (dim0, dim1).
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        with self.assertRaises(ValueError):
            self._run_scenario(
                x_layout,
                expected_map=(2, 1, 0),
                extra_args=(0,),
            )

        with self.assertRaises((ValueError, TypeError)):
            self._run_scenario(
                x_layout,
                expected_map=(2, 1, 0),
                extra_args=None,
            )


if __name__ == "__main__":
    unittest.main()
