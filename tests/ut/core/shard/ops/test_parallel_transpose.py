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
"""test ut for parallel transpose infer layout"""
import os
import unittest
from unittest.mock import patch
import numpy as np

# pylint: disable=wrong-import-position

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_transpose import TransposeDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


op_ms = TransposeDistributedOp("Transpose")
op_torch_permute = TransposeDistributedOp("permute")
op_torch_transpose = TransposeDistributedOp("transpose")
op_view = TransposeDistributedOp("TransposeView")


class TestParallelTranspose(unittest.TestCase):
    """Unit tests for TransposeDistributedOp."""
    def setUp(self) -> None:
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self) -> None:
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

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "cp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    def _make_2x2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=16)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2, 2), mesh_dim_names=("dp", "mp", "cp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_basic_transpose_operation(self, mock_platform):
        """
        Feature: Transpose distributed operator basic functionality
        Description: Test transpose operation with valid 2D layout and axis permutation
        Expectation: Output layout tensor map correctly permuted according to axis
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(1), Shard(0))
        x_layout = _build_layout(mesh, x_placements, 2)

        output_layout = op_ms.infer_layout((x_layout,), extra_args=((1, 0),))

        expected_map = (1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op_ms.get_expand_impl(None,output_layout, (x_layout,), ((1, 0),)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op_ms.get_expand_impl(None, output_layout, (x_layout,), ((1, 0),))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_3d_transpose_operation(self, mock_platform):
        """
        Feature: Transpose distributed operator with 3D tensor
        Description: Test transpose operation with 3D layout and complex axis permutation
        Expectation: Output layout tensor map correctly follows the given permutation
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(2), Shard(1), Shard(0))
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op_ms.infer_layout((x_layout,), extra_args=((2, 0, 1),))

        expected_map = (2, 0, 1)
        assert output_layout.to_dict()["tensor_map"] == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dimension_mismatch_error(self, mock_platform):
        """
        Feature: Transpose distributed operator error handling
        Description: Test transpose operation with mismatched tensor map and axis dimensions
        Expectation: Raise ValueError with appropriate message
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        with self.assertRaisesRegex(ValueError, "Input tensor shape and permutation must have the same size"):
            op_ms.infer_layout((x_layout,), extra_args=((1, 0, 2),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_negative_index_error(self, mock_platform):
        """
        Feature: Transpose distributed operator validation
        Description: Test MS-style transpose with negative index (not allowed in MS style logic)
        Expectation: Raise ValueError indicating invalid permutation
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("dp", "cp", "tp"))
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        with self.assertRaisesRegex(ValueError, "Invalid permutation"):
            op_ms.infer_layout((x_layout,), extra_args=((-1, 0, 1),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_duplicate_indices_error(self, mock_platform):
        """
        Feature: Transpose distributed operator uniqueness validation
        Description: Test transpose operation with duplicate indices in permutation
        Expectation: Raise ValueError indicating invalid permutation
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        with self.assertRaisesRegex(ValueError, "Invalid permutation"):
            op_ms.infer_layout((x_layout,), extra_args=((1, 1),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_permute_3d_basic(self, mock_platform):
        """
        Feature: Permute distributed operator basic functionality (3D)
        Description: PyTorch style permute moves dimensions on a 3D sharded tensor
        Expectation: Tensor map updated correctly following PyTorch semantics
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("tp", "cp", "dp"))
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        output_layout = op_torch_permute.infer_layout((x_layout,), extra_args=((2, 0, 1),))

        expected_map = (0, 2, 1)
        assert output_layout.to_dict()["tensor_map"] == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_permute_4d_complex(self, mock_platform):
        """
        Feature: Permute distributed operator with 4D tensor
        Description: Test permute operation with 4D layout to verify higher dimension support
        Expectation: Output layout tensor map correctly follows the given 4D permutation
        """
        mesh = self._make_2x2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2), Shard(3)), 4)

        output_layout = op_torch_permute.infer_layout((x_layout,), extra_args=((0, 3, 1, 2),))

        expected_map = (3, 0, 2, 1)
        assert output_layout.to_dict()["tensor_map"] == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_transpose_basic(self, mock_platform):
        """
        Feature: Torch Transpose distributed operator basic functionality
        Description: Swap two dimensions in a 2D sharded tensor
        Expectation: Tensor map dimensions swapped successfully
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        output_layout = op_torch_transpose.infer_layout((x_layout,), extra_args=(0, 1))

        expected_map = (0, 1)
        assert output_layout.to_dict()["tensor_map"] == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_transpose_negative_indices(self, mock_platform):
        """
        Feature: Torch Transpose negative indices support
        Description: Test transpose with negative dimension indices on a 3D sharded tensor
        Expectation: Negative indices are correctly interpreted and dimensions are swapped
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        output_layout = op_torch_transpose.infer_layout((x_layout,), extra_args=(0, -1))

        expected_map = (0, 1, 2)
        assert output_layout.to_dict()["tensor_map"] == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_transpose_out_of_bounds(self, mock_platform):
        """
        Feature: Torch Transpose bounds checking
        Description: Attempt to transpose dimensions using indices that exceed tensor rank
        Expectation: Raise ValueError indicating dimensions are out of bounds
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaisesRegex(ValueError, "Transpose dimensions out of bounds"):
            op_torch_transpose.infer_layout((x_layout,), extra_args=(0, 2))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_transpose_view_3d_basic(self, mock_platform):
        """
        Feature: TransposeView functionality
        Description: Verify TransposeView operation with standard 3D sharded layout
        Expectation: Output layout tensor map updated correctly according to the provided axis
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("tp", "cp", "dp"))
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

        output_layout = op_view.infer_layout((x_layout,), extra_args=((2, 0, 1),))

        expected_map = (0, 2, 1)
        assert output_layout.to_dict()["tensor_map"] == expected_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_identity_transpose(self, mock_platform):
        """
        Feature: Identity operation
        Description: Transpose with (0, 1, 2) on a 3D tensor to verify no change
        Expectation: Output layout remains identical to the input layout
        """
        mesh = self._make_2x2x2_mesh(mock_platform)

        x_placements = (Shard(0), Shard(1), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        output_layout = op_ms.infer_layout((x_layout,), extra_args=((0, 1, 2),))

        expected_map = (2, 1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map


class TestParallelTransposeExtView(unittest.TestCase):
    """Unit tests for TransposeDistributedOp (TransposeExtView variant)."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self) -> None:
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
        ext_view_op = TransposeDistributedOp("TransposeExtView")
        output_layout = ext_view_op.infer_layout((x_layout,), extra_args)
        assert output_layout.tensor_map == expected_map, (
            f"TransposeExtView failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

        assert ext_view_op.get_expand_impl(None, output_layout, (x_layout,), extra_args) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {ext_view_op.get_expand_impl(None, output_layout, (x_layout,), extra_args)}"
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

        self._run_scenario(x_layout, expected_map=(0, 1, 2), extra_args=(-1, -3))

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
            self._run_scenario(x_layout, expected_map=(2, 1, 0), extra_args=(-4, 0))

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
            self._run_scenario(x_layout, expected_map=(2, 1, 0), extra_args=(0,))

        with self.assertRaises((ValueError, TypeError)):
            self._run_scenario(x_layout, expected_map=(2, 1, 0), extra_args=None)


if __name__ == "__main__":
    unittest.main()
