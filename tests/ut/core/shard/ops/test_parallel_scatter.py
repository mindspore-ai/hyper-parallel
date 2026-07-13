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
"""parallel_scatter test"""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_scatter import ScatterDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ScatterDistributedOp("scatter")


class TestParallelScatter(unittest.TestCase):
    """Unit tests for ScatterDistributedOp."""
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

    def _create_mesh(self, mock_platform, shape=(2, 4), alias=("dp", "mp")):
        """Helper to create mesh and suppress pylint false positives."""
        self._setup_mock_platform(mock_platform, world_size=shape[0] * shape[1])
        return init_device_mesh(
            device_type="npu",
            mesh_shape=shape,
            mesh_dim_names=alias,
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_infer_layout_success(self, mock_platform):
        """
        Feature: Scatter on valid dimension
        Description: Scatter along a replicated dimension while another dimension is sharded.
                     Index and Src layouts match Input layout.
        Expectation: Output layout is identical to input layout.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        index_layout = input_layout
        src_layout = input_layout

        cache_values = [input_layout, 1, index_layout, src_layout]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"Scatter output layout mismatch: "
            f"expected={input_layout.tensor_map}, got={output_layout.tensor_map}"
        )
        assert output_layout.mesh_shape == input_layout.mesh_shape, (
            f"Scatter mesh shape mismatch: "
            f"expected={input_layout.mesh_shape}, got={output_layout.mesh_shape}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_sharded_dim(self, mock_platform):
        """
        Feature: Scatter on sharded dimension restriction
        Description: Attempt to scatter along a dimension that is sharded.
        Expectation: ValueError raised indicating scatter on sharded dimension is not supported.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, 0, input_layout, input_layout]

        with self.assertRaisesRegex(ValueError, "scatter dim should be replicated"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_index_mismatch(self, mock_platform):
        """
        Feature: Index layout validation
        Description: Index tensor layout does not match Input tensor layout.
        Expectation: ValueError raised.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        index_placements = (Replicate(), Replicate())
        index_layout = _build_layout(mesh, index_placements, 2)

        cache_values = [input_layout, 1, index_layout, input_layout]

        with self.assertRaisesRegex(ValueError, "index layout should match input layout"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_src_mismatch(self, mock_platform):
        """
        Feature: Source layout validation
        Description: Src tensor layout does not match Input tensor layout.
        Expectation: ValueError raised.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        src_placements = (Replicate(), Replicate())
        src_layout = _build_layout(mesh, src_placements, 2)

        cache_values = [input_layout, 1, input_layout, src_layout]

        with self.assertRaisesRegex(ValueError, "src layout should match input layout"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_negative_dim_normalization(self, mock_platform):
        """
        Feature: Dimension normalization
        Description: Use negative index for dim.
        Expectation: Correctly identifies the dimension and proceeds without errors.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, -1, input_layout, input_layout]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"Negative dim scatter failed: "
            f"expected={input_layout.tensor_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_negative_dim_sharded(self, mock_platform):
        """
        Feature: Dimension normalization with sharded check
        Description: Use negative index for dim that maps to a sharded dimension.
        Expectation: ValueError raised.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Replicate(), Shard(1))
        input_layout = _build_layout(mesh, input_placements, 2)

        assert input_layout.tensor_map[1] != -1, "Test setup error: Dimension 1 is not sharded"
        input_layout._partial = [None] * len(input_layout._partial)

        cache_values = [input_layout, -1, input_layout, input_layout]

        with self.assertRaisesRegex(ValueError, "scatter dim should be replicated"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_scalar_src(self, mock_platform):
        """
        Feature: Scalar source support
        Description: Src argument is not a tensor (None in layouts), representing a scalar.
        Expectation: Success, skipping src layout check.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, 1, input_layout, None]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"Scalar src scatter failed: "
            f"expected={input_layout.tensor_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_partial_input(self, mock_platform):
        """
        Feature: Partial input validation
        Description: Input tensor is in Partial state.
        Expectation: ValueError raised as scatter cannot operate on partial tensors.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Partial(), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, 1, input_layout, input_layout]

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_3d_complex_sharding(self, mock_platform):
        """
        Feature: 3D Tensor Scatter
        Description: 3D tensor sharded on dim0 and dim2, scatter on replicated dim1.
        Expectation: Success, output layout matches input.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "mp", "sp"),
            init_backend=False
        )
        input_placements = (Shard(0), Replicate(), Shard(2))
        input_layout = _build_layout(mesh, input_placements, 3)

        cache_values = [input_layout, 1, input_layout, input_layout]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"3D scatter failed: "
            f"expected={input_layout.tensor_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_all_replicated(self, mock_platform):
        """
        Feature: Fully replicated input
        Description: Input tensor is fully replicated on all devices.
        Expectation: Scatter allowed on any dimension.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Replicate(), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, 0, input_layout, input_layout]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"All replicated scatter failed: "
            f"expected={input_layout.tensor_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_invalid_dim_type(self, mock_platform):
        """
        Feature: Dim type validation
        Description: Pass a float instead of int for dim.
        Expectation: ValueError raised.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, 1.5, input_layout, input_layout]

        with self.assertRaisesRegex(ValueError, "dim should be an integer"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_dim_out_of_bounds_high(self, mock_platform):
        """
        Feature: Dim bounds check (Upper)
        Description: Dim index larger than ndim.
        Expectation: ValueError raised.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, 2, input_layout, input_layout]

        with self.assertRaisesRegex(ValueError, "should be in range"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_dim_out_of_bounds_low(self, mock_platform):
        """
        Feature: Dim bounds check (Lower)
        Description: Negative dim index smaller than -ndim.
        Expectation: ValueError raised.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, -3, input_layout, input_layout]

        with self.assertRaisesRegex(ValueError, "should be in range"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_none_input_layout(self, mock_platform):
        """
        Feature: None input layout validation
        Description: Input layout in cache_values is None.
        Expectation: ValueError raised.
        """
        cache_values = [None, 0, None, None]

        with self.assertRaisesRegex(ValueError, "should be a DTensor with a valid layout"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_null_index_layout(self, mock_platform):
        """
        Feature: Null index layout tolerance
        Description: Index layout passed as None.
        Expectation: Success (if dim check passes).
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        cache_values = [input_layout, 1, None, input_layout]
        output_layouts, _ = op.infer_layout(cache_values)
        output_layout = output_layouts[0]

        assert output_layout.tensor_map == input_layout.tensor_map, (
            f"Null index scatter failed: "
            f"expected={input_layout.tensor_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_sharded_src_mismatch_complex(self, mock_platform):
        """
        Feature: Src layout mismatch with replication
        Description: Input is fully replicated, but Src is sharded.
        Expectation: ValueError raised.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Replicate(), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        src_placements = (Shard(0), Replicate())
        src_layout = _build_layout(mesh, src_placements, 2)

        cache_values = [input_layout, 0, input_layout, src_layout]

        with self.assertRaisesRegex(ValueError, "src layout should match input layout"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_preprocess(self, mock_platform):
        """
        Feature: Preprocess for Scatter operator
        Description: Verify preprocess converts DTensor inputs to local and builds cache_values.
        Expectation: local_args contain local tensors, cache_values contain layouts + dim.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)
        index_layout = input_layout
        src_layout = input_layout

        mock_input = MagicMock()
        mock_input.layout = input_layout
        mock_input.to_local.return_value = MagicMock()
        mock_index = MagicMock()
        mock_index.layout = index_layout
        mock_index.to_local.return_value = MagicMock()
        mock_src = MagicMock()
        mock_src.layout = src_layout
        mock_src.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_input, 1, mock_index, mock_src), {}
        )

        assert not local_kwargs, (
            f"Expected local_kwargs to be empty, got {local_kwargs}"
        )
        assert len(cache_values) == 4, (
            f"Expected 4 cache_values, got {len(cache_values)}"
        )
        assert cache_values[0] is input_layout, (
            f"cache_values[0] should be input_layout, got {cache_values[0]}"
        )
        assert cache_values[1] == 1, (
            f"cache_values[1] should be dim=1, got {cache_values[1]}"
        )
        assert cache_values[2] is index_layout, (
            f"cache_values[2] should be index_layout, got {cache_values[2]}"
        )
        assert cache_values[3] is src_layout, (
            f"cache_values[3] should be src_layout, got {cache_values[3]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_preprocess_scalar_src(self, mock_platform):
        """
        Feature: Preprocess with scalar src
        Description: Verify preprocess handles scalar (non-DTensor) src correctly.
        Expectation: src passed through to local_args, cache_values[3] is None.
        """
        mesh = self._create_mesh(mock_platform)
        input_placements = (Shard(0), Replicate())
        input_layout = _build_layout(mesh, input_placements, 2)

        mock_input = MagicMock()
        mock_input.layout = input_layout
        mock_input.to_local.return_value = MagicMock()
        mock_index = MagicMock()
        mock_index.layout = input_layout
        mock_index.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_input, 1, mock_index, 3.14), {}
        )

        assert local_args[3] == 3.14, (
            f"Scalar src should pass through unchanged, got {local_args[3]}"
        )
        assert cache_values[3] is None, (
            f"Scalar src layout should be None, got {cache_values[3]}"
        )


if __name__ == "__main__":
    unittest.main()
    