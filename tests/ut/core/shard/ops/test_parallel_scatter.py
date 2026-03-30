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
from unittest.mock import patch
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_scatter import ScatterDistributedOp
from hyper_parallel.platform import get_platform
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

        layouts = (input_layout, None, index_layout, src_layout)
        extra_args = (1,)

        output_layout = op.infer_layout(layouts, extra_args=extra_args)

        assert output_layout.tensor_map == input_layout.tensor_map
        assert output_layout.mesh_shape == input_layout.mesh_shape

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, layouts, extra_args=extra_args) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, layouts, extra_args=extra_args)}"
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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (0,)

        with self.assertRaisesRegex(ValueError, "Scatter along sharded dimension 0 is not supported"):
            op.infer_layout(layouts, extra_args=extra_args)

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

        layouts = (input_layout, None, index_layout, input_layout)
        extra_args = (1,)

        with self.assertRaisesRegex(ValueError, "Index tensor layout .* must match input tensor layout"):
            op.infer_layout(layouts, extra_args=extra_args)

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

        layouts = (input_layout, None, input_layout, src_layout)
        extra_args = (1,)

        with self.assertRaisesRegex(ValueError, "Src tensor layout .* must match input tensor layout"):
            op.infer_layout(layouts, extra_args=extra_args)

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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (-1,)

        output_layout = op.infer_layout(layouts, extra_args=extra_args)
        assert output_layout.tensor_map == input_layout.tensor_map

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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (-1,)

        with self.assertRaisesRegex(ValueError, "Scatter along sharded dimension 1 is not supported"):
            op.infer_layout(layouts, extra_args=extra_args)

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

        layouts = (input_layout, None, input_layout, None)
        extra_args = (1,)

        output_layout = op.infer_layout(layouts, extra_args=extra_args)
        assert output_layout.tensor_map == input_layout.tensor_map

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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (1,)

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            op.infer_layout(layouts, extra_args=extra_args)

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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (1,)

        output_layout = op.infer_layout(layouts, extra_args=extra_args)
        assert output_layout.tensor_map == input_layout.tensor_map

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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (0,)

        output_layout = op.infer_layout(layouts, extra_args=extra_args)
        assert output_layout.tensor_map == input_layout.tensor_map

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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (1.5,)

        with self.assertRaisesRegex(ValueError, "'dim' must be an integer"):
            op.infer_layout(layouts, extra_args=extra_args)

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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (2,)

        with self.assertRaisesRegex(ValueError, "is out of bounds"):
            op.infer_layout(layouts, extra_args=extra_args)

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

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = (-3,)

        with self.assertRaisesRegex(ValueError, "is out of bounds"):
            op.infer_layout(layouts, extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_missing_dim_arg(self, mock_platform):
        """
        Feature: Extra args validation
        Description: extra_args tuple is empty (missing dim).
        Expectation: ValueError raised.
        """
        mesh = self._create_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(),), 1)

        layouts = (input_layout, None, input_layout, input_layout)
        extra_args = ()

        with self.assertRaisesRegex(ValueError, "requires 'dim' parameter"):
            op.infer_layout(layouts, extra_args=extra_args)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_fail_no_input_layout(self, mock_platform):
        """
        Feature: Layouts validation
        Description: layouts tuple is None or empty.
        Expectation: ValueError raised for empty tuple; TypeError for None.
        """
        extra_args = (0,)

        with self.assertRaisesRegex(ValueError, "requires a valid input tensor layout"):
            op.infer_layout((), extra_args=extra_args)

        with self.assertRaises(TypeError):
            op.infer_layout(None, extra_args=extra_args)

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

        layouts = (input_layout, None, None, input_layout)
        extra_args = (1,)

        output_layout = op.infer_layout(layouts, extra_args=extra_args)
        assert output_layout.tensor_map == input_layout.tensor_map

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

        layouts = (input_layout, None, input_layout, src_layout)
        extra_args = (0,)

        with self.assertRaisesRegex(ValueError, "Src tensor layout .* must match input tensor layout"):
            op.infer_layout(layouts, extra_args=extra_args)


if __name__ == "__main__":
    unittest.main()
