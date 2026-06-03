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
"""parallel_conv3d test"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_conv3d import Conv3dDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

# Initialize the operator
op = Conv3dDistributedOp("conv3d")
# Default extra_args for functional.conv3d: (stride, padding, dilation, groups)
default_extra_args = (1, 0, 1, 1)


class TestParallelConv3D(unittest.TestCase):
    """Unit tests for Conv3dDistributedOp."""

    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _cache_values(self, layouts, extra_args=default_extra_args):
        """Build Conv3d cache_values for the new infer_layout signature."""
        input_layout = layouts[0] if len(layouts) > 0 else None
        weight_layout = layouts[1] if len(layouts) > 1 else None
        bias_layout = layouts[2] if len(layouts) > 2 else None
        stride, padding, dilation, groups = extra_args
        return [input_layout, weight_layout, bias_layout, stride, padding, dilation, groups]

    def _infer_output_layout(self, layouts, extra_args=default_extra_args):
        """Infer Conv3d output layout and unwrap the single-output result."""
        output_layouts, _ = op.infer_layout(self._cache_values(layouts, extra_args))
        return output_layouts[0]

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

    def _make_2x4_mesh(self, mock_platform, mesh_dim_names=("dp", "sp")):
        """Set up mock and return a standard 2x4 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=mesh_dim_names)

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "sp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    def _make_1d_mesh(self, mock_platform, world_size=8, mesh_dim_names=("dp",)):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_preprocess_returns_local_args_and_cache_values(self, mock_platform):
        """
        Feature: New dispatch preprocessing
        Description: Conv3d preprocess converts DTensors to local tensors and builds cache_values.
        Expectation: MindSpore-compatible positional args are returned and cache_values carries layouts.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=4, mesh_dim_names=("dp",))
        in_layout = _build_layout(mesh, (Replicate(),), 5)
        w_layout = _build_layout(mesh, (Replicate(),), 5)
        b_layout = _build_layout(mesh, (Replicate(),), 1)

        input_tensor = MagicMock()
        input_tensor.layout = in_layout
        input_tensor.to_local.return_value = "local_input"
        weight = MagicMock()
        weight.layout = w_layout
        weight.to_local.return_value = "local_weight"
        bias = MagicMock()
        bias.layout = b_layout
        bias.to_local.return_value = "local_bias"

        local_args, local_kwargs, cache_values = op.preprocess(
            (input_tensor, weight),
            {"bias": bias, "stride": 2, "padding": 1, "dilation": 1, "groups": 1},
        )

        self.assertEqual(local_args, ("local_input", "local_weight", "local_bias", 2, 1, 1, 1))
        self.assertEqual(local_kwargs, {})
        self.assertEqual(cache_values, [in_layout, w_layout, b_layout, 2, 1, 1, 1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_data_and_spatial_parallelism(self, mock_platform):
        """
        Feature: Data Parallelism and Spatial Parallelism
        Description: Input is sharded on batch size (N) and depth (D), weight is replicated.
        Expectation: Output layout preserves sharding on N and D, and C_out is unsharded.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "sp"))

        # Input: (N, C_in, D, H, W) -> Sharded on N via "dp" and on D via "sp"
        in_placements = (Shard(0), Shard(2))
        in_layout = _build_layout(mesh, in_placements, 5)  # tensor_map: (1, -1, 0, -1, -1)

        # Weight: (C_out, C_in/groups, kD, kH, kW) -> Replicated
        w_placements = (Replicate(), Replicate())
        w_layout = _build_layout(mesh, w_placements, 5)    # tensor_map: (-1, -1, -1, -1, -1)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: (N, C_out, D_out, H_out, W_out)
        # N->dp(1), C_out->unsharded(-1), D->sp(0), H->-1, W->-1
        expected_map = (1, -1, 0, -1, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"DP + SP failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_column_parallelism(self, mock_platform):
        """
        Feature: Tensor Column Parallelism (sharded on output channel C_out)
        Description: Weight and bias are sharded on C_out (dim 0), input is replicated (or DP).
        Expectation: Output layout is sharded on C_out (dim 1) accordingly.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        # Input: Sharded on N via "dp", Replicated on "mp"
        in_placements = (Shard(0), Replicate())
        in_layout = _build_layout(mesh, in_placements, 5)  # tensor_map: (1, -1, -1, -1, -1)

        # Weight: Replicated on "dp", Sharded on C_out (dim 0) via "mp"
        w_placements = (Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 5)    # tensor_map: (-1, 0, -1, -1, -1)

        # Bias: Replicated on "dp", Sharded on C_out (dim 0) via "mp"
        b_placements = (Replicate(), Shard(0))
        b_layout = _build_layout(mesh, b_placements, 1)    # tensor_map: (0,)

        output_layout = self._infer_output_layout((in_layout, w_layout, b_layout), default_extra_args)

        # Output: (N, C_out, D_out, H_out, W_out)
        # N->dp(1), C_out->mp(0), D->-1, H->-1, W->-1
        expected_map = (1, 0, -1, -1, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"Column Parallelism failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_row_parallelism_partial_output(self, mock_platform):
        """
        Feature: Tensor Row Parallelism (sharded on input channel C_in)
        Description: Both input and weight are sharded on C_in (dim 1).
        Expectation: Output layout has partial sum status along the parallel mesh axis.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        # Input: Replicated on "dp", Sharded on C_in (dim 1) via "mp"
        in_placements = (Replicate(), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight: Replicated on "dp", Sharded on C_in (dim 1) via "mp"
        w_placements = (Replicate(), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output tensor map itself should be fully unsharded, but wrapped with Partial status on "mp"
        expected_map = (-1, -1, -1, -1, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            "Row parallelism tensor_map should be fully unsharded initially"
        )

        # 'dp' is mesh axis 0, 'mp' is mesh axis 1. We expect 'sum' reduction on 'mp'.
        self.assertEqual(
            output_layout.partial, [None, "sum"],
            f"Row parallelism failed to set partial status. Got {output_layout.partial}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_invalid_row_parallelism_axis_mismatch(self, mock_platform):
        """
        Feature: Invalid Row Parallelism validation
        Description: Input and weight are sharded on C_in, but mapped to different mesh dimensions.
        Expectation: ValueError is raised to prevent invalid mathematical reduction.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        # Input C_in sharded on "dp"
        in_placements = (Shard(1), Replicate())
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight C_in sharded on "mp"
        w_placements = (Replicate(), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 5)

        with self.assertRaisesRegex(ValueError, "Input C_in and Weight C_in must be sharded on the same axis."):
            op.infer_layout(self._cache_values((in_layout, w_layout), default_extra_args))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_invalid_column_parallelism_bias_mismatch(self, mock_platform):
        """
        Feature: Invalid Column Parallelism validation
        Description: Weight is sharded on C_out, but bias does not share the same sharding pattern.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        in_placements = (Replicate(), Replicate())
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight C_out sharded on "mp"
        w_placements = (Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 5)

        # Bias fully replicated
        b_placements = (Replicate(), Replicate())
        b_layout = _build_layout(mesh, b_placements, 1)

        with self.assertRaisesRegex(ValueError, "Weight C_out and Bias C_out must be sharded on the same axis."):
            op.infer_layout(self._cache_values((in_layout, w_layout, b_layout), default_extra_args))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_invalid_groups_with_sharded_c_in(self, mock_platform):
        """
        Feature: Grouped convolution constraint
        Description: Attempt to do Row Parallelism (sharded C_in) when groups > 1.
        Expectation: ValueError is raised since native group-wise scatter/gather is not strictly supported yet.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        in_placements = (Replicate(), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 5)

        w_placements = (Replicate(), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 5)

        # Set groups to 2
        extra_args_with_groups = (1, 0, 1, 2)

        with self.assertRaisesRegex(ValueError, "Sharding on C_in with groups > 1 is not supported."):
            op.infer_layout(self._cache_values((in_layout, w_layout), extra_args_with_groups))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_invalid_dimension_count(self, mock_platform):
        """
        Feature: Dimension validation
        Description: Pass inputs that are not exactly 5D (e.g., 4D tensor).
        Expectation: ValueError is raised enforcing 5D semantics for conv3d.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        # 4D input
        in_placements = (Replicate(), Replicate())
        in_layout = _build_layout(mesh, in_placements, 4)

        # 5D weight
        w_placements = (Replicate(), Replicate())
        w_layout = _build_layout(mesh, w_placements, 5)

        with self.assertRaisesRegex(ValueError, "Input and weight must be 5D."):
            op.infer_layout(self._cache_values((in_layout, w_layout), default_extra_args))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_missing_required_layouts(self, mock_platform):
        """
        Feature: Input arguments validation
        Description: Pass insufficient layout arguments (only one layout provided).
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "mp"))

        in_placements = (Replicate(), Replicate())
        in_layout = _build_layout(mesh, in_placements, 5)

        with self.assertRaisesRegex(ValueError, "Requires at least input and weight layouts."):
            op.infer_layout(self._cache_values((in_layout,), default_extra_args))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_pure_data_parallelism(self, mock_platform):
        """
        Feature: Pure Data Parallelism
        Description: Input is sharded on batch size (N), weight and bias are replicated.
        Expectation: Output layout preserves sharding on N, everything else is unsharded.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_dim_names=("dp",))

        # Input: Sharded on N (dim 0)
        in_placements = (Shard(0),)
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight and Bias: Replicated
        w_placements = (Replicate(),)
        w_layout = _build_layout(mesh, w_placements, 5)

        b_placements = (Replicate(),)
        b_layout = _build_layout(mesh, b_placements, 1)

        output_layout = self._infer_output_layout((in_layout, w_layout, b_layout), default_extra_args)

        # Output: (N, C_out, D, H, W) -> N sharded on "dp" (mesh dim 0)
        expected_map = (0, -1, -1, -1, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"Pure DP failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_spatial_parallelism_hw(self, mock_platform):
        """
        Feature: Spatial Parallelism on Height and Width
        Description: Input is sharded on H (dim 3) and W (dim 4), weight is fully replicated.
        Expectation: Output layout preserves sharding on H and W dimensions.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("sp_h", "sp_w"))

        # Input: Sharded on H (dim 3) via "sp_h" and W (dim 4) via "sp_w"
        in_placements = (Shard(3), Shard(4))
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight: Replicated
        w_placements = (Replicate(), Replicate())
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: (N, C_out, D, H, W) -> H sharded on sp_h(1), W sharded on sp_w(0)
        expected_map = (-1, -1, -1, 1, 0)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"SP on H and W failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_column_parallelism_no_bias(self, mock_platform):
        """
        Feature: Column Parallelism without Bias
        Description: Weight is sharded on C_out (dim 0), but no bias tensor is provided.
        Expectation: Operator infers layout correctly without throwing bias mismatch errors.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=4, mesh_dim_names=("tp",))

        in_placements = (Replicate(),)
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight: Sharded on C_out (dim 0)
        w_placements = (Shard(0),)
        w_layout = _build_layout(mesh, w_placements, 5)

        # Missing bias layout (only 2 items in tuple)
        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: C_out sharded on tp(0)
        expected_map = (-1, 0, -1, -1, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"Column Parallelism without bias failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_grouped_column_parallelism(self, mock_platform):
        """
        Feature: Grouped Convolution with Column Parallelism
        Description: groups > 1, Weight sharded on C_out (dim 0).
        Expectation: Passes successfully since Column Parallelism supports groups > 1 natively.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))

        in_placements = (Replicate(),)
        in_layout = _build_layout(mesh, in_placements, 5)

        w_placements = (Shard(0),)
        w_layout = _build_layout(mesh, w_placements, 5)

        # extra_args: (stride, padding, dilation, groups=4)
        extra_args_with_groups = (1, 0, 1, 4)

        output_layout = self._infer_output_layout((in_layout, w_layout), extra_args_with_groups)

        # Output: C_out sharded on tp(0)
        expected_map = (-1, 0, -1, -1, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"Grouped Column Parallelism failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_hybrid_dp_tp_sp(self, mock_platform):
        """
        Feature: 3D Hybrid Parallelism (DP + TP + SP)
        Description: 3D mesh. Input sharded on N (dp) and H (sp). Weight sharded on C_out (tp).
        Expectation: Output correctly combines all three sharding strategies.
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("dp", "tp", "sp"))

        # Input: DP on dim 0, SP on dim 3
        in_placements = (Shard(0), Replicate(), Shard(3))
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight: TP on dim 0
        w_placements = (Replicate(), Shard(0), Replicate())
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: N on dp(2), C_out on tp(1), H on sp(0)
        expected_map = (2, 1, -1, 0, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"3D Hybrid Parallelism failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_rejects_partial_inputs(self, mock_platform):
        """
        Feature: Rejection of Partial Inputs
        Description: Input tensor already has a 'Partial' state from a previous operation.
        Expectation: ValueError is raised because conv3d doesn't support computing directly on partial sums.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=2, mesh_dim_names=("tp",))

        in_placements = (Replicate(),)
        in_layout = _build_layout(mesh, in_placements, 5)
        # Artificially set a partial status
        in_layout.set_partial_by_dev_axis("tp", "sum")

        w_placements = (Replicate(),)
        w_layout = _build_layout(mesh, w_placements, 5)

        with self.assertRaisesRegex(ValueError, "Partial status which is not allowed"):
            op.infer_layout(self._cache_values((in_layout, w_layout), default_extra_args))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_fully_replicated(self, mock_platform):
        """
        Feature: Fully Replicated Execution
        Description: All inputs and weights are fully replicated.
        Expectation: Output layout is completely unsharded (all -1).
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=4, mesh_dim_names=("dp",))

        in_placements = (Replicate(),)
        in_layout = _build_layout(mesh, in_placements, 5)

        w_placements = (Replicate(),)
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        expected_map = (-1, -1, -1, -1, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"Fully replicated failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )


    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_spatial_parallelism_depth(self, mock_platform):
        """
        Feature: Spatial Parallelism on Depth
        Description: Input is sharded purely on the Depth (D) dimension.
        Expectation: Output layout preserves sharding on the D dimension.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=4, mesh_dim_names=("sp_d",))

        # Input: Sharded on D (dim 2)
        in_placements = (Shard(2),)
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight: Replicated
        w_placements = (Replicate(),)
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: D sharded on sp_d(0)
        expected_map = (-1, -1, 0, -1, -1)
        self.assertEqual(
            output_layout.to_dict()["tensor_map"], expected_map,
            f"SP on Depth failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_spatial_parallelism_width(self, mock_platform):
        """
        Feature: Spatial Parallelism on Width
        Description: Input is sharded purely on the Width (W) dimension.
        Expectation: Output layout preserves sharding on the W dimension.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=4, mesh_dim_names=("sp_w",))

        in_placements = (Shard(4),)
        in_layout = _build_layout(mesh, in_placements, 5)

        w_placements = (Replicate(),)
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        expected_map = (-1, -1, -1, -1, 0)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_spatial_parallelism_d_h_w(self, mock_platform):
        """
        Feature: 3D Spatial Parallelism
        Description: Input is sharded on Depth (D), Height (H), and Width (W) over a 3D Mesh.
        Expectation: Output layout mirrors the input's spatial sharding exactly.
        """
        mesh = self._make_2x2x2_mesh(mock_platform, mesh_dim_names=("sp_d", "sp_h", "sp_w"))

        # Input: Sharded on D(2), H(3), W(4)
        in_placements = (Shard(2), Shard(3), Shard(4))
        in_layout = _build_layout(mesh, in_placements, 5)

        w_placements = (Replicate(), Replicate(), Replicate())
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: D->sp_d(2), H->sp_h(1), W->sp_w(0)
        expected_map = (-1, -1, 2, 1, 0)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_hybrid_dp_cp(self, mock_platform):
        """
        Feature: Hybrid Data Parallelism and Column Parallelism
        Description: Input sharded on N, Weight sharded on C_out.
        Expectation: Output is sharded on both N and C_out.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "tp"))

        # Input: DP shards N (dim 0)
        in_placements = (Shard(0), Replicate())
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight: TP shards C_out (dim 0)
        w_placements = (Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: N->dp(1), C_out->tp(0)
        expected_map = (1, 0, -1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_hybrid_dp_rp(self, mock_platform):
        """
        Feature: Hybrid Data Parallelism and Row Parallelism
        Description: Input sharded on N and C_in. Weight sharded on C_in.
        Expectation: Output is sharded on N and has a partial sum status on the TP dimension.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "tp"))

        # Input: shards N on "dp", shards C_in on "tp"
        in_placements = (Shard(0), Shard(1))
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight: shards C_in on "tp"
        w_placements = (Replicate(), Shard(1))
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: N->dp(1), rest unsharded but partial on "tp"
        expected_map = (1, -1, -1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)
        self.assertEqual(output_layout.partial, [None, "sum"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_row_parallelism_with_bias(self, mock_platform):
        """
        Feature: Row Parallelism with Replicated Bias
        Description: Both input and weight are sharded on C_in. Bias is replicated.
        Expectation: Valid layout generation. Bias does not affect the partial sum state of the output.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=4, mesh_dim_names=("tp",))

        in_placements = (Shard(1),)
        in_layout = _build_layout(mesh, in_placements, 5)

        w_placements = (Shard(1),)
        w_layout = _build_layout(mesh, w_placements, 5)

        b_placements = (Replicate(),)
        b_layout = _build_layout(mesh, b_placements, 1)

        output_layout = self._infer_output_layout((in_layout, w_layout, b_layout), default_extra_args)

        expected_map = (-1, -1, -1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)
        self.assertEqual(output_layout.partial, ["sum"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_column_parallelism_1d_mesh(self, mock_platform):
        """
        Feature: Column Parallelism on a 1D Mesh
        Description: 1D mesh configuration where Weight is sharded on C_out.
        Expectation: Output layout is correctly sharded on C_out.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8, mesh_dim_names=("tp",))

        in_placements = (Replicate(),)
        in_layout = _build_layout(mesh, in_placements, 5)

        w_placements = (Shard(0),)
        w_layout = _build_layout(mesh, w_placements, 5)

        output_layout = self._infer_output_layout((in_layout, w_layout), default_extra_args)

        # Output: C_out sharded on tp(0)
        expected_map = (-1, 0, -1, -1, -1)
        self.assertEqual(output_layout.to_dict()["tensor_map"], expected_map)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_conv3d_invalid_bias_sharding(self, mock_platform):
        """
        Feature: Invalid Bias Sharding Validation
        Description: Bias is sharded on a different mesh dimension than Weight's C_out.
        Expectation: ValueError is raised protecting against mathematical shape mismatch.
        """
        mesh = self._make_2x4_mesh(mock_platform, mesh_dim_names=("dp", "tp"))

        in_placements = (Shard(0), Replicate())
        in_layout = _build_layout(mesh, in_placements, 5)

        # Weight sharded on C_out via "tp"
        w_placements = (Replicate(), Shard(0))
        w_layout = _build_layout(mesh, w_placements, 5)

        # Bias mistakenly sharded via "dp"
        b_placements = (Shard(0), Replicate())
        b_layout = _build_layout(mesh, b_placements, 1)

        with self.assertRaisesRegex(ValueError, "Weight C_out and Bias C_out must be sharded on the same axis"):
            op.infer_layout(self._cache_values((in_layout, w_layout, b_layout), default_extra_args))


if __name__ == "__main__":
    unittest.main()
