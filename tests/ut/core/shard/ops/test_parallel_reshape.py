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
"""parallel_reshape test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_reshape import ReshapeDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ReshapeDistributedOp("Reshape")
op_torch = ReshapeDistributedOp("reshape")
op_view = ReshapeDistributedOp("view")


class TestParallelReshape(unittest.TestCase):
    """Unit tests for ReshapeDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, mp, cp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "mp", "cp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_not_change_sharded_axis(self, mock_platform):
        """
        Feature: Reshape do not change sharded axis
        Description: Reshape do not change sharded axis
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        src_shape = (1024, 512, 512)
        dst_shape = (1024, 2, 256, 512)

        output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
        expected_map = (1, -1, -1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Reshape do not change sharded axis failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        expected_local_dst_shape = [512, 2, 256, 512]
        assert local_dst_shape == expected_local_dst_shape, (
            f"Reshape do not change sharded axis failed. Expected {expected_local_dst_shape}, "
            f"got {local_dst_shape}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_merge_sharded_axis(self, mock_platform):
        """
        Feature: Reshape merge shared axis with not shared axis
        Description: Reshape merge shared axis with not shared axis
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        src_shape = (4, 4, 8)
        dst_shape = (16, 8)

        output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Reshape do not change sharded axis failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        expected_local_dst_shape = [8, 8]
        assert local_dst_shape == expected_local_dst_shape, (
            f"Reshape do not change sharded axis failed. Expected {expected_local_dst_shape}, "
            f"got {local_dst_shape}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_split_sharded_axis(self, mock_platform):
        """
        Feature: Reshape split shared axis
        Description: Reshape do not change sharded axis
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        src_shape = (32, 128)
        dst_shape = (4, 8, 128)

        output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
        expected_map = (1, -1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Reshape do not change sharded axis failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        expected_local_dst_shape = [2, 8, 128]
        assert local_dst_shape == expected_local_dst_shape, (
            f"Reshape do not change sharded axis failed. Expected {expected_local_dst_shape}, "
            f"got {local_dst_shape}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_multi_axes_shared(self, mock_platform):
        """
        Feature: Reshape split, merge, resize axes
        Description: Reshape split, merge, resize axes
        Expectation: Success
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(1), Shard(3), Shard(0), Replicate(), Replicate()), 5)
        src_shape = (32, 6, 128, 28, 10)
        dst_shape = (4, 8, 2, 384, 280)

        output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
        expected_map = (0, -1, 2, -1, 1)
        assert output_layout.tensor_map == expected_map, (
            f"Reshape do not change sharded axis failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        expected_local_dst_shape = [2, 8, 1, 384, 140]
        assert local_dst_shape == expected_local_dst_shape, (
            f"Reshape do not change sharded axis failed. Expected {expected_local_dst_shape}, "
            f"got {local_dst_shape}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_can_not_reshape1(self, mock_platform):
        """
        Feature: Reshape can not be shared
        Description: Can not be reshaped
        Expectation: Fail
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(3), Replicate(), Replicate()), 4)
        src_shape = (4, 8, 4, 12)
        dst_shape = (4, 8, 12, 4)

        with self.assertRaises(ValueError):
            _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_can_not_reshape2(self, mock_platform):
        """
        Feature: Reshape can not be shared
        Description: Can not be reshaped
        Expectation: Fail
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(2), Replicate(), Replicate()), 4)
        src_shape = (4, 8, 12, 7)
        dst_shape = (4, 8, 2, 42)

        with self.assertRaises(ValueError):
            _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_dynamic_shape1(self, mock_platform):
        """
        Feature: Reshape parallel op with dynamic shape
        Description: Reshape parallel op with dynamic shape
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate(), Replicate()), 4)
        src_shape = (1024, -1, 256, 512)
        dst_shape = (1024, -1, 512)

        output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
        expected_map = (1, -1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Reshape do not change sharded axis failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        expected_local_dst_shape = [512, -1, 512]
        assert local_dst_shape == expected_local_dst_shape, (
            f"Reshape do not change sharded axis failed. Expected {expected_local_dst_shape}, "
            f"got {local_dst_shape}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_dynamic_shape2(self, mock_platform):
        """
        Feature: Reshape parallel op with dynamic shape
        Description: Reshape parallel op with dynamic shape
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        src_shape = (1024, -1, 512)
        dst_shape = (1024, -1, 256, 512)

        output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
        expected_map = (1, -1, -1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Reshape do not change sharded axis failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        expected_local_dst_shape = [512, -1, 256, 512]
        assert local_dst_shape == expected_local_dst_shape, (
            f"Reshape do not change sharded axis failed. Expected {expected_local_dst_shape}, "
            f"got {local_dst_shape}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_dynamic_shape3(self, mock_platform):
        """
        Feature: Reshape parallel op with dynamic shape
        Description: Reshape parallel op with dynamic shape
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        src_shape = (-1, 256, 512)
        dst_shape = (-1, 512)

        output_layout, local_dst_shape = op.infer_layout((x_layout,), (dst_shape, src_shape))
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Reshape do not change sharded axis failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )
        expected_local_dst_shape = [-1, 512]
        assert local_dst_shape == expected_local_dst_shape, (
            f"Reshape do not change sharded axis failed. Expected {expected_local_dst_shape}, "
            f"got {local_dst_shape}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_layout_dynamic_shape4(self, mock_platform):
        """
        Feature: Reshape parallel op with dynamic shape
        Description: Reshape parallel op with dynamic shape
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        src_shape = (-1, 512)
        dst_shape = (-1, 256, 512)

        with self.assertRaises(ValueError):
            _, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_reshape_basic_split(self, mock_platform):
        """
        Feature: PyTorch style reshape split sharded axis
        Description: Test splitting a sharded dimension into multiple dimensions (preserving layout)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        src_shape = (32, 64)
        dst_shape = (32, 16, 4)

        output_layout, local_dst_shape = op_torch.infer_layout((x_layout,), (dst_shape, src_shape))

        expected_map = (1, 0, -1)

        assert output_layout.tensor_map == expected_map, (
            f"Expected {expected_map}, got {output_layout.tensor_map}"
        )

        expected_local_dst_shape = [16, 4, 4]
        assert local_dst_shape == expected_local_dst_shape, (
            f"Expected local shape {expected_local_dst_shape}, got {local_dst_shape}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_reshape_dynamic_target(self, mock_platform):
        """
        Feature: PyTorch style reshape with -1 in target
        Description: Test passing -1 in destination shape to infer dimension automatically
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        src_shape = (32, 64)
        dst_shape = (32, -1)

        output_layout, local_dst_shape = op_torch.infer_layout((x_layout,), (dst_shape, src_shape))

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map

        expected_local_dst_shape = [16, 16]
        assert local_dst_shape == expected_local_dst_shape

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_reshape_flatten_unsharded(self, mock_platform):
        """
        Feature: PyTorch style reshape merge unsharded dims
        Description: Merging an unsharded dimension into a sharded one is NOT fully supported if it changes sharding cuts,
                     but merging unsharded dims into other unsharded dims or preserving blocks works.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(2), Replicate()), 3)

        src_shape = (16, 4, 8)
        dst_shape = (64, 8)

        output_layout, _ = op_torch.infer_layout((x_layout,), (dst_shape, src_shape))
        assert output_layout.tensor_map == (1, 0)

        dst_shape_2 = (16, 2, 2, 8)
        output_layout_2, _ = op_torch.infer_layout((x_layout,), (dst_shape_2, src_shape))
        assert output_layout_2.tensor_map == (1, -1, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_reshape_fail_missing_input_shape(self, mock_platform):
        """
        Feature: PyTorch style reshape exception
        Description: Verify that failing to provide input shape raises ValueError
        Expectation: ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        dst_shape = (32, 64)

        with self.assertRaisesRegex(ValueError, "reshape requires output shape and input shape."):
            op_torch.infer_layout((x_layout,), (dst_shape,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_reshape_fail_mismatch_total_size(self, mock_platform):
        """
        Feature: PyTorch style reshape validation
        Description: Verify shape element count mismatch validation
        Expectation: ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        src_shape = (10, 10)
        dst_shape = (20, 20)

        with self.assertRaises(ValueError):
            op_torch.infer_layout((x_layout,), (dst_shape, src_shape))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_view_layout_flatten_contiguous(self, mock_platform):
        """
        Feature: View operator flattening
        Description: Test flattening multiple dimensions where the inner dimension is not sharded.
                     Common case: x.view(batch_size, -1)
        Expectation: Success, preserving the sharding of the split point.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        src_shape = (32, 16, 8)
        dst_shape = (32, 128)

        output_layout, local_dst_shape = op_view.infer_layout((x_layout,), (dst_shape, src_shape))

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"View flatten failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

        expected_local_dst_shape = [16, 32]
        assert local_dst_shape == expected_local_dst_shape

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_view_layout_unflatten_split(self, mock_platform):
        """
        Feature: View operator unflattening (inverse of flatten)
        Description: Test expanding a sharded dimension into two, where the outer part keeps sharding
                     and the inner part becomes None (unsharded).
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        src_shape = (32, 128)
        dst_shape = (32, 16, 8)

        output_layout, local_dst_shape = op_view.infer_layout((x_layout,), (dst_shape, src_shape))

        expected_map = (1, 0, -1)
        assert output_layout.tensor_map == expected_map, (
            f"View unflatten failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

        expected_local_dst_shape = [16, 4, 8]
        assert local_dst_shape == expected_local_dst_shape

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_view_layout_dynamic_shape_inference(self, mock_platform):
        """
        Feature: View operator with -1 inference
        Description: Support x.view(-1, C) style calls similar to PyTorch
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        src_shape = (8, 4, 16)
        dst_shape = (8, -1)

        output_layout, local_dst_shape = op_view.infer_layout((x_layout,), (dst_shape, src_shape))

        expected_map = (1, 0)
        assert output_layout.tensor_map == expected_map

        expected_local_dst_shape = [4, 16]
        assert local_dst_shape == expected_local_dst_shape

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_view_layout_fail_shape_mismatch(self, mock_platform):
        """
        Feature: View operator validation
        Description: Ensure view raises error if total elements don't match (PyTorch behavior)
        Expectation: ValueError
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        src_shape = (10, 20)
        dst_shape = (10, 30)

        with self.assertRaisesRegex(ValueError, "total elements number"):
            op_view.infer_layout((x_layout,), (dst_shape, src_shape))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_with_partial_basic(self, mock_platform):
        """
        Feature: Reshape with partial input
        Description: Reshape should preserve partial status from input to output
        Expectation: Output layout has same partial status as input
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Partial(), Replicate()), 2)

        assert x_layout.is_partial(), "Input layout should have partial status"

        src_shape = (8, 16)
        dst_shape = (4, 4, 8)

        output_layout, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))

        assert output_layout.is_partial(), "Output layout should preserve partial status"
        assert output_layout.get_partial_by_dev_id("dp") == "sum", (
            f"Partial op should be 'sum', got {output_layout.get_partial_by_dev_id('dp')}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_with_partial_sharded_input(self, mock_platform):
        """
        Feature: Reshape with partial and sharded input
        Description: Reshape should preserve both sharding and partial status on different axes
        Expectation: Output has correct sharding and partial
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Partial()), 2)

        src_shape = (8, 16)
        dst_shape = (1, -1, 16)

        output_layout, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))

        assert output_layout.is_partial(), "Output layout should preserve partial status"
        assert output_layout.get_partial_by_dev_id("mp") == "sum", (
            f"Partial op should be 'sum', got {output_layout.get_partial_by_dev_id('mp')}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_with_partial_multi_axis(self, mock_platform):
        """
        Feature: Reshape with partial on multiple device axes
        Description: Reshape should preserve partial status on all axes
        Expectation: Output has partial on both device axes
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Partial(), Partial(), Replicate()), 2)

        src_shape = (4, 8)
        dst_shape = (2, 16)

        output_layout, _ = op.infer_layout((x_layout,), (dst_shape, src_shape))

        assert output_layout.is_partial(), "Output layout should preserve partial status"
        assert output_layout.get_partial_by_dev_id("dp") == "sum"
        assert output_layout.get_partial_by_dev_id("mp") == "sum"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_reshape_torch_with_partial(self, mock_platform):
        """
        Feature: torch.reshape with partial input
        Description: torch.reshape should preserve partial status
        Expectation: Output layout has partial status
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        src_shape = (4, 8)
        dst_shape = (2, 16)

        output_layout, _ = op_torch.infer_layout(
            (x_layout,), (dst_shape, src_shape)
        )

        assert output_layout.is_partial(), "Output layout should preserve partial status"
        assert output_layout.get_partial_by_dev_id("dp") == "sum"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_view_with_partial(self, mock_platform):
        """
        Feature: view with partial input
        Description: view should preserve partial status
        Expectation: Output layout has partial status
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        src_shape = (4, 8)
        dst_shape = (2, 16)

        output_layout, _ = op_view.infer_layout(
            (x_layout,), (dst_shape, src_shape)
        )

        assert output_layout.is_partial(), "Output layout should preserve partial status"
        assert output_layout.get_partial_by_dev_id("dp") == "sum"


if __name__ == "__main__":
    unittest.main()
