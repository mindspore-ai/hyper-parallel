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
"""Unit tests for SwiGLU distributed operator."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_swiglu import SwiGLUDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelSwiGLU(unittest.TestCase):
    """Unit tests for SwiGLUDistributedOp."""

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

    def _infer_single_layout(self, op, cache_values):
        """Infer and unpack a single output layout with the new cache_values API."""
        output_layouts, _ = op.infer_layout(cache_values)
        return output_layouts[0]

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
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False
        )

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "mp"),
            init_backend=False
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_data_parallel_success(self, mock_platform):
        """
        Feature: SwiGLU data parallel
        Description: Data parallel scenario with SwiGLU on unsharded axis
        Expectation: Success, output layout equals input layout
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 1, (4, 8)]
        output_layout = self._infer_single_layout(op, cache_values)

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Data parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test methods, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, ((output_layout,), None), cache_values) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, ((output_layout,), None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_model_parallel_success(self, mock_platform):
        """
        Feature: SwiGLU model parallel
        Description: Model parallel scenario with SwiGLU on unsharded batch dimension
        Expectation: Success, output layout equals input layout
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0, (4, 8)]
        output_layout = self._infer_single_layout(op, cache_values)

        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Model parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_hybrid_parallel_success(self, mock_platform):
        """
        Feature: SwiGLU hybrid parallel
        Description: Hybrid parallel scenario with SwiGLU on unsharded middle dimension
        Expectation: Success, output layout equals input layout
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 1, (4, 6, 8)]
        output_layout = self._infer_single_layout(op, cache_values)

        expected_map = (2, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Hybrid parallel test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_all_replicated(self, mock_platform):
        """
        Feature: SwiGLU all replicated
        Description: All dimensions replicated scenario
        Expectation: Success, output layout equals input layout
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0, (4, 8)]
        output_layout = self._infer_single_layout(op, cache_values)

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"All replicated test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_negative_dim(self, mock_platform):
        """
        Feature: SwiGLU negative dimension index
        Description: Test negative dimension index (dim=-1)
        Expectation: Success, output layout equals input layout
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, -1, (4, 8)]
        output_layout = self._infer_single_layout(op, cache_values)

        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Negative dim test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_split_axis_sharded_failure(self, mock_platform):
        """
        Feature: SwiGLU split axis sharded
        Description: Attempting SwiGLU on a sharded split axis
        Expectation: Raise ValueError with SwiGLU-specific error message
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "split axis"):
            op.infer_layout([x_layout, 0, (4, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_model_parallel_on_split_axis_failure(self, mock_platform):
        """
        Feature: SwiGLU model parallel on split axis
        Description: Model parallel scenario where the split axis is sharded
        Expectation: Raise ValueError
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "split axis"):
            op.infer_layout([x_layout, -1, (4, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_split_axis_odd_size_failure(self, mock_platform):
        """
        Feature: SwiGLU split axis odd size
        Description: SwiGLU split axis length is odd, cannot be halved
        Expectation: Raise ValueError
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "even size"):
            op.infer_layout([x_layout, 1, (4, 7)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_invalid_axis_type(self, mock_platform):
        """
        Feature: SwiGLU invalid axis type
        Description: Pass invalid type as axis argument
        Expectation: Raise ValueError
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "should be int"):
            op.infer_layout([x_layout, "invalid", (4, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_axis_out_of_range_failure(self, mock_platform):
        """
        Feature: SwiGLU axis out of range
        Description: Pass an axis index beyond the tensor's number of dimensions
        Expectation: Raise ValueError
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        with self.assertRaisesRegex(ValueError, "axis out of range"):
            op.infer_layout([x_layout, 3, (4, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_partial_input(self, mock_platform):
        """
        Feature: SwiGLU with partial input
        Description: Input with partial state
        Expectation: Raise ValueError since _allow_partial_inputs is False
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            op.infer_layout([x_layout, 1, (4, 8)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_preprocess(self, mock_platform):
        """
        Feature: SwiGLU preprocess
        Description: Convert DTensor input to local tensor and build cache_values
        Expectation: All runtime arguments are positional and cache contains layout, axis, and shape
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.shape = (4, 8)
        mock_local = MagicMock()
        mock_tensor.to_local.return_value = mock_local

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor,), {"dim": 1})

        assert local_args == (mock_local, 1), (
            f"Expected local_args=(mock_local, 1), got {local_args}"
        )
        assert not local_kwargs, (
            f"Expected empty local_kwargs, got {local_kwargs}"
        )
        assert cache_values == [x_layout, 1, (4, 8)], (
            f"Expected cache_values=[layout, 1, (4, 8)], got {cache_values}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_swiglu_3d_tensor(self, mock_platform):
        """
        Feature: SwiGLU on 3D tensor
        Description: Test SwiGLU on 3D tensor with mixed placements
        Expectation: Success
        """
        op = SwiGLUDistributedOp("Swiglu")
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, 2, (4, 6, 8)]
        output_layout = self._infer_single_layout(op, cache_values)

        expected_map = (2, 1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"3D tensor test failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )


if __name__ == "__main__":
    unittest.main()
