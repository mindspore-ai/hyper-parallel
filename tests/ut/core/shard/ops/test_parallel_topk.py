# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""parallel_topk test"""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_topk import TopKDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = TopKDistributedOp("TopK")
torch_op = TopKDistributedOp("topk")


class TestParallelTopK(unittest.TestCase):
    """Unit tests for TopKDistributedOp."""

    def setUp(self) -> None:
        """Clear global caches before each test to ensure isolation."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Restore global cache state after each test."""
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

    def _make_2x4x3_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4x3 (dp, mp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=24)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4, 3), mesh_dim_names=("dp", "mp", "tp"))

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2, 2),
                                mesh_dim_names=("dp", "tp", "mp"), init_backend=False)

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2 (dp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_layout_data_parallel(self, mock_platform):
        """
        Feature: TopK data parallel
        Description: Data parallel scenario (shard on first dim, topk on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert extra_info is None, (
            f"extra_info should be None, got {extra_info}"
        )
        values_layout, indices_layout = output_layouts
        expected_map = (1, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Data parallel test failed. Expected {expected_map}, "
            f"got values={values_layout.tensor_map}, indices={indices_layout.tensor_map}"
        )

        # TopKDistributedOp does not override get_expand_impl → always None.
        # Verified once here; other test cases omit this check as per testing conventions.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_layout_tensor_parallel(self, mock_platform):
        """
        Feature: TopK tensor parallel
        Description: Tensor parallel scenario (shard on first dim via mp, topk on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        cache_values = [x_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (0, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Tensor parallel test failed. Expected {expected_map}, "
            f"got values={values_layout.tensor_map}, indices={indices_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_layout_tensor_and_data_parallel(self, mock_platform):
        """
        Feature: TopK hybrid parallel
        Description: Hybrid data + tensor parallel scenario
        Expectation: Success
        """
        mesh = self._make_2x4x3_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        cache_values = [x_layout, None]
        output_layouts, extra_info = op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (2, 1, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Hybrid parallel test failed. Expected {expected_map}, "
            f"got values={values_layout.tensor_map}, indices={indices_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_partial_input_raises_error(self, mock_platform):
        """
        Feature: TopKDistributedOp rejects inputs with Partial status.
        Description: Input has Partial status set on dp axis.
        Expectation: ValueError is raised about Partial status not being allowed.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")
        cache_values = [x_layout, 1]
        with self.assertRaisesRegex(ValueError, "Partial status"):
            torch_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_topk_layout_data_parallel(self, mock_platform):
        """
        Feature: TopK data parallel (torch path)
        Description: Data parallel scenario with torch.topk op name
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        cache_values = [x_layout, None]
        output_layouts, extra_info = torch_op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (1, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Torch data parallel test failed. Expected {expected_map}, "
            f"got values={values_layout.tensor_map}, indices={indices_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_topk_layout_tensor_parallel(self, mock_platform):
        """
        Feature: TopK tensor parallel (torch path)
        Description: Tensor parallel scenario with torch.topk op name
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        cache_values = [x_layout, None]
        output_layouts, extra_info = torch_op.infer_layout(cache_values)

        values_layout, indices_layout = output_layouts
        expected_map = (0, -1)
        assert values_layout.tensor_map == indices_layout.tensor_map == expected_map, (
            f"Torch tensor parallel test failed. Expected {expected_map}, "
            f"got values={values_layout.tensor_map}, indices={indices_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_topk_layout_mixed_parallel_invalid(self, mock_platform):
        """
        Feature: Test topk on a sharded dimension
        Description: Test topk on a sharded dimension
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        cache_values = [x_layout, None]
        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            torch_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_topk_layout_error_dim_out_of_range(self, mock_platform):
        """
        Feature: Test indicating a invalid dim
        Description: Test indicating a invalid dim
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        cache_values = [x_layout, 5]
        with self.assertRaisesRegex(ValueError, "dimension out of range"):
            torch_op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_preprocess_torch(self, mock_platform):
        """
        Feature: TopKDistributedOp preprocess for PyTorch path.
        Description: PyTorch torch.topk has no keyword-only params; all params in local_args.
        Expectation: local_kwargs is empty; local_args has 5 elements (tensor, k, dim, largest, sorted).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = torch_op.preprocess((mock_tensor, 3), {})

        assert not local_kwargs, (
            f"For PyTorch 'topk', local_kwargs should be empty, got {local_kwargs}"
        )
        assert len(local_args) == 5, (
            f"For PyTorch 'topk', local_args should have 5 elements "
            f"(tensor, k, dim, largest, sorted), got {len(local_args)}"
        )
        assert local_args[1] == 3, (
            f"k should be 3, got {local_args[1]}"
        )
        assert local_args[2] == -1, (
            f"dim default should be -1, got {local_args[2]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_preprocess_mindspore(self, mock_platform):
        """
        Feature: TopKDistributedOp preprocess for MindSpore Primitive path.
        Description: MindSpore TopkExt Primitive takes all positional args.
        Expectation: local_kwargs is empty; local_args has 5 elements (tensor, k, dim, largest, sorted).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor, 5, 0), {})

        assert not local_kwargs, (
            f"For MindSpore 'TopkExt', local_kwargs should be empty, got {local_kwargs}"
        )
        assert len(local_args) == 5, (
            f"For MindSpore 'TopkExt', local_args should have 5 elements "
            f"(tensor, k, dim, largest, sorted), got {len(local_args)}"
        )
        assert local_args[1] == 5, (
            f"k should be 5, got {local_args[1]}"
        )
        assert local_args[2] == 0, (
            f"dim should be 0, got {local_args[2]}"
        )


    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_topk_multiaxis_tuple_sharded_dim_error(self, mock_platform):
        """
        Feature: TopKDistributedOp rejects StridedShard multi-axis mapping on topk dim.
        Description: Dim 0 is mapped to a tuple via StridedShard + Shard combo; topk on dim 0 should fail.
        Expectation: ValueError is raised about topk along a sharded dimension.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        # Build a layout where dim 0 gets a multi-axis tuple mapping via StridedShard.
        # Use two Shard placements on the same tensor dim to produce a multi-axis mapping.
        from hyper_parallel.core.dtensor.placement_types import StridedShard  # pylint: disable=C0415
        x_layout = _build_layout(mesh, (StridedShard(0, split_factor=2), Shard(0), Replicate()), 2)

        cache_values = [x_layout, 0]
        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            torch_op.infer_layout(cache_values)




if __name__ == "__main__":
    unittest.main()
    