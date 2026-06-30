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
"""parallel_scatter_update test"""
import unittest
from unittest.mock import patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_scatter_update import (
    ScatterUpdateDistributedOp,
    _normalize_scatter_update_args,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


op = ScatterUpdateDistributedOp("ScatterUpdate")


class TestParallelScatterUpdate(unittest.TestCase):
    """Unit tests for ScatterUpdateDistributedOp."""

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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp")
        )

    def _make_2x2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2x2 (dp, cp, mp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=16)
        return init_device_mesh(
            device_type="npu", mesh_shape=(2, 2, 2, 2), mesh_dim_names=("dp", "cp", "mp", "tp")
        )

    def _run_infer(self, input_layout, indices_layout, updates_layout):
        """Helper to run infer_layout with cache_values and return output layout."""
        cache_values = [input_layout, indices_layout, updates_layout]
        output_layouts, extra_info = op.infer_layout(cache_values)
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        return output_layouts[0]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_data_parallel_1(self, mock_platform):
        """
        Feature: Data parallel on dimension 1.
        Description: input[1] sharded on dp, updates follows.
        Expectation: output matches input layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Replicate()), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Shard(1), Replicate(), Replicate()), 3)

        output_layout = self._run_infer(input_layout, indices_layout, updates_layout)
        expected_map = (-1, 2, -1)
        assert output_layout.tensor_map == expected_map, (
            f"ScatterUpdate failed in scenario 'Data Parallel on dim 1'. "
            f"Expected {expected_map}, got {output_layout.tensor_map}"
        )

        # Since get_expand_impl is not overridden, it returns None by default.
        # The same applies to other test cases, so it is unnecessary to test its return value.
        cache_values = [input_layout, indices_layout, updates_layout]
        self.assertIsNone(
            op.get_expand_impl(None, ((output_layout,), None), cache_values),
            "get_expand_impl should return None",
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_model_parallel_2(self, mock_platform):
        """
        Feature: Model parallel on dimension 2.
        Description: input[2] sharded on mp, updates follows.
        Expectation: output matches input layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)

        output_layout = self._run_infer(input_layout, indices_layout, updates_layout)
        expected_map = (-1, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"ScatterUpdate failed in scenario 'Model Parallel on dim 2'. "
            f"Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_hybrid_parallel_3(self, mock_platform):
        """
        Feature: Hybrid parallel on dimensions 1 and 2.
        Description: input[1] on dp, input[2] on mp, updates follows.
        Expectation: output matches input layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Shard(2)), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Shard(1), Replicate(), Shard(2)), 3)

        output_layout = self._run_infer(input_layout, indices_layout, updates_layout)
        expected_map = (-1, 2, 0)
        assert output_layout.tensor_map == expected_map, (
            f"ScatterUpdate failed in scenario 'Hybrid Parallel (DP+MP)'. "
            f"Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_multi_dim_indices_4(self, mock_platform):
        """
        Feature: Multi-dimensional indices.
        Description: indices is 2D, updates first 2 dims cannot be sharded.
        Expectation: output matches input layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Shard(2)), 3)
        indices_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        updates_layout = _build_layout(mesh, (Shard(2), Replicate(), Shard(3), Replicate()), 4)

        output_layout = self._run_infer(input_layout, indices_layout, updates_layout)
        expected_map = (-1, 2, 0)
        assert output_layout.tensor_map == expected_map, (
            f"ScatterUpdate failed in scenario 'Multi-dimensional indices (2D)'. "
            f"Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_three_dim_input_5(self, mock_platform):
        """
        Feature: Three-dimensional input with complex sharding.
        Description: input[1] on cp, input[2] on mp, 3D indices.
        Expectation: output matches input layout.
        """
        mesh = self._make_2x2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Shard(1), Shard(2), Shard(3)), 4)
        indices_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        updates_layout = _build_layout(
            mesh, (Replicate(), Shard(3), Shard(4), Shard(5), Replicate(), Replicate()), 6
        )

        output_layout = self._run_infer(input_layout, indices_layout, updates_layout)
        expected_map = (-1, 2, 1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"ScatterUpdate failed in scenario 'Three-dimensional input with 3D indices'. "
            f"Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_replicate_all_6(self, mock_platform):
        """
        Feature: Full replication.
        Description: all tensors replicated across devices.
        Expectation: output is replicated.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        output_layout = self._run_infer(input_layout, indices_layout, updates_layout)
        expected_map = (-1, -1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"ScatterUpdate failed in scenario 'Full replication'. "
            f"Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_error_input_first_dim_sharded_7(self, mock_platform):
        """
        Feature: Error case - input first dimension sharded.
        Description: input[0] cannot be sharded.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)

        cache_values = [input_layout, indices_layout, updates_layout]
        with self.assertRaisesRegex(ValueError, "first dimension of input cannot be sharded"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_error_indices_sharded_8(self, mock_platform):
        """
        Feature: Error case - indices sharded.
        Description: indices cannot be sharded.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Replicate()), 3)
        indices_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        updates_layout = _build_layout(mesh, (Shard(1), Replicate(), Replicate()), 3)

        cache_values = [input_layout, indices_layout, updates_layout]
        with self.assertRaisesRegex(ValueError, "indices cannot be sharded"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_error_updates_prefix_sharded_9(self, mock_platform):
        """
        Feature: Error case - updates prefix sharded.
        Description: updates first n dims (n=len(indices)) cannot be sharded.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Replicate()), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Shard(1), Shard(0), Replicate()), 3)

        cache_values = [input_layout, indices_layout, updates_layout]
        with self.assertRaisesRegex(ValueError, "dimensions of updates cannot be sharded"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_error_updates_mismatch_10(self, mock_platform):
        """
        Feature: Error case - updates sharding mismatch.
        Description: updates[indices_ndim:] must match input[1:].
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Shard(2)), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Shard(2), Replicate(), Shard(1)), 3)

        cache_values = [input_layout, indices_layout, updates_layout]
        with self.assertRaisesRegex(ValueError, "updates sharding must match input"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_error_updates_rank_mismatch_11(self, mock_platform):
        """
        Feature: Error case - updates rank mismatch.
        Description: updates rank must equal indices_ndim + input_ndim - 1.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Shard(2)), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Shard(1), Replicate()), 2)

        cache_values = [input_layout, indices_layout, updates_layout]
        with self.assertRaisesRegex(ValueError, "updates rank mismatch"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_error_indices_not_dtensor(self, mock_platform):
        """
        Feature: Error case - indices is not a DTensor.
        Description: When input is a DTensor, indices must also be a DTensor.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Replicate()), 3)
        updates_layout = _build_layout(mesh, (Shard(1), Replicate(), Replicate()), 3)
        # indices_layout is None when indices is a plain Tensor
        cache_values = [input_layout, None, updates_layout]
        with self.assertRaisesRegex(ValueError, "indices must be a DTensor"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_error_updates_not_dtensor(self, mock_platform):
        """
        Feature: Error case - updates is not a DTensor.
        Description: When input is a DTensor, updates must also be a DTensor.
        Expectation: raise ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Replicate()), 3)
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        # updates_layout is None when updates is a plain Tensor
        cache_values = [input_layout, indices_layout, None]
        with self.assertRaisesRegex(ValueError, "updates must be a DTensor"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_update_partial_propagation_12(self, mock_platform):
        """
        Feature: Partial status propagation.
        Description: input with partial status, output should inherit.
        Expectation: output layout has same partial status.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        input_layout = _build_layout(mesh, (Shard(1), Replicate(), Shard(2)), 3)
        input_layout.set_partial_by_dev_axis("cp", "sum")
        indices_layout = _build_layout(mesh, (Replicate(),), 1)
        updates_layout = _build_layout(mesh, (Shard(1), Replicate(), Shard(2)), 3)

        output_layout = self._run_infer(input_layout, indices_layout, updates_layout)

        assert output_layout.get_partial_by_dev_id("cp") == "sum", (
            "Partial status should be propagated from input to output"
        )

    def test_normalize_scatter_update_args_all_positional(self):
        """
        Feature: ScatterUpdate argument normalization
        Description: Test that _normalize_scatter_update_args correctly normalizes args
        Expectation: Returns normalized args tuple and empty kwargs
        """
        x = object()
        idx = object()
        upd = object()
        args, kwargs = _normalize_scatter_update_args(x, idx, upd)
        assert args == (x, idx, upd), f"Expected (x, idx, upd), got {args}"
        assert not kwargs, f"Expected empty kwargs, got {kwargs}"


if __name__ == "__main__":
    unittest.main()
    