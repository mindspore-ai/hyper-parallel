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
"""parallel_argsort test"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_argsort import ArgsortDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ArgsortDistributedOp("argsort")
op_ms = ArgsortDistributedOp("ArgSort")


class TestParallelArgsort(unittest.TestCase):
    """Unit tests for ArgsortDistributedOp."""

    def setUp(self) -> None:
        """Clear global caches before each test to ensure test isolation."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Clear global caches after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 4),
                                mesh_dim_names=("dp", "mp"), init_backend=False)

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, tp, mp) mesh via init_device_mesh."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 8
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()
        return init_device_mesh(device_type="cpu", mesh_shape=(2, 2, 2),
                                mesh_dim_names=("dp", "tp", "mp"), init_backend=False)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_inference_basic(self, mock_platform):
        """
        Feature: Argsort on an unsharded dimension
        Description: Perform argsort along the last dimension (dim=-1), which is fully replicated.
        Expectation: Output layout is identical to the input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, -1]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert extra_info is None, (
            f"Argsort extra_info should be None, got {extra_info}"
        )
        output_layout = output_layouts[0]
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"Basic argsort failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

        # ArgsortDistributedOp does not override get_expand_impl → always None.
        # Verified once here; other test cases omit this check as per testing conventions.
        assert op.get_expand_impl(None, (output_layouts, None), cache_values) is None, (
            f"get_expand_impl should return None for argsort, "
            f"got {op.get_expand_impl(None, (output_layouts, None), cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_inference_specific_dim(self, mock_platform):
        """
        Feature: Argsort on a specific unsharded dimension with extra kwargs
        Description: Perform argsort on dim=0, with descending=True. dim=0 is Replicate.
        Expectation: Output layout is identical to the input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert extra_info is None, (
            f"Argsort extra_info should be None, got {extra_info}"
        )
        output_layout = output_layouts[0]
        expected_map = (-1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Specific dim argsort failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_inference_negative_dim(self, mock_platform):
        """
        Feature: Argsort handling of negative dimensions
        Description: Perform argsort on a 3D tensor using dim=-2, which maps to the middle
            unsharded dimension.
        Expectation: Resolves the negative dimension correctly and returns the identical layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Replicate(), Shard(2))
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, -2]
        output_layouts, extra_info = op.infer_layout(cache_values)

        assert extra_info is None, (
            f"Argsort extra_info should be None, got {extra_info}"
        )
        output_layout = output_layouts[0]
        expected_map = (2, -1, 0)
        assert output_layout.tensor_map == expected_map, (
            f"Negative dim argsort failed. Expected {expected_map}, "
            f"got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_invalid_sharded_dim(self, mock_platform):
        """
        Feature: Argsort on a sharded dimension
        Description: Attempt to perform argsort along a dimension that is currently sharded.
        Expectation: ValueError is raised preventing the mathematically incorrect operation.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 0]
        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_invalid_out_of_bounds_dim(self, mock_platform):
        """
        Feature: Argsort with out-of-bounds dimension
        Description: Attempt to perform argsort using a dimension index larger than tensor rank.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, 2]
        with self.assertRaisesRegex(ValueError, "dimension out of range"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_partial_input_raises_error(self, mock_platform):
        """
        Feature: ArgsortDistributedOp rejects inputs with Partial status.
        Description: Input has Partial status set on dp axis (pending AllReduce).
        Expectation: ValueError is raised about Partial status not being allowed.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        cache_values = [x_layout, 1]
        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_layout_multiaxis_tuple_sharded_dim_error(self, mock_platform):
        """
        Feature: ArgsortDistributedOp rejects StridedShard multi-axis mapping on sort dim.
        Description: Dim 0 is mapped to a tuple of mesh axes via StridedShard + Shard(0) combo.
        Expectation: ValueError is raised about sorting along a sharded dimension.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        # pylint: disable=C0415
        from hyper_parallel.core.dtensor.placement_types import StridedShard
        x_layout = _build_layout(mesh, (StridedShard(0, split_factor=2), Shard(0), Replicate()), 2)

        cache_values = [x_layout, 0]
        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_preprocess_torch_stable_in_kwargs(self, mock_platform):
        """
        Feature: ArgsortDistributedOp preprocess routes stable into kwargs for PyTorch.
        Description: PyTorch torch.argsort declares stable keyword-only (after *); op_name is 'argsort'.
        Expectation: local_kwargs contains dim, descending, stable; local_args has 1 element (tensor only).
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor,), {})

        assert local_kwargs == {'dim': -1, 'descending': False, 'stable': False}, (
            f"For PyTorch 'argsort', local_kwargs should be "
            f"{{'dim': -1, 'descending': False, 'stable': False}}, got local_kwargs={local_kwargs}"
        )
        assert len(local_args) == 1, (
            f"For PyTorch 'argsort', local_args should have 1 element (tensor only), "
            f"got {len(local_args)}"
        )
        assert cache_values[1] == -1, (
            f"Default dim should be -1, got {cache_values[1]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_argsort_preprocess_mindspore_stable_in_args(self, mock_platform):
        """
        Feature: ArgsortDistributedOp preprocess routes stable into positional args for MindSpore.
        Description: MindSpore ArgSort Primitive does not accept kwargs; op_name is 'ArgSort'.
        Expectation: local_kwargs is empty; local_args has 4 elements with stable as the 4th arg.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op_ms.preprocess((mock_tensor,), {})

        assert not local_kwargs, (
            f"For MindSpore 'ArgSort', local_kwargs should be empty, got {local_kwargs}"
        )
        assert len(local_args) == 4, (
            f"For MindSpore 'ArgSort', local_args should have 4 elements "
            f"(tensor, dim, descending, stable), got {len(local_args)}"
        )
        assert local_args[3] is False, (
            f"stable default should be False, got {local_args[3]}"
        )


if __name__ == "__main__":
    unittest.main()
