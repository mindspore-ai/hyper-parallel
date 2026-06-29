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
"""parallel_nonzero test"""
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_nonzero import NonzeroDistributedOp
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = NonzeroDistributedOp("nonzero")


class TestParallelNonzero(unittest.TestCase):
    """Unit tests for NonzeroDistributedOp."""
    def setUp(self) -> None:
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
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

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "tp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nonzero_layout_inference_basic(self, mock_platform):
        """
        Feature: Nonzero on fully replicated input (as_tuple=False)
        Description: Input is a fully replicated 2D tensor. Nonzero returns a single 2D tensor.
        Expectation: Output layout is a single fully replicated 2D layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        cache_values = [x_layout, False]
        infer_result = op.infer_layout(cache_values)
        output_layouts, extra_info = infer_result

        assert extra_info is None, (
            f"Nonzero extra_info should be None, got {extra_info}"
        )
        assert not isinstance(output_layouts, tuple) or len(output_layouts) == 1, (
            f"Expected a single output layout for as_tuple=False, "
            f"got {output_layouts}"
        )
        output_layout = output_layouts[0]

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Basic nonzero failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test cases, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, infer_result, cache_values) is None, (
            f"get_expand_impl should return None for {op.op_name}, "
            f"got {op.get_expand_impl(None, infer_result, cache_values)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nonzero_layout_inference_as_tuple(self, mock_platform):
        """
        Feature: Nonzero on fully replicated input (as_tuple=True)
        Description: Input is a fully replicated 3D tensor. Nonzero with as_tuple=True
                     returns a tuple of 1D tensors matching input ndim.
        Expectation: Output is a tuple of 3 fully replicated 1D layouts.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Replicate(), Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        cache_values = [x_layout, True]
        infer_result = op.infer_layout(cache_values)
        output_layouts, extra_info = infer_result

        assert extra_info is None, (
            f"Nonzero extra_info should be None, got {extra_info}"
        )
        assert isinstance(output_layouts, tuple), (
            "Expected output to be a tuple of layouts"
        )
        assert len(output_layouts) == 3, (
            f"Expected 3 layouts for a 3D input, got {len(output_layouts)}"
        )

        expected_map = (-1,)
        for i, out_layout in enumerate(output_layouts):
            assert out_layout.to_dict()["tensor_map"] == expected_map, (
                f"Tuple output {i} layout mismatch. Expected {expected_map}, "
                f"got {out_layout.to_dict()['tensor_map']}"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nonzero_layout_invalid_sharded_input(self, mock_platform):
        """
        Feature: Nonzero on sharded input
        Description: Attempt to run nonzero on a tensor with a sharded dimension.
                     Since nonzero generates data-dependent dynamic shapes, it is unsafe.
        Expectation: ValueError raised with clear message.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        x_layout._partial = [None] * len(x_layout._partial)

        cache_values = [x_layout, False]
        with self.assertRaisesRegex(ValueError, "input tensor should be fully replicated"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nonzero_layout_invalid_none_input(self, mock_platform):
        """
        Feature: Nonzero with None input layout
        Description: Attempt to run nonzero with a None input layout.
        Expectation: ValueError raised with clear message.
        """
        self._make_2x4_mesh(mock_platform)

        cache_values = [None, False]
        with self.assertRaisesRegex(ValueError, "input_layout should be a valid Layout"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nonzero_layout_invalid_partial_input(self, mock_platform):
        """
        Feature: Nonzero on partial input
        Description: Attempt to run nonzero on a tensor with a Partial state.
        Expectation: ValueError raised by the base class check.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        cache_values = [x_layout, False]
        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nonzero_preprocess_basic(self, mock_platform):
        """
        Feature: NonzeroDistributedOp preprocess routes as_tuple into kwargs.
        Description: torch.nonzero declares as_tuple as keyword-only (after *);
                     op_name is 'nonzero'. Verifies local_kwargs receives as_tuple,
                     local_args contains the to_local'd tensor, and cache_values
                     contains layout and as_tuple.
        Expectation: local_kwargs has 'as_tuple'; cache_values has layout and as_tuple.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess((mock_tensor,), {})

        assert local_kwargs == {'as_tuple': False}, (
            f"For torch.nonzero, local_kwargs should be {{'as_tuple': False}}, "
            f"got local_kwargs={local_kwargs}"
        )
        assert len(local_args) == 1, (
            f"For torch.nonzero, local_args should have 1 element, "
            f"got {len(local_args)}"
        )
        assert cache_values[0] is x_layout, (
            f"cache_values[0] should be the input layout, "
            f"got {cache_values[0]}"
        )
        assert cache_values[1] is False, (
            f"cache_values[1] should be False (default as_tuple), "
            f"got {cache_values[1]}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nonzero_preprocess_as_tuple_true(self, mock_platform):
        """
        Feature: NonzeroDistributedOp preprocess with as_tuple=True.
        Description: When as_tuple=True is passed, it should be preserved in
                     local_kwargs and cache_values.
        Expectation: local_kwargs and cache_values reflect as_tuple=True.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        mock_tensor = MagicMock()
        mock_tensor.layout = x_layout
        mock_tensor.to_local.return_value = MagicMock()

        local_args, local_kwargs, cache_values = op.preprocess(
            (mock_tensor,), {'as_tuple': True}
        )

        assert local_kwargs == {'as_tuple': True}, (
            f"For torch.nonzero with as_tuple=True, local_kwargs should be "
            f"{{'as_tuple': True}}, got local_kwargs={local_kwargs}"
        )
        assert cache_values[1] is True, (
            f"cache_values[1] should be True, got {cache_values[1]}"
        )


if __name__ == "__main__":
    unittest.main()