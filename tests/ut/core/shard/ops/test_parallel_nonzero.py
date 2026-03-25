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
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_nonzero import NonzeroDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = NonzeroDistributedOp("nonzero")


class TestParallelNonzero(unittest.TestCase):
    """Unit tests for NonzeroDistributedOp."""
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

        output_layout = op.infer_layout((x_layout,))

        expected_map = (-1, -1)
        assert not isinstance(output_layout, tuple)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Basic nonzero failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
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

        output_layouts = op.infer_layout((x_layout,), extra_args=(True,))

        expected_map = (-1,)

        assert isinstance(output_layouts, tuple), "Expected output to be a tuple of layouts"
        assert len(output_layouts) == 3, "Expected 3 layouts for a 3D input"

        for i, out_layout in enumerate(output_layouts):
            assert out_layout.to_dict()["tensor_map"] == expected_map, (
                f"Tuple output {i} layout mismatch. Expected {expected_map}, got {out_layout.to_dict()['tensor_map']}"
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

        with self.assertRaisesRegex(ValueError, "input tensor must be fully replicated"):
            op.infer_layout((x_layout,))

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

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            op.infer_layout((x_layout,))


if __name__ == "__main__":
    unittest.main()
