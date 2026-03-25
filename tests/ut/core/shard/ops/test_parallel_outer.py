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
"""parallel_outer test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_outer import OuterDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = OuterDistributedOp("outer")


class TestParallelOuter(unittest.TestCase):
    """Unit tests for OuterDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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
        """Set up mock and return a standard 2x4 (dp, tp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_outer_layout_inference_both_replicated(self, mock_platform):
        """
        Feature: Outer product with fully replicated inputs
        Description: Compute outer product of two 1-D tensors that are fully replicated.
        Expectation: Output layout is a 2-D tensor that is fully replicated.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x1_placements = (Replicate(), Replicate())
        x2_placements = (Replicate(), Replicate())

        x1_layout = _build_layout(mesh, x1_placements, 1)
        x2_layout = _build_layout(mesh, x2_placements, 1)

        output_layout = op.infer_layout((x1_layout, x2_layout))

        expected_map = (-1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Replicated outer failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_outer_layout_inference_orthogonal_sharding(self, mock_platform):
        """
        Feature: Outer product with orthogonal sharding
        Description: Compute outer product of two 1-D tensors sharded on different mesh dimensions.
        Expectation: Output layout inherits dim0 sharding from input1 and dim1 sharding from input2.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x1_placements = (Shard(0), Replicate())
        x1_layout = _build_layout(mesh, x1_placements, 1)

        x2_placements = (Replicate(), Shard(0))
        x2_layout = _build_layout(mesh, x2_placements, 1)

        output_layout = op.infer_layout((x1_layout, x2_layout))

        expected_map = (1, 0)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Orthogonal sharding failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_outer_layout_inference_partial_sharding(self, mock_platform):
        """
        Feature: Outer product with partial sharding
        Description: Compute outer product where only the first input is sharded.
        Expectation: Output inherits sharding for dim0, and is unsharded for dim1.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x1_placements = (Shard(0), Replicate())
        x1_layout = _build_layout(mesh, x1_placements, 1)

        x2_placements = (Replicate(), Replicate())
        x2_layout = _build_layout(mesh, x2_placements, 1)

        output_layout = op.infer_layout((x1_layout, x2_layout))

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Partial sharding failed. Expected {expected_map}, "
            f"got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_outer_layout_invalid_overlapping_sharding(self, mock_platform):
        """
        Feature: Validate overlapping sharding
        Description: Attempt to compute outer product when both inputs are sharded on the same mesh dimension.
        Expectation: ValueError is raised because it violates orthogonal placement rules.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        placements = (Shard(0), Replicate())
        layout = _build_layout(mesh, placements, 1)

        with self.assertRaisesRegex(ValueError, "the two inputs cannot be sharded on the same device mesh dimension"):
            op.infer_layout((layout, layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_outer_layout_invalid_not_1d_tensor(self, mock_platform):
        """
        Feature: Validate input dimensionality
        Description: Attempt to compute outer product using inputs that are not exactly 1-D.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x1_placements = (Replicate(), Replicate())
        x1_layout = _build_layout(mesh, x1_placements, 1)

        x2_placements = (Replicate(), Replicate())
        x2_layout = _build_layout(mesh, x2_placements, 2)

        with self.assertRaisesRegex(ValueError, "requires exactly 1-D tensors as inputs"):
            op.infer_layout((x1_layout, x2_layout))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_outer_layout_invalid_num_inputs(self, mock_platform):
        """
        Feature: Validate number of inputs
        Description: Attempt to compute outer product without exactly two layouts.
        Expectation: ValueError is raised.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        x1_placements = (Replicate(), Replicate())
        x1_layout = _build_layout(mesh, x1_placements, 1)

        with self.assertRaisesRegex(ValueError, "requires exactly 2 input layouts"):
            op.infer_layout((x1_layout,))

        with self.assertRaisesRegex(ValueError, "requires exactly 2 input layouts"):
            op.infer_layout((x1_layout, x1_layout, x1_layout))


if __name__ == "__main__":
    unittest.main()
