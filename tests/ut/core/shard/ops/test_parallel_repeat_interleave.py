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
"""parallel_repeat_interleave test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_repeat_interleave import RepeatInterleaveDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = RepeatInterleaveDistributedOp("repeat_interleave")


class TestParallelRepeatInterleave(unittest.TestCase):
    """Unit tests for RepeatInterleaveDistributedOp."""
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
    def test_torch_repeat_interleave_layout_data_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave data parallel
        Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        repeats = 2
        dim = 1
        output_layout = op.infer_layout((x_layout,), (repeats, dim))

        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Data Parallel with torch repeat_interleave test failed. Expected {expected_map},"
            f" got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_tensor_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave tensor parallel
        Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(0))
        x_layout = _build_layout(mesh, x_placements, 2)
        repeats = 2
        dim = 1
        output_layout = op.infer_layout((x_layout,), (repeats, dim))
        expected_map = (0, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Tensor Parallel with torch repeat_interleave test failed. Expected {expected_map},"
            f" got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_with_tensor_layout_data_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave data parallel
        Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        repeats_tensor = [2, 1, 1, 1]

        dim = 1
        output_layout = op.infer_layout((x_layout,), (repeats_tensor, dim))
        expected_map = (1, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Data Parallel with torch repeat_interleave test failed. Expected {expected_map},"
            f" got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_with_tensor_layout_tensor_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave tensor parallel
        Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(0))
        x_layout = _build_layout(mesh, x_placements, 2)
        repeats_tensor = [2, 1, 1, 1]

        dim = 1
        output_layout = op.infer_layout((x_layout,), (repeats_tensor, dim))
        expected_map = (0, -1)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Tensor Parallel with torch repeat_interleave test failed. Expected {expected_map},"
            f" got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_dim_none_layout_data_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave data parallel
        Description: Data parallel scenario (shard on first dim, repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)
        repeats = 2
        output_layout = op.infer_layout((x_layout,), (repeats,))
        expected_map = (1,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Data Parallel with dim None repeat_interleave test failed. Expected {expected_map},"
            f" got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_dim_none_layout_tensor_parallel(self, mock_platform):
        """
        Feature: RepeatInterleave tensor parallel
        Description: Tensor parallel scenario (shard on first dim with 'tp', repeat on last unsharded dim)
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Shard(0))
        x_layout = _build_layout(mesh, x_placements, 2)

        repeats = 2
        output_layout = op.infer_layout((x_layout,), (repeats,))
        expected_map = (0,)
        assert output_layout.to_dict()["tensor_map"] == expected_map, (
            f"Tensor Parallel dim None repeat_interleave test failed. Expected {expected_map},"
            f" got {output_layout.to_dict()['tensor_map']}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_sharded_dim_error(self, mock_platform):
        """
        Feature: RepeatInterleave on sharded dimension
        Description: Repeat on a sharded dimension should raise error
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1))
        x_layout = _build_layout(mesh, x_placements, 2)

        repeats = 2
        dim = 1
        with self.assertRaisesRegex(ValueError, "Cannot perform sharding on params along the chosen dim"):
            op.infer_layout((x_layout,), (repeats, dim))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_torch_repeat_interleave_layout_error_dim_out_of_range(self, mock_platform):
        """
        Feature: Test indicating a invalid dim
        Description: Test indicating a invalid dim.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        repeats = 2
        dim = 5

        with self.assertRaisesRegex(ValueError, "Dimension out of range"):
            op.infer_layout((x_layout,), extra_args=(repeats, dim))


if __name__ == "__main__":
    unittest.main()
