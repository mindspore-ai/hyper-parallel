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
"""parallel_pad test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_pad import PadDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = PadDistributedOp("pad")


class TestParallelPad(unittest.TestCase):
    """Unit tests for PadDistributedOp."""
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
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, tp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_pad_infer_layout_success_unsharded(self, mock_platform):
        """
        Feature: Pad unsharded dimension
        Description: Pad the last dimension which is Replicated.
        Expectation: Output layout is identical to input layout.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        pad = (1, 1)
        output_layout = op.infer_layout((x_layout,), extra_args=(pad,))

        assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test classes, so it is unnecessary to test its return value.
        assert op.get_expand_impl(None, output_layout, (x_layout,), extra_args=(pad,)) is None, (
            f"get_expand_impl test failed. Expected None, "
            f"got {op.get_expand_impl(None, output_layout, (x_layout,), extra_args=(pad,))}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_pad_infer_layout_fail_sharded(self, mock_platform):
        """
        Feature: Pad sharded dimension
        Description: Attempt to pad a dimension that is Sharded with non-zero values.
        Expectation: Raises ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        pad = (0, 0, 1, 1)

        with self.assertRaisesRegex(ValueError, "does not support padding on a sharded dimension"):
            op.infer_layout((x_layout,), extra_args=(pad,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_pad_infer_layout_mixed_dims(self, mock_platform):
        """
        Feature: Pad specific dimension in multi-dim tensor
        Description: Pad only the replicated dimension in a 3D tensor with mixed sharding.
        Expectation: Success for unsharded pad, Failure for sharded pad.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_placements = (Shard(0), Shard(1), Replicate())
        x_layout = _build_layout(mesh, x_placements, 3)

        pad_success = (1, 1)
        output_layout = op.infer_layout((x_layout,), extra_args=(pad_success,))
        assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]

        pad_fail = (0, 0, 1, 1)
        with self.assertRaisesRegex(ValueError, "does not support padding on a sharded dimension"):
            op.infer_layout((x_layout,), extra_args=(pad_fail,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_pad_infer_layout_zero_padding_on_sharded(self, mock_platform):
        """
        Feature: Zero padding on sharded dimension
        Description: If pad values are 0, it should be allowed even if dimension is sharded.
        Expectation: Success.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, x_placements, 2)

        pad = (1, 1, 0, 0)

        output_layout = op.infer_layout((x_layout,), extra_args=(pad,))
        assert output_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_pad_infer_layout_invalid_args(self, mock_platform):
        """
        Feature: Validate arguments
        Description: Check invalid pad tuple length or size mismatches.
        Expectation: Raises ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)
        x_placements = (Replicate(),)
        x_layout = _build_layout(mesh, x_placements, 1)

        with self.assertRaisesRegex(ValueError, "Pad tuple length must be even"):
            op.infer_layout((x_layout,), extra_args=((1, 2, 3),))

        with self.assertRaisesRegex(ValueError, "but tensor only has"):
            op.infer_layout((x_layout,), extra_args=((1, 1, 1, 1),))


if __name__ == "__main__":
    unittest.main()
