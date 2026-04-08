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
"""Unit tests for zeros_like distributed op (layout inference)."""
import os
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_elementwise import ElementWiseDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelZerosLike(unittest.TestCase):
    """Unit tests for ElementWiseDistributedOp used by zeros_like.

    zeros_like is a single-input creation op: its output layout must always
    equal the input layout regardless of placement (single-input early-return
    in ElementWiseDistributedOp.infer_layout).  Two placement variants cover
    the passthrough path; the partial-input case covers the error guard that
    runs before the early-return.
    """

    def setUp(self):
        """Clear global caches and initialise platform before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()
        self.op = ElementWiseDistributedOp("zeros_like")

    def tearDown(self):
        """Restore global state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        """Configure minimal mock-platform attributes needed by init_device_mesh.

        Args:
            mock_platform: MagicMock injected by @patch.
            world_size: Value returned by mock_platform.get_world_size().
        """
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size

    def _make_2x4_mesh(self, mock_platform):
        """Return a 2x4 (dp, mp) mesh with mocked backend.

        Args:
            mock_platform: MagicMock injected by @patch.

        Returns:
            DeviceMesh with shape (2, 4) and dim names ("dp", "mp").
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu",
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_zeros_like_data_parallel(self, mock_platform):
        """
        Feature: zeros_like data parallel
        Description: Input sharded on batch dim; output layout must match input.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Shard(0), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        output_layout = self.op.infer_layout((x_layout,), {})

        # tensor_map for (Shard(0), Replicate()) on 2x4 ("dp", "mp"):
        # mesh axis 0 (dp) shards tensor dim 0  -> mapped index = 2-1-0 = 1
        # mesh axis 1 (mp) is Replicate         -> -1
        expected_map = (1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"zeros_like data parallel layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

        # Since `get_expand_impl` is not overridden, it returns None by default.
        # The same applies to other test cases, so it is unnecessary to test
        # its return value again.
        assert self.op.get_expand_impl(None, output_layout, (x_layout,), {}) is None, (
            f"get_expand_impl should return None, "
            f"got={self.op.get_expand_impl(None, output_layout, (x_layout,), {})}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_zeros_like_all_replicated(self, mock_platform):
        """
        Feature: zeros_like all replicated
        Description: Fully replicated input; output must also be fully replicated.
        Expectation: Success
        """
        mesh = self._make_2x4_mesh(mock_platform)
        placements = (Replicate(), Replicate())
        x_layout = _build_layout(mesh, placements, 2)

        output_layout = self.op.infer_layout((x_layout,), {})

        expected_map = (-1, -1)
        assert output_layout.tensor_map == expected_map, (
            f"zeros_like all-replicated layout mismatch: "
            f"expected={expected_map}, got={output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_zeros_like_partial_input_raises(self, mock_platform):
        """
        Feature: zeros_like partial input error
        Description: Input with Partial status must be rejected because
            ElementWiseDistributedOp does not allow partial inputs by default.
        Expectation: ValueError is raised
        """
        mesh = self._make_2x4_mesh(mock_platform)
        # Replicate on both axes so set_partial_by_dev_axis succeeds
        x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        x_layout.set_partial_by_dev_axis("dp", "sum")

        with self.assertRaisesRegex(ValueError, "has Partial status which is not allowed"):
            self.op.infer_layout((x_layout,), {})


if __name__ == "__main__":
    unittest.main()
