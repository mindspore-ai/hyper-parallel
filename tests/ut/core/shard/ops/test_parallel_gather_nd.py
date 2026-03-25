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
"""parallel_gather_nd test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_gather import GatherNdDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelGatherNd(unittest.TestCase):
    """Unit tests for GatherNdDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method.

        Clears global caches to ensure test isolation and initializes
        the platform for testing.
        """
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.platform = get_platform()
        self.op = GatherNdDistributedOp("GatherNd")

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

    def _make_1d_mesh(self, mock_platform, world_size=8):
        """Set up mock and return a standard 1D mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    def _make_2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x4 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2_mesh(self, mock_platform, mesh_dim_names=("dp", "cp", "mp")):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_data_parallel(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params is replicated; indices is sharded on the first dimension and replicated on the last dimension.
        Expectation: Output inherits sharding from indices[:-1].
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        out_layout = self.op.infer_layout((p_layout, i_layout), (([16, 64], [16, 2]),))
        expected_map = (0,)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd data parallel failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_model_parallel(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params is replicated; indices is sharded on the first dimension by model-parallel axis.
        Expectation: Output inherits sharding from indices[:-1].
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        out_layout = self.op.infer_layout((p_layout, i_layout), (([16, 64], [16, 2]),))
        expected_map = (0,)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd model parallel failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_replicated(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params and indices are both replicated.
        Expectation: Output is replicated.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        out_layout = self.op.infer_layout((p_layout, i_layout), (([16, 64], [16, 2]),))
        expected_map = (-1,)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd replicated failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_indices_last_dim_sharded_error(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Indices last dimension is sharded, which is not allowed.
        Expectation: Raise ValueError.
        """
        mesh = self._make_2x4_mesh(mock_platform)

        p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        i_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

        with self.assertRaises(ValueError):
            self.op.infer_layout((p_layout, i_layout), (([16, 64], [16, 2]),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_params_sharded_error(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params is sharded, which is not allowed.
        Expectation: Raise ValueError.
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        i_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        with self.assertRaises(ValueError):
            self.op.infer_layout((p_layout, i_layout), (([16, 64], [16, 2]),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_3d_indices_no_extra_args(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: 3D indices without extra_args.
        Expectation: Output inherits only indices[:-1] sharding.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)

        p_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        i_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

        out_layout = self.op.infer_layout((p_layout, i_layout), (([16, 64, 32], [16, 8, 2]),))
        expected_map = (2, 1, -1)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd 3D indices failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_gathernd_params_is_none_layout(self, mock_platform):
        """
        Feature: GatherNd layout inference.
        Description: Params layout is None (plain Tensor), indices is sharded.
        Expectation: Output inherits sharding from indices[:-1].
        """
        mesh = self._make_1d_mesh(mock_platform, world_size=8)

        p_layout = None
        i_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

        out_layout = self.op.infer_layout((p_layout, i_layout), (([16, 64], [16, 2]),))
        expected_map = (0,)
        assert out_layout.tensor_map == expected_map, (
            f"GatherNd params None failed. Expected {expected_map}, got {out_layout.tensor_map}"
        )


if __name__ == "__main__":
    unittest.main()
