# Copyright 2025 Huawei Technologies Co., Ltd
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
"""parallel_batch_matmul_ext test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_matmul import BatchMatMulExtDistributedOp
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestParallelBatchMatMulExt(unittest.TestCase):
    """Unit tests for BatchMatMulExtDistributedOp."""
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

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 (dp, cp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "mp"))

    def _run_scenario(self, x_layout, w_layout, expected_map):
        """Infer layout of BatchMatmul operator"""
        op = BatchMatMulExtDistributedOp("BatchMatMulExt")
        output_layout = op.infer_layout((x_layout, w_layout), None)
        assert output_layout.tensor_map == expected_map, (
            f"BatchMatMulExt failed. Expected {expected_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_data_parallel(self, mock_platform):
        """
        Feature: Data parallel in python shard.
        Description: Test data parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_model_parallel(self, mock_platform):
        """
        Feature: Model parallel in python shard.
        Description: Test model parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(2)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(-1, -1, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_hybrid_parallel(self, mock_platform):
        """
        Feature: Hybrid parallel in python shard.
        Description: Test hybrid parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)
        w_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, 1, 0))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_tensor_parallel(self, mock_platform):
        """
        Feature: Tensor parallel in python shard.
        Description: Test tensor parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_hybrid_tensor_parallel(self, mock_platform):
        """
        Feature: Tensor parallel in python shard.
        Description: Test tensor parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)
        w_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, 1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bmm_ext_multi_shard_tensor_parallel(self, mock_platform):
        """
        Feature: Multi shard tensor parallel in python shard.
        Description: Test multi shard tensor parallel in python shard.
        Expectation: Run success.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
        w_layout = _build_layout(mesh, (Replicate(), Shard(2), Shard(1)), 3)

        self._run_scenario(x_layout, w_layout, expected_map=(2, -1, 1))


if __name__ == "__main__":
    unittest.main()
