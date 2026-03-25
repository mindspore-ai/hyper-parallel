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
"""parallel_scaled_dot_product_attention unit test"""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout, Layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_scaled_dot_product_attention import (
    ScaledDotProductAttentionDistributedOp,
)
from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = ScaledDotProductAttentionDistributedOp("scaled_dot_product_attention")


class TestParallelScaledDotProductAttention(unittest.TestCase):
    """Unit tests for ScaledDotProductAttentionDistributedOp."""
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
        """Set up mock and return a standard 2x2x2 (dp, sp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "sp", "mp"))

    def _make_4x2_mesh(self, mock_platform):
        """Set up mock and return a standard 4x2 (dp, mp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))

    def _make_2x2x2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2x4 (dp, sp, mp, pp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=32)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2, 4), mesh_dim_names=("dp", "sp", "mp", "pp"))

    def _run_scenario(self, q_layout, k_layout, v_layout, expected_out_map):
        """Infer layout and verify attention output tensor_map."""
        output_layout = op.infer_layout((q_layout, k_layout, v_layout), [])
        assert output_layout.tensor_map == expected_out_map, (
            f"Expected {expected_out_map}, got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_no_parallel_1(self, mock_platform):
        """
        Feature: Layout inference with no parallelism.
        Description: All dimensions replicated on a 3D mesh, BNSD 4D input.
        Expectation: Output tensor_map is all -1.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (-1, -1, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_data_parallel_2(self, mock_platform):
        """
        Feature: Layout inference with data parallelism.
        Description: BNSD batch dimension sharded on dp axis.
        Expectation: Output batch dimension remains sharded on dp.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Shard(0), Replicate(), Replicate())
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        self._run_scenario(q_layout, k_layout, v_layout, (2, -1, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_head_parallel_3(self, mock_platform):
        """
        Feature: Layout inference with head parallelism.
        Description: BNSD head dimension sharded on mp axis.
        Expectation: Output head dimension remains sharded on mp.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Replicate(), Replicate(), Shard(1))
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        self._run_scenario(q_layout, k_layout, v_layout, (-1, 0, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_sequence_parallel_4(self, mock_platform):
        """
        Feature: Layout inference with sequence parallelism.
        Description: BNSD query sequence sharded on sp axis, KV replicated.
        Expectation: Output sequence dimension sharded on sp.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Shard(2), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (-1, -1, 1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_hybrid_dp_mp_5(self, mock_platform):
        """
        Feature: Layout inference with hybrid DP + MP.
        Description: BNSD batch on dp, head on mp.
        Expectation: Both dimensions remain sharded.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Shard(0), Replicate(), Shard(1))
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        self._run_scenario(q_layout, k_layout, v_layout, (2, 0, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_hybrid_dp_sp_mp_6(self, mock_platform):
        """
        Feature: Layout inference with full hybrid DP + SP + MP.
        Description: BNSD with all three parallel strategies, KV only dp+mp.
        Expectation: Output preserves dp+sp+mp sharding from query.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
        k_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 4)
        v_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (2, 0, 1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_kv_different_layout_7(self, mock_platform):
        """
        Feature: Layout inference with different KV sharding.
        Description: Key has sp sharding but Value does not.
        Expectation: Layout inference succeeds based on query layout.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
        k_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
        v_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (2, 0, 1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_dp_mp_2d_mesh_8(self, mock_platform):
        """
        Feature: Layout inference with DP + MP on a 2D mesh.
        Description: BNSD batch on dp, head on mp, 2D mesh (4, 2).
        Expectation: Output preserves sharding on both dimensions.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(1))
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        self._run_scenario(q_layout, k_layout, v_layout, (1, 0, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_dp_sp_2d_mesh_9(self, mock_platform):
        """
        Feature: Layout inference with DP + SP on a 2D mesh.
        Description: BNSD batch on dp, sequence on sp, head replicated.
        Expectation: Output preserves batch and sequence sharding.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2)), 4)
        k_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
        v_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (1, -1, 0, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_single_device_10(self, mock_platform):
        """
        Feature: Layout inference with single device mesh.
        Description: Mesh size is 1, all dimensions replicated.
        Expectation: Output tensor_map is all -1.
        """
        self._setup_mock_platform(mock_platform, world_size=1)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(1,), mesh_dim_names=("dp",))
        q_layout = _build_layout(mesh, (Replicate(),), 4)
        k_layout = _build_layout(mesh, (Replicate(),), 4)
        v_layout = _build_layout(mesh, (Replicate(),), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (-1, -1, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_large_world_size_11(self, mock_platform):
        """
        Feature: Layout inference with large world size.
        Description: 32 devices on a 4D mesh (2,2,2,4) with dp+sp+mp+pp axes.
        Expectation: Output scales correctly with 4D mesh.
        """
        mesh = self._make_2x2x2x4_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1), Replicate()), 4)
        k_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1), Replicate()), 4)
        v_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1), Replicate()), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (3, 1, 2, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_single_output_layout_12(self, mock_platform):
        """
        Feature: Layout inference returns a single output layout.
        Description: SDPA returns only attention_out.
        Expectation: Output is a single Layout, not a tuple.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(1))
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        output_layout = op.infer_layout((q_layout, k_layout, v_layout), [])

        assert isinstance(output_layout, Layout), "Should return a single Layout"
        assert output_layout.tensor_map == (1, 0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_head_dim_always_replicated_13(self, mock_platform):
        """
        Feature: Layout inference never shards head_dim.
        Description: Verify the last dimension (D) is always -1 in the output.
        Expectation: Output head_dim (dim 3) is -1.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(1)), 4)
        k_layout = _build_layout(mesh, (Shard(0), Shard(1)), 4)
        v_layout = _build_layout(mesh, (Shard(0), Shard(1)), 4)

        output_layout = op.infer_layout((q_layout, k_layout, v_layout), [])

        tensor_map = output_layout.tensor_map
        assert tensor_map[3] == -1, "Head dim (dim 3) should not be sharded"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_3d_input_head_parallel_14(self, mock_platform):
        """
        Feature: Layout inference for 3D input [N, S, D].
        Description: 3D tensor without batch dimension, head on mp.
        Expectation: Output preserves 3D sharding structure.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",))
        q_layout = _build_layout(mesh, (Shard(0),), 3)
        k_layout = _build_layout(mesh, (Shard(0),), 3)
        v_layout = _build_layout(mesh, (Shard(0),), 3)

        self._run_scenario(q_layout, k_layout, v_layout, (0, -1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_3d_input_sp_mp_15(self, mock_platform):
        """
        Feature: Layout inference for 3D input with SP + MP.
        Description: 3D tensor [N, S, D] with head on mp and seq on sp.
        Expectation: Output preserves head and sequence sharding.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
        q_layout = _build_layout(mesh, (Shard(1), Shard(0)), 3)
        k_layout = _build_layout(mesh, (Replicate(), Shard(0)), 3)
        v_layout = _build_layout(mesh, (Replicate(), Shard(0)), 3)

        self._run_scenario(q_layout, k_layout, v_layout, (0, 1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_output_matches_query_layout_16(self, mock_platform):
        """
        Feature: Output layout always matches query layout.
        Description: Verify that the output tensor_map is identical to query tensor_map.
        Expectation: Output tensor_map equals query tensor_map.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
        k_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 4)

        output_layout = op.infer_layout((q_layout, k_layout, v_layout), [])

        assert output_layout.tensor_map == q_layout.tensor_map, (
            "Output tensor_map should match query tensor_map"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_sp_only_on_query_17(self, mock_platform):
        """
        Feature: Layout inference with SP only applied to query.
        Description: BNSD with query seq on sp, KV fully replicated.
        Expectation: Output preserves query sequence sharding.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
        q_layout = _build_layout(mesh, (Shard(2),), 4)
        k_layout = _build_layout(mesh, (Replicate(),), 4)
        v_layout = _build_layout(mesh, (Replicate(),), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (-1, -1, 0, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_sp_mp_2d_mesh_18(self, mock_platform):
        """
        Feature: Layout inference with SP + MP on a 2D mesh.
        Description: BNSD with query seq on sp, head on mp, KV only mp.
        Expectation: Output preserves both sp and mp sharding.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
        q_layout = _build_layout(mesh, (Shard(2), Shard(1)), 4)
        k_layout = _build_layout(mesh, (Replicate(), Shard(1)), 4)
        v_layout = _build_layout(mesh, (Replicate(), Shard(1)), 4)

        self._run_scenario(q_layout, k_layout, v_layout, (-1, 0, 1, -1))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_error_query_layout_none_19(self, mock_platform):
        """
        Feature: Error handling when query layout is None.
        Description: Pass None as query layout.
        Expectation: ValueError raised.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
        k_layout = _build_layout(mesh, (Replicate(),), 4)
        v_layout = _build_layout(mesh, (Replicate(),), 4)

        with self.assertRaisesRegex(ValueError, "Query layout cannot be None"):
            op.infer_layout((None, k_layout, v_layout), [])


if __name__ == "__main__":
    unittest.main()
