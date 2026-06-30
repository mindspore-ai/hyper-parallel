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

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE, Layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_scaled_dot_product_attention import (
    ScaledDotProductAttentionDistributedOp,
)
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
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clean up after each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

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
        cache_values = [q_layout, k_layout, v_layout]
        output_layouts, extra_info = op.infer_layout(cache_values)
        output_layout = output_layouts[0]
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
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
    def test_sdpa_kv_different_layout_raises_7(self, mock_platform):
        """
        Feature: Layout inference rejects different KV sharding.
        Description: Key has mp sharding but Value does not.
        Expectation: ValueError raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2), Replicate(), Replicate()), 4)
        k_layout = _build_layout(mesh, (Shard(0), Shard(2), Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate(), Replicate()), 4)

        with self.assertRaisesRegex(ValueError, "Key and Value must have identical"):
            op.infer_layout([q_layout, k_layout, v_layout])

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

        output_layouts, _ = op.infer_layout([q_layout, k_layout, v_layout])
        output_layout = output_layouts[0]

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

        output_layouts, _ = op.infer_layout([q_layout, k_layout, v_layout])
        output_layout = output_layouts[0]

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
        Description: Verify that the output tensor_map is identical to query tensor_map
            when Q, K, V have consistent sharding.
        Expectation: Output tensor_map equals query tensor_map.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
        k_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 4)
        v_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 4)

        output_layouts, _ = op.infer_layout([q_layout, k_layout, v_layout])

        assert output_layouts[0].tensor_map == q_layout.tensor_map, (
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

        with self.assertRaisesRegex(ValueError, "query layout should not be None"):
            op.infer_layout([None, k_layout, v_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_kv_seq_sharding_not_supported_20(self, mock_platform):
        """
        Feature: Error handling when KV sequence is sharded alongside query sequence.
        Description: Query seq sharded on sp, KV seq also sharded on sp.
        Expectation: NotImplementedError raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Replicate(), Shard(2), Replicate())
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        with self.assertRaisesRegex(NotImplementedError, "KV sequence sharding"):
            op.infer_layout([q_layout, k_layout, v_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_kv_seq_sharding_query_replicated_21(self, mock_platform):
        """
        Feature: Error handling when KV seq is sharded while query seq is replicated.
        Description: Query replicated, KV seq sharded on sp.
        Expectation: NotImplementedError raised (KV seq sharding unsupported unconditionally).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Shard(2), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Shard(2), Replicate()), 4)

        with self.assertRaisesRegex(NotImplementedError, "KV sequence sharding"):
            op.infer_layout([q_layout, k_layout, v_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_partial_key_raises_22(self, mock_platform):
        """
        Feature: Error handling when Key has Partial status.
        Description: Query and Value are replicated, Key has Partial(sum).
        Expectation: ValueError raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        k_layout.set_partial_by_dev_axis("dp", "sum")
        v_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)

        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout([q_layout, k_layout, v_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_mesh_mismatch_raises_23(self, mock_platform):
        """
        Feature: Error handling when K/V mesh differs from Query mesh.
        Description: Query on (2,2,2) mesh, Key on (4,2) mesh.
        Expectation: ValueError raised.
        """
        mesh_q = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh_q, (Replicate(), Replicate(), Replicate()), 4)

        self._setup_mock_platform(mock_platform, world_size=8)
        mesh_kv = init_device_mesh(
            device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp")
        )
        k_layout = _build_layout(mesh_kv, (Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh_kv, (Replicate(), Replicate()), 4)

        with self.assertRaisesRegex(ValueError, "mesh must match"):
            op.infer_layout([q_layout, k_layout, v_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_batch_sharded_q_plain_kv_raises_24(self, mock_platform):
        """
        Feature: Mixed DTensor/plain Tensor validation.
        Description: DTensor Q with batch sharding, plain K/V.
        Expectation: ValueError raised (batch/head must not be sharded with plain K/V).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 4)
        with self.assertRaisesRegex(ValueError, "batch and head dimensions must not be sharded"):
            op.infer_layout([q_layout, None, None])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_head_sharded_q_plain_kv_raises_25(self, mock_platform):
        """
        Feature: Mixed DTensor/plain Tensor validation.
        Description: DTensor Q with head sharding, plain K/V.
        Expectation: ValueError raised (batch/head must not be sharded with plain K/V).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 4)
        with self.assertRaisesRegex(ValueError, "batch and head dimensions must not be sharded"):
            op.infer_layout([q_layout, None, None])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_seq_sharded_q_plain_kv_success_26(self, mock_platform):
        """
        Feature: Mixed DTensor/plain Tensor validation.
        Description: DTensor Q with seq sharding (SP), plain K/V.
        Expectation: infer_layout succeeds, output preserves seq sharding.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Shard(2), Replicate()), 4)
        output_layouts, extra_info = op.infer_layout([q_layout, None, None])
        output_layout = output_layouts[0]
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        assert output_layout.tensor_map == (-1, -1, 1, -1), (
            f"Expected (-1, -1, 1, -1), got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_replicated_q_plain_kv_success_27(self, mock_platform):
        """
        Feature: Mixed DTensor/plain Tensor validation.
        Description: DTensor Q fully replicated, plain K/V.
        Expectation: infer_layout succeeds, output is replicated.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        output_layouts, extra_info = op.infer_layout([q_layout, None, None])
        output_layout = output_layouts[0]
        assert extra_info is None, f"extra_info should be None, got {extra_info}"
        assert output_layout.tensor_map == (-1, -1, -1, -1), (
            f"Expected (-1, -1, -1, -1), got {output_layout.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_asymmetric_kv_raises_28(self, mock_platform):
        """
        Feature: Mixed DTensor/plain Tensor validation.
        Description: K is DTensor, V is plain Tensor — asymmetric K/V.
        Expectation: ValueError raised (K/V must both be DTensor or both plain).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        with self.assertRaisesRegex(ValueError, "Key and Value must both be DTensors"):
            op.infer_layout([q_layout, k_layout, None])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_asymmetric_kv_reverse_raises_29(self, mock_platform):
        """
        Feature: Mixed DTensor/plain Tensor validation.
        Description: K is plain Tensor, V is DTensor — asymmetric K/V.
        Expectation: ValueError raised (K/V must both be DTensor or both plain).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        with self.assertRaisesRegex(ValueError, "Key and Value must both be DTensors"):
            op.infer_layout([q_layout, None, v_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_ndim_mismatch_raises_30(self, mock_platform):
        """
        Feature: ndim validation.
        Description: Q and V are 4D, but K is 3D.
        Expectation: ValueError raised (Q/K/V must have the same rank).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
        v_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
        with self.assertRaisesRegex(ValueError, "must have the same rank"):
            op.infer_layout([q_layout, k_layout, v_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_ndim_invalid_raises_31(self, mock_platform):
        """
        Feature: ndim validation.
        Description: Q is 2D (invalid for SDPA).
        Expectation: ValueError raised (only 3D or 4D supported).
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
        q_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        with self.assertRaisesRegex(ValueError, "only 3D or 4D inputs are supported"):
            op.infer_layout([q_layout, q_layout, q_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sdpa_dim_sharding_not_supported_32(self, mock_platform):
        """
        Feature: D-dimension sharding prohibition.
        Description: Q with the last embedding dimension sharded.
        Expectation: NotImplementedError raised.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(3)), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(3)), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(3)), 4)
        with self.assertRaisesRegex(NotImplementedError, "sharding the last embedding dimension"):
            op.infer_layout([q_layout, k_layout, v_layout])


class TestSdpaHelperMethods(unittest.TestCase):
    """
    Feature: ScaledDotProductAttentionDistributedOp helper methods
    Description: Test _normalize_dim_map, _get_dims, _get_dim_split_num,
                 _get_split_info, _validate_sharding_consistency, and get_expand_impl.
    Expectation: Correct values and exceptions for each path.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )

    def _make_2x4_mesh(self, mock_platform):
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )

    def test_normalize_dim_map_none_returns_string_none(self):
        """_normalize_dim_map(None) returns the string 'None'."""
        result = ScaledDotProductAttentionDistributedOp._normalize_dim_map(None)
        self.assertEqual(result, "None")

    def test_normalize_dim_map_non_none_returns_value(self):
        """_normalize_dim_map with non-None value returns the value unchanged."""
        self.assertEqual(
            ScaledDotProductAttentionDistributedOp._normalize_dim_map("dp"), "dp"
        )
        self.assertEqual(
            ScaledDotProductAttentionDistributedOp._normalize_dim_map(0), 0
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_dims_3d_tensor(self, mock_platform):
        """_get_dims returns {head:0, seq:1, dim:2} for a 3D layout."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",)
        )
        layout = _build_layout(mesh, (Replicate(),), 3)
        dims = op._get_dims(layout)
        self.assertEqual(dims["head"], 0)
        self.assertEqual(dims["seq"], 1)
        self.assertEqual(dims["dim"], 2)
        self.assertNotIn("batch", dims)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_dims_4d_tensor(self, mock_platform):
        """_get_dims returns {batch:0, head:1, seq:2, dim:3} for a 4D layout."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",)
        )
        q_layout = _build_layout(mesh, (Replicate(),), 4)
        dims = op._get_dims(q_layout)
        self.assertEqual(dims["batch"], 0)
        self.assertEqual(dims["head"], 1)
        self.assertEqual(dims["seq"], 2)
        self.assertEqual(dims["dim"], 3)

    def test_get_dims_non_3d_defaults_to_4d(self):
        """_get_dims returns 4D mapping when tensor_map is not length 3."""
        from unittest.mock import MagicMock
        mock_layout = MagicMock()
        mock_layout.tensor_map = (None, None, None, None, None)
        dims = op._get_dims(mock_layout)
        self.assertEqual(dims, {"batch": 0, "head": 1, "seq": 2, "dim": 3})
        mock_layout.tensor_map = None
        dims = op._get_dims(mock_layout)
        self.assertEqual(dims, {"batch": 0, "head": 1, "seq": 2, "dim": 3})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_dim_split_num_no_alias_tensor_map(self, mock_platform):
        """_get_dim_split_num returns 1 when layout has no alias_tensor_map."""
        from unittest.mock import MagicMock
        mock_layout = MagicMock()
        del mock_layout.alias_tensor_map
        result = op._get_dim_split_num(mock_layout, 0)
        self.assertEqual(result, 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_dim_split_num_dim_out_of_range_returns_one(self, mock_platform):
        """_get_dim_split_num returns 1 when dim_idx >= len(alias_tensor_map)."""
        from unittest.mock import MagicMock
        mock_layout = MagicMock()
        mock_layout.alias_tensor_map = ("dp",)
        result = op._get_dim_split_num(mock_layout, 5)
        self.assertEqual(result, 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_dim_split_num_none_mapping_returns_one(self, mock_platform):
        """_get_dim_split_num returns 1 for a 'None' mapped dimension."""
        from unittest.mock import MagicMock
        mock_layout = MagicMock()
        mock_layout.alias_tensor_map = ("None", "dp")
        result = op._get_dim_split_num(mock_layout, 0)
        self.assertEqual(result, 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_dim_split_num_string_mapping_returns_device_count(self, mock_platform):
        """_get_dim_split_num returns device count for a string-mapped dimension."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )
        layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
        result = op._get_dim_split_num(layout, 0)
        self.assertEqual(result, 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_split_info_all_replicated(self, mock_platform):
        """_get_split_info returns 1 for all dims when layout is fully replicated."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )
        layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        dims = {"batch": 0, "head": 1, "seq": 2}
        split_info = op._get_split_info(layout, dims)
        self.assertEqual(split_info["batch"], 1)
        self.assertEqual(split_info["head"], 1)
        self.assertEqual(split_info["seq"], 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_sharding_consistency_none_key_returns(self, mock_platform):
        """_validate_sharding_consistency does nothing when key_layout is None."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )
        q_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
        op._validate_sharding_consistency(q_layout, None, {"batch": 0, "head": 1, "dim": 3})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_sharding_consistency_mismatch_raises(self, mock_platform):
        """_validate_sharding_consistency raises ValueError for batch mismatch."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )
        q_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        dims = {"batch": 0, "head": 1, "seq": 2, "dim": 3}
        with self.assertRaisesRegex(ValueError, "identical batch sharding"):
            op._validate_sharding_consistency(q_layout, k_layout, dims)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_expand_impl_none_query_returns_none(self, mock_platform):
        """get_expand_impl with None query layout returns None."""
        result = op.get_expand_impl(None, None, [None])
        self.assertIsNone(result)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_key_value_mismatch_raises(self, mock_platform):
        """infer_layout raises ValueError when Key and Value have different tensor_maps."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )
        q_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
        k_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Shard(1)), 4)
        with self.assertRaisesRegex(ValueError, "Key and Value must have identical"):
            op.infer_layout([q_layout, k_layout, v_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_expand_impl_returns_callable(self, mock_platform):
        """get_expand_impl with valid layouts returns a callable expanded_impl."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )
        q_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        impl = op.get_expand_impl(lambda *a, **k: "result", None, [q_layout, k_layout, v_layout])
        self.assertTrue(callable(impl))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_get_expand_impl_callable_no_sp_calls_func(self, mock_platform):
        """expanded_impl with no sequence parallelism calls func directly."""
        from unittest.mock import MagicMock
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )
        q_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)

        func_calls = []

        def mock_func(*args, **kwargs):
            func_calls.append(kwargs)
            return "attention_output"

        impl = op.get_expand_impl(mock_func, None, [q_layout, k_layout, v_layout])
        query_mock = MagicMock()
        key_mock = MagicMock()
        value_mock = MagicMock()
        result = impl(query_mock, key_mock, value_mock)
        self.assertEqual(result, "attention_output")
        self.assertEqual(len(func_calls), 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_q_k_sharding_mismatch_raises(self, mock_platform):
        """infer_layout raises ValueError when Q/K sharding mismatch on non-seq dims."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp")
        )
        q_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)

        with self.assertRaisesRegex(ValueError, "identical batch sharding"):
            op.infer_layout([q_layout, k_layout, v_layout])

    def test_get_split_info_no_alias_tensor_map_returns_defaults(self):
        """_get_split_info returns all-1 defaults when alias_tensor_map is None."""
        from unittest.mock import MagicMock
        mock_layout = MagicMock()
        mock_layout.alias_tensor_map = None
        dims = {"batch": 0, "head": 1, "seq": 2}
        result = op._get_split_info(mock_layout, dims)
        self.assertEqual(result, {"batch": 1, "head": 1, "seq": 1})

    def test_build_causal_mask_for_chunk_shape_and_values(self):
        """_build_causal_mask_for_chunk returns correct shape and causal pattern."""
        import torch
        result = ScaledDotProductAttentionDistributedOp._build_causal_mask_for_chunk(
            local_q_len=4, kv_len=8, split_id=1, device=torch.device("cpu")
        )
        self.assertEqual(result.shape, (4, 8))
        self.assertTrue(result[0, 3])
        self.assertFalse(result[0, 5])

    def test_adjust_attn_mask_causal_split_id_zero_preserves_causal(self):
        """_adjust_attn_mask_for_sp with is_causal=True and split_id=0 keeps is_causal=True."""
        import torch
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        adj_mask, adj_causal, adj_key, adj_value = op._adjust_attn_mask_for_sp(
            None, True, key, value, 0, 4, 2, 8, 2, torch.device("cpu")
        )
        self.assertIsNone(adj_mask)
        self.assertTrue(adj_causal)
        self.assertEqual(adj_key.shape[2], 4)

    def test_adjust_attn_mask_causal_split_id_nonzero_builds_mask(self):
        """_adjust_attn_mask_for_sp with is_causal=True and split_id=1 builds causal mask."""
        import torch
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        adj_mask, adj_causal, adj_key, adj_value = op._adjust_attn_mask_for_sp(
            None, True, key, value, 1, 4, 2, 8, 2, torch.device("cpu")
        )
        self.assertIsNotNone(adj_mask)
        self.assertFalse(adj_causal)

    def test_adjust_attn_mask_explicit_mask_2d_sliced(self):
        """_adjust_attn_mask_for_sp with explicit 2D attn_mask slices local Q range."""
        import torch
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        global_q_len = 8
        local_q_len = 4
        attn_mask = torch.ones(global_q_len, 8)
        adj_mask, adj_causal, adj_key, adj_value = op._adjust_attn_mask_for_sp(
            attn_mask, False, key, value, 0, local_q_len, 2, 8, 2, torch.device("cpu")
        )
        self.assertEqual(adj_mask.shape[0], local_q_len)

    def test_adjust_attn_mask_explicit_mask_4d_sliced(self):
        """_adjust_attn_mask_for_sp with explicit 4D attn_mask slices local Q range."""
        import torch
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        global_q_len = 8
        local_q_len = 4
        attn_mask = torch.ones(2, 4, global_q_len, 8)
        adj_mask, adj_causal, adj_key, adj_value = op._adjust_attn_mask_for_sp(
            attn_mask, False, key, value, 1, local_q_len, 2, 8, 2, torch.device("cpu")
        )
        self.assertEqual(adj_mask.shape[2], local_q_len)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.ops.parallel_scaled_dot_product_attention.platform")
    def test_expanded_impl_with_sequence_parallelism(self, mock_sdpa_platform, mock_mesh_platform):
        """expanded_impl with SP active calls _adjust_attn_mask_for_sp and then func."""
        import torch
        self._setup_mock_platform(mock_mesh_platform, world_size=8)
        mock_sdpa_platform.get_rank.return_value = 0
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp")
        )
        q_layout = _build_layout(mesh, (Shard(2), Replicate()), 4)
        k_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)
        v_layout = _build_layout(mesh, (Replicate(), Replicate()), 4)

        func_calls = []

        def mock_func(*args, **kwargs):
            func_calls.append(1)
            return torch.zeros(2, 4, 2, 64)

        impl = op.get_expand_impl(mock_func, None, [q_layout, k_layout, v_layout])
        query = torch.randn(2, 4, 2, 64)
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        impl(query, key, value)
        self.assertEqual(len(func_calls), 1)

    def test_validate_sharding_consistency_none_alias_tensor_map_returns(self):
        """_validate_sharding_consistency returns silently when alias_tensor_map is None."""
        from unittest.mock import MagicMock
        mock_q = MagicMock()
        mock_q.alias_tensor_map = None
        mock_k = MagicMock()
        mock_k.alias_tensor_map = ("something",)
        op._validate_sharding_consistency(mock_q, mock_k, {"batch": 0})

    def test_validate_sharding_consistency_both_none_alias_tensor_map_returns(self):
        """_validate_sharding_consistency skips when both alias_tensor_maps are None."""
        from unittest.mock import MagicMock
        mock_q = MagicMock()
        mock_q.alias_tensor_map = None
        mock_k = MagicMock()
        mock_k.alias_tensor_map = None
        op._validate_sharding_consistency(mock_q, mock_k, {"batch": 0})


if __name__ == "__main__":
    unittest.main()
