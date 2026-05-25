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
"""parallel_ms_flash_attention_score unit test"""
import os
import unittest
from unittest.mock import patch
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_ms_flash_attention_score import (
    FlashAttentionScoreDistributedOp,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = FlashAttentionScoreDistributedOp("FlashAttentionScore")

ATTENTION_OUT_IDX = 3
SOFTMAX_MAX_IDX = 0
SOFTMAX_SUM_IDX = 1
SOFTMAX_OUT_IDX = 2


class TestMsFlashAttentionScore(unittest.TestCase):
    """Unit tests for FlashAttentionScoreDistributedOp."""
    def setUp(self):
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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

    def _make_8_mesh(self, mock_platform, mesh_dim_name="dp"):
        """Set up mock and return a standard 8-element mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=(mesh_dim_name,))

    def _make_4_mesh(self, mock_platform, mesh_dim_name="sp"):
        """Set up mock and return a standard 4-element mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=(mesh_dim_name,))

    def _make_2x4_mesh(self, mock_platform, mesh_dim_names=("dp", "sp")):
        """Set up mock and return a standard 2x4 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=mesh_dim_names)

    def _make_2x2x2x4_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2x4 (dp, sp, mp, pp) mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=32)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2, 4), mesh_dim_names=("dp", "sp", "mp", "pp"))

    def _make_extra_args(self, head_num=16, input_layout="BSH", sparse_mode=0):
        """Build extra_args matching MS pyboost scalar parameter order."""
        return [head_num, 1.0, 0.125, 2147483647, 2147483647, 0,
                input_layout, sparse_mode]

    def _run_scenario(self, mock_platform, q_placements, k_placements, v_placements,
                      ndim, expected_out_map, extra_args, expect_expand_impl=True):
        """Infer layout and verify attention output tensor_map and get_expand_impl."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, q_placements, ndim)
        k_layout = _build_layout(mesh, k_placements, ndim)
        v_layout = _build_layout(mesh, v_placements, ndim)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), extra_args)
        attention_out_layout = output_layouts[ATTENTION_OUT_IDX]
        assert attention_out_layout.tensor_map == expected_out_map, (
            f"Expected {expected_out_map}, got {attention_out_layout.tensor_map}"
        )

        impl = op.get_expand_impl(None, output_layouts, (q_layout, k_layout, v_layout), extra_args)
        if expect_expand_impl:
            assert callable(impl), f"Expected callable, got {type(impl)}"
        else:
            assert impl is None, f"Expected None, got {type(impl)}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_no_parallel_1(self, mock_platform):
        """
        Feature: Layout inference with no parallelism.
        Description: All dimensions replicated on a 3D mesh.
        Expectation: Output tensor_map is all -1.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Replicate(), Replicate(), Replicate())
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (-1, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_data_parallel_2(self, mock_platform):
        """
        Feature: Layout inference with data parallelism.
        Description: BSH batch dimension sharded on dp axis.
        Expectation: Output batch dimension remains sharded on dp.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Shard(0), Replicate(), Replicate())
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_head_parallel_3(self, mock_platform):
        """
        Feature: Layout inference with head parallelism.
        Description: BSH hidden dimension sharded on mp axis.
        Expectation: Output hidden dimension remains sharded on mp.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Replicate(), Replicate(), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (-1, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sequence_parallel_4(self, mock_platform):
        """
        Feature: Layout inference with sequence parallelism.
        Description: BSH query sequence sharded on sp axis, KV not sharded.
        Expectation: Output sequence dimension sharded on sp.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_placements = (Replicate(), Shard(1), Replicate())
        kv_placements = (Replicate(), Replicate(), Replicate())
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (-1, 1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_hybrid_dp_mp_5(self, mock_platform):
        """
        Feature: Layout inference with hybrid DP + MP.
        Description: BSH batch on dp, hidden on mp.
        Expectation: Both dimensions remain sharded.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Shard(0), Replicate(), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_hybrid_dp_sp_mp_6(self, mock_platform):
        """
        Feature: Layout inference with full hybrid DP + SP + MP.
        Description: BSH with all three parallel strategies, KV only dp+mp.
        Expectation: Output preserves dp+sp+mp sharding from query.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_placements = (Shard(0), Shard(1), Shard(2))
        kv_placements = (Shard(0), Replicate(), Shard(2))
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, 1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_kv_different_layout_7(self, mock_platform):
        """
        Feature: Error handling for K/V inconsistent tensor_map in get_expand_impl.
        Description: Key and Value have different tensor_map (sp sharding mismatch).
        Expectation: infer_layout succeeds, get_expand_impl raises ValueError.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_placements = (Shard(0), Shard(1), Shard(2))
        k_placements = (Shard(0), Shard(1), Shard(2))
        v_placements = (Shard(0), Replicate(), Shard(2))
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, k_placements, 3)
        v_layout = _build_layout(mesh, v_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, 1, 0)

        with self.assertRaisesRegex(ValueError, "Key and Value must have identical sharding"):
            op.get_expand_impl(None, output_layouts, (q_layout, k_layout, v_layout), self._make_extra_args())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_bnsd_layout_8(self, mock_platform):
        """
        Feature: Layout inference for BNSD input layout.
        Description: 4D tensor with dp on batch, mp on head.
        Expectation: Output preserves 4D sharding structure.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(1))
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="BNSD"))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (1, 0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sbh_layout_9(self, mock_platform):
        """
        Feature: Layout inference for SBH input layout.
        Description: Sequence-first 3D layout with head parallelism on mp.
        Expectation: Output preserves SBH dimension order with mp sharding.
        """
        mesh = self._make_8_mesh(mock_platform, "mp")
        placements = (Shard(2),)
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="SBH"))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (-1, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_bsnd_layout_10(self, mock_platform):
        """
        Feature: Layout inference for BSND input layout.
        Description: 4D tensor with dp on batch, mp on head (dim 2).
        Expectation: Output preserves 4D sharding structure.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="BSND"))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (1, -1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sparse_mode_0_11(self, mock_platform):
        """
        Feature: Layout inference with sparse_mode=0.
        Description: defaultMask mode passed via extra_args.
        Expectation: Layout inference succeeds normally.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Shard(0), Replicate(), Replicate())
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(sparse_mode=0))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sparse_mode_2_sp_12(self, mock_platform):
        """
        Feature: Layout inference with sparse_mode=2 and sequence parallelism.
        Description: leftUpCausal mode with query seq sharded on sp.
        Expectation: Layout inference succeeds with seq sharding preserved.
        """
        mesh = self._make_8_mesh(mock_platform, "sp")
        q_placements = (Shard(1),)
        kv_placements = (Replicate(),)
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(sparse_mode=2))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (-1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sparse_mode_3_sp_13(self, mock_platform):
        """
        Feature: Layout inference with sparse_mode=3 and SP.
        Description: rightDownCausal with query seq sharded on sp.
        Expectation: Layout inference succeeds with seq sharding preserved.
        """
        mesh = self._make_4_mesh(mock_platform, "sp")
        q_placements = (Shard(1),)
        kv_placements = (Replicate(),)
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(sparse_mode=3))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (-1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sparse_mode_4_band_14(self, mock_platform):
        """
        Feature: Layout inference with sparse_mode=4 (band) and DP.
        Description: Band mask mode with data parallelism only.
        Expectation: Layout inference succeeds normally.
        """
        mesh = self._make_8_mesh(mock_platform, "dp")
        placements = (Shard(0),)
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(sparse_mode=4))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_count_15(self, mock_platform):
        """
        Feature: Layout inference returns 4 output layouts.
        Description: Q/K/V all have the same dp+mp sharding.
        Expectation: Returns exactly 4 layouts; attention_out matches query.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Shard(0), Replicate(), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())

        assert len(output_layouts) == 4, f"Expected 4 output layouts, got {len(output_layouts)}"
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_bsh_16(self, mock_platform):
        """
        Feature: All 4 output layouts for BSH with dp+mp.
        Description: Verify attention_out, softmax_max, softmax_sum, and softmax_out layouts.
        Expectation: attention_out matches query; softmax_max/sum are 4D; softmax_out is empty.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())

        softmax_max = output_layouts[SOFTMAX_MAX_IDX]
        softmax_sum = output_layouts[SOFTMAX_SUM_IDX]
        softmax_out = output_layouts[SOFTMAX_OUT_IDX]
        attention_out = output_layouts[ATTENTION_OUT_IDX]

        assert attention_out.tensor_map == q_layout.tensor_map
        assert attention_out.tensor_map == (1, -1, 0)
        assert len(softmax_max.tensor_map) == 4
        assert len(softmax_sum.tensor_map) == 4
        assert softmax_max.tensor_map == (1, 0, -1, -1)
        assert softmax_sum.tensor_map == (1, 0, -1, -1)
        assert softmax_out.tensor_map == ()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_multi_dimensional_mesh_17(self, mock_platform):
        """
        Feature: Layout inference with 3D device mesh.
        Description: BSH with dp+sp+mp on a (2,2,2) mesh, all inputs fully sharded.
        Expectation: Output preserves all three sharding axes.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Shard(0), Shard(1), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, 1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_single_device_18(self, mock_platform):
        """
        Feature: Layout inference with single device mesh.
        Description: Mesh size is 1, all dimensions replicated.
        Expectation: Output tensor_map is all -1.
        """
        self._setup_mock_platform(mock_platform, world_size=1)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(1,), mesh_dim_names=("dp",))
        placements = (Replicate(),)
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (-1, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_large_world_size_19(self, mock_platform):
        """
        Feature: Layout inference with large world size.
        Description: 32 devices on a 4D mesh (2,2,2,4) with dp+sp+mp+pp axes.
        Expectation: Output scales correctly with 4D mesh.
        """
        mesh = self._make_2x2x2x4_mesh(mock_platform)
        q_placements = (Shard(0), Shard(1), Shard(2), Replicate())
        kv_placements = (Shard(0), Replicate(), Shard(2), Replicate())
        q_layout = _build_layout(mesh, q_placements, 4)
        k_layout = _build_layout(mesh, kv_placements, 4)
        v_layout = _build_layout(mesh, kv_placements, 4)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(head_num=32))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (3, 2, 1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_bnsd_20(self, mock_platform):
        """
        Feature: Softmax output layouts for BNSD input.
        Description: BNSD with dp on batch, mp on head.
        Expectation: softmax_max/sum tensor_map is (dp, mp, -1, -1).
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(1))
        q_layout = _build_layout(mesh, placements, 4)
        k_layout = _build_layout(mesh, placements, 4)
        v_layout = _build_layout(mesh, placements, 4)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="BNSD"))

        attention_out = output_layouts[ATTENTION_OUT_IDX]
        softmax_max = output_layouts[SOFTMAX_MAX_IDX]
        softmax_sum = output_layouts[SOFTMAX_SUM_IDX]

        assert attention_out.tensor_map == (1, 0, -1, -1)
        assert softmax_max.tensor_map == (1, 0, -1, -1)
        assert softmax_sum.tensor_map == (1, 0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_sbh_21(self, mock_platform):
        """
        Feature: Softmax output layouts for SBH input.
        Description: SBH with dp on batch (dim 1), mp on hidden (dim 2).
        Expectation: softmax_max/sum tensor_map is (dp, mp, -1, -1).
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(1), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="SBH"))

        attention_out = output_layouts[ATTENTION_OUT_IDX]
        softmax_max = output_layouts[SOFTMAX_MAX_IDX]
        softmax_sum = output_layouts[SOFTMAX_SUM_IDX]

        assert attention_out.tensor_map == (-1, 1, 0)
        assert softmax_max.tensor_map == (1, 0, -1, -1)
        assert softmax_sum.tensor_map == (1, 0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_tnd_22(self, mock_platform):
        """
        Feature: Softmax output layouts for TND input.
        Description: TND with dp on token dim (dim 0), mp on head dim (dim 1).
        Expectation: softmax_max/sum tensor_map is (dp, mp, -1, -1).
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(1))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="TND"))

        attention_out = output_layouts[ATTENTION_OUT_IDX]
        softmax_max = output_layouts[SOFTMAX_MAX_IDX]
        softmax_sum = output_layouts[SOFTMAX_SUM_IDX]

        assert attention_out.tensor_map == (1, 0, -1)
        assert softmax_max.tensor_map == (1, 0, -1, -1)
        assert softmax_sum.tensor_map == (1, 0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_mixed_parallel_23(self, mock_platform):
        """
        Feature: Softmax output layouts with full DP + SP + MP parallelism.
        Description: BSH with dp on batch, sp on seq, mp on hidden.
        Expectation: softmax_max/sum tensor_map is (dp, mp, sp, -1).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_placements = (Shard(0), Shard(1), Shard(2))
        kv_placements = (Shard(0), Replicate(), Shard(2))
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())

        attention_out = output_layouts[ATTENTION_OUT_IDX]
        softmax_max = output_layouts[SOFTMAX_MAX_IDX]
        softmax_sum = output_layouts[SOFTMAX_SUM_IDX]

        assert attention_out.tensor_map == (2, 1, 0)
        assert softmax_max.tensor_map == (2, 0, 1, -1)
        assert softmax_sum.tensor_map == (2, 0, 1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_softmax_no_sharding_24(self, mock_platform):
        """
        Feature: Softmax output layouts with no sharding.
        Description: All dimensions replicated on a single-axis mesh.
        Expectation: softmax_max/sum tensor_map is all -1.
        """
        mesh = self._make_8_mesh(mock_platform, "dp")
        placements = (Replicate(),)
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())

        softmax_max = output_layouts[SOFTMAX_MAX_IDX]
        softmax_sum = output_layouts[SOFTMAX_SUM_IDX]

        assert softmax_max.tensor_map == (-1, -1, -1, -1)
        assert softmax_sum.tensor_map == (-1, -1, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_tnd_sp_25(self, mock_platform):
        """
        Feature: Layout inference for TND with sequence parallelism.
        Description: TND layout with T (token) dim sharded on sp, N (head) dim sharded on mp.
        Expectation: Output preserves T-dim sharding on sp and N-dim sharding on mp.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
        q_placements = (Shard(0), Shard(1))
        kv_placements = (Replicate(), Shard(1))
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="TND"))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_tnd_softmax_output_sp_26(self, mock_platform):
        """
        Feature: Softmax output layouts for TND with sequence parallelism.
        Description: TND with sp on T (dim 0), mp on N (dim 1).
        Expectation: softmax_max/sum preserve sp and mp sharding.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
        q_placements = (Shard(0), Shard(1))
        kv_placements = (Replicate(), Shard(1))
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="TND"))

        attention_out = output_layouts[ATTENTION_OUT_IDX]
        softmax_max = output_layouts[SOFTMAX_MAX_IDX]
        softmax_sum = output_layouts[SOFTMAX_SUM_IDX]

        assert attention_out.tensor_map == (1, 0, -1)
        assert len(softmax_max.tensor_map) == 4
        assert len(softmax_sum.tensor_map) == 4

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_bsnd_sp_27(self, mock_platform):
        """
        Feature: Layout inference for BSND with sequence parallelism.
        Description: 4D BSND tensor with dp on batch (dim 0), sp on seq (dim 1), mp on head (dim 2).
        Expectation: Output preserves all three sharding axes on 4D tensor.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_placements = (Shard(0), Shard(1), Shard(2))
        kv_placements = (Shard(0), Replicate(), Shard(2))
        q_layout = _build_layout(mesh, q_placements, 4)
        k_layout = _build_layout(mesh, kv_placements, 4)
        v_layout = _build_layout(mesh, kv_placements, 4)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="BSND"))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, 1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_bnsd_sp_28(self, mock_platform):
        """
        Feature: Layout inference for BNSD with sequence parallelism.
        Description: 4D BNSD tensor with dp on batch (dim 0), mp on head (dim 1), sp on seq (dim 2).
        Expectation: Output preserves dp+mp+sp sharding on 4D tensor.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_placements = (Shard(0), Shard(2), Shard(1))
        kv_placements = (Shard(0), Replicate(), Shard(1))
        q_layout = _build_layout(mesh, q_placements, 4)
        k_layout = _build_layout(mesh, kv_placements, 4)
        v_layout = _build_layout(mesh, kv_placements, 4)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="BNSD"))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (2, 0, 1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_ring_attention_29(self, mock_platform):
        """
        Feature: Layout inference with both Q and KV sequence-sharded.
        Description: BSH with sp sharding on seq dim for Q, K, and V (ring attention).
        Expectation: Layout inference succeeds; output seq dim sharded on sp.
        """
        mesh = self._make_2x4_mesh(mock_platform, ("dp", "sp"))
        placements = (Shard(0), Shard(1))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_dp_only_single_axis_30(self, mock_platform):
        """
        Feature: Layout inference with pure DP on a single-axis mesh.
        Description: Only data parallelism, no model or sequence parallelism.
        Expectation: Only batch dimension sharded.
        """
        mesh = self._make_8_mesh(mock_platform, "dp")
        placements = (Shard(0),)
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_mp_only_single_axis_31(self, mock_platform):
        """
        Feature: Layout inference with pure MP on a single-axis mesh.
        Description: Only model parallelism on hidden dim, no DP or SP.
        Expectation: Only hidden dimension sharded.
        """
        mesh = self._make_8_mesh(mock_platform, "mp")
        placements = (Shard(2),)
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args())
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (-1, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sbh_sp_32(self, mock_platform):
        """
        Feature: Layout inference for SBH with sequence parallelism.
        Description: SBH layout with sp on seq (dim 0) and mp on hidden (dim 2).
        Expectation: Output preserves SBH ordering with sp and mp sharding.
        """
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
        q_placements = (Shard(0), Shard(2))
        kv_placements = (Replicate(), Shard(2))
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="SBH"))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (1, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_head_num_1_33(self, mock_platform):
        """
        Feature: Layout inference with head_num=1.
        Description: Single-head attention with mp sharding.
        Expectation: Layout inference succeeds; sharding recorded as specified.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(head_num=1))
        assert output_layouts[ATTENTION_OUT_IDX].tensor_map == (1, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_invalid_input_layout_34(self, mock_platform):
        """
        Feature: Graceful handling for unsupported input_layout string.
        Description: extra_args specifies an unsupported input_layout.
        Expectation: Layout inference completes without error; attention_out matches query.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="INVALID"))

        attention_out = output_layouts[ATTENTION_OUT_IDX]
        assert attention_out.tensor_map == q_layout.tensor_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_missing_extra_args_35(self, mock_platform):
        """
        Feature: Error handling for missing extra_args.
        Description: extra_args is empty, missing required head_num and input_layout.
        Expectation: Raises an appropriate error.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Replicate())
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        with self.assertRaises((IndexError, TypeError, ValueError, RuntimeError)):
            op.infer_layout((q_layout, k_layout, v_layout), [])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_ndim_mismatch_layout_36(self, mock_platform):
        """
        Feature: Graceful handling for tensor ndim vs input_layout mismatch.
        Description: 3D tensor with BNSD (4D) input_layout in extra_args.
        Expectation: Layout inference completes without error; output has 3D tensor_map.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_layouts = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="BNSD"))

        attention_out = output_layouts[ATTENTION_OUT_IDX]
        assert len(attention_out.tensor_map) == 3

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_integer_input_layout_37(self, mock_platform):
        """
        Feature: Layout inference with integer input_layout enum.
        Description: extra_args specifies input_layout as integer 0 (BSH).
        Expectation: Equivalent to passing "BSH" string.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        output_int = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout=0))
        output_str = op.infer_layout((q_layout, k_layout, v_layout), self._make_extra_args(input_layout="BSH"))

        assert output_int[ATTENTION_OUT_IDX].tensor_map == output_str[ATTENTION_OUT_IDX].tensor_map

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_query_layout_none_38(self, mock_platform):
        """
        Feature: get_expand_impl returns None when query layout is None.
        Description: Query layout is None, which causes get_expand_impl to return None.
        Expectation: infer_layout raises ValueError, get_expand_impl returns None.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Replicate())
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        with self.assertRaisesRegex(ValueError, "Query layout cannot be None"):
            op.infer_layout((None, k_layout, v_layout), self._make_extra_args())

        impl = op.get_expand_impl(None, None, (None, k_layout, v_layout), self._make_extra_args())
        assert impl is None


if __name__ == "__main__":
    unittest.main()
