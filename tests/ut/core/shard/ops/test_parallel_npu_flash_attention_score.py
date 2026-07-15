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
"""parallel_npu_fusion_attention unit test"""
import os
import unittest
import warnings
from unittest.mock import patch, MagicMock
import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_npu_flash_attention_score import (
    NPUFlashAttentionScoreDistributedOp,
    SPARSE_ALL_MASK,
    SPARSE_LEFT_UP_CAUSAL,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = NPUFlashAttentionScoreDistributedOp("npu_fusion_attention")


class TestParallelNpuFlashAttentionScore(unittest.TestCase):
    """Unit tests for NPUFlashAttentionScoreDistributedOp."""
    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
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

    def _run_scenario(self, mock_platform, q_placements, k_placements, v_placements,
                      ndim, expected_out_map, input_layout):
        """Infer layout and verify attention output tensor_map."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, q_placements, ndim)
        k_layout = _build_layout(mesh, k_placements, ndim)
        v_layout = _build_layout(mesh, v_placements, ndim)

        cache_values = [q_layout, k_layout, v_layout, input_layout]
        infer_result = op.infer_layout(cache_values)
        attention_out_layout = infer_result[0][0]
        assert attention_out_layout.to_dict()["tensor_map"] == expected_out_map, (
            f"Expected {expected_out_map}, got {attention_out_layout.to_dict()['tensor_map']}"
        )

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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (-1, -1, -1)

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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (2, -1, -1)

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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (-1, -1, 0)

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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (-1, 1, -1)



    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_kv_different_layout_7(self, mock_platform):
        """
        Feature: Layout inference with different KV sharding.
        Description: Key has sp sharding but Value does not.
        Expectation: Layout inference raises ValueError for K/V sharding mismatch.
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        q_placements = (Shard(0), Shard(1), Shard(2))
        k_placements = (Shard(0), Shard(1), Shard(2))
        v_placements = (Shard(0), Replicate(), Shard(2))
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, k_placements, 3)
        v_layout = _build_layout(mesh, v_placements, 3)

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        # K/V tensor_map mismatch is now checked in _validate_input_layouts during infer_layout
        with self.assertRaises(ValueError):
            op.infer_layout(cache_values)


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

        cache_values = [q_layout, k_layout, v_layout, "SBH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (-1, -1, 0)

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

        cache_values = [q_layout, k_layout, v_layout, "BSND"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (1, -1, 0, -1)


    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sparse_mode_2_15(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (-1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_multiple_inputs_same_layout_16(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator

        assert len(output_layouts) == 4, "Should return 4 output layouts"
        assert output_layouts[0].to_dict()["tensor_map"] == (2, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_bsh_17(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator

        attention_out, softmax_max, softmax_sum, softmax_out = output_layouts

        assert attention_out.to_dict()["tensor_map"] == q_layout.to_dict()["tensor_map"]
        assert attention_out.to_dict()["tensor_map"] == (1, -1, 0)
        assert len(softmax_max.to_dict()["tensor_map"]) == 4
        assert len(softmax_sum.to_dict()["tensor_map"]) == 4
        assert softmax_max.to_dict()["tensor_map"] == (1, 0, -1, -1)
        assert softmax_sum.to_dict()["tensor_map"] == (1, 0, -1, -1)
        assert softmax_out.to_dict()["tensor_map"] == ()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_multi_dimensional_mesh_18(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (2, 1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_single_device_19(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (-1, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_large_world_size_20(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (3, 2, 1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_bnsd_21(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BNSD"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator

        attention_out, softmax_max, softmax_sum, _ = output_layouts

        assert attention_out.to_dict()["tensor_map"] == (1, 0, -1, -1)
        assert softmax_max.to_dict()["tensor_map"] == (1, 0, -1, -1)
        assert softmax_sum.to_dict()["tensor_map"] == (1, 0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_sbh_22(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "SBH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator

        attention_out, softmax_max, softmax_sum, _ = output_layouts

        assert attention_out.to_dict()["tensor_map"] == (-1, 1, 0)
        assert softmax_max.to_dict()["tensor_map"] == (1, 0, -1, -1)
        assert softmax_sum.to_dict()["tensor_map"] == (1, 0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_tnd_23(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "TND"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator

        attention_out, softmax_max, softmax_sum, _ = output_layouts

        assert attention_out.to_dict()["tensor_map"] == (1, 0, -1)
        assert softmax_max.to_dict()["tensor_map"] == (1, 0, -1)
        assert softmax_sum.to_dict()["tensor_map"] == (1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_output_layouts_mixed_parallel_24(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator

        attention_out, softmax_max, softmax_sum, _ = output_layouts

        assert attention_out.to_dict()["tensor_map"] == (2, 1, 0)
        assert softmax_max.to_dict()["tensor_map"] == (2, 0, 1, -1)
        assert softmax_sum.to_dict()["tensor_map"] == (2, 0, 1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_softmax_no_sharding_25(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator

        _, softmax_max, softmax_sum, _ = output_layouts

        assert softmax_max.to_dict()["tensor_map"] == (-1, -1, -1, -1)
        assert softmax_sum.to_dict()["tensor_map"] == (-1, -1, -1, -1)





    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_bsnd_sp_31(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSND"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (2, 1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_bnsd_sp_32(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BNSD"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (2, 0, 1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_ring_attention_33(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sparse_mode_3_sp_34(self, mock_platform):
        """
        Feature: Layout inference with sparse_mode=3 (rightDownToLeftUp) and SP.
        Description: Causal variant with reversed direction, query seq sharded on sp.
        Expectation: Layout inference succeeds with seq sharding preserved.
        """
        mesh = self._make_4_mesh(mock_platform, "sp")
        q_placements = (Shard(1),)
        kv_placements = (Replicate(),)
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (-1, 0, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sparse_mode_4_band_35(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_invalid_input_layout_36(self, mock_platform):
        """
        Feature: Error handling for invalid input_layout string.
        Description: extra_args specifies an unsupported input_layout.
        Expectation: Raises an appropriate error.
        """
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        with self.assertRaises((ValueError, KeyError, RuntimeError)):
            cache_values = [q_layout, k_layout, v_layout, "INVALID"]
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_missing_extra_args_37(self, mock_platform):
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
            cache_values = [q_layout, k_layout, v_layout]
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_ndim_mismatch_layout_38(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BNSD"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        attention_out_layout = output_layouts[0]
        assert len(attention_out_layout.to_dict()["tensor_map"]) == 3


    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_dp_only_no_mp_no_sp_40(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (0, -1, -1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_mp_only_no_dp_no_sp_41(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (-1, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_sbh_sp_42(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "SBH"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator
        assert output_layouts[0].to_dict()["tensor_map"] == (1, -1, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_tnd_softmax_output_sp_43(self, mock_platform):
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

        cache_values = [q_layout, k_layout, v_layout, "TND"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]
        # Verify get_expand_impl returns callable for this operator
        impl = op.get_expand_impl(None, infer_result, cache_values)
        assert callable(impl), f"Expected callable, got {type(impl)}"
        # Verify get_expand_impl returns callable for this operator

        attention_out, softmax_max, softmax_sum, _ = output_layouts

        assert attention_out.to_dict()["tensor_map"] == (1, 0, -1)
        assert len(softmax_max.to_dict()["tensor_map"]) == 3
        assert len(softmax_sum.to_dict()["tensor_map"]) == 3


    # ========================================================================
    # Batch 1: Static / pure-logic method tests (no tensors or meshes needed)
    # ========================================================================

    def test_normalize_dim_map_none(self):
        """_normalize_dim_map: None returns "None"."""
        result = NPUFlashAttentionScoreDistributedOp._normalize_dim_map(None)
        assert result == "None", f"Expected 'None', got {result}"

    def test_normalize_dim_map_string(self):
        """_normalize_dim_map: string returns same string."""
        result = NPUFlashAttentionScoreDistributedOp._normalize_dim_map("sp")
        assert result == "sp", f"Expected 'sp', got {result}"

    def test_get_seq_dim_idx_seq_present(self):
        """_get_seq_dim_idx: returns 'seq' index when present."""
        result = op._get_seq_dim_idx({"seq": 1, "batch": 0})
        assert result == 1, f"Expected 1, got {result}"

    def test_get_seq_dim_idx_total_present(self):
        """_get_seq_dim_idx: returns 'total' index when 'seq' absent."""
        result = op._get_seq_dim_idx({"total": 0, "head": 1})
        assert result == 0, f"Expected 0, got {result}"

    def test_get_seq_dim_idx_neither_present(self):
        """_get_seq_dim_idx: returns None when neither 'seq' nor 'total' present."""
        result = op._get_seq_dim_idx({"batch": 0, "head": 1})
        assert result is None, f"Expected None, got {result}"

    def test_is_attn_mask_compressed(self):
        """_is_attn_mask_compressed: only 2/3/4 (causal/band) are compressed."""
        for mode, expected in [(0, False), (1, False), (2, True), (3, True), (4, True)]:
            with self.subTest(sparse_mode=mode):
                assert op._is_attn_mask_compressed(mode) is expected

    def test_adjust_head_num_valid(self):
        """_adjust_head_num: valid division returns local head count."""
        result = op._adjust_head_num(32, 4)
        assert result == 8, f"Expected 8, got {result}"

    def test_adjust_head_num_zero_split(self):
        """_adjust_head_num: head_split_num <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            op._adjust_head_num(32, 0)

    def test_adjust_head_num_not_divisible(self):
        """_adjust_head_num: non-divisible raises ValueError."""
        with self.assertRaises(ValueError):
            op._adjust_head_num(33, 4)

    def test_truncate_result_short_tuple(self):
        """_truncate_result: tuple with < 4 elements returned as-is."""
        result = NPUFlashAttentionScoreDistributedOp._truncate_result((1, 2))
        assert result == (1, 2), f"Expected (1, 2), got {result}"

    def test_truncate_result_long_tuple(self):
        """_truncate_result: tuple with > 4 elements truncated to first 4."""
        result = NPUFlashAttentionScoreDistributedOp._truncate_result((1, 2, 3, 4, 5, 6))
        assert result == (1, 2, 3, 4), f"Expected (1, 2, 3, 4), got {result}"

    def test_truncate_result_not_tuple(self):
        """_truncate_result: non-tuple returned as-is (scalar)."""
        result = NPUFlashAttentionScoreDistributedOp._truncate_result(42)
        assert result == 42, f"Expected 42, got {result}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_create_default_softmax_layout(self, mock_platform):
        """_create_default_softmax_layout: returns all-replicated 4D layout."""
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)

        softmax_layout = op._create_default_softmax_layout(q_layout)
        assert softmax_layout.to_dict()["tensor_map"] == (-1, -1, -1, -1), (
            f"Expected (-1, -1, -1, -1), got {softmax_layout.to_dict()['tensor_map']}"
        )


    # ========================================================================
    # Batch 2: _validate_input_layouts error paths
    # ========================================================================

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_input_layouts_query_none(self, mock_platform):
        """_validate_input_layouts: query_layout=None raises ValueError."""
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)
        layout_dims = op._layout_dims
        with self.assertRaises(ValueError):
            NPUFlashAttentionScoreDistributedOp._validate_input_layouts(
                None, k_layout, v_layout, "BSH", layout_dims, "npu_fusion_attention"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_input_layouts_kv_mismatch(self, mock_platform):
        """_validate_input_layouts: K/V tensor_map mismatch raises ValueError."""
        mesh = self._make_4x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
        k_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
        v_layout = _build_layout(mesh, (Shard(1), Shard(0)), 3)
        layout_dims = op._layout_dims
        with self.assertRaises(ValueError):
            NPUFlashAttentionScoreDistributedOp._validate_input_layouts(
                q_layout, k_layout, v_layout, "BSH", layout_dims, "npu_fusion_attention"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_input_layouts_valid(self, mock_platform):
        """_validate_input_layouts: valid layouts pass without error."""
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)
        layout_dims = op._layout_dims
        # Should not raise
        NPUFlashAttentionScoreDistributedOp._validate_input_layouts(
            q_layout, k_layout, v_layout, "BSH", layout_dims, "npu_fusion_attention"
        )


    # ========================================================================
    # Batch 3: Sharding consistency error branches
    # ========================================================================

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_sharding_consistency_key_none(self, mock_platform):
        """_validate_sharding_consistency: key_layout=None returns early."""
        mesh = self._make_4x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
        layout_dims = op._layout_dims
        # Should not raise
        NPUFlashAttentionScoreDistributedOp._validate_sharding_consistency(
            q_layout, None, "BSH", layout_dims, "npu_fusion_attention"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_sharding_consistency_batch_mismatch(self, mock_platform):
        """_validate_sharding_consistency: Q/K batch sharding mismatch raises ValueError."""
        mesh = self._make_4x2_mesh(mock_platform)
        # Q: batch sharded on dp, K: batch replicated
        q_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
        k_layout = _build_layout(mesh, (Replicate(), Shard(2)), 3)
        layout_dims = op._layout_dims
        with self.assertRaises(ValueError):
            NPUFlashAttentionScoreDistributedOp._validate_sharding_consistency(
                q_layout, k_layout, "BSH", layout_dims, "npu_fusion_attention"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_sharding_consistency_hidden_mismatch(self, mock_platform):
        """_validate_sharding_consistency: Q/K hidden sharding mismatch raises ValueError."""
        mesh = self._make_4x2_mesh(mock_platform)
        # Q: hidden sharded on mp, K: hidden replicated
        q_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
        k_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
        layout_dims = op._layout_dims
        with self.assertRaises(ValueError):
            NPUFlashAttentionScoreDistributedOp._validate_sharding_consistency(
                q_layout, k_layout, "BSH", layout_dims, "npu_fusion_attention"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_validate_sharding_consistency_dim_mismatch(self, mock_platform):
        """_validate_sharding_consistency: BNSD Q/K dim sharding mismatch raises ValueError."""
        mesh = self._make_4x2_mesh(mock_platform)
        # Q: batch+dim sharded, K: batch only
        q_layout = _build_layout(mesh, (Shard(0), Shard(3)), 4)
        k_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
        layout_dims = op._layout_dims
        with self.assertRaises(ValueError):
            NPUFlashAttentionScoreDistributedOp._validate_sharding_consistency(
                q_layout, k_layout, "BSND", layout_dims, "npu_fusion_attention"
            )

    # ========================================================================
    # Batch 4: _validate_atten_mask and _validate_pse_configuration
    # ========================================================================

    def test_validate_atten_mask_none_all_mask_error(self):
        """_validate_atten_mask: None mask + allMask (sparse_mode=1) raises ValueError."""
        with self.assertRaises(ValueError):
            op._validate_atten_mask(None, SPARSE_ALL_MASK, "BSH")

    def test_validate_atten_mask_wrong_dims(self):
        """_validate_atten_mask: only 2D/4D supported; 3D or 5D raises ValueError."""
        for shape in [(8, 8, 8), (1, 2, 3, 4, 5)]:
            with self.subTest(mask_shape=shape):
                mock_mask = MagicMock()
                mock_mask.shape = shape
                with self.assertRaises(ValueError):
                    op._validate_atten_mask(mock_mask, 0, "BSH")

    def test_validate_atten_mask_varlen_4d_error(self):
        """_validate_atten_mask: varlen with 4D mask raises ValueError."""
        mock_mask = MagicMock()
        mock_mask.shape = (1, 2, 8, 8)
        with self.assertRaises(ValueError):
            op._validate_atten_mask(mock_mask, 0, "BSH", is_varlen=True)

    def test_validate_atten_mask_varlen_2d_ok(self):
        """_validate_atten_mask: varlen with 2D mask passes."""
        mock_mask = MagicMock()
        mock_mask.shape = (8, 8)
        op._validate_atten_mask(mock_mask, 0, "BSH", is_varlen=True)

    def test_validate_atten_mask_compressed_warning(self):
        """_validate_atten_mask: compressed mode with non-2048 shape warns."""
        mock_mask = MagicMock()
        mock_mask.shape = (1, 1, 1024, 1024)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            op._validate_atten_mask(mock_mask, SPARSE_LEFT_UP_CAUSAL, "BSH")
            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"

    def test_validate_pse_none_ok(self):
        """_validate_pse_configuration: None PSE passes."""
        op._validate_pse_configuration(None, 0)

    def test_validate_pse_wrong_dims(self):
        """_validate_pse_configuration: only 3D/4D supported; 2D or 5D raises ValueError."""
        for shape in [(8, 8), (1, 2, 3, 4, 5)]:
            with self.subTest(pse_shape=shape):
                mock_pse = MagicMock()
                mock_pse.shape = shape
                with self.assertRaises(ValueError):
                    op._validate_pse_configuration(mock_pse, 0)

    def test_validate_pse_alibi_warning(self):
        """_validate_pse_configuration: 4D PSE with dim2=1024 warns (Alibi scenario)."""
        mock_pse = MagicMock()
        mock_pse.shape = (1, 2, 1024, 64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            op._validate_pse_configuration(mock_pse, 0)
            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"

    def test_validate_pse_3d_valid(self):
        """_validate_pse_configuration: valid 3D PSE passes."""
        mock_pse = MagicMock()
        mock_pse.shape = (2, 8, 64)
        op._validate_pse_configuration(mock_pse, 0)


    # ========================================================================
    # Batch 5: _check_seq_sharding_compatibility and _compute_sparse_params
    # ========================================================================

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_check_seq_sharding_key_none(self, mock_platform):
        """_check_seq_sharding_compatibility: key_layout=None returns early."""
        mesh = self._make_4x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(0), Shard(1)), 3)
        op._check_seq_sharding_compatibility(q_layout, None, "BSH", 1, 2, 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_check_seq_sharding_non_tnd_kv_seq_split(self, mock_platform):
        """_check_seq_sharding_compatibility: non-TND + kv_seq_split_num>1 -> NotImplementedError."""
        mesh = self._make_4x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(1), Replicate()), 3)
        k_layout = _build_layout(mesh, (Shard(1), Replicate()), 3)
        with self.assertRaises(NotImplementedError):
            op._check_seq_sharding_compatibility(q_layout, k_layout, "BSH", 1, 2, 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_check_seq_sharding_diff_kv_split_error(self, mock_platform):
        """_check_seq_sharding_compatibility: different Q/K seq shard with kv_split>1 raises."""
        mesh = self._make_4x2_mesh(mock_platform)
        q_layout = _build_layout(mesh, (Shard(1), Shard(0)), 3)
        k_layout = _build_layout(mesh, (Shard(1), Replicate()), 3)
        with self.assertRaises(NotImplementedError):
            op._check_seq_sharding_compatibility(q_layout, k_layout, "BSH", 1, 2, 2)

    def test_compute_sparse_params_unknown_mode_unchanged(self):
        """_compute_sparse_params: unknown sparse_mode returns unchanged params."""
        result = op._compute_sparse_params(99, 100, 200, 0, 4, 8, 32, 32)
        assert result == (99, 100, 200), f"Expected (99, 100, 200), got {result}"

    def test_compute_sparse_params_all_mask_unchanged(self):
        """_compute_sparse_params: ALL_MASK returns unchanged params."""
        result = op._compute_sparse_params(1, 100, 200, 0, 4, 8, 32, 32)
        assert result == (1, 100, 200), f"Expected (1, 100, 200), got {result}"

    def test_compute_sparse_params_default_mask_left_up(self):
        """_compute_sparse_params: DEFAULT_MASK with LEFT_UP update adjusts tokens."""
        # split_id=0, split_num=4, local_q_len=8, global_q_len=32, global_kv_len=32
        result = op._compute_sparse_params(0, 100, 200, 0, 4, 8, 32, 32)
        # new_pre = 100 + (-0*8) = 100, new_next = 200 + (0*8) = 200
        assert result == (0, 100, 200), f"Expected (0, 100, 200), got {result}"

    def test_compute_sparse_params_left_up_causal(self):
        """_compute_sparse_params: LEFT_UP_CAUSAL converts to BAND with adjusted tokens."""
        # split_id=1, split_num=4, local_q_len=8, global_q_len=32, global_kv_len=32
        result = op._compute_sparse_params(2, 0, 0, 1, 4, 8, 32, 32)
        # pre = global_kv_len + offset = 32 + (32 - (1+1)*8) = 32 + 16 = 48
        # next = 0 + (-offset) = 0 + (-16) = -16
        # new_sparse_mode = 4 (BAND)
        assert result == (4, 48, -16), f"Expected (4, 48, -16), got {result}"

    def test_compute_sparse_params_right_down_causal(self):
        """_compute_sparse_params: RIGHT_DOWN_CAUSAL converts to BAND with adjusted tokens."""
        # split_id=0, split_num=4, local_q_len=8, global_q_len=32, global_kv_len=32
        result = op._compute_sparse_params(3, 0, 0, 0, 4, 8, 32, 32)
        # pre = global_kv_len + offset = 32 + (4-0-1)*8 = 32 + 24 = 56
        # next = 0 + (-offset) = -24
        assert result == (4, 56, -24), f"Expected (4, 56, -24), got {result}"

    def test_compute_sparse_params_band_mode(self):
        """_compute_sparse_params: BAND keeps mode, adjusts tokens per update_mode."""
        # sparse_mode=4 (BAND) -> update_mode=RIGHT_DOWN_TO_RIGHT_DOWN(3)
        # split_id=2, split_num=4, local_q_len=8, global_q_len=32, global_kv_len=32
        result = op._compute_sparse_params(4, 10, 20, 2, 4, 8, 32, 32)
        # offset = (4-2-1)*8 = 8; pre = 10+8 = 18; next = 20+(-8) = 12
        assert result == (4, 18, 12), f"Expected (4, 18, 12), got {result}"


    # ========================================================================
    # Batch 6: infer_layout error paths
    # ========================================================================

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_non_string_input_layout(self, mock_platform):
        """infer_layout: non-string input_layout raises ValueError."""
        mesh = self._make_4x2_mesh(mock_platform)
        placements = (Shard(0), Shard(2))
        q_layout = _build_layout(mesh, placements, 3)
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        cache_values = [q_layout, k_layout, v_layout, 42]
        with self.assertRaises(ValueError):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_partial_input_rejected(self, mock_platform):
        """infer_layout: Partial input layout raises ValueError via _check_partial_inputs."""
        mesh = self._make_2x2x2_mesh(mock_platform)
        placements = (Replicate(), Replicate(), Replicate())
        q_layout = _build_layout(mesh, placements, 3)
        q_layout.set_partial_by_dev_axis("sp", "sum")
        k_layout = _build_layout(mesh, placements, 3)
        v_layout = _build_layout(mesh, placements, 3)

        cache_values = [q_layout, k_layout, v_layout, "BSH"]
        with self.assertRaises(ValueError):
            op.infer_layout(cache_values)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_flash_attention_tnd_softmax_wrap_output_rank_match(self, mock_platform):
        """Verify TND softmax outputs remain rank-consistent after wrapping."""
        self._setup_mock_platform(mock_platform, world_size=8)
        mesh = init_device_mesh(
            device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp")
        )
        q_placements = (Shard(0), Shard(1))
        kv_placements = (Replicate(), Shard(1))
        q_layout = _build_layout(mesh, q_placements, 3)
        k_layout = _build_layout(mesh, kv_placements, 3)
        v_layout = _build_layout(mesh, kv_placements, 3)

        cache_values = [q_layout, k_layout, v_layout, "TND"]
        infer_result = op.infer_layout(cache_values)
        output_layouts = infer_result[0]

        # NPU output order: attention_out, softmax_max, softmax_sum, softmax_out.
        local_outputs = (
            np.empty((2, 2, 32), dtype=np.float32),
            np.empty((2, 2, 8), dtype=np.float32),
            np.empty((2, 2, 8), dtype=np.float32),
            np.empty((), dtype=np.float32),
        )

        def _wrap_with_layout(local_tensor, device_mesh, placements, layout):
            """Validate the metadata passed by wrap_output and return a test double."""
            assert device_mesh is layout.mesh
            assert placements == layout.placements
            local_shape = local_tensor.shape
            tensor_map = layout.to_dict()["tensor_map"]
            assert len(local_shape) == len(tensor_map)
            assert len(layout.get_global_shape(local_shape)) == len(local_shape)
            wrapped_output = MagicMock()
            wrapped_output.to_local.return_value = local_tensor
            wrapped_output.layout = layout
            return wrapped_output

        with patch(
            "hyper_parallel.core.dtensor.dtensor.DTensor",
            side_effect=_wrap_with_layout,
        ) as mock_dtensor:
            wrapped = op.wrap_output(local_outputs, output_layouts)

        assert mock_dtensor.call_count == len(output_layouts)
        assert wrapped[1].layout.to_dict()["tensor_map"] == (1, 0, -1)
        assert wrapped[2].layout.to_dict()["tensor_map"] == (1, 0, -1)
        assert len(wrapped[1].to_local().shape) == 3
        assert len(wrapped[2].to_local().shape) == 3


if __name__ == "__main__":
    unittest.main()
