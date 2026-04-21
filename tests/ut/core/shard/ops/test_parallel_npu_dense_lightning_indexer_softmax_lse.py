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
"""Unit tests for NpuDenseLightningIndexerSoftmaxLseDistributedOp."""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.shard.ops.parallel_npu_dense_lightning_indexer_softmax_lse import (
    NpuDenseLightningIndexerSoftmaxLseDistributedOp,
    _normalize_softmax_lse_args,
    _adjust_bsnd_key,
    _adjust_tnd_seq_lens,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class TestNpuDenseLightningIndexerSoftmaxLse(unittest.TestCase):
    """Unit tests for NpuDenseLightningIndexerSoftmaxLseDistributedOp."""

    def setUp(self):
        """Clear global state before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        """Clear global state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        """Configure mock platform for mesh creation."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.split_group.return_value = MagicMock()
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.asnumpy() if hasattr(t, "asnumpy") else np.array(t)
        )

    def _make_1d_mesh(self, mock_platform, size=8, name="dp"):
        """Return a 1-D mesh of the given size."""
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(device_type="npu", mesh_shape=(size,), mesh_dim_names=(name,))

    def _make_2x4_dp_cp_mesh(self, mock_platform):
        """Return a 2×4 (dp, cp) mesh — 8 devices."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "cp"))

    @staticmethod
    def _get_op():
        return get_distributed_op("npu_dense_lightning_indexer_softmax_lse")

    # ------------------------------------------------------------------
    # _normalize_softmax_lse_args
    # ------------------------------------------------------------------

    def test_normalize_args_positional_defaults_1(self):
        """
        Feature: _normalize_softmax_lse_args fills in default values.
        Description: Call with only the 3 mandatory positional args.
        Expectation: Optional args default to None/BSND/3/INT64_MAX.
        """
        q, k, w = object(), object(), object()
        args, kwargs = _normalize_softmax_lse_args(q, k, w)
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertIs(args[2], w)
        self.assertIsNone(args[3], msg=f"actual_seq_qlen default should be None, got {args[3]}")
        self.assertIsNone(args[4], msg=f"actual_seq_klen default should be None, got {args[4]}")
        self.assertEqual(args[5], 'BSND', msg=f"layout default should be 'BSND', got {args[5]}")
        self.assertEqual(args[6], 3, msg=f"sparse_mode default should be 3, got {args[6]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    def test_normalize_args_layout_tnd_2(self):
        """
        Feature: _normalize_softmax_lse_args accepts layout keyword.
        Description: Pass layout='TND' as kwarg.
        Expectation: args[5] == 'TND'.
        """
        q, k, w = object(), object(), object()
        args, _ = _normalize_softmax_lse_args(q, k, w, layout='TND')
        self.assertEqual(args[5], 'TND', msg=f"args[5] should be 'TND', got {args[5]}")

    def test_normalize_args_seq_lens_3(self):
        """
        Feature: _normalize_softmax_lse_args passes through seq len lists.
        Description: Pass actual_seq_qlen and actual_seq_klen.
        Expectation: seq len lists land at the correct positions.
        """
        q, k, w = object(), object(), object()
        qlen, klen = [4, 8], [4, 8]
        args, _ = _normalize_softmax_lse_args(q, k, w, actual_seq_qlen=qlen, actual_seq_klen=klen)
        self.assertIs(args[3], qlen, msg=f"args[3] should be qlen, got {args[3]}")
        self.assertIs(args[4], klen, msg=f"args[4] should be klen, got {args[4]}")

    def test_normalize_args_returns_empty_kwargs_4(self):
        """
        Feature: _normalize_softmax_lse_args always returns empty kwargs.
        Description: Pass any combination of kwargs.
        Expectation: Returned kwargs dict is always empty.
        """
        q, k, w = object(), object(), object()
        _, kwargs = _normalize_softmax_lse_args(q, k, w, layout='TND', sparse_mode=3)
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def test_adjust_bsnd_key_split0_5(self):
        """
        Feature: _adjust_bsnd_key slices S2 to causal window.
        Description: split_id=0, S1_local=4 → key[:, :4, ...]
        Expectation: returned shape has S2=4.
        """
        k = np.ones((2, 16, 1, 128), dtype=np.float32)
        sliced = _adjust_bsnd_key(k, local_q_s1=4, split_id=0)
        self.assertEqual(sliced.shape[1], 4, msg=f"split_id=0 should give S2=4, got {sliced.shape[1]}")

    def test_adjust_bsnd_key_split2_6(self):
        """
        Feature: _adjust_bsnd_key slices S2 to causal window.
        Description: split_id=2, S1_local=4 → key[:, :12, ...]
        Expectation: returned shape has S2=12.
        """
        k = np.ones((2, 16, 1, 128), dtype=np.float32)
        sliced = _adjust_bsnd_key(k, local_q_s1=4, split_id=2)
        self.assertEqual(sliced.shape[1], 12, msg=f"split_id=2 should give S2=12, got {sliced.shape[1]}")

    def test_adjust_bsnd_key_preserves_other_dims_7(self):
        """
        Feature: _adjust_bsnd_key preserves B, N2index, D dimensions.
        Description: Slice only the S2 dimension.
        Expectation: B, N2index, D remain unchanged.
        """
        k = np.ones((2, 16, 3, 64), dtype=np.float32)
        sliced = _adjust_bsnd_key(k, local_q_s1=4, split_id=1)
        self.assertEqual(sliced.shape[0], 2, msg=f"B should be unchanged, got {sliced.shape[0]}")
        self.assertEqual(sliced.shape[2], 3, msg=f"N2index should be unchanged, got {sliced.shape[2]}")
        self.assertEqual(sliced.shape[3], 64, msg=f"D should be unchanged, got {sliced.shape[3]}")

    def test_adjust_tnd_seq_lens_rank0_8(self):
        """
        Feature: _adjust_tnd_seq_lens adjusts cumulative lens for cp_rank=0.
        Description: Two batches of 4 q-tokens, 8 k-tokens; cp=0 (no token offset).
        Expectation: new_q[-1]==local_q.shape[0] (last element is local max);
                     new_k[-1]==local_k.shape[0].
        """
        import torch
        local_q = torch.zeros((4, 1, 128))
        local_k = torch.zeros((8, 1, 128))
        actual_seq_qlen = torch.tensor([4, 8], dtype=torch.int32)
        actual_seq_klen = torch.tensor([8, 16], dtype=torch.int32)
        new_q, new_k = _adjust_tnd_seq_lens(
            local_q, local_k, actual_seq_qlen, actual_seq_klen, cp_rank=0)
        self.assertEqual(new_q[-1].item(), local_q.shape[0],
                         msg=f"new_q[-1] should equal T1_local={local_q.shape[0]}, got {new_q[-1]}")
        self.assertEqual(new_k[-1].item(), local_k.shape[0],
                         msg=f"new_k[-1] should equal T2_local={local_k.shape[0]}, got {new_k[-1]}")

    # ------------------------------------------------------------------
    # YAML registration
    # ------------------------------------------------------------------

    def test_yaml_registration_9(self):
        """
        Feature: YAML loader registers NpuDenseLightningIndexerSoftmaxLseDistributedOp.
        Description: Call get_distributed_op with the op name.
        Expectation: Returns a non-None NpuDenseLightningIndexerSoftmaxLseDistributedOp instance.
        """
        op = self._get_op()
        self.assertIsNotNone(op, msg="npu_dense_lightning_indexer_softmax_lse should be registered")
        self.assertIsInstance(
            op, NpuDenseLightningIndexerSoftmaxLseDistributedOp,
            msg=f"Expected NpuDenseLightningIndexerSoftmaxLseDistributedOp, got {type(op)}"
        )

    # ------------------------------------------------------------------
    # preprocess — MindSpore path
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.shard.ops.parallel_npu_dense_lightning_indexer_softmax_lse.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_ms_9_positional_args_10(self, mock_mesh_plat, mock_op_plat):
        """
        Feature: preprocess packs all 9 args positionally on MindSpore.
        Description: Patch platform type to MINDSPORE, call preprocess.
        Expectation: local_args has 9 elements; local_kwargs is empty.
        """
        self._setup_mock_platform(mock_mesh_plat)
        mock_op_plat.platform_type = PlatformType.MINDSPORE
        mesh = init_device_mesh("npu", (1,), mesh_dim_names=("dp",))
        layout_obj = _build_layout(mesh, (Replicate(),), 4)

        def _mock_dtensor(layout):
            dt = MagicMock(spec=DTensor)
            dt.to_local.return_value = MagicMock()
            dt.layout = layout
            return dt

        dtq = _mock_dtensor(layout_obj)
        dtk = _mock_dtensor(layout_obj)
        dtw = _mock_dtensor(_build_layout(mesh, (Replicate(),), 3))

        op = self._get_op()
        local_args, local_kwargs, cache_values = op.preprocess((dtq, dtk, dtw), {})

        self.assertEqual(len(local_args), 9, msg=f"MS path should pack 9 positional args, got {len(local_args)}")
        self.assertEqual(local_kwargs, {}, msg=f"MS path should have no kwargs, got {local_kwargs}")
        self.assertEqual(len(cache_values), 4, msg=f"cache_values should have 4 elements, got {len(cache_values)}")
        self.assertEqual(cache_values[3], 'BSND', msg=f"cache_values[3] should be 'BSND', got {cache_values[3]}")

    @patch("hyper_parallel.core.shard.ops.parallel_npu_dense_lightning_indexer_softmax_lse.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_torch_3_positional_args_11(self, mock_mesh_plat, mock_op_plat):
        """
        Feature: preprocess puts optional args in kwargs on PyTorch.
        Description: Patch platform type to PYTORCH, pass layout='TND'.
        Expectation: local_args has 3 elements; layout='TND' in local_kwargs.
        """
        self._setup_mock_platform(mock_mesh_plat)
        mock_op_plat.platform_type = PlatformType.PYTORCH
        mesh = init_device_mesh("npu", (1,), mesh_dim_names=("dp",))

        def _mock_dtensor(layout):
            dt = MagicMock(spec=DTensor)
            dt.to_local.return_value = MagicMock()
            dt.layout = layout
            return dt

        dtq = _mock_dtensor(_build_layout(mesh, (Replicate(),), 4))
        dtk = _mock_dtensor(_build_layout(mesh, (Replicate(),), 4))
        dtw = _mock_dtensor(_build_layout(mesh, (Replicate(),), 3))

        op = self._get_op()
        local_args, local_kwargs, cache_values = op.preprocess((dtq, dtk, dtw), {'layout': 'TND'})

        self.assertEqual(len(local_args), 3, msg=f"Torch path should have 3 positional args, got {len(local_args)}")
        self.assertEqual(local_kwargs.get('layout'), 'TND', msg=f"layout should be TND in kwargs, got {local_kwargs}")
        self.assertIn('actual_seq_qlen', local_kwargs)
        self.assertIn('actual_seq_klen', local_kwargs)
        self.assertEqual(cache_values[3], 'TND', msg=f"cache_values[3] should be 'TND', got {cache_values[3]}")

    # ------------------------------------------------------------------
    # infer_layout — BSND
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_replicated_output_shape_12(self, mock_platform):
        """
        Feature: infer_layout BSND replicated produces correct output tensor_map.
        Description: All inputs replicated on 1-D mesh.
        Expectation: Two outputs, tensor_map = (-1,-1,-1) each.
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        q = _build_layout(mesh, (Replicate(),), 4)
        k = _build_layout(mesh, (Replicate(),), 4)
        w = _build_layout(mesh, (Replicate(),), 3)
        op = self._get_op()
        result, extra = op.infer_layout([q, k, w, 'BSND'])

        self.assertIsNone(extra)
        self.assertEqual(len(result), 2, msg=f"Should return 2 outputs, got {len(result)}")
        self.assertEqual(result[0].tensor_map, (-1, -1, -1),
                         msg=f"Expected (-1,-1,-1), got {result[0].tensor_map}")
        self.assertEqual(result[1].tensor_map, (-1, -1, -1),
                         msg=f"Expected (-1,-1,-1), got {result[1].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_dp_only_13(self, mock_platform):
        """
        Feature: infer_layout BSND with batch-dimension DP.
        Description: q/k/w all have B sharded on dp (1-D mesh, size 4).
        Expectation: Output tensor_map[0] == 0 (batch from dp); S1 index replication.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")
        q = _build_layout(mesh, (Shard(0),), 4)
        k = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'BSND'])

        # Output (B, N2index, S1): B propagates, N2index replicated, S1 from q dim 1
        self.assertEqual(result[0].tensor_map[0], q.tensor_map[0],
                         msg=f"B should propagate: expected {q.tensor_map[0]}, got {result[0].tensor_map[0]}")
        self.assertEqual(result[0].tensor_map[1], -1,
                         msg=f"N2index should be replicated, got {result[0].tensor_map[1]}")
        self.assertEqual(result[0].tensor_map[2], q.tensor_map[1],
                         msg=f"S1 should come from q_tm[1]: expected {q.tensor_map[1]}, got {result[0].tensor_map[2]}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_dp_cp_mindformers_scenario_14(self, mock_platform):
        """
        Feature: infer_layout BSND with dp=2, cp=4 (the standard shard scenario).
        Description: q_index on (dp,cp,None,None), k_index on (dp,None,None,None), w on (dp,cp,None).
        Expectation: Output (B,N2index,S1) has B on dp (index 1), N2 replicated, S1 on cp (index 0).
        """
        mesh = self._make_2x4_dp_cp_mesh(mock_platform)
        q = _build_layout(mesh, (Shard(0), Shard(1)), 4)   # (1,0,-1,-1)
        k = _build_layout(mesh, (Shard(0), Replicate()), 4)  # (1,-1,-1,-1)
        w = _build_layout(mesh, (Shard(0), Shard(1)), 3)     # (1,0,-1)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'BSND'])

        # q_tm[0]=1 (B→dp), q_tm[1]=0 (S1→cp) → out=(1,-1,0)
        self.assertEqual(result[0].tensor_map, (1, -1, 0),
                         msg=f"Expected (1,-1,0), got {result[0].tensor_map}")
        self.assertEqual(result[1].tensor_map, (1, -1, 0),
                         msg=f"Expected (1,-1,0), got {result[1].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_outputs_are_independent_copies_15(self, mock_platform):
        """
        Feature: Both output layouts are independent deepcopies.
        Description: Check that the two returned layout objects are distinct.
        Expectation: result[0] is not result[1].
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        q = _build_layout(mesh, (Shard(0),), 4)
        k = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'BSND'])

        self.assertIsNot(result[0], result[1],
                         msg="Both output layouts must be distinct objects (deepcopy)")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_output_has_placements_16(self, mock_platform):
        """
        Feature: infer_layout sets placements on output layouts.
        Description: Both outputs must have non-None placements.
        Expectation: result[i].placements is not None for all outputs.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        q = _build_layout(mesh, (Shard(0),), 4)
        k = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'BSND'])

        for i, layout in enumerate(result):
            self.assertIsNotNone(layout.placements,
                                 msg=f"output[{i}].placements should not be None")

    # ------------------------------------------------------------------
    # infer_layout — TND
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_replicated_17(self, mock_platform):
        """
        Feature: infer_layout TND replicated produces correct output tensor_map.
        Description: All inputs replicated on 1-D mesh.
        Expectation: Two outputs, tensor_map = (-1,-1) each (N2index, T1 dims).
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        q = _build_layout(mesh, (Replicate(),), 3)
        k = _build_layout(mesh, (Replicate(),), 3)
        w = _build_layout(mesh, (Replicate(),), 2)
        op = self._get_op()
        result, extra = op.infer_layout([q, k, w, 'TND'])

        self.assertIsNone(extra)
        self.assertEqual(len(result), 2, msg=f"Should return 2 outputs, got {len(result)}")
        # TND out shape (N2index, T1): N2 replicated, T1 from q dim 0
        self.assertEqual(result[0].tensor_map, (-1, -1),
                         msg=f"Replicated TND output should be (-1,-1), got {result[0].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_dp_cp_merged_18(self, mock_platform):
        """
        Feature: infer_layout TND with T1 sharded on merged dp_cp axis.
        Description: q_index (T1,N1index,D) on dp_cp axis; k_index replicated.
        Expectation: Output (N2index,T1) has N2 replicated, T1 sharded on dp_cp (index 0).
        """
        mesh = self._make_1d_mesh(mock_platform, size=8, name="dp_cp")
        q = _build_layout(mesh, (Shard(0),), 3)   # (0,-1,-1)
        k = _build_layout(mesh, (Replicate(),), 3)  # (-1,-1,-1)
        w = _build_layout(mesh, (Shard(0),), 2)    # (0,-1)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'TND'])

        # out (N2index, T1): N2 replicated, T1 from q_tm[0]=0 → out_tm=(-1, 0)
        self.assertEqual(result[0].tensor_map, (-1, 0),
                         msg=f"Expected (-1,0) for TND dp_cp, got {result[0].tensor_map}")

    # ------------------------------------------------------------------
    # infer_layout — partial raises
    # ------------------------------------------------------------------

    def test_infer_layout_raises_on_partial_19(self):
        """
        Feature: Partial inputs are rejected.
        Description: Pass a layout with is_partial() returning True.
        Expectation: Raises ValueError.
        """
        partial_layout = MagicMock()
        partial_layout.is_partial.return_value = True
        normal_layout = MagicMock()
        normal_layout.is_partial.return_value = False
        op = self._get_op()
        with self.assertRaises(ValueError):
            op.infer_layout([partial_layout, normal_layout, normal_layout, 'BSND'])

    # ------------------------------------------------------------------
    # infer_layout — BSND validation errors
    # ------------------------------------------------------------------

    def _make_mock_layouts(self, q_tm, k_tm, w_tm, partial=False):
        """Return (q_layout, k_layout, w_layout) mocks with given tensor_maps."""
        def _mock(tm):
            m = MagicMock()
            m.is_partial.return_value = partial
            m.tensor_map = tm
            return m
        return _mock(q_tm), _mock(k_tm), _mock(w_tm)

    def test_infer_layout_bsnd_rejects_n1index_sharded_20(self):
        """
        Feature: BSND N1index (dim 2) sharding rejected.
        Description: query_index has tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N1index'.
        """
        q, k, w = self._make_mock_layouts((0, -1, 0, -1), (0, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1index"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_d_sharded_21(self):
        """
        Feature: BSND D (dim 3) sharding rejected.
        Description: query_index has tensor_map[3] != -1.
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, 0), (0, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_s2_sharded_22(self):
        """
        Feature: BSND S2 (dim 1) of key_index sharding rejected.
        Description: key_index has tensor_map[1] != -1.
        Expectation: Raises ValueError mentioning 'S2'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, 0, -1, -1), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "S2"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_n2index_sharded_23(self):
        """
        Feature: BSND N2index (dim 2) of key_index sharding rejected.
        Description: key_index has tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N2index'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, -1, 0, -1), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N2index"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_key_d_sharded_24(self):
        """
        Feature: BSND D (dim 3) of key_index sharding rejected.
        Description: key_index has tensor_map[3] != -1.
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, -1, -1, 0), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_batch_mismatch_25(self):
        """
        Feature: BSND B sharding mismatch between query_index and key_index rejected.
        Description: query_index B on mesh dim 0, key_index B on mesh dim 1.
        Expectation: Raises ValueError mentioning 'B (dim 0) sharding'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, -1), (1, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"B \(dim 0\) sharding of query_index and key_index should match"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_b_mismatch_26(self):
        """
        Feature: BSND B mismatch between weights and query_index rejected.
        Description: weights.B on mesh dim 1, query_index.B on mesh dim 0.
        Expectation: Raises ValueError mentioning 'B (dim 0) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, -1), (0, -1, -1, -1), (1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"B \(dim 0\) sharding of weights"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_s1_mismatch_27(self):
        """
        Feature: BSND S1 mismatch between weights and query_index rejected.
        Description: weights.S1 on dim 0, query_index.S1 on dim 1.
        Expectation: Raises ValueError mentioning 'S1 (dim 1) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, 1, -1, -1), (0, -1, -1, -1), (0, 0, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"S1 \(dim 1\) sharding of weights"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_n1_sharded_28(self):
        """
        Feature: BSND N1index (dim 2) of weights sharding rejected.
        Description: weights.tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N1index'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, -1), (0, -1, -1, -1), (0, -1, 0))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1index"):
            op.infer_layout([q, k, w, 'BSND'])

    # ------------------------------------------------------------------
    # infer_layout — TND validation errors
    # ------------------------------------------------------------------

    def test_infer_layout_tnd_rejects_n1_sharded_29(self):
        """
        Feature: TND N1index (dim 1) of query_index sharding rejected.
        Description: query_index has tensor_map[1] != -1 (TND).
        Expectation: Raises ValueError mentioning 'N1index'.
        """
        q, k, w = self._make_mock_layouts((-1, 0, -1), (-1, -1, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1index"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_d_sharded_30(self):
        """
        Feature: TND D (dim 2) of query_index sharding rejected.
        Description: query_index has tensor_map[2] != -1 (TND).
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, 0), (-1, -1, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_key_n2index_sharded_31(self):
        """
        Feature: TND N2index (dim 1) of key_index sharding rejected.
        Description: key_index has tensor_map[1] != -1 (TND).
        Expectation: Raises ValueError mentioning 'N2index'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1), (-1, 0, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N2index"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_key_d_sharded_32(self):
        """
        Feature: TND D (dim 2) of key_index sharding rejected.
        Description: key_index has tensor_map[2] != -1 (TND).
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1), (-1, -1, 0), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_weights_t1_mismatch_33(self):
        """
        Feature: TND T1 sharding mismatch between weights and query_index rejected.
        Description: weights.T1 on mesh dim 1, query_index.T1 on mesh dim 0.
        Expectation: Raises ValueError mentioning 'T1 (dim 0) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1), (-1, -1, -1), (1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"T1 \(dim 0\) sharding of weights should match query_index"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_weights_n1_sharded_34(self):
        """
        Feature: TND N1index (dim 1) of weights sharding rejected.
        Description: weights.tensor_map[1] != -1 (TND).
        Expectation: Raises ValueError mentioning 'N1index'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1), (-1, -1, -1), (-1, 0))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1index"):
            op.infer_layout([q, k, w, 'TND'])

    # ------------------------------------------------------------------
    # get_expand_impl
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_impl_bsnd_no_cp_returns_none_35(self, mock_platform):
        """
        Feature: get_expand_impl returns None when S1 is not sharded (no CP).
        Description: BSND q_index with S1 replicated (tensor_map[1]==-1).
        Expectation: get_expand_impl returns None.
        Since get_expand_impl is overridden, we verify it once for the no-CP case;
        other no-CP tests do not repeat this check.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        q = _build_layout(mesh, (Shard(0),), 4)   # S1 not sharded
        k = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'BSND'])

        assert op.get_expand_impl(None, result, [q, k, w, 'BSND']) is None, (
            "get_expand_impl should return None when S1 is not sharded (no BSND CP)"
        )

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_impl_tnd_no_cp_returns_callable_36(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: get_expand_impl always returns callable for TND (DP batch slicing is always needed).
        Description: TND with same T1 sharding on q and k.
        Expectation: get_expand_impl returns a callable wrapper.
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=4)
        mock_layout_plat.get_rank.return_value = 0
        mesh = init_device_mesh("npu", (4,), mesh_dim_names=("dp",))
        q = _build_layout(mesh, (Shard(0),), 3)
        k = _build_layout(mesh, (Shard(0),), 3)
        w = _build_layout(mesh, (Shard(0),), 2)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'TND'])

        impl = op.get_expand_impl(None, result, [q, k, w, 'TND'])
        assert callable(impl), (
            f"get_expand_impl should return callable for TND, got {type(impl)}"
        )

    def test_expand_impl_bsnd_cp_returns_callable_37(self):
        """
        Feature: get_expand_impl returns callable when S1 is sharded (BSND+CP).
        Description: q_index tensor_map[1] != -1 triggers CP path.
        Expectation: Callable is returned.
        """
        q_mock = MagicMock()
        q_mock.tensor_map = (-1, 0, -1, -1)
        # MagicMock's alias_tensor_map has len=0 → _get_split_id short-circuits to 0.
        op = self._get_op()
        impl = op.get_expand_impl(None, None, [q_mock, MagicMock(), MagicMock(), 'BSND'])
        self.assertTrue(callable(impl), msg=f"Expected callable for BSND+CP, got {type(impl)}")

    @patch("hyper_parallel.core.shard.ops.parallel_npu_dense_lightning_indexer_softmax_lse.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.dtensor.layout.platform")
    def test_expand_impl_tnd_cp_returns_callable_38(self, mock_layout_plat, mock_mesh_plat, mock_op_plat):
        """
        Feature: get_expand_impl returns callable when q_split > k_split (TND+CP).
        Description: q on 8-device dp_cp mesh, k replicated → q_split=8 > k_split=1.
        Expectation: Callable is returned.
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=8)
        mock_op_plat.get_rank.return_value = 0
        mock_layout_plat.get_rank.return_value = 0

        mesh_8 = init_device_mesh("npu", (8,), mesh_dim_names=("dp_cp",))
        q = _build_layout(mesh_8, (Shard(0),), 3)    # alias_tensor_map[0]="dp_cp" → split=8
        k = _build_layout(mesh_8, (Replicate(),), 3)  # alias_tensor_map[0]="None" → split=1

        op = self._get_op()
        impl = op.get_expand_impl(None, None, [q, k, MagicMock(), 'TND'])
        self.assertTrue(callable(impl), msg=f"Expected callable for TND+CP, got {type(impl)}")




if __name__ == "__main__":
    unittest.main()
