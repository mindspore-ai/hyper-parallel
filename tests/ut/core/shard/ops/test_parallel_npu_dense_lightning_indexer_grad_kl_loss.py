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
"""Unit tests for NpuDenseLightningIndexerGradKlLossDistributedOp."""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE, DTensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_npu_dense_lightning_indexer_grad_kl_loss import (
    NpuDenseLightningIndexerGradKlLossDistributedOp,
    _normalize_dense_grad_kl_loss_args,
    _to_local,
    _to_local_seq_len,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class TestNpuDenseLightningIndexerGradKlLoss(unittest.TestCase):
    """Unit tests for NpuDenseLightningIndexerGradKlLossDistributedOp."""

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
        return get_distributed_op("npu_dense_lightning_indexer_grad_kl_loss")

    # ------------------------------------------------------------------
    # _normalize_dense_grad_kl_loss_args
    # ------------------------------------------------------------------

    def test_normalize_args_positional_defaults_1(self):
        """
        Feature: _normalize_dense_grad_kl_loss_args fills in default optional values.
        Description: Call with only the 10 mandatory positional args.
        Expectation: Optional args default to None/BSND/3/INT64_MAX; kwargs empty.
        """
        q, k, qi, ki, w = object(), object(), object(), object(), object()
        sm, ss, smi, ssi, scale = object(), object(), object(), object(), 1.0
        args, kwargs = _normalize_dense_grad_kl_loss_args(q, k, qi, ki, w, sm, ss, smi, ssi, scale)
        self.assertIs(args[0], q)
        self.assertIs(args[2], qi)
        self.assertIs(args[9], scale)
        self.assertIsNone(args[10], msg=f"query_rope default should be None, got {args[10]}")
        self.assertIsNone(args[11], msg=f"key_rope default should be None, got {args[11]}")
        self.assertIsNone(args[12], msg=f"actual_seq_qlen default should be None, got {args[12]}")
        self.assertIsNone(args[13], msg=f"actual_seq_klen default should be None, got {args[13]}")
        self.assertEqual(args[14], 'BSND', msg=f"layout default should be 'BSND', got {args[14]}")
        self.assertEqual(args[15], 3, msg=f"sparse_mode default should be 3, got {args[15]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    def test_normalize_args_returns_18_elements_2(self):
        """
        Feature: _normalize_dense_grad_kl_loss_args returns 18-element tuple.
        Description: Call with 10 mandatory args and verify tuple length.
        Expectation: len(args) == 18.
        """
        objs = [object() for _ in range(10)]
        args, _ = _normalize_dense_grad_kl_loss_args(*objs)
        self.assertEqual(len(args), 18, msg=f"Expected 18 positional args, got {len(args)}")

    def test_normalize_args_layout_tnd_3(self):
        """
        Feature: _normalize_dense_grad_kl_loss_args accepts layout keyword.
        Description: Pass layout='TND' as kwarg.
        Expectation: args[14] == 'TND'.
        """
        objs = [object() for _ in range(10)]
        args, _ = _normalize_dense_grad_kl_loss_args(*objs, layout='TND')
        self.assertEqual(args[14], 'TND', msg=f"args[14] should be 'TND', got {args[14]}")

    def test_normalize_args_seq_lens_4(self):
        """
        Feature: _normalize_dense_grad_kl_loss_args passes through seq len lists.
        Description: Pass actual_seq_qlen and actual_seq_klen.
        Expectation: seq len lists land at indices 12 and 13.
        """
        objs = [object() for _ in range(10)]
        qlen, klen = [4, 8], [4, 8]
        args, _ = _normalize_dense_grad_kl_loss_args(*objs, actual_seq_qlen=qlen, actual_seq_klen=klen)
        self.assertIs(args[12], qlen, msg=f"args[12] should be qlen, got {args[12]}")
        self.assertIs(args[13], klen, msg=f"args[13] should be klen, got {args[13]}")

    # ------------------------------------------------------------------
    # _to_local helper
    # ------------------------------------------------------------------

    def test_to_local_with_dtensor_5(self):
        """
        Feature: _to_local extracts the local tensor from a DTensor.
        Description: Pass a mock DTensor; to_local() should be called.
        Expectation: Result is the value returned by to_local().
        """
        local_tensor = MagicMock()
        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = local_tensor
        result = _to_local(dt)
        dt.to_local.assert_called_once()
        self.assertIs(result, local_tensor, msg=f"Expected local tensor, got {result}")

    def test_to_local_with_none_6(self):
        """
        Feature: _to_local passes None (optional input not provided) through.
        Description: Pass None.
        Expectation: None is returned.
        """
        self.assertIsNone(_to_local(None), msg="None should pass through _to_local")

    def test_to_local_rejects_plain_tensor_6b(self):
        """
        Feature: _to_local rejects non-DTensor tensor inputs.
        Description: Every input except actual_seq_lengths must be a DTensor so its
            layout can be inferred; a plain tensor has no to_local() method.
        Expectation: AttributeError is raised.
        """
        with self.assertRaises(AttributeError):
            _to_local([1, 2, 3])

    def test_to_local_seq_len_with_dtensor_6c(self):
        """
        Feature: _to_local_seq_len extracts the local tensor from a DTensor.
        Expectation: Result is the value returned by to_local().
        """
        local_tensor = MagicMock()
        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = local_tensor
        self.assertIs(_to_local_seq_len(dt), local_tensor)

    def test_to_local_seq_len_passes_plain_tensor_6d(self):
        """
        Feature: _to_local_seq_len passes a non-DTensor actual_seq_lengths through.
        Description: actual_seq_lengths are network-built and not guaranteed to be
            DTensors, so a plain tensor is passed through unchanged.
        Expectation: The same object is returned without modification.
        """
        plain = [1, 2, 3]
        self.assertIs(_to_local_seq_len(plain), plain,
                      msg=f"plain actual_seq_lengths should pass through, got {plain}")

    # ------------------------------------------------------------------
    # YAML registration
    # ------------------------------------------------------------------

    def test_yaml_registration_7(self):
        """
        Feature: YAML loader registers NpuDenseLightningIndexerGradKlLossDistributedOp.
        Description: Call get_distributed_op with the op name.
        Expectation: Returns a non-None NpuDenseLightningIndexerGradKlLossDistributedOp instance.
        """
        op = self._get_op()
        self.assertIsNotNone(op, msg="npu_dense_lightning_indexer_grad_kl_loss should be registered")
        self.assertIsInstance(
            op, NpuDenseLightningIndexerGradKlLossDistributedOp,
            msg=f"Expected NpuDenseLightningIndexerGradKlLossDistributedOp, got {type(op)}"
        )

    # ------------------------------------------------------------------
    # preprocess — MindSpore path
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.shard.ops.parallel_npu_dense_lightning_indexer_grad_kl_loss.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_ms_18_positional_args_8(self, mock_mesh_plat, mock_op_plat):
        """
        Feature: preprocess packs all 18 args positionally on MindSpore.
        Description: Patch platform type to MINDSPORE, call preprocess with 10 DTensor args.
        Expectation: local_args has 18 elements; local_kwargs is empty; cache_values has 4 elements.
        """
        self._setup_mock_platform(mock_mesh_plat)
        mock_op_plat.platform_type = PlatformType.MINDSPORE
        mesh = init_device_mesh("npu", (1,), mesh_dim_names=("dp",))
        layout_4d = _build_layout(mesh, (Replicate(),), 4)
        layout_3d = _build_layout(mesh, (Replicate(),), 3)

        def _mock_dtensor(layout):
            dt = MagicMock(spec=DTensor)
            dt.to_local.return_value = MagicMock()
            dt.layout = layout
            return dt

        dtq = _mock_dtensor(layout_4d)
        dtk = _mock_dtensor(layout_4d)
        dtqi = _mock_dtensor(layout_4d)
        dtki = _mock_dtensor(layout_4d)
        dtw = _mock_dtensor(layout_3d)
        dtsm = _mock_dtensor(layout_3d)
        dtss = _mock_dtensor(layout_3d)
        dtsmi = _mock_dtensor(layout_3d)
        dtssi = _mock_dtensor(layout_3d)

        op = self._get_op()
        local_args, local_kwargs, cache_values = op.preprocess(
            (dtq, dtk, dtqi, dtki, dtw, dtsm, dtss, dtsmi, dtssi, 1.0), {}
        )

        self.assertEqual(len(local_args), 18, msg=f"MS path should pack 18 positional args, got {len(local_args)}")
        self.assertEqual(local_kwargs, {}, msg=f"MS path should have no kwargs, got {local_kwargs}")
        self.assertEqual(len(cache_values), 4, msg=f"cache_values should have 4 elements, got {len(cache_values)}")
        self.assertEqual(cache_values[3], 'BSND', msg=f"cache_values[3] should be 'BSND', got {cache_values[3]}")

    @patch("hyper_parallel.core.shard.ops.parallel_npu_dense_lightning_indexer_grad_kl_loss.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_torch_10_positional_args_9(self, mock_mesh_plat, mock_op_plat):
        """
        Feature: preprocess puts optional args in kwargs on PyTorch.
        Description: Patch platform type to PYTORCH, pass layout='TND'.
        Expectation: local_args has 10 elements; optional args are in local_kwargs.
        """
        self._setup_mock_platform(mock_mesh_plat)
        mock_op_plat.platform_type = PlatformType.PYTORCH
        mesh = init_device_mesh("npu", (1,), mesh_dim_names=("dp",))
        layout_4d = _build_layout(mesh, (Replicate(),), 4)
        layout_3d = _build_layout(mesh, (Replicate(),), 3)

        def _mock_dtensor(layout):
            dt = MagicMock(spec=DTensor)
            dt.to_local.return_value = MagicMock()
            dt.layout = layout
            return dt

        dtq = _mock_dtensor(layout_4d)
        dtk = _mock_dtensor(layout_4d)
        dtqi = _mock_dtensor(layout_4d)
        dtki = _mock_dtensor(layout_4d)
        dtw = _mock_dtensor(layout_3d)
        dtsm = _mock_dtensor(layout_3d)
        dtss = _mock_dtensor(layout_3d)
        dtsmi = _mock_dtensor(layout_3d)
        dtssi = _mock_dtensor(layout_3d)

        op = self._get_op()
        local_args, local_kwargs, cache_values = op.preprocess(
            (dtq, dtk, dtqi, dtki, dtw, dtsm, dtss, dtsmi, dtssi, 1.0),
            {'layout': 'TND'}
        )

        self.assertEqual(len(local_args), 10, msg=f"Torch path should have 10 positional args, got {len(local_args)}")
        self.assertEqual(local_kwargs.get('layout'), 'TND', msg=f"layout should be TND in kwargs, got {local_kwargs}")
        self.assertIn('actual_seq_qlen', local_kwargs)
        self.assertIn('actual_seq_klen', local_kwargs)
        self.assertEqual(cache_values[3], 'TND', msg=f"cache_values[3] should be 'TND', got {cache_values[3]}")

    # ------------------------------------------------------------------
    # infer_layout — BSND
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_replicated_four_outputs_10(self, mock_platform):
        """
        Feature: infer_layout BSND all-replicated produces 4 outputs.
        Description: All inputs replicated on 1-D mesh.
        Expectation: 4 outputs returned; d_q/d_k/d_w replicated; loss tensor_map=(-1,).
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        qi = _build_layout(mesh, (Replicate(),), 4)
        ki = _build_layout(mesh, (Replicate(),), 4)
        w = _build_layout(mesh, (Replicate(),), 3)
        op = self._get_op()
        result, extra = op.infer_layout([qi, ki, w, 'BSND'])

        self.assertIsNone(extra)
        self.assertEqual(len(result), 4, msg=f"Should return 4 outputs, got {len(result)}")
        self.assertEqual(result[0].tensor_map, (-1, -1, -1, -1),
                         msg=f"d_q_index replicated expected (-1,-1,-1,-1), got {result[0].tensor_map}")
        self.assertEqual(result[1].tensor_map, (-1, -1, -1, -1),
                         msg=f"d_k_index replicated expected (-1,-1,-1,-1), got {result[1].tensor_map}")
        self.assertEqual(result[2].tensor_map, (-1, -1, -1),
                         msg=f"d_weights replicated expected (-1,-1,-1), got {result[2].tensor_map}")
        self.assertEqual(result[3].tensor_map, (-1,),
                         msg=f"loss tensor_map should be (-1,), got {result[3].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_dp_only_inherits_input_layouts_11(self, mock_platform):
        """
        Feature: infer_layout BSND with batch-dimension DP — outputs inherit inputs.
        Description: q_index/k_index sharded on B (dim 0); w sharded on B and S1.
        Expectation: d_q_index inherits qi layout; d_k_index inherits ki layout;
                     d_weights inherits w layout; loss is replicated.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")
        qi = _build_layout(mesh, (Shard(0),), 4)  # B sharded
        ki = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([qi, ki, w, 'BSND'])

        self.assertEqual(result[0].tensor_map, qi.tensor_map,
                         msg=f"d_q_index should inherit qi layout, expected {qi.tensor_map}, "
                             f"got {result[0].tensor_map}")
        self.assertEqual(result[1].tensor_map, ki.tensor_map,
                         msg=f"d_k_index should inherit ki layout, expected {ki.tensor_map}, "
                             f"got {result[1].tensor_map}")
        self.assertEqual(result[2].tensor_map, w.tensor_map,
                         msg=f"d_weights should inherit w layout, expected {w.tensor_map}, "
                             f"got {result[2].tensor_map}")
        self.assertEqual(result[3].tensor_map, (-1,),
                         msg=f"loss should be replicated (-1,), got {result[3].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_dp_cp_mindformers_scenario_12(self, mock_platform):
        """
        Feature: infer_layout BSND with dp=2, cp=4 — standard shard scenario.
        Description: qi on (dp,cp,None,None), ki on (dp,None,None,None), w on (dp,cp,None).
        Expectation: d_q_index=(1,0,-1,-1), d_k_index=(1,-1,-1,-1), d_w=(1,0,-1), loss=(-1,).
        """
        mesh = self._make_2x4_dp_cp_mesh(mock_platform)
        qi = _build_layout(mesh, (Shard(0), Shard(1)), 4)  # (1,0,-1,-1)
        ki = _build_layout(mesh, (Shard(0), Replicate()), 4)  # (1,-1,-1,-1)
        w = _build_layout(mesh, (Shard(0), Shard(1)), 3)   # (1,0,-1)
        op = self._get_op()
        result, _ = op.infer_layout([qi, ki, w, 'BSND'])

        self.assertEqual(result[0].tensor_map, (1, 0, -1, -1),
                         msg=f"d_q_index should be (1,0,-1,-1), got {result[0].tensor_map}")
        self.assertEqual(result[1].tensor_map, (1, -1, -1, -1),
                         msg=f"d_k_index should be (1,-1,-1,-1), got {result[1].tensor_map}")
        self.assertEqual(result[2].tensor_map, (1, 0, -1),
                         msg=f"d_weights should be (1,0,-1), got {result[2].tensor_map}")
        self.assertEqual(result[3].tensor_map, (-1,),
                         msg=f"loss should be (-1,), got {result[3].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_outputs_are_independent_copies_13(self, mock_platform):
        """
        Feature: The four output layouts are independent deepcopies.
        Description: Mutate one output layout and check others are unaffected.
        Expectation: All four result objects are distinct; d_q_index is not qi input.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        qi = _build_layout(mesh, (Shard(0),), 4)
        ki = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([qi, ki, w, 'BSND'])

        self.assertIsNot(result[0], qi, msg="d_q_index must be a deepcopy, not the input layout itself")
        self.assertIsNot(result[0], result[1], msg="d_q_index and d_k_index must be distinct objects")
        self.assertIsNot(result[0], result[2], msg="d_q_index and d_weights must be distinct objects")
        self.assertIsNot(result[0], result[3], msg="d_q_index and loss must be distinct objects")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_outputs_have_placements_14(self, mock_platform):
        """
        Feature: infer_layout sets placements on all four output layouts.
        Description: All four outputs must have non-None placements.
        Expectation: result[i].placements is not None for all i in 0..3.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        qi = _build_layout(mesh, (Shard(0),), 4)
        ki = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([qi, ki, w, 'BSND'])

        for i, layout in enumerate(result):
            self.assertIsNotNone(layout.placements,
                                 msg=f"output[{i}].placements should not be None")

    # ------------------------------------------------------------------
    # infer_layout — TND
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_replicated_15(self, mock_platform):
        """
        Feature: infer_layout TND all-replicated produces 4 replicated outputs.
        Description: All inputs replicated on 1-D mesh with TND layout.
        Expectation: 4 outputs; d_q/d_k/d_w tensor maps replicated; loss=(-1,).
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        qi = _build_layout(mesh, (Replicate(),), 3)
        ki = _build_layout(mesh, (Replicate(),), 3)
        w = _build_layout(mesh, (Replicate(),), 2)
        op = self._get_op()
        result, extra = op.infer_layout([qi, ki, w, 'TND'])

        self.assertIsNone(extra)
        self.assertEqual(len(result), 4, msg=f"Should return 4 outputs, got {len(result)}")
        self.assertEqual(result[0].tensor_map, (-1, -1, -1),
                         msg=f"d_q_index TND replicated should be (-1,-1,-1), got {result[0].tensor_map}")
        self.assertEqual(result[3].tensor_map, (-1,),
                         msg=f"loss tensor_map should be (-1,), got {result[3].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_dp_cp_merged_16(self, mock_platform):
        """
        Feature: infer_layout TND with T1 sharded on merged dp_cp axis.
        Description: q_index (T1,N1index,D) on dp_cp; k_index replicated; w on dp_cp.
        Expectation: Outputs inherit input layouts; d_k_index and loss are Partial (CP active).
        """
        mesh = self._make_1d_mesh(mock_platform, size=8, name="dp_cp")
        qi = _build_layout(mesh, (Shard(0),), 3)   # (0,-1,-1)
        ki = _build_layout(mesh, (Replicate(),), 3)  # (-1,-1,-1)
        w = _build_layout(mesh, (Shard(0),), 2)    # (0,-1)
        op = self._get_op()
        result, _ = op.infer_layout([qi, ki, w, 'TND'])

        self.assertEqual(result[0].tensor_map, (0, -1, -1),
                         msg=f"d_q_index should inherit (0,-1,-1), got {result[0].tensor_map}")
        self.assertEqual(result[1].tensor_map, (-1, -1, -1),
                         msg=f"d_k_index should inherit (-1,-1,-1), got {result[1].tensor_map}")
        self.assertEqual(result[2].tensor_map, (0, -1),
                         msg=f"d_weights should inherit (0,-1), got {result[2].tensor_map}")
        self.assertEqual(result[3].tensor_map, (-1,),
                         msg=f"loss should be (-1,), got {result[3].tensor_map}")
        self.assertTrue(result[1].is_partial(), (
            f"d_k_index should be Partial in TND+CP (q_split=8 > k_split=1), "
            f"got partial={result[1].partial}"
        ))
        self.assertTrue(result[3].is_partial(), (
            f"loss should be Partial in TND+CP, "
            f"got partial={result[3].partial}"
        ))

    # ------------------------------------------------------------------
    # infer_layout — partial raises
    # ------------------------------------------------------------------

    def test_infer_layout_raises_on_partial_17(self):
        """
        Feature: Partial inputs are rejected.
        Description: Pass a layout mock with is_partial() returning True.
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
        """Return (qi_layout, ki_layout, w_layout) mocks with given tensor_maps."""
        def _mock(tm):
            m = MagicMock()
            m.is_partial.return_value = partial
            m.tensor_map = tm
            return m
        return _mock(q_tm), _mock(k_tm), _mock(w_tm)

    def test_infer_layout_bsnd_rejects_n1index_sharded_18(self):
        """
        Feature: BSND N1index (dim 2) of query_index sharding rejected.
        Description: query_index has tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N1index'.
        """
        q, k, w = self._make_mock_layouts((0, -1, 0, -1), (0, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1index"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_d_sharded_19(self):
        """
        Feature: BSND D (dim 3) of query_index sharding rejected.
        Description: query_index has tensor_map[3] != -1.
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, 0), (0, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_s2_sharded_20(self):
        """
        Feature: BSND S2 (dim 1) of key_index sharding rejected.
        Description: key_index has tensor_map[1] != -1.
        Expectation: Raises ValueError mentioning 'S2'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, 0, -1, -1), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "S2"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_n2index_sharded_21(self):
        """
        Feature: BSND N2index (dim 2) of key_index sharding rejected.
        Description: key_index has tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N2index'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, -1, 0, -1), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N2index"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_key_d_sharded_22(self):
        """
        Feature: BSND D (dim 3) of key_index sharding rejected.
        Description: key_index has tensor_map[3] != -1.
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, -1, -1, 0), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_batch_mismatch_23(self):
        """
        Feature: BSND B sharding mismatch between query_index and key_index rejected.
        Description: query_index B on mesh dim 0, key_index B on mesh dim 1.
        Expectation: Raises ValueError mentioning 'B (dim 0) sharding'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, -1), (1, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"B \(dim 0\) sharding of query_index and key_index should match"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_b_mismatch_24(self):
        """
        Feature: BSND B mismatch between weights and query_index rejected.
        Description: weights.B on mesh dim 1, query_index.B on mesh dim 0.
        Expectation: Raises ValueError mentioning 'B (dim 0) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, -1), (0, -1, -1, -1), (1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"B \(dim 0\) sharding of weights"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_s1_mismatch_25(self):
        """
        Feature: BSND S1 mismatch between weights and query_index rejected.
        Description: weights.S1 on dim 0, query_index.S1 on dim 1.
        Expectation: Raises ValueError mentioning 'S1 (dim 1) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, 1, -1, -1), (0, -1, -1, -1), (0, 0, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"S1 \(dim 1\) sharding of weights"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_n1_sharded_26(self):
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

    def test_infer_layout_tnd_rejects_n1_sharded_27(self):
        """
        Feature: TND N1index (dim 1) of query_index sharding rejected.
        Description: query_index has tensor_map[1] != -1 (TND).
        Expectation: Raises ValueError mentioning 'N1index'.
        """
        q, k, w = self._make_mock_layouts((-1, 0, -1), (-1, -1, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1index"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_d_sharded_28(self):
        """
        Feature: TND D (dim 2) of query_index sharding rejected.
        Description: query_index has tensor_map[2] != -1 (TND).
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, 0), (-1, -1, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_key_n2index_sharded_29(self):
        """
        Feature: TND N2index (dim 1) of key_index sharding rejected.
        Description: key_index has tensor_map[1] != -1 (TND).
        Expectation: Raises ValueError mentioning 'N2index'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1), (-1, 0, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N2index"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_key_d_sharded_30(self):
        """
        Feature: TND D (dim 2) of key_index sharding rejected.
        Description: key_index has tensor_map[2] != -1 (TND).
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1), (-1, -1, 0), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_weights_t1_mismatch_31(self):
        """
        Feature: TND T1 sharding mismatch between weights and query_index rejected.
        Description: weights.T1 on mesh dim 1, query_index.T1 on mesh dim 0.
        Expectation: Raises ValueError mentioning 'T1 (dim 0) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1), (-1, -1, -1), (1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"T1 \(dim 0\) sharding of weights should match query_index"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_weights_n1_sharded_32(self):
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
    def test_expand_impl_bsnd_no_cp_returns_none_33(self, mock_platform):
        """
        Feature: get_expand_impl returns None when S1 of query_index is not sharded (no BSND CP).
        Description: query_index with S1 replicated (tensor_map[1]==-1).
        Expectation: get_expand_impl returns None.
        Since get_expand_impl is overridden, we verify it once for the no-CP case;
        other no-CP tests do not repeat this check.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        qi = _build_layout(mesh, (Shard(0),), 4)   # S1 not sharded
        ki = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()

        assert op.get_expand_impl(None, None, [qi, ki, w, 'BSND']) is None, (
            "get_expand_impl should return None when S1 is not sharded (no BSND CP)"
        )

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_impl_tnd_no_cp_returns_callable_34(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: get_expand_impl always returns callable for TND (DP batch slicing is always needed).
        Description: TND with same T1 sharding on qi and ki.
        Expectation: get_expand_impl returns a callable wrapper.
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=4)
        mock_layout_plat.get_rank.return_value = 0
        mesh = init_device_mesh("npu", (4,), mesh_dim_names=("dp",))
        qi = _build_layout(mesh, (Shard(0),), 3)
        ki = _build_layout(mesh, (Shard(0),), 3)
        w = _build_layout(mesh, (Shard(0),), 2)
        op = self._get_op()

        impl = op.get_expand_impl(None, None, [qi, ki, w, 'TND'])
        assert callable(impl), (
            f"get_expand_impl should return callable for TND, got {type(impl)}"
        )

    def test_expand_impl_bsnd_cp_returns_callable_35(self):
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

    @patch("hyper_parallel.core.shard.ops.parallel_npu_dense_lightning_indexer_grad_kl_loss.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.dtensor.layout.platform")
    def test_expand_impl_tnd_cp_returns_callable_36(self, mock_layout_plat, mock_mesh_plat, mock_op_plat):
        """
        Feature: get_expand_impl returns callable when q_split > k_split (TND+CP).
        Description: qi on 8-device dp_cp mesh, ki replicated → q_split=8 > k_split=1.
        Expectation: Callable is returned.
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=8)
        mock_op_plat.get_rank.return_value = 0
        mock_layout_plat.get_rank.return_value = 0

        mesh_8 = init_device_mesh("npu", (8,), mesh_dim_names=("dp_cp",))
        qi = _build_layout(mesh_8, (Shard(0),), 3)    # alias_tensor_map[0]="dp_cp" → split=8
        ki = _build_layout(mesh_8, (Replicate(),), 3)  # alias_tensor_map[0]="None" → split=1

        op = self._get_op()
        impl = op.get_expand_impl(None, None, [qi, ki, MagicMock(), 'TND'])
        self.assertTrue(callable(impl), msg=f"Expected callable for TND+CP, got {type(impl)}")



if __name__ == "__main__":
    unittest.main()
