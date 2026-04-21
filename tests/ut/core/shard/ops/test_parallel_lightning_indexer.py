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
"""Unit tests for LightningIndexerDistributedOp."""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.shard.ops.parallel_lightning_indexer import (
    LightningIndexerDistributedOp,
    _normalize_lightning_indexer_args,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class TestLightningIndexer(unittest.TestCase):
    """Unit tests for LightningIndexerDistributedOp."""

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
        return get_distributed_op("LightningIndexer")

    # ------------------------------------------------------------------
    # _normalize_lightning_indexer_args
    # ------------------------------------------------------------------

    def test_normalize_args_defaults_1(self):
        """
        Feature: _normalize_lightning_indexer_args fills in default values.
        Description: Call with only the 3 mandatory positional args.
        Expectation: Query/key/weights in positional; optional args in kwargs.
        """
        q, k, w = object(), object(), object()
        args, kwargs = _normalize_lightning_indexer_args(q, k, w)
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertIs(args[2], w)
        self.assertEqual(len(args), 3)
        self.assertIsNone(kwargs['actual_seq_lengths_query'])
        self.assertIsNone(kwargs['actual_seq_lengths_key'])
        self.assertIsNone(kwargs['block_table'])
        self.assertEqual(kwargs['layout_query'], 'BSND')
        self.assertEqual(kwargs['layout_key'], 'BSND')
        self.assertEqual(kwargs['sparse_count'], 2048)
        self.assertEqual(kwargs['sparse_mode'], 3)
        self.assertEqual(kwargs['return_value'], False)

    def test_normalize_args_layout_tnd_2(self):
        """
        Feature: _normalize_lightning_indexer_args accepts layout keyword.
        Description: Pass layout_query='TND' as kwarg.
        Expectation: kwargs['layout_query'] == 'TND'.
        """
        q, k, w = object(), object(), object()
        _, kwargs = _normalize_lightning_indexer_args(q, k, w, layout_query='TND')
        self.assertEqual(kwargs['layout_query'], 'TND')

    def test_normalize_args_return_value_3(self):
        """
        Feature: _normalize_lightning_indexer_args accepts return_value keyword.
        Description: Pass return_value=True as kwarg.
        Expectation: kwargs['return_value'] == True.
        """
        q, k, w = object(), object(), object()
        _, kwargs = _normalize_lightning_indexer_args(q, k, w, return_value=True)
        self.assertEqual(kwargs['return_value'], True)

    # ------------------------------------------------------------------
    # YAML registration
    # ------------------------------------------------------------------

    def test_yaml_registration_4(self):
        """
        Feature: YAML loader registers LightningIndexerDistributedOp.
        Description: Call get_distributed_op with the op name.
        Expectation: Returns a non-None LightningIndexerDistributedOp instance.
        """
        op = self._get_op()
        self.assertIsNotNone(op, msg="LightningIndexer should be registered")
        self.assertIsInstance(
            op, LightningIndexerDistributedOp,
            msg=f"Expected LightningIndexerDistributedOp, got {type(op)}"
        )

    # ------------------------------------------------------------------
    # preprocess — MindSpore path
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.shard.ops.parallel_lightning_indexer.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_3_positional_args_5(self, mock_mesh_plat, mock_op_plat):
        """
        Feature: preprocess packs q/k/w as positional, rest as kwargs.
        Description: Patch platform type to MINDSPORE, call preprocess.
        Expectation: local_args has 3 elements; kwargs hold 10 optional args.
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

        self.assertEqual(len(local_args), 3,
                         msg=f"positional args should be 3 (q,k,w), got {len(local_args)}")
        self.assertIn('actual_seq_lengths_query', local_kwargs)
        self.assertIn('layout_query', local_kwargs)
        self.assertEqual(local_kwargs['layout_query'], 'BSND')
        self.assertEqual(len(cache_values), 4,
                         msg=f"cache_values should have 4 elements, got {len(cache_values)}")
        self.assertEqual(cache_values[3], 'BSND',
                         msg=f"cache_values[3] should be 'BSND', got {cache_values[3]}")

    @patch("hyper_parallel.core.shard.ops.parallel_lightning_indexer.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_torch_3_positional_args_6(self, mock_mesh_plat, mock_op_plat):
        """
        Feature: preprocess puts optional args in kwargs on PyTorch.
        Description: Patch platform type to PYTORCH, pass layout_query='TND'.
        Expectation: local_args has 3 elements; layout_query='TND' in local_kwargs.
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
        local_args, local_kwargs, cache_values = op.preprocess(
            (dtq, dtk, dtw), {'layout_query': 'TND'}
        )

        self.assertEqual(len(local_args), 3,
                         msg=f"Torch path should have 3 positional args, got {len(local_args)}")
        self.assertEqual(local_kwargs.get('layout_query'), 'TND',
                         msg=f"layout_query should be TND in kwargs, got {local_kwargs}")
        self.assertIn('actual_seq_lengths_query', local_kwargs)
        self.assertIn('actual_seq_lengths_key', local_kwargs)
        self.assertEqual(cache_values[3], 'TND',
                         msg=f"cache_values[3] should be 'TND', got {cache_values[3]}")

    # ------------------------------------------------------------------
    # infer_layout — BSND
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_replicated_7(self, mock_platform):
        """
        Feature: infer_layout BSND replicated produces correct output tensor_map.
        Description: All inputs replicated on 1-D mesh.
        Expectation: Two outputs, tensor_map = (-1,-1,-1,-1) each.
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        q = _build_layout(mesh, (Replicate(),), 4)
        k = _build_layout(mesh, (Replicate(),), 4)
        w = _build_layout(mesh, (Replicate(),), 3)
        op = self._get_op()
        result, extra = op.infer_layout([q, k, w, 'BSND'])

        self.assertIsNone(extra)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].tensor_map, (-1, -1, -1, -1),
                         msg=f"Expected (-1,-1,-1,-1), got {result[0].tensor_map}")
        self.assertEqual(result[1].tensor_map, (-1, -1, -1, -1),
                         msg=f"Expected (-1,-1,-1,-1), got {result[1].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_dp_only_8(self, mock_platform):
        """
        Feature: infer_layout BSND with batch-dimension DP.
        Description: q/k/w all have B sharded on dp (1-D mesh, size 4).
        Expectation: Output tensor_map[0] == 0 (batch from dp); S1, N2, sparse_count replicated.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")
        q = _build_layout(mesh, (Shard(0),), 4)
        k = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'BSND'])

        self.assertEqual(result[0].tensor_map[0], q.tensor_map[0],
                         msg=(f"B should propagate: expected {q.tensor_map[0]}, "
                              f"got {result[0].tensor_map[0]}"))
        self.assertEqual(result[0].tensor_map[1], -1,
                         msg=f"S1 should be replicated when not sharded, got {result[0].tensor_map[1]}")
        self.assertEqual(result[0].tensor_map[2], -1,
                         msg=f"N2 should be replicated, got {result[0].tensor_map[2]}")
        self.assertEqual(result[0].tensor_map[3], -1,
                         msg=f"sparse_count should be replicated, got {result[0].tensor_map[3]}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_dp_cp_9(self, mock_platform):
        """
        Feature: infer_layout BSND with dp=2, cp=4.
        Description: q_index on (dp,cp,None,None), k_index on (dp,None,None,None),
                     w on (dp,cp,None).
        Expectation: Output (B,S1,N2,SC) has B on dp, S1 on cp, N2/SC replicated.
        """
        mesh = self._make_2x4_dp_cp_mesh(mock_platform)
        q = _build_layout(mesh, (Shard(0), Shard(1)), 4)
        k = _build_layout(mesh, (Shard(0), Replicate()), 4)
        w = _build_layout(mesh, (Shard(0), Shard(1)), 3)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'BSND'])

        # q_tm[0]=1 (B→dp), q_tm[1]=0 (S1→cp) → out=(1,0,-1,-1)
        self.assertEqual(result[0].tensor_map, (1, 0, -1, -1),
                         msg=f"Expected (1,0,-1,-1), got {result[0].tensor_map}")
        self.assertEqual(result[1].tensor_map, (1, 0, -1, -1),
                         msg=f"Expected (1,0,-1,-1), got {result[1].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_outputs_independent_copies_10(self, mock_platform):
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
    def test_infer_layout_bsnd_output_has_placements_11(self, mock_platform):
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
    def test_infer_layout_tnd_replicated_12(self, mock_platform):
        """
        Feature: infer_layout TND replicated produces correct output tensor_map.
        Description: All inputs replicated on 1-D mesh.
        Expectation: Two outputs, tensor_map = (-1,-1,-1) each.
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        q = _build_layout(mesh, (Replicate(),), 3)
        k = _build_layout(mesh, (Replicate(),), 3)
        w = _build_layout(mesh, (Replicate(),), 2)
        op = self._get_op()
        result, extra = op.infer_layout([q, k, w, 'TND'])

        self.assertIsNone(extra)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].tensor_map, (-1, -1, -1),
                         msg=f"Replicated TND output should be (-1,-1,-1), got {result[0].tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_dp_cp_merged_13(self, mock_platform):
        """
        Feature: infer_layout TND with T1 sharded on merged dp_cp axis.
        Description: q_index (T1,N1,D) on dp_cp axis; k_index replicated.
        Expectation: Output (T1,N2,sparse_count): T1 on dp_cp, N2 replicated, sparse_count replicated.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8, name="dp_cp")
        q = _build_layout(mesh, (Shard(0),), 3)
        k = _build_layout(mesh, (Replicate(),), 3)
        w = _build_layout(mesh, (Shard(0),), 2)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'TND'])

        self.assertEqual(result[0].tensor_map, (0, -1, -1),
                         msg=f"Expected (0,-1,-1) for TND dp_cp, got {result[0].tensor_map}")

    # ------------------------------------------------------------------
    # infer_layout — partial raises
    # ------------------------------------------------------------------

    def test_infer_layout_raises_on_partial_14(self):
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

    def test_infer_layout_bsnd_rejects_n1_sharded_15(self):
        """
        Feature: BSND N1 (dim 2) sharding rejected.
        Description: query has tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N1'.
        """
        q, k, w = self._make_mock_layouts((0, -1, 0, -1), (0, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_d_sharded_16(self):
        """
        Feature: BSND D (dim 3) sharding rejected.
        Description: query has tensor_map[3] != -1.
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, 0), (0, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_s2_sharded_17(self):
        """
        Feature: BSND S2 (dim 1) of key sharding rejected.
        Description: key has tensor_map[1] != -1.
        Expectation: Raises ValueError mentioning 'S2'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, 0, -1, -1), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "S2"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_n2_sharded_18(self):
        """
        Feature: BSND N2 (dim 2) of key sharding rejected.
        Description: key has tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N2'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, -1, 0, -1), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N2"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_key_d_sharded_19(self):
        """
        Feature: BSND D (dim 3) of key sharding rejected.
        Description: key has tensor_map[3] != -1.
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1, -1), (-1, -1, -1, 0), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_batch_mismatch_20(self):
        """
        Feature: BSND B sharding mismatch between query and key rejected.
        Description: query B on mesh dim 0, key B on mesh dim 1.
        Expectation: Raises ValueError mentioning 'B (dim 0) sharding'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, -1), (1, -1, -1, -1), (0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError,
                                     r"B \(dim 0\) sharding of query and key should match"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_b_mismatch_21(self):
        """
        Feature: BSND B mismatch between weights and query rejected.
        Description: weights.B on mesh dim 1, query.B on mesh dim 0.
        Expectation: Raises ValueError mentioning 'B (dim 0) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, -1), (0, -1, -1, -1), (1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"B \(dim 0\) sharding of weights"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_s1_mismatch_22(self):
        """
        Feature: BSND S1 mismatch between weights and query rejected.
        Description: weights.S1 on dim 0, query.S1 on dim 1.
        Expectation: Raises ValueError mentioning 'S1 (dim 1) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, 1, -1, -1), (0, -1, -1, -1), (0, 0, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"S1 \(dim 1\) sharding of weights"):
            op.infer_layout([q, k, w, 'BSND'])

    def test_infer_layout_bsnd_rejects_weights_n1_sharded_23(self):
        """
        Feature: BSND N1 (dim 2) of weights sharding rejected.
        Description: weights.tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N1'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1, -1), (0, -1, -1, -1), (0, -1, 0))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1"):
            op.infer_layout([q, k, w, 'BSND'])

    # ------------------------------------------------------------------
    # infer_layout — TND validation errors
    # ------------------------------------------------------------------

    def test_infer_layout_tnd_rejects_n1_sharded_24(self):
        """
        Feature: TND N1 (dim 1) of query sharding rejected.
        Description: query has tensor_map[1] != -1 (TND).
        Expectation: Raises ValueError mentioning 'N1'.
        """
        q, k, w = self._make_mock_layouts((-1, 0, -1), (-1, -1, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_d_sharded_25(self):
        """
        Feature: TND D (dim 2) of query sharding rejected.
        Description: query has tensor_map[2] != -1 (TND).
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, 0), (-1, -1, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_key_n2_sharded_26(self):
        """
        Feature: TND N2 (dim 1) of key sharding rejected.
        Description: key has tensor_map[1] != -1 (TND).
        Expectation: Raises ValueError mentioning 'N2'.
        """
        q, k, w = self._make_mock_layouts((-1, -1, -1), (-1, 0, -1), (-1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N2"):
            op.infer_layout([q, k, w, 'TND'])

    def test_infer_layout_tnd_rejects_weights_t1_mismatch_27(self):
        """
        Feature: TND T1 sharding mismatch between weights and query rejected.
        Description: weights.T1 on mesh dim 1, query.T1 on mesh dim 0.
        Expectation: Raises ValueError mentioning 'T1 (dim 0) sharding of weights'.
        """
        q, k, w = self._make_mock_layouts((0, -1, -1), (-1, -1, -1), (1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError,
                                     r"T1 \(dim 0\) sharding of weights should match query"):
            op.infer_layout([q, k, w, 'TND'])

    # ------------------------------------------------------------------
    # get_expand_impl
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_impl_bsnd_no_cp_returns_none_28(self, mock_platform):
        """
        Feature: get_expand_impl returns None when S1 is not sharded (no CP).
        Description: BSND q_index with S1 replicated.
        Since get_expand_impl is overridden, we verify it once for the no-CP case.
        Expectation: get_expand_impl returns None.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        q = _build_layout(mesh, (Shard(0),), 4)
        k = _build_layout(mesh, (Shard(0),), 4)
        w = _build_layout(mesh, (Shard(0),), 3)
        op = self._get_op()
        result, _ = op.infer_layout([q, k, w, 'BSND'])

        assert op.get_expand_impl(None, result, [q, k, w, 'BSND']) is None, (
            "get_expand_impl should return None when S1 is not sharded (no BSND CP)"
        )

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_expand_impl_tnd_no_cp_returns_callable_29(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: get_expand_impl returns callable for TND even without CP.
        Description: TND with same T1 sharding on q and k. Seq_len adjustment is always needed.
        Expectation: get_expand_impl returns a callable.
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

    def test_expand_impl_bsnd_cp_returns_callable_30(self):
        """
        Feature: get_expand_impl returns callable when S1 is sharded (BSND+CP).
        Description: q_index tensor_map[1] != -1 triggers CP path.
        Expectation: Callable is returned.
        """
        q_mock = MagicMock()
        q_mock.tensor_map = (-1, 0, -1, -1)
        op = self._get_op()
        impl = op.get_expand_impl(None, None, [q_mock, MagicMock(), MagicMock(), 'BSND'])
        self.assertTrue(callable(impl),
                        msg=f"Expected callable for BSND+CP, got {type(impl)}")

    @patch("hyper_parallel.core.shard.ops.parallel_lightning_indexer.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.dtensor.layout.platform")
    def test_expand_impl_tnd_cp_returns_callable_31(self, mock_layout_plat, mock_mesh_plat, mock_op_plat):
        """
        Feature: get_expand_impl returns callable when q_split > k_split (TND+CP).
        Description: q on 8-device dp_cp mesh, k replicated → q_split=8 > k_split=1.
        Expectation: Callable is returned.
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=8)
        mock_op_plat.get_rank.return_value = 0
        mock_layout_plat.get_rank.return_value = 0

        mesh_8 = init_device_mesh("npu", (8,), mesh_dim_names=("dp_cp",))
        q = _build_layout(mesh_8, (Shard(0),), 3)
        k = _build_layout(mesh_8, (Replicate(),), 3)

        op = self._get_op()
        impl = op.get_expand_impl(None, None, [q, k, MagicMock(), 'TND'])
        self.assertTrue(callable(impl),
                        msg=f"Expected callable for TND+CP, got {type(impl)}")


if __name__ == "__main__":
    unittest.main()
