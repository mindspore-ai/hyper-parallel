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
"""Unit tests for SparseFlashAttentionDistributedOp."""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_npu_sparse_flash_attention import (
    SparseFlashAttentionDistributedOp,
    _normalize_sfa_args,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestSparseFlashAttentionDistributedOp(unittest.TestCase):
    """Unit tests for SparseFlashAttentionDistributedOp."""

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
        """Return a 1-D mesh of given size."""
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(device_type="npu", mesh_shape=(size,), mesh_dim_names=(name,))

    def _make_2x4_dp_cp_mesh(self, mock_platform):
        """Return a 2×4 (dp, cp) mesh — 8 devices."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "cp"))

    @staticmethod
    def _get_op():
        return get_distributed_op("npu_sparse_flash_attention")

    @staticmethod
    def _get_op_ms():
        return get_distributed_op("SparseFlashAttention")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mock_layouts_bsnd(self, q_tm, k_tm, v_tm=None, si_tm=None):
        """Return (q, k, v, si) mock layouts with given tensor_maps for BSND."""
        def _mock(tm):
            m = MagicMock()
            m.is_partial.return_value = False
            m.tensor_map = tm
            return m
        if v_tm is None:
            v_tm = k_tm
        if si_tm is None:
            si_tm = (q_tm[0], q_tm[1], -1, -1)
        return _mock(q_tm), _mock(k_tm), _mock(v_tm), _mock(si_tm)

    def _mock_layouts_tnd(self, q_tm, k_tm, v_tm=None, si_tm=None):
        """Return (q, k, v, si) mock layouts with given tensor_maps for TND."""
        def _mock(tm):
            m = MagicMock()
            m.is_partial.return_value = False
            m.tensor_map = tm
            return m
        if v_tm is None:
            v_tm = k_tm
        if si_tm is None:
            si_tm = (q_tm[0], -1, -1)
        return _mock(q_tm), _mock(k_tm), _mock(v_tm), _mock(si_tm)

    # ------------------------------------------------------------------
    # _normalize_sfa_args
    # ------------------------------------------------------------------

    def test_normalize_args_defaults_1(self):
        """
        Feature: _normalize_sfa_args fills in default values.
        Description: Call with only the 5 mandatory positional args.
        Expectation: Optional args get correct defaults.
        """
        q, k, v, si = object(), object(), object(), object()
        args, kwargs = _normalize_sfa_args(q, k, v, si, 1.0)
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertIs(args[2], v)
        self.assertIs(args[3], si)
        self.assertEqual(args[4], 1.0)
        self.assertIsNone(args[5], msg=f"block_table default should be None, got {args[5]}")
        self.assertEqual(args[11], 'BSND', msg=f"layout_query default should be 'BSND', got {args[11]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    def test_normalize_args_layout_tnd_2(self):
        """
        Feature: _normalize_sfa_args accepts layout_query keyword.
        Description: Pass layout_query='TND' as kwarg.
        Expectation: args[11] == 'TND'.
        """
        q, k, v, si = object(), object(), object(), object()
        args, _ = _normalize_sfa_args(q, k, v, si, 0.5, layout_query='TND')
        self.assertEqual(args[11], 'TND', msg=f"args[11] should be 'TND', got {args[11]}")

    def test_normalize_args_returns_empty_kwargs_3(self):
        """
        Feature: _normalize_sfa_args always returns empty kwargs dict.
        Description: Pass any combination of kwargs.
        Expectation: Returned kwargs dict is always empty.
        """
        q, k, v, si = object(), object(), object(), object()
        _, kwargs = _normalize_sfa_args(q, k, v, si, 0.5, layout_query='TND', sparse_mode=0)
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    # ------------------------------------------------------------------
    # YAML registration
    # ------------------------------------------------------------------

    def test_yaml_registration_torch_6(self):
        """
        Feature: YAML loader registers SparseFlashAttentionDistributedOp (torch name).
        Description: Call get_distributed_op with snake_case op name.
        Expectation: Returns a SparseFlashAttentionDistributedOp instance.
        """
        op = self._get_op()
        self.assertIsNotNone(op, msg="npu_sparse_flash_attention should be registered")
        self.assertIsInstance(
            op, SparseFlashAttentionDistributedOp,
            msg=f"Expected SparseFlashAttentionDistributedOp, got {type(op)}"
        )

    def test_yaml_registration_mindspore_7(self):
        """
        Feature: YAML loader registers SparseFlashAttentionDistributedOp (MindSpore name).
        Description: Call get_distributed_op with CamelCase op name.
        Expectation: Returns a SparseFlashAttentionDistributedOp instance.
        """
        op = self._get_op_ms()
        self.assertIsNotNone(op, msg="SparseFlashAttention should be registered")
        self.assertIsInstance(
            op, SparseFlashAttentionDistributedOp,
            msg=f"Expected SparseFlashAttentionDistributedOp, got {type(op)}"
        )

    # ------------------------------------------------------------------
    # infer_layout — BSND positive cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bsnd_all_replicated_8(self, mock_platform):
        """
        Feature: infer_layout BSND all replicated produces all-(-1) tensor_maps.
        Description: All inputs replicated on 1-D size-1 mesh.
        Expectation: attention_out/softmax_max/softmax_sum all have fully -1 tensor_maps.
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        q = _build_layout(mesh, (Replicate(),), 4)
        k = _build_layout(mesh, (Replicate(),), 4)
        v = _build_layout(mesh, (Replicate(),), 4)
        si = _build_layout(mesh, (Replicate(),), 4)

        op = self._get_op()
        (attn, smax, ssum), extra = op.infer_layout([q, k, v, si, 'BSND'])

        self.assertIsNone(extra)
        self.assertEqual(attn.tensor_map, (-1, -1, -1, -1),
                         msg=f"BSND replicated attn: expected (-1,-1,-1,-1), got {attn.tensor_map}")
        self.assertEqual(smax.tensor_map, (-1, -1, -1, -1),
                         msg=f"BSND replicated smax: expected (-1,-1,-1,-1), got {smax.tensor_map}")
        self.assertEqual(ssum.tensor_map, (-1, -1, -1, -1),
                         msg=f"BSND replicated ssum: expected (-1,-1,-1,-1), got {ssum.tensor_map}")

        # get_expand_impl returns None when S1 (dim 1) is not sharded (no CP).
        # For the replicated case, S1 is not sharded, so None is expected.
        assert op.get_expand_impl(None, (attn, smax, ssum), [q, k, v, si, 'BSND']) is None, (
            f"BSND replicated get_expand_impl should return None (S1 not sharded), "
            f"got {op.get_expand_impl(None, (attn, smax, ssum), [q, k, v, si, 'BSND'])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bsnd_dp_success_9(self, mock_platform):
        """
        Feature: infer_layout BSND with B-dim data parallel.
        Description: All inputs B-sharded on 1-D dp mesh (size 4).
        Expectation: attention_out tensor_map[0]==0; softmax_max tensor_map==(0,-1,−1,−1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")
        q = _build_layout(mesh, (Shard(0),), 4)
        k = _build_layout(mesh, (Shard(0),), 4)
        v = _build_layout(mesh, (Shard(0),), 4)
        si = _build_layout(mesh, (Shard(0),), 4)

        op = self._get_op()
        (attn, smax, ssum), _ = op.infer_layout([q, k, v, si, 'BSND'])

        self.assertEqual(attn.tensor_map, (0, -1, -1, -1),
                         msg=f"BSND DP attn: expected (0,-1,-1,-1), got {attn.tensor_map}")
        # softmax (B, N2, S1, N1/N2): B=q_tm[0]=0, N2=-1, S1=q_tm[1]=-1, N1/N2=-1
        self.assertEqual(smax.tensor_map, (0, -1, -1, -1),
                         msg=f"BSND DP smax: expected (0,-1,-1,-1), got {smax.tensor_map}")
        self.assertEqual(ssum.tensor_map, (0, -1, -1, -1),
                         msg=f"BSND DP ssum: expected (0,-1,-1,-1), got {ssum.tensor_map}")

        impl = op.get_expand_impl(None, (attn, smax, ssum), [q, k, v, si, 'BSND'])
        self.assertIsNone(impl, msg=(
            f"BSND DP get_expand_impl should return None (S1 not sharded), got {impl}"
        ))

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bsnd_cp_success_10(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: infer_layout BSND with S1-dim context parallel.
        Description: q/si sharded on S1, k/v replicated; 1-D cp mesh.
        Expectation: attention_out tensor_map==(−1,0,−1,−1); softmax_max==(−1,−1,0,−1);
            get_expand_impl returns callable (BSND+CP slices k/v to causal window).
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=4)
        mock_layout_plat.get_rank.return_value = 0

        mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("cp",))
        q = _build_layout(mesh, (Shard(1),), 4)    # (B,S1,N1,D) → S1 sharded
        k = _build_layout(mesh, (Replicate(),), 4)
        v = _build_layout(mesh, (Replicate(),), 4)
        si = _build_layout(mesh, (Shard(1),), 4)   # S1 matches query

        op = self._get_op()
        (attn, smax, ssum), _ = op.infer_layout([q, k, v, si, 'BSND'])

        # q_tm = (-1, 0, -1, -1)
        self.assertEqual(attn.tensor_map, (-1, 0, -1, -1),
                         msg=f"BSND CP attn: expected (-1,0,-1,-1), got {attn.tensor_map}")
        # softmax (B, N2, S1, N1/N2): q_tm[0]=-1, -1, q_tm[1]=0, -1
        self.assertEqual(smax.tensor_map, (-1, -1, 0, -1),
                         msg=f"BSND CP smax: expected (-1,-1,0,-1), got {smax.tensor_map}")

        # S1 is sharded → k/v are sliced to the causal window per rank.
        impl = op.get_expand_impl(None, (attn, smax, ssum), [q, k, v, si, 'BSND'])
        assert callable(impl), (
            f"BSND CP get_expand_impl should return callable, got {type(impl)}"
        )

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bsnd_dp_cp_success_11(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: infer_layout BSND with B-dim DP and S1-dim CP.
        Description: 2×4 (dp, cp) mesh; q B-sharded on dp, S1-sharded on cp.
        Expectation: attention_out tensor_map==(1,0,−1,−1); softmax_max==(1,−1,0,−1);
            get_expand_impl returns callable (BSND+CP slices k/v to causal window).
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=8)
        mock_layout_plat.get_rank.return_value = 0

        mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "cp"))
        q = _build_layout(mesh, (Shard(0), Shard(1)), 4)    # (B,S1,N1,D)
        k = _build_layout(mesh, (Shard(0), Replicate()), 4)
        v = _build_layout(mesh, (Shard(0), Replicate()), 4)
        si = _build_layout(mesh, (Shard(0), Shard(1)), 4)   # B+S1 matches q

        op = self._get_op()
        (attn, smax, ssum), _ = op.infer_layout([q, k, v, si, 'BSND'])

        # q_tm = (1, 0, -1, -1)
        self.assertEqual(attn.tensor_map, (1, 0, -1, -1),
                         msg=f"BSND DP+CP attn: expected (1,0,-1,-1), got {attn.tensor_map}")
        # softmax (B, N2, S1, N1/N2): (q_tm[0]=1, -1, q_tm[1]=0, -1)
        self.assertEqual(smax.tensor_map, (1, -1, 0, -1),
                         msg=f"BSND DP+CP smax: expected (1,-1,0,-1), got {smax.tensor_map}")
        self.assertEqual(ssum.tensor_map, (1, -1, 0, -1),
                         msg=f"BSND DP+CP ssum: expected (1,-1,0,-1), got {ssum.tensor_map}")

        # S1 is sharded → k/v are sliced to the causal window per rank.
        impl = op.get_expand_impl(None, (attn, smax, ssum), [q, k, v, si, 'BSND'])
        assert callable(impl), (
            f"BSND DP+CP get_expand_impl should return callable, got {type(impl)}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_output_independent_copies_12(self, mock_platform):
        """
        Feature: All three output layouts are independent deepcopy objects.
        Description: Check that returned layout objects are distinct.
        Expectation: attn, smax, ssum are all distinct objects.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        q = _build_layout(mesh, (Shard(0),), 4)
        k = _build_layout(mesh, (Shard(0),), 4)
        v = _build_layout(mesh, (Shard(0),), 4)
        si = _build_layout(mesh, (Shard(0),), 4)

        op = self._get_op()
        (attn, smax, ssum), _ = op.infer_layout([q, k, v, si, 'BSND'])

        self.assertIsNot(attn, smax,
                         msg="attention_out and softmax_max must be distinct layout objects")
        self.assertIsNot(smax, ssum,
                         msg="softmax_max and softmax_sum must be distinct layout objects")

    # ------------------------------------------------------------------
    # infer_layout — TND positive cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tnd_all_replicated_13(self, mock_platform):
        """
        Feature: infer_layout TND all replicated produces all-(-1) tensor_maps.
        Description: All inputs replicated on 1-D size-1 mesh.
        Expectation: attention_out/softmax_max/softmax_sum all have fully -1 tensor_maps.
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        q = _build_layout(mesh, (Replicate(),), 3)
        k = _build_layout(mesh, (Replicate(),), 3)
        v = _build_layout(mesh, (Replicate(),), 3)
        si = _build_layout(mesh, (Replicate(),), 3)

        op = self._get_op()
        (attn, smax, ssum), extra = op.infer_layout([q, k, v, si, 'TND'])

        self.assertIsNone(extra)
        self.assertEqual(attn.tensor_map, (-1, -1, -1),
                         msg=f"TND replicated attn: expected (-1,-1,-1), got {attn.tensor_map}")
        # softmax (N2, T1, N1/N2): q_tm[0]=-1
        self.assertEqual(smax.tensor_map, (-1, -1, -1),
                         msg=f"TND replicated smax: expected (-1,-1,-1), got {smax.tensor_map}")
        self.assertEqual(ssum.tensor_map, (-1, -1, -1),
                         msg=f"TND replicated ssum: expected (-1,-1,-1), got {ssum.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tnd_dp_success_14(self, mock_platform):
        """
        Feature: infer_layout TND with T1-dim data parallel.
        Description: query/si sharded on T1 (dim 0); 1-D dp mesh (size 4).
        Expectation: attention_out tensor_map==(0,-1,-1); softmax_max==(−1,0,−1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")
        q = _build_layout(mesh, (Shard(0),), 3)
        k = _build_layout(mesh, (Shard(0),), 3)
        v = _build_layout(mesh, (Shard(0),), 3)
        si = _build_layout(mesh, (Shard(0),), 3)

        op = self._get_op()
        (attn, smax, ssum), _ = op.infer_layout([q, k, v, si, 'TND'])

        self.assertEqual(attn.tensor_map, (0, -1, -1),
                         msg=f"TND DP attn: expected (0,-1,-1), got {attn.tensor_map}")
        # softmax (N2, T1, N1/N2): (-1, q_tm[0]=0, -1)
        self.assertEqual(smax.tensor_map, (-1, 0, -1),
                         msg=f"TND DP smax: expected (-1,0,-1), got {smax.tensor_map}")
        self.assertEqual(ssum.tensor_map, (-1, 0, -1),
                         msg=f"TND DP ssum: expected (-1,0,-1), got {ssum.tensor_map}")

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tnd_cp_expand_impl_callable_15(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: get_expand_impl returns callable when q T1 sharded more than k (TND+CP).
        Description: q sharded on 8-device dp_cp mesh, k replicated → q_split=8 > k_split=1.
        Expectation: get_expand_impl returns a callable.
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=8)
        mock_layout_plat.get_rank.return_value = 0

        mesh = init_device_mesh("npu", (8,), mesh_dim_names=("dp_cp",))
        q = _build_layout(mesh, (Shard(0),), 3)
        k = _build_layout(mesh, (Replicate(),), 3)
        v = _build_layout(mesh, (Replicate(),), 3)
        si = _build_layout(mesh, (Shard(0),), 3)

        op = self._get_op()
        result, _ = op.infer_layout([q, k, v, si, 'TND'])
        impl = op.get_expand_impl(None, result, [q, k, v, si, 'TND'])
        self.assertTrue(callable(impl),
                        msg=f"TND+CP get_expand_impl should return callable, got {type(impl)}")

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tnd_dp_cp_2d_mesh_expand_impl_callable_16(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: get_expand_impl returns callable for TND with 2-D dp+cp mesh.
        Description: 2×4 (dp, cp) mesh; q T1 sharded by BOTH dp and cp (combined 8-way
            split); k T2 sharded by dp only (2-way split). q_split=8 > k_split=2 triggers
            _tnd_cp_impl. Verifies the multi-axis get_split_id fix is plumbed correctly.
        Expectation: get_expand_impl returns callable; attention_out and softmax_max have
            multi-axis tensor_map entries for T1.
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=8)
        mock_layout_plat.get_rank.return_value = 0

        mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "cp"))
        # dp AND cp both shard T1 of q/si → combined 8-way split
        q = _build_layout(mesh, (Shard(0), Shard(0)), 3)
        # dp shards T2 of k; cp Replicate → 2-way split
        k = _build_layout(mesh, (Shard(0), Replicate()), 3)
        v = _build_layout(mesh, (Shard(0), Replicate()), 3)
        si = _build_layout(mesh, (Shard(0), Shard(0)), 3)

        op = self._get_op()
        (attn, smax, ssum), _ = op.infer_layout([q, k, v, si, 'TND'])

        # attention_out = deepcopy(q_layout); T1 tensor_map[0] is a tuple of mesh axes
        # (cp axis=1 first, dp axis=0 second) — both must be present for combined sharding.
        self.assertIsInstance(attn.tensor_map[0], tuple,
                              msg=f"TND dp+cp attn T1 must be a tuple, got {attn.tensor_map[0]!r}")
        self.assertIn(0, attn.tensor_map[0],
                      msg=f"TND dp+cp attn T1 tuple must include dp axis 0, got {attn.tensor_map[0]!r}")
        self.assertIn(1, attn.tensor_map[0],
                      msg=f"TND dp+cp attn T1 tuple must include cp axis 1, got {attn.tensor_map[0]!r}")
        # softmax (N2, T1, N1/N2): tensor_map[1] = q_tm[0] (same multi-axis tuple)
        self.assertIsInstance(smax.tensor_map[1], tuple,
                              msg=f"TND dp+cp smax T1 must be a tuple, got {smax.tensor_map[1]!r}")
        self.assertIn(0, smax.tensor_map[1],
                      msg=f"TND dp+cp smax T1 tuple must include dp axis 0, got {smax.tensor_map[1]!r}")
        self.assertIn(1, smax.tensor_map[1],
                      msg=f"TND dp+cp smax T1 tuple must include cp axis 1, got {smax.tensor_map[1]!r}")

        impl = op.get_expand_impl(None, (attn, smax, ssum), [q, k, v, si, 'TND'])
        self.assertTrue(callable(impl),
                        msg=f"TND dp+cp get_expand_impl should return callable, got {type(impl)}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tnd_no_cp_returns_callable_17(self, mock_platform):
        """
        Feature: get_expand_impl returns callable when q and k equally sharded TND.
        Description: q and k both sharded on same T1 dimension (DP only, no CP offset).
            Even without CP, _tnd_cp_impl is returned to clamp actual_seq_lengths to
            T1_local (cp_rank=0 path). This is required because the caller passes
            full (replicated) seq_lengths; the wrapper clips them to the local slice.
        Expectation: get_expand_impl returns callable with cp_rank=0.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")
        q = _build_layout(mesh, (Shard(0),), 3)
        k = _build_layout(mesh, (Shard(0),), 3)
        v = _build_layout(mesh, (Shard(0),), 3)
        si = _build_layout(mesh, (Shard(0),), 3)

        op = self._get_op()
        result, _ = op.infer_layout([q, k, v, si, 'TND'])
        impl = op.get_expand_impl(None, result, [q, k, v, si, 'TND'])
        self.assertTrue(callable(impl),
                        msg=f"TND DP-only get_expand_impl should return callable, got {impl}")

    # ------------------------------------------------------------------
    # infer_layout — negative / error cases
    # ------------------------------------------------------------------

    def test_partial_input_raises_18(self):
        """
        Feature: Partial inputs are rejected.
        Description: Pass a layout with is_partial() returning True.
        Expectation: Raises ValueError.
        """
        partial_layout = MagicMock()
        partial_layout.is_partial.return_value = True
        normal_layout = MagicMock()
        normal_layout.is_partial.return_value = False
        normal_layout.tensor_map = (-1, -1, -1, -1)

        op = self._get_op()
        with self.assertRaises(ValueError):
            op.infer_layout([partial_layout, normal_layout, normal_layout, normal_layout, 'BSND'])

    def test_pa_bsnd_raises_19(self):
        """
        Feature: PA_BSND layout is not supported in distributed mode.
        Description: Pass layout_str='PA_BSND'.
        Expectation: Raises ValueError mentioning 'PA_BSND'.
        """
        q, k, v, si = self._mock_layouts_bsnd((-1, -1, -1, -1), (-1, -1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "PA_BSND"):
            op.infer_layout([q, k, v, si, 'PA_BSND'])

    def test_bsnd_n1_sharded_raises_20(self):
        """
        Feature: BSND N1 (dim 2) of query sharding is forbidden.
        Description: query tensor_map[2] != -1 (TP head sharding attempt).
        Expectation: Raises ValueError mentioning 'N1'.
        """
        q, k, v, si = self._mock_layouts_bsnd((0, -1, 0, -1), (0, -1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1"):
            op.infer_layout([q, k, v, si, 'BSND'])

    def test_tnd_n1_sharded_raises_21(self):
        """
        Feature: TND N1 (dim 1) of query sharding is forbidden.
        Description: query tensor_map[1] != -1 (TP head sharding attempt).
        Expectation: Raises ValueError mentioning 'N1'.
        """
        q, k, v, si = self._mock_layouts_tnd((-1, 0, -1), (-1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N1"):
            op.infer_layout([q, k, v, si, 'TND'])

    def test_query_d_sharded_raises_22(self):
        """
        Feature: BSND D (dim 3) of query sharding is forbidden.
        Description: query tensor_map[3] != -1.
        Expectation: Raises ValueError mentioning 'D'.
        """
        q, k, v, si = self._mock_layouts_bsnd((-1, -1, -1, 0), (-1, -1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([q, k, v, si, 'BSND'])

    def test_key_s2_sharded_raises_23(self):
        """
        Feature: BSND S2 (dim 1) of key sharding is forbidden.
        Description: key tensor_map[1] != -1.
        Expectation: Raises ValueError mentioning 'S2'.
        """
        q, k, v, si = self._mock_layouts_bsnd((-1, -1, -1, -1), (-1, 0, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "S2"):
            op.infer_layout([q, k, v, si, 'BSND'])

    def test_key_n2_sharded_raises_24(self):
        """
        Feature: BSND N2 (dim 2) of key sharding is forbidden.
        Description: key tensor_map[2] != -1.
        Expectation: Raises ValueError mentioning 'N2'.
        """
        q, k, v, si = self._mock_layouts_bsnd((-1, -1, -1, -1), (-1, -1, 0, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N2"):
            op.infer_layout([q, k, v, si, 'BSND'])

    def test_batch_mismatch_raises_25(self):
        """
        Feature: BSND B sharding mismatch between key and query is rejected.
        Description: query B on mesh dim 0, key B on mesh dim 1.
        Expectation: Raises ValueError mentioning 'B (dim 0) sharding of key'.
        """
        q, k, v, si = self._mock_layouts_bsnd((0, -1, -1, -1), (1, -1, -1, -1))
        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"B \(dim 0\) sharding of key"):
            op.infer_layout([q, k, v, si, 'BSND'])

    def test_si_s1_mismatch_raises_26(self):
        """
        Feature: BSND S1 mismatch between sparse_indices and query is rejected.
        Description: query S1 on mesh dim 0, sparse_indices S1 on mesh dim 1.
        Expectation: Raises ValueError mentioning 'S1 (dim 1) sharding of sparse_indices'.
        """
        q = MagicMock()
        q.is_partial.return_value = False
        q.tensor_map = (0, 0, -1, -1)
        k = MagicMock()
        k.is_partial.return_value = False
        k.tensor_map = (0, -1, -1, -1)
        v = MagicMock()
        v.is_partial.return_value = False
        v.tensor_map = (0, -1, -1, -1)
        si = MagicMock()
        si.is_partial.return_value = False
        si.tensor_map = (0, 1, -1, -1)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, r"S1 \(dim 1\) sharding of sparse_indices"):
            op.infer_layout([q, k, v, si, 'BSND'])


if __name__ == "__main__":
    unittest.main()
