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
"""Unit tests for RotaryPositionEmbeddingDistributedOp."""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_rotary_position_embedding import (
    RotaryPositionEmbeddingDistributedOp,
    _normalize_npu_rotary_mul_args,
    _normalize_rpe_args,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestNormalizeRpeArgs(unittest.TestCase):
    """Unit tests for _normalize_rpe_args."""

    def test_three_positional_args_mode_defaults_to_zero_1(self):
        """
        Feature: _normalize_rpe_args fills in default mode=0.
        Description: Call with only x, cos, sin positional args.
        Expectation: args == (x, cos, sin, 0), kwargs == {}.
        """
        x, cos, sin = object(), object(), object()
        args, kwargs = _normalize_rpe_args(x, cos, sin)
        self.assertIs(args[0], x, msg=f"args[0] should be x, got {args[0]}")
        self.assertIs(args[1], cos, msg=f"args[1] should be cos, got {args[1]}")
        self.assertIs(args[2], sin, msg=f"args[2] should be sin, got {args[2]}")
        self.assertEqual(args[3], 0, msg=f"default mode should be 0, got {args[3]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    def test_mode_keyword_arg_2(self):
        """
        Feature: _normalize_rpe_args accepts mode as keyword argument.
        Description: Call with mode=1 as kwarg.
        Expectation: args[3] == 1.
        """
        x, cos, sin = object(), object(), object()
        args, _ = _normalize_rpe_args(x, cos, sin, mode=1)
        self.assertEqual(args[3], 1, msg=f"mode kwarg should be 1, got {args[3]}")

    def test_mode_positional_arg_3(self):
        """
        Feature: _normalize_rpe_args accepts mode as positional argument.
        Description: Call with mode=2 as positional arg.
        Expectation: args[3] == 2.
        """
        x, cos, sin = object(), object(), object()
        args, kwargs = _normalize_rpe_args(x, cos, sin, 2)
        self.assertEqual(args[3], 2, msg=f"mode positional should be 2, got {args[3]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")


class TestYamlRegistration(unittest.TestCase):
    """Unit tests for YAML-driven op registration."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def test_pascal_case_registration_4(self):
        """
        Feature: YAML registers RotaryPositionEmbeddingDistributedOp (MindSpore name).
        Description: Call get_distributed_op with PascalCase op name.
        Expectation: Returns a RotaryPositionEmbeddingDistributedOp instance.
        """
        op = get_distributed_op("RotaryPositionEmbedding")
        self.assertIsNotNone(
            op,
            msg="RotaryPositionEmbedding should be registered via YAML"
        )
        self.assertIsInstance(
            op, RotaryPositionEmbeddingDistributedOp,
            msg=f"Expected RotaryPositionEmbeddingDistributedOp, got {type(op)}"
        )


class TestInferLayoutPositive(unittest.TestCase):
    """Positive layout inference tests for RotaryPositionEmbeddingDistributedOp."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    @staticmethod
    def _get_op():
        return get_distributed_op("RotaryPositionEmbedding")

    def _setup_mock_platform(self, mock_platform, world_size=8):
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.split_group.return_value = MagicMock()
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.asnumpy() if hasattr(t, "asnumpy") else np.array(t)
        )

    def _make_1d_mesh(self, mock_platform, size=4, name="dp"):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(device_type="npu", mesh_shape=(size,), mesh_dim_names=(name,))

    def _make_2d_mesh(self, mock_platform, shape=(2, 2), names=("dp", "tp")):
        self._setup_mock_platform(mock_platform, world_size=shape[0] * shape[1])
        return init_device_mesh(device_type="npu", mesh_shape=shape, mesh_dim_names=names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_all_replicated_4d_5(self, mock_platform):
        """
        Feature: infer_layout with all inputs replicated (4-D).
        Description: 1-D size-1 mesh, all Replicate.
        Expectation: output.tensor_map == (-1, -1, -1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        x = _build_layout(mesh, (Replicate(),), 4)
        cos = _build_layout(mesh, (Replicate(),), 4)
        sin = _build_layout(mesh, (Replicate(),), 4)

        op = self._get_op()
        (out,), extra = op.infer_layout([x, cos, sin])

        self.assertIsNone(extra)
        self.assertEqual(
            out.tensor_map, (-1, -1, -1, -1),
            msg=f"All-replicated output should be (-1,-1,-1,-1), got {out.tensor_map}"
        )
        # get_expand_impl is not overridden; verify once here that it returns None.
        # All other tests do not need to repeat this check.
        assert op.get_expand_impl(None, (out,), [x, cos, sin]) is None, (
            f"get_expand_impl should return None for RPE (no expand logic), "
            f"got {op.get_expand_impl(None, (out,), [x, cos, sin])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dp_b_cos_replicated_6(self, mock_platform):
        """
        Feature: infer_layout BNSD B-dim DP, cos/sin broadcast (11SD shape).
        Description: x Shard(0) on 1-D dp mesh, cos/sin Replicate.
        Expectation: output.tensor_map == (0, -1, -1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")
        x = _build_layout(mesh, (Shard(0),), 4)
        cos = _build_layout(mesh, (Replicate(),), 4)
        sin = _build_layout(mesh, (Replicate(),), 4)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        self.assertEqual(
            out.tensor_map, (0, -1, -1, -1),
            msg=f"DP-B output should be (0,-1,-1,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tp_n_cos_broadcast_7(self, mock_platform):
        """
        Feature: infer_layout BNSD N-dim TP, cos/sin in 11SD shape (N=1 broadcast).
        Description: x Shard(1) on 1-D tp mesh, cos/sin Replicate (N=1 in cos).
        Expectation: output.tensor_map == (-1, 0, -1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="tp")
        x = _build_layout(mesh, (Shard(1),), 4)
        cos = _build_layout(mesh, (Replicate(),), 4)
        sin = _build_layout(mesh, (Replicate(),), 4)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        self.assertEqual(
            out.tensor_map, (-1, 0, -1, -1),
            msg=f"TP-N output should be (-1,0,-1,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cp_s_cos_same_shard_8(self, mock_platform):
        """
        Feature: infer_layout BNSD S-dim CP, cos/sin sharded same as x.
        Description: x Shard(2), cos/sin Shard(2) on 1-D sp mesh.
        Expectation: output.tensor_map == (-1, -1, 0, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="sp")
        x = _build_layout(mesh, (Shard(2),), 4)
        cos = _build_layout(mesh, (Shard(2),), 4)
        sin = _build_layout(mesh, (Shard(2),), 4)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        self.assertEqual(
            out.tensor_map, (-1, -1, 0, -1),
            msg=f"CP-S output should be (-1,-1,0,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cp_s_cos_replicated_9(self, mock_platform):
        """
        Feature: infer_layout BNSD S-dim CP, cos/sin Replicate (S=1 broadcast case).
        Description: x Shard(2), cos/sin Replicate on 1-D sp mesh.
        Expectation: output.tensor_map == (-1, -1, 0, -1), no error raised.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="sp")
        x = _build_layout(mesh, (Shard(2),), 4)
        cos = _build_layout(mesh, (Replicate(),), 4)
        sin = _build_layout(mesh, (Replicate(),), 4)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        self.assertEqual(
            out.tensor_map, (-1, -1, 0, -1),
            msg=f"CP-S with cos Replicate should be (-1,-1,0,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dp_tp_2d_mesh_10(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: infer_layout BNSD with DP+TP on 2-D mesh.
        Description: mesh (2,2) dp×tp; x Shard(B)×Shard(N), cos Shard(B)×Replicate.
        Expectation:
          x.tensor_map == (1, 0, -1, -1)  [B→dp(axis1), N→tp(axis0)]
          output.tensor_map == (1, 0, -1, -1).
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=4)
        mock_layout_plat.get_rank.return_value = 0

        mesh = self._make_2d_mesh(mock_mesh_plat, shape=(2, 2), names=("dp", "tp"))
        x = _build_layout(mesh, (Shard(0), Shard(1)), 4)    # B on dp, N on tp
        cos = _build_layout(mesh, (Shard(0), Replicate()), 4)  # B on dp, N replicated
        sin = _build_layout(mesh, (Shard(0), Replicate()), 4)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        # dp is mesh-axis 1, tp is mesh-axis 0 in tensor_map convention
        self.assertEqual(
            out.tensor_map, (1, 0, -1, -1),
            msg=f"DP+TP output should be (1,0,-1,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dp_cp_2d_mesh_11(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: infer_layout BNSD with DP+CP on 2-D mesh.
        Description: mesh (2,2) dp×sp; x Shard(B)×Shard(S), cos Shard(B)×Shard(S).
        Expectation: output.tensor_map == (1, -1, 0, -1).
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=4)
        mock_layout_plat.get_rank.return_value = 0

        mesh = self._make_2d_mesh(mock_mesh_plat, shape=(2, 2), names=("dp", "sp"))
        x = _build_layout(mesh, (Shard(0), Shard(2)), 4)    # B on dp, S on sp
        cos = _build_layout(mesh, (Shard(0), Shard(2)), 4)
        sin = _build_layout(mesh, (Shard(0), Shard(2)), 4)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        # dp→axis1, sp→axis0; B→tm[0]=1, S→tm[2]=0
        self.assertEqual(
            out.tensor_map, (1, -1, 0, -1),
            msg=f"DP+CP output should be (1,-1,0,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.layout.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tp_cp_2d_mesh_12(self, mock_mesh_plat, mock_layout_plat):
        """
        Feature: infer_layout BNSD with TP+CP on 2-D mesh.
        Description: mesh (2,2) tp×sp; x Shard(N)×Shard(S), cos Replicate×Shard(S).
        Expectation: output.tensor_map == (-1, 1, 0, -1).
        """
        self._setup_mock_platform(mock_mesh_plat, world_size=4)
        mock_layout_plat.get_rank.return_value = 0

        mesh = self._make_2d_mesh(mock_mesh_plat, shape=(2, 2), names=("tp", "sp"))
        x = _build_layout(mesh, (Shard(1), Shard(2)), 4)       # N on tp, S on sp
        cos = _build_layout(mesh, (Replicate(), Shard(2)), 4)   # N=1 broadcast, S on sp
        sin = _build_layout(mesh, (Replicate(), Shard(2)), 4)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        # tp→axis1, sp→axis0; N→tm[1]=1, S→tm[2]=0
        self.assertEqual(
            out.tensor_map, (-1, 1, 0, -1),
            msg=f"TP+CP output should be (-1,1,0,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_all_replicated_3d_tnd_13(self, mock_platform):
        """
        Feature: infer_layout 3-D TND all replicated.
        Description: 1-D size-1 mesh, ndim=3 (TND), all Replicate.
        Expectation: output.tensor_map == (-1, -1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        x = _build_layout(mesh, (Replicate(),), 3)
        cos = _build_layout(mesh, (Replicate(),), 3)
        sin = _build_layout(mesh, (Replicate(),), 3)

        op = self._get_op()
        (out,), extra = op.infer_layout([x, cos, sin])

        self.assertIsNone(extra)
        self.assertEqual(
            out.tensor_map, (-1, -1, -1),
            msg=f"TND all-replicated output should be (-1,-1,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tnd_dp_t_14(self, mock_platform):
        """
        Feature: infer_layout 3-D TND with T-dim DP.
        Description: x Shard(0) on 1-D dp mesh, ndim=3.
        Expectation: output.tensor_map == (0, -1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")
        x = _build_layout(mesh, (Shard(0),), 3)
        cos = _build_layout(mesh, (Replicate(),), 3)
        sin = _build_layout(mesh, (Replicate(),), 3)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        self.assertEqual(
            out.tensor_map, (0, -1, -1),
            msg=f"TND T-dim DP output should be (0,-1,-1), got {out.tensor_map}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_output_is_independent_deepcopy_15(self, mock_platform):
        """
        Feature: infer_layout output is an independent deep copy of x_layout.
        Description: Modifying output should not affect x_layout.
        Expectation: out is not x (different object).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x = _build_layout(mesh, (Shard(0),), 4)
        cos = _build_layout(mesh, (Replicate(),), 4)
        sin = _build_layout(mesh, (Replicate(),), 4)

        op = self._get_op()
        (out,), _ = op.infer_layout([x, cos, sin])

        self.assertIsNot(
            out, x,
            msg="Output layout must be an independent copy, not the same object as x_layout"
        )


class TestInferLayoutNegative(unittest.TestCase):
    """Negative layout inference tests — error cases."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    @staticmethod
    def _get_op():
        return get_distributed_op("RotaryPositionEmbedding")

    @staticmethod
    def _mock_layout(tensor_map, is_partial=False):
        m = MagicMock()
        m.is_partial.return_value = is_partial
        m.tensor_map = tensor_map
        return m

    def test_x_d_sharded_raises_16(self):
        """
        Feature: x D (last dim) sharding is forbidden.
        Description: x.tensor_map[-1] != -1.
        Expectation: Raises ValueError mentioning 'D'.
        """
        x = self._mock_layout((-1, -1, -1, 0))
        cos = self._mock_layout((-1, -1, -1, -1))
        sin = self._mock_layout((-1, -1, -1, -1))

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([x, cos, sin])

    def test_cos_d_sharded_raises_17(self):
        """
        Feature: cos D (last dim) sharding is forbidden.
        Description: cos.tensor_map[-1] != -1.
        Expectation: Raises ValueError mentioning 'D' and 'cos'.
        """
        x = self._mock_layout((-1, -1, -1, -1))
        cos = self._mock_layout((-1, -1, -1, 0))
        sin = self._mock_layout((-1, -1, -1, -1))

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "D"):
            op.infer_layout([x, cos, sin])

    def test_sin_d_sharded_raises_18(self):
        """
        Feature: sin D (last dim) sharding is forbidden.
        Description: sin.tensor_map[-1] != -1.
        Expectation: Raises ValueError mentioning 'D' and 'sin'.
        """
        x = self._mock_layout((-1, -1, -1, -1))
        cos = self._mock_layout((-1, -1, -1, -1))
        sin = self._mock_layout((-1, -1, -1, 0))

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "sin"):
            op.infer_layout([x, cos, sin])

    def test_cos_b_mismatch_raises_19(self):
        """
        Feature: cos B sharding inconsistent with x is rejected.
        Description: x.tensor_map[0]=0, cos.tensor_map[0]=1 (different mesh axes).
        Expectation: Raises ValueError mentioning 'cos'.
        """
        x = self._mock_layout((0, -1, -1, -1))
        cos = self._mock_layout((1, -1, -1, -1))
        sin = self._mock_layout((-1, -1, -1, -1))

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "cos"):
            op.infer_layout([x, cos, sin])

    def test_sin_s_mismatch_raises_20(self):
        """
        Feature: sin S sharding inconsistent with x is rejected.
        Description: x.tensor_map[2]=0, sin.tensor_map[2]=1 (different mesh axes).
        Expectation: Raises ValueError mentioning 'sin'.
        """
        x = self._mock_layout((-1, -1, 0, -1))
        cos = self._mock_layout((-1, -1, -1, -1))
        sin = self._mock_layout((-1, -1, 1, -1))

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "sin"):
            op.infer_layout([x, cos, sin])

    def test_cos_n_sharded_x_n_replicated_raises_21(self):
        """
        Feature: cos N sharded when x N is replicated is rejected.
        Description: x.tensor_map[1]=-1, cos.tensor_map[1]=0.
        Expectation: Raises ValueError mentioning 'cos'.
        """
        x = self._mock_layout((-1, -1, -1, -1))
        cos = self._mock_layout((-1, 0, -1, -1))
        sin = self._mock_layout((-1, -1, -1, -1))

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "cos"):
            op.infer_layout([x, cos, sin])

    def test_partial_input_raises_22(self):
        """
        Feature: Partial inputs are rejected.
        Description: x.is_partial() returns True.
        Expectation: Raises ValueError.
        """
        x = self._mock_layout((-1, -1, -1, -1), is_partial=True)
        cos = self._mock_layout((-1, -1, -1, -1))
        sin = self._mock_layout((-1, -1, -1, -1))

        op = self._get_op()
        with self.assertRaises(ValueError):
            op.infer_layout([x, cos, sin])


class TestNormalizeNpuRotaryMulArgs(unittest.TestCase):
    """Unit tests for _normalize_npu_rotary_mul_args."""

    def test_no_rotary_mode_defaults_to_mode_0(self):
        """
        Feature: _normalize_npu_rotary_mul_args fills default mode=0.
        Description: Call with only x, cos, sin (no rotary_mode kwarg).
        Expectation: args == (x, cos, sin, 0), kwargs == {}.
        """
        x, cos, sin = object(), object(), object()
        args, kwargs = _normalize_npu_rotary_mul_args(x, cos, sin)
        self.assertIs(args[0], x, msg=f"args[0] should be x, got {args[0]}")
        self.assertIs(args[1], cos, msg=f"args[1] should be cos, got {args[1]}")
        self.assertIs(args[2], sin, msg=f"args[2] should be sin, got {args[2]}")
        self.assertEqual(args[3], 0, msg=f"default mode should be 0, got {args[3]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    def test_rotary_mode_none_defaults_to_mode_0(self):
        """
        Feature: _normalize_npu_rotary_mul_args rotary_mode=None → mode=0.
        Description: Call with rotary_mode=None kwarg.
        Expectation: args[3] == 0.
        """
        x, cos, sin = object(), object(), object()
        args, kwargs = _normalize_npu_rotary_mul_args(x, cos, sin, rotary_mode=None)
        self.assertEqual(args[3], 0, msg=f"rotary_mode=None should give mode=0, got {args[3]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    def test_rotary_mode_half_maps_to_mode_0(self):
        """
        Feature: _normalize_npu_rotary_mul_args rotary_mode='half' → mode=0.
        Description: Call with rotary_mode='half' kwarg.
        Expectation: args[3] == 0.
        """
        x, cos, sin = object(), object(), object()
        args, kwargs = _normalize_npu_rotary_mul_args(x, cos, sin, rotary_mode="half")
        self.assertEqual(args[3], 0, msg=f"rotary_mode='half' should give mode=0, got {args[3]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    def test_rotary_mode_interleave_maps_to_mode_1(self):
        """
        Feature: _normalize_npu_rotary_mul_args rotary_mode='interleave' → mode=1.
        Description: Call with rotary_mode='interleave' kwarg.
        Expectation: args[3] == 1.
        """
        x, cos, sin = object(), object(), object()
        args, _ = _normalize_npu_rotary_mul_args(x, cos, sin, rotary_mode="interleave")
        self.assertEqual(args[3], 1, msg=f"rotary_mode='interleave' should give mode=1, got {args[3]}")

    def test_unsupported_rotary_mode_raises(self):
        """
        Feature: _normalize_npu_rotary_mul_args rejects unsupported rotary_mode.
        Description: Call with rotary_mode='invalid'.
        Expectation: Raises ValueError mentioning 'unsupported'.
        """
        x, cos, sin = object(), object(), object()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _normalize_npu_rotary_mul_args(x, cos, sin, rotary_mode="invalid")


class TestPreprocessRouting(unittest.TestCase):
    """Unit tests for preprocess cross-platform routing."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    @staticmethod
    def _setup_mock_platform(mock_platform, world_size=4):
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.split_group.return_value = MagicMock()
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.asnumpy() if hasattr(t, "asnumpy") else np.array(t)
        )

    @staticmethod
    def _make_dtensor_mock(mesh, ndim, shard_dim=None):
        """Build a mock DTensor with a real Layout attached."""
        from hyper_parallel.core.dtensor.dtensor import DTensor
        if shard_dim is not None:
            p = tuple(Shard(shard_dim) if i == shard_dim else Replicate()
                      for i in range(ndim))
        else:
            p = tuple(Replicate() for _ in range(ndim))
        ly = _build_layout(mesh, p, ndim)
        dt = MagicMock(spec=DTensor)
        dt.layout = ly
        dt.to_local.return_value = "local"
        return dt

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_primitive_mode_in_local_args(self, mock_platform):
        """
        Feature: Primitive preprocess puts mode in local_args, local_kwargs empty.
        Description: op_name="RotaryPositionEmbedding", mode=2 passed as kwarg.
        Expectation: local_args[3] == 2, local_kwargs == {}.
        """
        self._setup_mock_platform(mock_platform)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
        x_dt = self._make_dtensor_mock(mesh, 4)
        cos_dt = self._make_dtensor_mock(mesh, 4)
        sin_dt = self._make_dtensor_mock(mesh, 4)

        op = RotaryPositionEmbeddingDistributedOp("RotaryPositionEmbedding")
        local_args, local_kwargs, cv = op.preprocess(
            (x_dt, cos_dt, sin_dt), {"mode": 2}
        )

        self.assertEqual(len(local_args), 4,
                         msg=f"Primitive: local_args should have 4 entries, got {len(local_args)}")
        self.assertEqual(local_kwargs, {},
                         msg=f"Primitive: local_kwargs should be empty, got {local_kwargs}")
        self.assertEqual(len(cv), 3,
                         msg=f"cache_values should have 3 entries, got {len(cv)}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_pytorch_rotary_mode_in_local_kwargs(self, mock_platform):
        """
        Feature: PyTorch preprocess puts rotary_mode in local_kwargs.
        Description: op_name="npu_rotary_mul", rotary_mode="interleave".
        Expectation: local_args has 3 tensors, local_kwargs={'rotary_mode': 'interleave'}.
        """
        self._setup_mock_platform(mock_platform)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
        x_dt = self._make_dtensor_mock(mesh, 4)
        cos_dt = self._make_dtensor_mock(mesh, 4)
        sin_dt = self._make_dtensor_mock(mesh, 4)

        op = RotaryPositionEmbeddingDistributedOp("npu_rotary_mul")
        local_args, local_kwargs, cv = op.preprocess(
            (x_dt, cos_dt, sin_dt), {"rotary_mode": "interleave"}
        )

        self.assertEqual(len(local_args), 3,
                         msg=(
                             f"PyTorch: local_args should have 3 tensors only, "
                             f"got {local_args}"
                         ))
        self.assertEqual(local_kwargs, {"rotary_mode": "interleave"},
                         msg=(
                             f"PyTorch: local_kwargs should contain rotary_mode='interleave', "
                             f"got {local_kwargs}"
                         ))
        self.assertEqual(len(cv), 3,
                         msg=f"cache_values should have 3 entries, got {len(cv)}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_pytorch_no_rotary_mode_empty_kwargs(self, mock_platform):
        """
        Feature: PyTorch preprocess with no rotary_mode → empty local_kwargs.
        Description: op_name="npu_rotary_mul", no rotary_mode kwarg.
        Expectation: local_args has 3 tensors, local_kwargs == {}.
        """
        self._setup_mock_platform(mock_platform)
        mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
        x_dt = self._make_dtensor_mock(mesh, 4)
        cos_dt = self._make_dtensor_mock(mesh, 4)
        sin_dt = self._make_dtensor_mock(mesh, 4)

        op = RotaryPositionEmbeddingDistributedOp("npu_rotary_mul")
        local_args, local_kwargs, cv = op.preprocess(
            (x_dt, cos_dt, sin_dt), {}
        )

        self.assertEqual(len(local_args), 3,
                         msg=f"PyTorch no-rotary-mode: local_args should have 3 tensors, got {local_args}")
        self.assertEqual(local_kwargs, {},
                         msg=f"PyTorch no-rotary-mode: local_kwargs should be empty, got {local_kwargs}")
        self.assertEqual(len(cv), 3,
                         msg=f"cache_values should have 3 entries, got {len(cv)}")


class TestNpuRotaryMulYamlRegistration(unittest.TestCase):
    """YAML registration tests for npu_rotary_mul."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def test_npu_rotary_mul_registration(self):
        """
        Feature: YAML registers npu_rotary_mul.
        Description: get_distributed_op("npu_rotary_mul").
        Expectation: Returns a RotaryPositionEmbeddingDistributedOp instance.
        """
        op = get_distributed_op("npu_rotary_mul")
        self.assertIsNotNone(
            op, msg="npu_rotary_mul should be registered via YAML"
        )
        self.assertIsInstance(
            op, RotaryPositionEmbeddingDistributedOp,
            msg=(
                f"Expected RotaryPositionEmbeddingDistributedOp, "
                f"got {type(op)}"
            )
        )

    def test_ms_primitive_op_names_contains_rotary(self):
        """
        Feature: _MS_PRIMITIVE_OP_NAMES contains RotaryPositionEmbedding.
        Description: Read class attribute.
        Expectation: 'RotaryPositionEmbedding' in _MS_PRIMITIVE_OP_NAMES.
        """
        self.assertIn(
            "RotaryPositionEmbedding",
            RotaryPositionEmbeddingDistributedOp._MS_PRIMITIVE_OP_NAMES,
            msg=(
                "_MS_PRIMITIVE_OP_NAMES should contain 'RotaryPositionEmbedding', "
                f"got {RotaryPositionEmbeddingDistributedOp._MS_PRIMITIVE_OP_NAMES}"
            )
        )


if __name__ == "__main__":
    unittest.main()
