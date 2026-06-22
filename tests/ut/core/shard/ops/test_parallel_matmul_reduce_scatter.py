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
"""Unit tests for MatmulReduceScatterDistributedOp."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_matmul_reduce_scatter import (
    MatmulReduceScatterDistributedOp,
    _normalize_mrs_args,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestMatmulReduceScatterDistributedOp(unittest.TestCase):
    """Unit tests for MatmulReduceScatterDistributedOp."""

    def setUp(self) -> None:
        """Clear global state before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
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

    def _make_1d_mesh(self, mock_platform, size=4, name="tp"):
        """Return a 1-D mesh of given size."""
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(device_type="npu", mesh_shape=(size,), mesh_dim_names=(name,),
                                init_backend=False)

    def _make_2d_mesh(self, mock_platform, shape=(2, 4), names=("dp", "tp")):
        """Return a 2-D mesh."""
        self._setup_mock_platform(mock_platform, world_size=shape[0] * shape[1])
        return init_device_mesh(device_type="npu", mesh_shape=shape, mesh_dim_names=names,
                                init_backend=False)

    @staticmethod
    def _get_op():
        return get_distributed_op("MatmulReduceScatter")

    # ------------------------------------------------------------------
    # _normalize_mrs_args
    # ------------------------------------------------------------------

    def test_normalize_args_defaults_1(self):
        """
        Feature: _normalize_mrs_args fills in default values.
        Description: Call with only 4 mandatory positional args.
        Expectation: Optional args get correct defaults.
        """
        x1, x2, group, world_size = object(), object(), "hccl_world_group", 2
        args, kwargs = _normalize_mrs_args(x1, x2, group, world_size)
        self.assertIs(args[0], x1)
        self.assertIs(args[1], x2)
        self.assertEqual(args[2], "hccl_world_group")
        self.assertEqual(args[3], 2)
        self.assertEqual(args[4], 'sum', msg=f"reduce_op default should be 'sum', got {args[4]}")
        self.assertIsNone(args[5], msg=f"bias default should be None, got {args[5]}")
        self.assertEqual(args[6], 0, msg=f"comm_turn default should be 0, got {args[6]}")
        self.assertFalse(args[7], msg=f"trans_input default should be False, got {args[7]}")
        self.assertFalse(args[8], msg=f"trans_x2 default should be False, got {args[8]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    # ------------------------------------------------------------------
    # YAML registration
    # ------------------------------------------------------------------

    def test_yaml_registration_mindspore_2(self):
        """
        Feature: YAML loader registers MatmulReduceScatterDistributedOp.
        Description: Call get_distributed_op with MindSpore CamelCase op name.
        Expectation: Returns a MatmulReduceScatterDistributedOp instance.
        """
        op = self._get_op()
        self.assertIsNotNone(op, msg="MatmulReduceScatter should be registered")
        self.assertIsInstance(
            op, MatmulReduceScatterDistributedOp,
            msg=f"Expected MatmulReduceScatterDistributedOp, got {type(op)}"
        )

    # ------------------------------------------------------------------
    # infer_layout — positive cases
    # cache_values format: [x1_layout, x2_layout, trans_x2]
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_mrs_basic_tp_3(self, mock_platform):
        """
        Feature: infer_layout basic TP case.
        Description: x1 k-dim (dim 1) sharded on tp (mesh_dim 0); x2 k-dim (dim 0) sharded on tp.
                     Pure TP: x1 m-dim (dim 0) is Replicate.
        Expectation: output tensor_map (0, -1) — m-dim sharded by tp (ReduceScatter), n-dim Replicate.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="tp")
        # x1 (m, k): m Replicate, k Shard(1) on tp
        x1_layout = _build_layout(mesh, (Shard(1),), 2)
        # x2 (k, n): k Shard(0) on tp, n Replicate
        x2_layout = _build_layout(mesh, (Shard(0),), 2)

        op = self._get_op()
        out_layout, extra = op.infer_layout([x1_layout, x2_layout, False])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (0, -1),
                         msg=f"Basic TP: expected (0, -1), got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_mrs_trans_x2_true_4(self, mock_platform):
        """
        Feature: infer_layout trans_x2=True.
        Description: x2 physical shape is (n, k); trans_x2=True means x2 dim 1 is k.
                     x2 Shard(1) means n-dim (dim 0) is Replicate, k-dim (dim 1) is sharded on tp.
        Expectation: output tensor_map (0, -1) — m-dim sharded by tp, n-dim Replicate.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="tp")
        # x1 (m, k): k-dim (tensor dim 1) sharded on tp (mesh_dim 0)
        x1_layout = _build_layout(mesh, (Shard(1),), 2)
        # x2 (n, k): k-dim is tensor dim 1; Shard(1) shards k on tp (mesh_dim 0)
        x2_layout = _build_layout(mesh, (Shard(1),), 2)

        op = self._get_op()
        out_layout, extra = op.infer_layout([x1_layout, x2_layout, True])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (0, -1),
                         msg=f"trans_x2=True: expected (0, -1), got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_mrs_dp_tp_mesh_5(self, mock_platform):
        """
        Feature: infer_layout 2D (dp=2, tp=4) mesh, x1 m sharded by dp, k sharded by tp.
        Description: x1: m-dim (tensor dim 0) sharded by dp (mesh_dim 0),
                         k-dim (tensor dim 1) sharded by tp (mesh_dim 1).
                     x2: k-dim (tensor dim 0) sharded by tp (mesh_dim 1), n-dim Replicate.
        Expectation: output tensor_map ((1, 0), -1) — m jointly sharded by dp (outer) + tp (inner), n Replicate.
        """
        mesh = self._make_2d_mesh(mock_platform, shape=(2, 4), names=("dp", "tp"))
        # x1 (m, k): m sharded by dp (mesh_dim 0), k sharded by tp (mesh_dim 1)
        x1_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
        # x2 (k, n): k sharded by tp (mesh_dim 1), n Replicate
        x2_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

        op = self._get_op()
        out_layout, extra = op.infer_layout([x1_layout, x2_layout, False])

        self.assertIsNone(extra)
        # tensor_map uses reversed mesh_dim indexing: value 0 = tp (innermost), value 1 = dp (outermost).
        # ReduceScatter converts k-sharding (tp, value 0) to m-sharding;
        # dp m-sharding (value 1) is outer, tp-derived m-sharding (value 0) is inner → output_m = (1, 0).
        self.assertEqual(out_layout.tensor_map, ((1, 0), -1),
                         msg=f"DP+TP mesh: expected ((1, 0), -1), got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_mrs_x2_n_sharded_6(self, mock_platform):
        """
        Feature: infer_layout x2 n-dim sharded.
        Description: 2D mesh: x2 k sharded by tp (mesh_dim 1), n sharded by dp (mesh_dim 0).
        Expectation: output tensor_map (0, 1) — m sharded by tp, n sharded by dp.
        """
        mesh = self._make_2d_mesh(mock_platform, shape=(2, 4), names=("dp", "tp"))
        # x1 (m, k): m Replicate, k sharded by tp (mesh_dim 1)
        x1_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        # x2 (k, n): k sharded by tp (mesh_dim 1), n sharded by dp (mesh_dim 0)
        x2_layout = _build_layout(mesh, (Shard(1), Shard(0)), 2)

        op = self._get_op()
        out_layout, extra = op.infer_layout([x1_layout, x2_layout, False])

        self.assertIsNone(extra)
        # tensor_map value 0 = tp (innermost), value 1 = dp (outermost).
        # output_m = 0 (tp ReduceScatter on m), output_n = 1 (dp shards n).
        self.assertEqual(out_layout.tensor_map, (0, 1),
                         msg=f"x2 n sharded: expected (0, 1), got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_mrs_x2_n_replicate_7(self, mock_platform):
        """
        Feature: infer_layout x2 n-dim Replicate.
        Description: Pure TP with Replicate n-dim.
        Expectation: output tensor_map (0, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="tp")
        x1_layout = _build_layout(mesh, (Shard(1),), 2)
        x2_layout = _build_layout(mesh, (Shard(0),), 2)

        op = self._get_op()
        out_layout, extra = op.infer_layout([x1_layout, x2_layout, False])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (0, -1),
                         msg=f"n Replicate: expected (0, -1), got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_output_independent_copy_8(self, mock_platform):
        """
        Feature: infer_layout returns an independent deep copy of output layout.
        Description: Verify the returned layout is a new object, not a reference to any input.
        Expectation: output_layout is a distinct object from x1_layout and x2_layout.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="tp")
        x1_layout = _build_layout(mesh, (Shard(1),), 2)
        x2_layout = _build_layout(mesh, (Shard(0),), 2)

        op = self._get_op()
        out_layout, _ = op.infer_layout([x1_layout, x2_layout, False])

        self.assertIsNot(out_layout, x1_layout,
                         msg="output_layout must not be the same object as x1_layout")
        self.assertIsNot(out_layout, x2_layout,
                         msg="output_layout must not be the same object as x2_layout")

    # ------------------------------------------------------------------
    # infer_layout — error cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_input_raises_9(self, mock_platform):
        """
        Feature: infer_layout rejects Partial inputs.
        Description: x1 layout has Partial status.
        Expectation: ValueError with 'Partial status' message.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="tp")
        x1_layout = _build_layout(mesh, (Replicate(),), 2)
        x1_layout.set_partial_by_dev_axis("tp", 'sum')
        x2_layout = _build_layout(mesh, (Shard(0),), 2)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout([x1_layout, x2_layout, False])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_k_replicate_raises_10(self, mock_platform):
        """
        Feature: infer_layout rejects x1 k-dim (dim 1) Replicate.
        Description: k must be sharded for ReduceScatter to be semantically correct.
        Expectation: ValueError mentioning x1 k-dim must be Shard.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="tp")
        x1_layout = _build_layout(mesh, (Replicate(),), 2)
        x2_layout = _build_layout(mesh, (Replicate(),), 2)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "x1 k-dim.*must be Shard"):
            op.infer_layout([x1_layout, x2_layout, False])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_k_mismatch_raises_11(self, mock_platform):
        """
        Feature: infer_layout rejects x2 k-dim layout mismatch with x1 k-dim.
        Description: x1 k-dim sharded on mesh_dim 0, x2 k-dim Replicate → mismatch.
        Expectation: ValueError mentioning k layout must match.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="tp")
        x1_layout = _build_layout(mesh, (Shard(1),), 2)
        x2_layout = _build_layout(mesh, (Replicate(),), 2)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "x2 dim.*\\(k\\) layout must match"):
            op.infer_layout([x1_layout, x2_layout, False])


if __name__ == "__main__":
    unittest.main()
