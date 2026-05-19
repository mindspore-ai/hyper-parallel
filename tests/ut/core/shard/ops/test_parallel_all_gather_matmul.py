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
"""Unit tests for AllGatherMatmulDistributedOp."""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_all_gather_matmul import (
    AllGatherMatmulDistributedOp,
    _normalize_agm_args,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestAllGatherMatmulDistributedOp(unittest.TestCase):
    """Unit tests for AllGatherMatmulDistributedOp."""

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
        return get_distributed_op("AllGatherMatmul")

    # ------------------------------------------------------------------
    # _normalize_agm_args
    # ------------------------------------------------------------------

    def test_normalize_args_defaults_1(self):
        """
        Feature: _normalize_agm_args fills in default values.
        Description: Call with only 4 mandatory positional args.
        Expectation: Optional args get correct defaults.
        """
        x1, x2, group, world_size = object(), object(), "hccl_world_group", 2
        args, kwargs = _normalize_agm_args(x1, x2, group, world_size)
        self.assertIs(args[0], x1)
        self.assertIs(args[1], x2)
        self.assertEqual(args[2], "hccl_world_group")
        self.assertEqual(args[3], 2)
        self.assertIsNone(args[4], msg=f"bias default should be None, got {args[4]}")
        self.assertEqual(args[5], 0, msg=f"gather_index default should be 0, got {args[5]}")
        self.assertTrue(args[6], msg=f"gather_output default should be True, got {args[6]}")
        self.assertEqual(args[7], 0, msg=f"comm_turn default should be 0, got {args[7]}")
        self.assertFalse(args[8], msg=f"trans_input default should be False, got {args[8]}")
        self.assertFalse(args[9], msg=f"trans_x2 default should be False, got {args[9]}")
        self.assertEqual(kwargs, {}, msg=f"kwargs should be empty, got {kwargs}")

    # ------------------------------------------------------------------
    # YAML registration
    # ------------------------------------------------------------------

    def test_yaml_registration_mindspore_2(self):
        """
        Feature: YAML loader registers AllGatherMatmulDistributedOp.
        Description: Call get_distributed_op with MindSpore CamelCase op name.
        Expectation: Returns an AllGatherMatmulDistributedOp instance.
        """
        op = self._get_op()
        self.assertIsNotNone(op, msg="AllGatherMatmul should be registered")
        self.assertIsInstance(
            op, AllGatherMatmulDistributedOp,
            msg=f"Expected AllGatherMatmulDistributedOp, got {type(op)}"
        )

    # ------------------------------------------------------------------
    # infer_layout — positive cases
    # cache_values format: [x1_layout, x2_layout, trans_x2, gather_output]
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_replicated_3(self, mock_platform):
        """
        Feature: infer_layout all replicated.
        Description: Both inputs fully replicated.
        Expectation: output and gather_out tensor_maps are both (-1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Replicate(),), 2)
        x2_layout = _build_layout(mesh, (Replicate(),), 2)

        op = self._get_op()
        (out_layout, gather_layout), extra = op.infer_layout(
            [x1_layout, x2_layout, False, True])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (-1, -1),
                         msg=f"Replicated output: expected (-1,-1), got {out_layout.tensor_map}")
        self.assertEqual(gather_layout.tensor_map, (-1, -1),
                         msg=f"Replicated gather_out: expected (-1,-1), got {gather_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_input_shard0_x2_replicate_4(self, mock_platform):
        """
        Feature: infer_layout x1 Shard(0) on tp, x2 Replicate.
        Description: x1 m-dim sharded by AllGather; x2 n-dim Replicate.
        Expectation: output tensor_map (-1, -1); gather_out tensor_map (-1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Shard(0),), 2)
        x2_layout = _build_layout(mesh, (Replicate(),), 2)

        op = self._get_op()
        (out_layout, gather_layout), extra = op.infer_layout(
            [x1_layout, x2_layout, False, True])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (-1, -1),
                         msg=f"Expected (-1,-1), got {out_layout.tensor_map}")
        self.assertEqual(gather_layout.tensor_map, (-1, -1),
                         msg=f"Expected (-1,-1), got {gather_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_input_shard0_x2_shard1_5(self, mock_platform):
        """
        Feature: infer_layout x1 Shard(0) on tp, x2 Shard(1) on n.
        Description: x2 n-dim sharded on tp mesh_dim (index 0).
        Expectation: output tensor_map (-1, 0); gather_out tensor_map (-1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Shard(0),), 2)
        x2_layout = _build_layout(mesh, (Shard(1),), 2)

        op = self._get_op()
        (out_layout, gather_layout), extra = op.infer_layout(
            [x1_layout, x2_layout, False, True])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (-1, 0),
                         msg=f"Expected (-1, 0), got {out_layout.tensor_map}")
        self.assertEqual(gather_layout.tensor_map, (-1, -1),
                         msg=f"Expected (-1,-1), got {gather_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_trans_x2_false_n_sharded_6(self, mock_platform):
        """
        Feature: infer_layout trans_x2=False, x2 Shard(1).
        Description: trans_x2=False: n is x2 dim 1; x2 dim 1 sharded on tp (index 0).
        Expectation: output tensor_map (-1, 0).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Replicate(),), 2)
        x2_layout = _build_layout(mesh, (Shard(1),), 2)

        op = self._get_op()
        (out_layout, _), extra = op.infer_layout([x1_layout, x2_layout, False, True])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (-1, 0),
                         msg=f"trans_x2=False n sharded: expected (-1, 0), got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_trans_x2_true_n_sharded_7(self, mock_platform):
        """
        Feature: infer_layout trans_x2=True, x2 Shard(0) on n.
        Description: trans_x2=True: n is x2 dim 0; x2 physical shape is (n, k);
                     x2 Shard(0) means n is sharded on tp mesh_dim (index 0).
        Expectation: output tensor_map (-1, 0).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Replicate(),), 2)
        # trans_x2=True: x2 shape is (n, k); Shard(0) shards n on tp (mesh_dim index 0)
        x2_layout = _build_layout(mesh, (Shard(0),), 2)

        op = self._get_op()
        (out_layout, _), extra = op.infer_layout([x1_layout, x2_layout, True, True])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (-1, 0),
                         msg=f"trans_x2=True n sharded: expected (-1, 0), got {out_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_gather_output_false_8(self, mock_platform):
        """
        Feature: infer_layout gather_output=False.
        Description: When gather_output is False, gather_out layout should still be (-1, -1).
        Expectation: gather_out tensor_map is (-1, -1).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Shard(0),), 2)
        x2_layout = _build_layout(mesh, (Replicate(),), 2)

        op = self._get_op()
        (_, gather_layout), extra = op.infer_layout([x1_layout, x2_layout, False, False])

        self.assertIsNone(extra)
        self.assertEqual(gather_layout.tensor_map, (-1, -1),
                         msg=f"gather_output=False: expected (-1,-1), got {gather_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_2d_mesh_dp_tp_9(self, mock_platform):
        """
        Feature: infer_layout 2D (dp=2, tp=4) mesh.
        Description: x1 Shard(0) on tp only; x2 Replicate. AllGather on tp axis.
        Expectation: output tensor_map (-1, -1); gather_out tensor_map (-1, -1).
        """
        mesh = self._make_2d_mesh(mock_platform, shape=(2, 4), names=("dp", "tp"))
        # x1: m-dim sharded on tp (mesh_dim index 1), k-dim Replicate
        x1_layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)
        x2_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        op = self._get_op()
        (out_layout, gather_layout), extra = op.infer_layout(
            [x1_layout, x2_layout, False, True])

        self.assertIsNone(extra)
        self.assertEqual(out_layout.tensor_map, (-1, -1),
                         msg=f"2D mesh output: expected (-1,-1), got {out_layout.tensor_map}")
        self.assertEqual(gather_layout.tensor_map, (-1, -1),
                         msg=f"2D mesh gather_out: expected (-1,-1), got {gather_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_output_independent_copies_10(self, mock_platform):
        """
        Feature: infer_layout returns independent deep copies for output and gather_out.
        Description: Mutate output_layout.tensor_map; gather_out_layout must be unaffected.
        Expectation: The two layouts are separate objects with independent state.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Replicate(),), 2)
        x2_layout = _build_layout(mesh, (Shard(1),), 2)

        op = self._get_op()
        (out_layout, gather_layout), _ = op.infer_layout(
            [x1_layout, x2_layout, False, True])

        self.assertIsNot(out_layout, gather_layout,
                         msg="output_layout and gather_out_layout must be distinct objects")
        self.assertEqual(out_layout.tensor_map, (-1, 0))
        self.assertEqual(gather_layout.tensor_map, (-1, -1))

    # ------------------------------------------------------------------
    # infer_layout — error cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partial_input_raises_11(self, mock_platform):
        """
        Feature: infer_layout rejects Partial inputs.
        Description: x1 layout has Partial status.
        Expectation: ValueError with 'Partial status' message.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Replicate(),), 2)
        x1_layout.set_partial_by_dev_axis("tp", 'sum')
        x2_layout = _build_layout(mesh, (Replicate(),), 2)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout([x1_layout, x2_layout, False, True])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_k_sharded_output_partial_12(self, mock_platform):
        """
        Feature: infer_layout k-dim sharded on both x1 and x2 — output is Partial.
        Description: x1 Shard(1) on tp (k-dim), x2 Shard(0) on tp (k-dim); placements match.
                     Output carries Partial(sum) on tp; gather_out k-dim follows x1's k placement.
        Expectation: output is_partial() True, tensor_map (-1,-1); gather_out tensor_map (-1, 0).
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        # x1 Shard(1): tensor_dim 1 (k) sharded on tp → tensor_map = (-1, 0)
        x1_layout = _build_layout(mesh, (Shard(1),), 2)
        # x2 Shard(0): tensor_dim 0 (k, trans_x2=False) sharded on tp → tensor_map = (0, -1)
        x2_layout = _build_layout(mesh, (Shard(0),), 2)

        op = self._get_op()
        (out_layout, gather_layout), extra = op.infer_layout(
            [x1_layout, x2_layout, False, True])

        self.assertIsNone(extra)
        self.assertTrue(out_layout.is_partial(),
                        msg="k sharded: output must carry Partial status")
        self.assertEqual(out_layout.tensor_map, (-1, -1),
                         msg=f"k sharded: expected output tensor_map (-1,-1), got {out_layout.tensor_map}")
        self.assertEqual(gather_layout.tensor_map, (-1, 0),
                         msg=f"k sharded: expected gather_out tensor_map (-1,0), got {gather_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_agm_k_mismatch_raises_13(self, mock_platform):
        """
        Feature: infer_layout rejects mismatched k-dim placements between x1 and x2.
        Description: x1 k-dim is Replicate but x2 k-dim is Shard on tp.
        Expectation: ValueError mentioning k-dim placement must match.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4)
        x1_layout = _build_layout(mesh, (Replicate(),), 2)   # k-dim = Replicate (-1)
        x2_layout = _build_layout(mesh, (Shard(0),), 2)      # k-dim (dim 0) = tp (0)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "k-dim.*placement.*must match"):
            op.infer_layout([x1_layout, x2_layout, False, True])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_input_m_multi_mesh_raises_14(self, mock_platform):
        """
        Feature: infer_layout rejects x1 m-dim jointly sharded across multiple mesh dims.
        Description: trans_input=False: x1 m-dim is dim 0; jointly sharded across dp+tp.
        Expectation: ValueError mentioning jointly sharded is not supported.
        """
        mesh = self._make_2d_mesh(mock_platform, shape=(2, 4), names=("dp", "tp"))
        x1_layout = _build_layout(mesh, (Shard(0), Shard(0)), 2)
        x2_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "jointly sharded"):
            op.infer_layout([x1_layout, x2_layout, False, True])

if __name__ == "__main__":
    unittest.main()
