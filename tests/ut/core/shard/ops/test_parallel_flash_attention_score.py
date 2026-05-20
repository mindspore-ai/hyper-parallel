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
"""Unit tests for ParallelFlashAttention (parallel_flash_attention_score.py)"""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.shard.ops.parallel_flash_attention_score import ParallelFlashAttention
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _make_mesh(mock_platform, mesh_shape, mesh_dim_names):
    EXISTING_COMM_GROUPS.clear()
    _DEVICE_MESH_MAP.clear()
    mock_platform.get_rank.return_value = 0
    mock_platform.get_world_size.return_value = int(np.prod(mesh_shape))
    mock_platform.tensor_to_numpy.side_effect = (
        lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
    )
    return init_device_mesh(
        device_type="npu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        init_backend=False,
    )


class TestParallelFlashAttentionInit(unittest.TestCase):
    """
    Feature: ParallelFlashAttention initialization
    Description: Test various parameter combinations and validation in __init__.
    Expectation: TND layout raises ValueError; valid layouts initialize normally.
    """

    def test_default_init_bsh(self):
        """BSH layout initializes without error."""
        fa = ParallelFlashAttention(head_num=16)
        self.assertIsNone(fa.in_layout)
        self.assertIsNone(fa.out_layout)

    def test_bnsd_layout_init(self):
        """BNSD layout initializes without error."""
        fa = ParallelFlashAttention(head_num=8, input_layout="BNSD")
        self.assertIsNone(fa.in_layout)

    def test_sbh_layout_init(self):
        """SBH layout initializes without error."""
        fa = ParallelFlashAttention(head_num=4, input_layout="SBH")
        self.assertIsNone(fa.in_layout)

    def test_bsnd_layout_init(self):
        """BSND layout initializes without error."""
        fa = ParallelFlashAttention(head_num=8, input_layout="BSND")
        self.assertIsNone(fa.in_layout)

    def test_tnd_layout_raises(self):
        """TND layout raises ValueError (VarLen not supported)."""
        with self.assertRaises(ValueError) as ctx:
            ParallelFlashAttention(head_num=8, input_layout="TND")
        self.assertIn("FlashAttentionVarLen", str(ctx.exception))

    def test_custom_params(self):
        """Custom keep_prob, scalar_value, pre/next_tokens are stored."""
        fa = ParallelFlashAttention(
            head_num=16,
            keep_prob=0.9,
            scalar_value=0.125,
            pre_tokens=1024,
            next_tokens=512,
            sparse_mode=2,
        )
        self.assertEqual(fa._keep_prob, 0.9)
        self.assertEqual(fa._scalar_value, 0.125)
        self.assertEqual(fa._pre_tokens, 1024)
        self.assertEqual(fa._next_tokens, 512)
        self.assertEqual(fa._sparse_mode, 2)


class TestInferDimByLayout(unittest.TestCase):
    """
    Feature: ParallelFlashAttention._infer_dim_by_layout
    Description: Infer batch, seq, and head_num dimensions from layout string.
    Expectation: Correct dimension indices returned; missing dims raise ValueError.
    """

    def test_bsh_dims(self):
        """BSH returns batch=0, seq=1, head=2."""
        b, s, h = ParallelFlashAttention._infer_dim_by_layout("BSH")
        self.assertEqual(b, 0)
        self.assertEqual(s, 1)
        self.assertEqual(h, 2)

    def test_bnsd_dims(self):
        """BNSD returns batch=0, head=1, seq=2."""
        b, s, h = ParallelFlashAttention._infer_dim_by_layout("BNSD")
        self.assertEqual(b, 0)
        self.assertEqual(s, 2)
        self.assertEqual(h, 1)

    def test_sbh_dims(self):
        """SBH returns batch=1, seq=0, head=2."""
        b, s, h = ParallelFlashAttention._infer_dim_by_layout("SBH")
        self.assertEqual(b, 1)
        self.assertEqual(s, 0)
        self.assertEqual(h, 2)

    def test_missing_batch_raises(self):
        """Layout without B raises ValueError."""
        with self.assertRaises(ValueError):
            ParallelFlashAttention._infer_dim_by_layout("SNH")

    def test_missing_seq_raises(self):
        """Layout without S raises ValueError."""
        with self.assertRaises(ValueError):
            ParallelFlashAttention._infer_dim_by_layout("BNH")

    def test_missing_head_raises(self):
        """Layout without H or N raises ValueError."""
        with self.assertRaises(ValueError):
            ParallelFlashAttention._infer_dim_by_layout("BSX")


class TestInsertNoneForNonTensorArg(unittest.TestCase):
    """
    Feature: ParallelFlashAttention._insert_none_for_non_tensor_arg
    Description: Expand layout list to match all args, inserting None for non-Tensor args.
    Expectation: Correct placement of layouts; mismatch raises ValueError.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_tensor_arg(self, mock_platform):
        """One Tensor arg and one layout → result has layout at tensor position."""
        from mindspore import Tensor as MSTensor
        import mindspore as ms

        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 3)

        tensor_arg = MSTensor(np.zeros((2, 4, 8), dtype=np.float16))
        non_tensor_arg = 16  # head_num (int)
        args = (tensor_arg, non_tensor_arg)

        result = ParallelFlashAttention._insert_none_for_non_tensor_arg([layout], args)
        self.assertEqual(len(result), len(args))
        self.assertIs(result[0], layout)
        self.assertIsNone(result[1])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_layout_count_mismatch_raises(self, mock_platform):
        """Mismatch between layout count and tensor arg count raises ValueError."""
        from mindspore import Tensor as MSTensor

        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout1 = _build_layout(mesh, (Replicate(),), 3)
        layout2 = _build_layout(mesh, (Replicate(),), 3)

        tensor_arg = MSTensor(np.zeros((2, 4, 8), dtype=np.float16))
        args = (tensor_arg, 16)  # 1 tensor, 1 non-tensor

        with self.assertRaises(ValueError):
            ParallelFlashAttention._insert_none_for_non_tensor_arg([layout1, layout2], args)

    def test_empty_layouts_raises(self):
        """Empty layouts list raises ValueError."""
        with self.assertRaises(ValueError):
            ParallelFlashAttention._insert_none_for_non_tensor_arg([], (1, 2))


class TestShardMethod(unittest.TestCase):
    """
    Feature: ParallelFlashAttention.shard
    Description: Set in_layout and out_layout; validate types and prevent repeated calls.
    Expectation: Valid layouts are stored; invalid types raise ValueError; repeated call raises ValueError.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_valid_shard_sets_layouts(self, mock_platform):
        """Valid in/out strategies are stored as layouts."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 3)
        fa = ParallelFlashAttention(head_num=8)
        fa.shard((layout,), (layout,))
        self.assertEqual(fa.in_layout, (layout,))
        self.assertEqual(fa.out_layout, (layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_repeated_shard_raises(self, mock_platform):
        """Calling shard twice raises ValueError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 3)
        fa = ParallelFlashAttention(head_num=8)
        fa.shard((layout,), (layout,))
        with self.assertRaises(ValueError):
            fa.shard((layout,), (layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_invalid_in_strategy_raises(self, mock_platform):
        """Non-tuple in_strategy raises ValueError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 3)
        fa = ParallelFlashAttention(head_num=8)
        with self.assertRaises(ValueError):
            fa.shard("invalid", (layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_in_strategy_with_non_layout_raises(self, mock_platform):
        """Tuple containing non-Layout in_strategy raises ValueError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 3)
        fa = ParallelFlashAttention(head_num=8)
        with self.assertRaises(ValueError):
            fa.shard((layout, "not_a_layout"), (layout,))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_invalid_out_strategy_raises(self, mock_platform):
        """Non-tuple out_strategy raises ValueError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 3)
        fa = ParallelFlashAttention(head_num=8)
        with self.assertRaises(ValueError):
            fa.shard((layout,), "invalid")


class TestGetSplitNum(unittest.TestCase):
    """
    Feature: ParallelFlashAttention._get_split_num
    Description: Get the number of splits along a specified tensor dimension.
    Expectation: Returns correct split count based on layout's alias_tensor_map.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sharded_dim_returns_two(self, mock_platform):
        """Batch-sharded dimension returns 2 for a 2-device dp axis."""
        mesh = _make_mesh(mock_platform, (2, 4), ("dp", "mp"))
        layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
        split_num = ParallelFlashAttention._get_split_num(layout, 0)
        self.assertEqual(split_num, 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sharded_on_mp_returns_four(self, mock_platform):
        """Head-sharded dimension on mp(4 devices) returns 4."""
        mesh = _make_mesh(mock_platform, (2, 4), ("dp", "mp"))
        layout = _build_layout(mesh, (Replicate(), Shard(2)), 3)
        split_num = ParallelFlashAttention._get_split_num(layout, 2)
        self.assertEqual(split_num, 4)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_invalid_map_raises(self, mock_platform):
        """Invalid tensor map entry (non-str, non-tuple) raises ValueError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Shard(0),), 2)
        layout._alias_tensor_map = (123,)
        with self.assertRaises(ValueError):
            ParallelFlashAttention._get_split_num(layout, 0)


class TestGetDeviceMeshFromLayouts(unittest.TestCase):
    """
    Feature: ParallelFlashAttention._get_device_mesh_from_layouts
    Description: Extract device_mesh from the first non-None layout.
    Expectation: Returns the mesh of the first non-None layout; raises ValueError if all None.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_returns_mesh_from_in_layout(self, mock_platform):
        """Returns device mesh from first non-None in_layout."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 3)
        fa = ParallelFlashAttention(head_num=8)
        fa.in_layout = (layout,)
        fa.out_layout = (layout,)
        result = fa._get_device_mesh_from_layouts()
        self.assertEqual(result.mesh_shape, mesh.mesh_shape)
        self.assertEqual(result.mesh_dim_names, mesh.mesh_dim_names)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_falls_back_to_out_layout(self, mock_platform):
        """Falls back to out_layout when in_layout has only None."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 3)
        fa = ParallelFlashAttention(head_num=8)
        fa.in_layout = (None,)
        fa.out_layout = (layout,)
        result = fa._get_device_mesh_from_layouts()
        self.assertEqual(result.mesh_shape, mesh.mesh_shape)
        self.assertEqual(result.mesh_dim_names, mesh.mesh_dim_names)

    def test_all_none_raises(self):
        """All None layouts raises ValueError."""
        fa = ParallelFlashAttention(head_num=8)
        fa.in_layout = (None,)
        fa.out_layout = (None,)
        with self.assertRaises(ValueError):
            fa._get_device_mesh_from_layouts()


class TestConstructValidation(unittest.TestCase):
    """
    Feature: ParallelFlashAttention.construct
    Description: Calling construct without setting shard strategy raises ValueError.
    Expectation: ValueError raised when in_layout or out_layout is None.
    """

    def test_construct_without_shard_raises(self):
        """construct() raises ValueError when shard() has not been called."""
        fa = ParallelFlashAttention(head_num=8)
        mock_query = MagicMock()
        mock_query.shape = (2, 4, 64)
        with self.assertRaises(ValueError) as ctx:
            fa.construct(mock_query, MagicMock(), MagicMock(), None)
        self.assertIn("shard", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
