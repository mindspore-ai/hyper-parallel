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
"""Unit tests for MC2 parallel styles and MC2Linear conversion."""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.tensor_parallel.mc2 import (
    AllGatherMatmulFunction,
    MC2Linear,
    _move_dim_to_front,
    _move_front_to_dim,
    _normalize_sequence_dim,
)
from hyper_parallel.core.tensor_parallel.mc2_style import (
    MC2ColwiseParallel,
    MC2RowwiseParallel,
)
from hyper_parallel.core.tensor_parallel.style import ColwiseParallel, RowwiseParallel


class _FakeMesh:
    """Minimal mesh stub for style.apply unit tests."""

    @staticmethod
    def get_group():
        return "tp-group"

    @staticmethod
    def size():
        return 4


class _Ctx:
    """Minimal autograd ctx stub for AllGatherMatmulFunction.backward."""

    def __init__(self, gathered, w, hcom="hcom", world_size=2, has_bias=False):
        self.saved_tensors = (gathered, w)
        self.hcom = hcom
        self.world_size = world_size
        self.has_bias = has_bias


class TestMC2Linear(unittest.TestCase):
    """Tests for MC2Linear in-place conversion and configuration."""

    def test_from_linear_preserves_parameters(self):
        """
        Feature: MC2Linear.from_linear keeps weight identity
        Description: convert nn.Linear in place
        Expectation: same module object, same weight Parameter, class becomes MC2Linear
        """
        linear = nn.Linear(8, 16, bias=False)
        weight = linear.weight
        result = MC2Linear.from_linear(linear)
        self.assertIs(result, linear)
        self.assertIsInstance(result, MC2Linear)
        self.assertIs(result.weight, weight)
        self.assertIn("weight", result._parameters)
        self.assertIs(result._parameters["weight"], weight)

    def test_from_linear_rejects_non_linear(self):
        """
        Feature: MC2Linear.from_linear type check
        Description: pass nn.ReLU
        Expectation: TypeError
        """
        with self.assertRaises(TypeError):
            MC2Linear.from_linear(nn.ReLU())

    def test_configure_mc2_modes(self):
        """
        Feature: MC2Linear.configure_mc2
        Description: configure all_gather / reduce_scatter and reject invalid mode
        Expectation: attributes set; invalid mode raises ValueError
        """
        linear = MC2Linear.from_linear(nn.Linear(4, 8, bias=True))
        linear.configure_mc2("all_gather", "g", 2, sequence_dim=1)
        self.assertEqual(linear.mc2_mode, "all_gather")
        self.assertEqual(linear.mc2_group, "g")
        self.assertEqual(linear.mc2_world_size, 2)
        self.assertEqual(linear.mc2_sequence_dim, 1)

        linear.configure_mc2("reduce_scatter", "g2", 4, sequence_dim=0)
        self.assertEqual(linear.mc2_mode, "reduce_scatter")

        with self.assertRaises(ValueError):
            linear.configure_mc2("all_reduce", "g", 2)


class TestMC2StyleInit(unittest.TestCase):
    """Constructor validation for MC2 styles."""

    def test_mc2_colwise_requires_sharded_input(self):
        """
        Feature: MC2ColwiseParallel layout validation
        Description: Replicate input layout
        Expectation: ValueError mentioning sharded input
        """
        with self.assertRaisesRegex(ValueError, "sharded input"):
            MC2ColwiseParallel(input_layouts=Replicate())

    def test_mc2_rowwise_requires_sharded_output(self):
        """
        Feature: MC2RowwiseParallel layout validation
        Description: Replicate output layout
        Expectation: ValueError mentioning sharded output
        """
        with self.assertRaisesRegex(ValueError, "sharded output"):
            MC2RowwiseParallel(output_layouts=Replicate())

    def test_mc2_colwise_keeps_sequence_sharding(self):
        """
        Feature: MC2ColwiseParallel desired_input_layouts
        Description: construct with Shard(1) input
        Expectation: desired_input_layouts equals input_layouts (no AG redistribute)
        """
        style = MC2ColwiseParallel(input_layouts=Shard(1), use_local_output=False)
        self.assertEqual(style.input_layouts, (Shard(1),))
        self.assertEqual(style.desired_input_layouts, (Shard(1),))
        self.assertEqual(style._sequence_dim, 1)
        self.assertIsInstance(style, ColwiseParallel)

    def test_mc2_rowwise_records_sequence_dim(self):
        """
        Feature: MC2RowwiseParallel sequence dim from output layout
        Description: construct with Shard(1) output
        Expectation: _sequence_dim is 1; subclass of RowwiseParallel
        """
        style = MC2RowwiseParallel(output_layouts=Shard(1), use_local_output=False)
        self.assertEqual(style.output_layouts, (Shard(1),))
        self.assertEqual(style._sequence_dim, 1)
        self.assertIsInstance(style, RowwiseParallel)


class TestMC2StyleApply(unittest.TestCase):
    """apply() replaces Linear before delegating to the base style."""

    def test_mc2_colwise_replaces_linear_in_place(self):
        """
        Feature: MC2ColwiseParallel.apply Linear replacement
        Description: mock base apply and call MC2ColwiseParallel.apply
        Expectation: module becomes MC2Linear with all_gather mode and mesh group/size
        """
        style = MC2ColwiseParallel(input_layouts=Shard(0), use_local_output=False)
        linear = nn.Linear(4, 8, bias=False)
        weight = linear.weight

        with patch.object(ColwiseParallel, "apply", lambda self, module, device_mesh: module):
            result = style.apply(linear, _FakeMesh())

        self.assertIs(result, linear)
        self.assertIsInstance(result, MC2Linear)
        self.assertIs(result.weight, weight)
        self.assertEqual(result.mc2_mode, "all_gather")
        self.assertEqual(result.mc2_group, "tp-group")
        self.assertEqual(result.mc2_world_size, 4)
        self.assertEqual(result.mc2_sequence_dim, 0)

    def test_mc2_rowwise_replaces_linear_in_place(self):
        """
        Feature: MC2RowwiseParallel.apply Linear replacement
        Description: mock base apply and call MC2RowwiseParallel.apply
        Expectation: module becomes MC2Linear with reduce_scatter mode
        """
        style = MC2RowwiseParallel(output_layouts=Shard(1), use_local_output=False)
        linear = nn.Linear(8, 4, bias=True)
        weight = linear.weight

        with patch.object(RowwiseParallel, "apply", lambda self, module, device_mesh: module):
            result = style.apply(linear, _FakeMesh())

        self.assertIs(result, linear)
        self.assertIsInstance(result, MC2Linear)
        self.assertIs(result.weight, weight)
        self.assertEqual(result.mc2_mode, "reduce_scatter")
        self.assertEqual(result.mc2_group, "tp-group")
        self.assertEqual(result.mc2_world_size, 4)
        self.assertEqual(result.mc2_sequence_dim, 1)

    def test_mc2_style_rejects_non_linear(self):
        """
        Feature: MC2 styles reject unsupported modules
        Description: apply to nn.Embedding
        Expectation: NotImplementedError
        """
        style = MC2ColwiseParallel(input_layouts=Shard(0))
        with self.assertRaises(NotImplementedError):
            style.apply(nn.Embedding(10, 4), _FakeMesh())


class TestAllGatherMatmulBackwardFused(unittest.TestCase):
    """Column-parallel backward should call fused matmul_reduce_scatter (MindFormers)."""

    def test_column_backward_uses_npu_mm_reduce_scatter_base(self):
        """
        Feature: AllGatherMatmulFunction.backward uses fused MRS by default
        Description: mock torch_npu; run backward with n_local-sized weight
        Expectation: npu_mm_reduce_scatter_base called once with (grad_out, w)
        """
        m_full, n_local, k = 8, 256, 32
        grad_out = torch.randn(m_full, n_local)
        gathered = torch.randn(m_full, k)
        w = torch.randn(n_local, k)
        fake_dx = torch.randn(m_full // 2, k)

        mock_npu = MagicMock()
        mock_npu.npu_mm_reduce_scatter_base.return_value = fake_dx

        ctx = _Ctx(gathered, w, hcom="fake-hcom", world_size=2, has_bias=False)
        with patch(
            "hyper_parallel.core.tensor_parallel.mc2._require_torch_npu",
            return_value=mock_npu,
        ):
            grad_x, grad_w, *_ = AllGatherMatmulFunction.backward(ctx, grad_out)

        mock_npu.npu_mm_reduce_scatter_base.assert_called_once()
        args, kwargs = mock_npu.npu_mm_reduce_scatter_base.call_args
        # contiguous() may allocate a copy; check layout contract.
        self.assertEqual(tuple(args[0].shape), (m_full, n_local))
        self.assertIs(args[1], w)
        self.assertEqual(args[2], "fake-hcom")
        self.assertEqual(args[3], 2)
        self.assertEqual(kwargs.get("reduce_op"), "sum")
        self.assertIs(grad_x, fake_dx)
        self.assertEqual(tuple(grad_w.shape), (n_local, k))


class TestMC2SequenceDimLayout(unittest.TestCase):
    """SP sequence-dim helpers used before flatten + fused AG/RS."""

    def test_normalize_sequence_dim(self):
        """
        Feature: sequence_dim normalization
        Description: positive / negative dims and out-of-range
        Expectation: resolved index or RuntimeError
        """
        self.assertEqual(_normalize_sequence_dim(1, 2), 1)
        self.assertEqual(_normalize_sequence_dim(-1, 2), 1)
        with self.assertRaises(RuntimeError):
            _normalize_sequence_dim(2, 2)

    def test_move_dim_roundtrip(self):
        """
        Feature: move sequence dim to front and back
        Description: permute [B, S, H] <-> [S, B, H]
        Expectation: round-trip equals original; front layout is seq-first
        """
        x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        y = _move_dim_to_front(x, 1)
        self.assertEqual(tuple(y.shape), (4, 2, 3))
        self.assertTrue(torch.equal(y[0], x[:, 0, :]))
        self.assertTrue(torch.equal(_move_front_to_dim(y, 1), x))

    def test_batch_first_all_gather_needs_seq_front(self):
        """
        Feature: flatten+AG layout for Shard(1) batch-first activations
        Description: simulate 2-rank AG on dim0 after flatten; compare naive vs seq-front
        Expectation: only seq-front path reconstructs global [B, S, H]
        """
        batch, seq, hidden, world = 2, 8, 4, 2
        full = torch.arange(batch * seq * hidden, dtype=torch.float32).reshape(
            batch, seq, hidden
        )
        seq_local = seq // world
        shards = [
            full[:, r * seq_local:(r + 1) * seq_local].contiguous() for r in range(world)
        ]

        # Naive flatten then cat (old MC2Linear bug): wrong token order.
        naive = torch.cat([s.reshape(-1, hidden) for s in shards], dim=0)
        naive_out = naive.reshape(batch, seq, hidden)
        self.assertFalse(torch.equal(naive_out, full))

        # Move seq to front, flatten, AG, reshape, restore (fixed path).
        gathered = torch.cat(
            [_move_dim_to_front(s, 1).reshape(-1, hidden) for s in shards], dim=0
        )
        seq_first = gathered.reshape(seq, batch, hidden)
        fixed = _move_front_to_dim(seq_first, 1)
        self.assertTrue(torch.equal(fixed, full))


if __name__ == "__main__":
    unittest.main()
