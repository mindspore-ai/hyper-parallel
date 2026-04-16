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
"""Unit tests for ``hyper_parallel.core.expert_parallel.expert_parallel``.

Tests cover:
- C1: _generate_permute_indices, _permute, _unpermute (pure tensor, no mocks)
- C2: ExpertParallel._token_dispatch and _token_combine (mocked platform collectives)
- C3: TensorParallel._partition_fn (mocked DTensor helpers)
- C4: ExpertTensorParallel._partition_fn and dispatch/combine delegation

All tests run on CPU without any distributed setup.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.expert_parallel.expert_parallel import (
    _generate_permute_indices,
    _permute,
    _unpermute,
    ExpertParallel,
    TensorParallel,
    ExpertTensorParallel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_device_mesh(ep_size: int, local_rank: int = 0):
    """Return a minimal DeviceMesh mock for EP unit tests."""
    mesh = MagicMock()
    mesh.size.return_value = ep_size
    mesh.get_group.return_value = None  # group not used when platform is mocked
    mesh.get_local_rank.return_value = local_rank
    return mesh


def _make_mock_module(num_experts: int = 4, dim: int = 8, hidden_dim: int = 16):
    """Create a minimal nn.Module mimic with w1, w2, w3 parameters."""
    from torch import nn

    class _FakeExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim))
            self.w2 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim))
            self.w3 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim))

    return _FakeExperts()


# ---------------------------------------------------------------------------
# C1: _generate_permute_indices
# ---------------------------------------------------------------------------

class TestGeneratePermuteIndices(unittest.TestCase):
    """Unit tests for ``_generate_permute_indices``."""

    def test_basic_2ranks_2experts(self):
        """
        Feature: _generate_permute_indices basic case
        Description: 2 ranks, 2 experts per rank — verify expert-major ordering.
            counts[r*E + e] = tokens from rank r for local expert e.
            rank0->expert0: 3, rank0->expert1: 2, rank1->expert0: 1, rank1->expert1: 4
        Expectation: num_tok = [4, 6], total permuted indices = 10
        """
        counts = torch.tensor([3, 2, 1, 4])
        perm_indices, num_tok = _generate_permute_indices(counts, experts_per_rank=2, num_ranks=2)
        self.assertEqual(int(num_tok[0]), 4, f"expert 0 should get 4 tokens, got {num_tok[0]}")
        self.assertEqual(int(num_tok[1]), 6, f"expert 1 should get 6 tokens, got {num_tok[1]}")
        self.assertEqual(perm_indices.numel(), 10, f"total tokens should be 10, got {perm_indices.numel()}")

    def test_permuted_indices_select_correct_tokens(self):
        """
        Feature: _generate_permute_indices index correctness
        Description: perm_indices correctly maps expert-major -> rank-major positions.
            rank-major buffer: positions 0-2 = rank0*expert0, 3-4 = rank0*expert1,
            5 = rank1*expert0, 6-9 = rank1*expert1
        Expectation: first 4 entries (expert-0 block) are from positions {0, 1, 2, 5}
        """
        counts = torch.tensor([3, 2, 1, 4])
        perm_indices, _ = _generate_permute_indices(counts, experts_per_rank=2, num_ranks=2)
        expert0_sources = set(perm_indices[:4].tolist())
        self.assertEqual(
            expert0_sources, {0, 1, 2, 5},
            f"Expert-0 sources should be {{0,1,2,5}}, got {expert0_sources}"
        )

    def test_zero_tokens_for_some_experts(self):
        """
        Feature: _generate_permute_indices with zero-token experts
        Description: Zero-token experts produce correct zero counts and no extra indices.
        Expectation: num_tok = [5, 3], total indices = 8
        """
        counts = torch.tensor([5, 0, 0, 3])
        perm_indices, num_tok = _generate_permute_indices(counts, experts_per_rank=2, num_ranks=2)
        self.assertEqual(int(num_tok[0]), 5, f"expert 0: {num_tok[0]}, expected 5")
        self.assertEqual(int(num_tok[1]), 3, f"expert 1: {num_tok[1]}, expected 3")
        self.assertEqual(perm_indices.numel(), 8, f"total tokens: {perm_indices.numel()}, expected 8")

    def test_all_zero_tokens(self):
        """
        Feature: _generate_permute_indices all-zero counts
        Description: All-zero token counts return empty indices.
        Expectation: perm_indices is empty, num_tok sums to 0
        """
        counts = torch.zeros(4, dtype=torch.long)
        perm_indices, num_tok = _generate_permute_indices(counts, experts_per_rank=2, num_ranks=2)
        self.assertEqual(
            perm_indices.numel(), 0,
            f"Expected empty perm_indices, got numel={perm_indices.numel()}"
        )
        self.assertEqual(int(num_tok.sum()), 0, f"Expected all-zero num_tok, got {num_tok}")

    def test_uneven_token_distribution(self):
        """
        Feature: _generate_permute_indices uneven distribution
        Description: 3 ranks, 3 experts each with uneven token counts.
        Expectation: expert 0 gets 8, expert 1 gets 8, expert 2 gets 7, total 23
        """
        counts = torch.tensor([2, 5, 1, 0, 3, 4, 6, 0, 2])  # 3 ranks x 3 experts
        perm_indices, num_tok = _generate_permute_indices(counts, experts_per_rank=3, num_ranks=3)
        self.assertEqual(int(num_tok[0]), 8, f"expert 0: {num_tok[0]}, expected 8")
        self.assertEqual(int(num_tok[1]), 8, f"expert 1: {num_tok[1]}, expected 8")
        self.assertEqual(int(num_tok[2]), 7, f"expert 2: {num_tok[2]}, expected 7")
        self.assertEqual(perm_indices.numel(), 23, f"total tokens: {perm_indices.numel()}, expected 23")


# ---------------------------------------------------------------------------
# C1: _permute
# ---------------------------------------------------------------------------

class TestPermute(unittest.TestCase):
    """Unit tests for ``_permute``."""

    def test_output_shapes(self):
        """
        Feature: _permute output shapes
        Description: _permute returns tensors with correct shapes for 2 ranks, 2 experts.
        Expectation: orig_shape = (10, dim), permuted_x.shape = (10, dim),
            perm_idx.numel() = 10, num_tok.shape = (2,)
        """
        dim = 4
        counts = torch.tensor([3, 2, 1, 4])  # 2 ranks, 2 local experts
        total = int(counts.sum())
        x = torch.randn(total, dim)
        orig_shape, permuted_x, perm_idx, num_tok = _permute(
            x, counts, ep_degree=2, num_local_experts=2
        )
        self.assertEqual(
            orig_shape, (total, dim),
            f"orig_shape: {orig_shape}, expected ({total}, {dim})"
        )
        self.assertEqual(
            permuted_x.shape, (total, dim),
            f"permuted_x shape: {permuted_x.shape}, expected ({total}, {dim})"
        )
        self.assertEqual(
            perm_idx.numel(), total,
            f"perm_idx.numel()={perm_idx.numel()}, expected {total}"
        )
        self.assertEqual(
            num_tok.shape, (2,),
            f"num_tok shape: {num_tok.shape}, expected (2,)"
        )

    def test_permuted_content_matches_index(self):
        """
        Feature: _permute content correctness
        Description: permuted_x[i] == x[perm_idx[i]] for all i.
        Expectation: every row of permuted_x equals the indexed row of x
        """
        dim = 6
        counts = torch.tensor([2, 3, 4, 1])
        x = torch.arange(int(counts.sum()) * dim, dtype=torch.float).view(-1, dim)
        _, permuted_x, perm_idx, _ = _permute(x, counts, ep_degree=2, num_local_experts=2)
        for i in range(permuted_x.shape[0]):
            self.assertTrue(
                torch.equal(permuted_x[i], x[perm_idx[i]]),
                f"permuted_x[{i}] != x[perm_idx[{i}]]"
            )

    def test_empty_input_returns_zero_tensor(self):
        """
        Feature: _permute with all-zero counts
        Description: When all token counts are zero, permuted_x should be empty.
        Expectation: permuted_x.numel() = 0
        """
        dim = 4
        counts = torch.zeros(4, dtype=torch.long)
        x = torch.zeros(0, dim)
        _, permuted_x, perm_idx, num_tok = _permute(x, counts, ep_degree=2, num_local_experts=2)
        self.assertEqual(
            permuted_x.numel(), 0,
            f"Expected empty permuted_x, got numel={permuted_x.numel()}"
        )
        self.assertEqual(int(num_tok.sum()), 0, f"num_tok should be all-zero, got {num_tok}")


# ---------------------------------------------------------------------------
# C1: _unpermute
# ---------------------------------------------------------------------------

class TestUnpermute(unittest.TestCase):
    """Unit tests for ``_unpermute``."""

    def test_round_trip(self):
        """
        Feature: _unpermute round-trip
        Description: _unpermute(_permute(x)) reproduces x exactly.
        Expectation: restored tensor is allclose to original x
        """
        dim = 5
        counts = torch.tensor([3, 2, 1, 4])
        x = torch.randn(int(counts.sum()), dim)
        orig_shape, permuted_x, perm_idx, _ = _permute(x, counts, ep_degree=2, num_local_experts=2)
        restored = _unpermute(permuted_x, orig_shape, perm_idx)
        self.assertTrue(
            torch.allclose(restored, x, atol=1e-6),
            f"Round-trip failed: max diff={(restored - x).abs().max():.2e}"
        )

    def test_round_trip_with_zero_tokens(self):
        """
        Feature: _unpermute round-trip with zero-token experts
        Description: Round-trip works when some experts have zero tokens.
        Expectation: restored tensor is allclose to original x
        """
        dim = 3
        counts = torch.tensor([4, 0, 0, 6])
        x = torch.randn(10, dim)
        orig_shape, permuted_x, perm_idx, _ = _permute(x, counts, ep_degree=2, num_local_experts=2)
        restored = _unpermute(permuted_x, orig_shape, perm_idx)
        self.assertTrue(
            torch.allclose(restored, x, atol=1e-6),
            f"Round-trip with zeros failed: max diff={(restored - x).abs().max():.2e}"
        )

    def test_gradient_flows_through_unpermute(self):
        """
        Feature: _unpermute gradient flow
        Description: Gradient flows back through _unpermute via scatter indexing.
        Expectation: x.grad is not None after backward
        """
        dim = 4
        counts = torch.tensor([2, 3, 1, 2])
        total = int(counts.sum())
        x = torch.randn(total, dim, requires_grad=True)
        _, permuted_x, perm_idx, _ = _permute(x, counts, ep_degree=2, num_local_experts=2)
        restored = _unpermute(permuted_x, (total, dim), perm_idx)
        restored.sum().backward()
        self.assertIsNotNone(x.grad, "Gradient should flow through _permute and _unpermute")


# ---------------------------------------------------------------------------
# C2: ExpertParallel._token_dispatch (mocked platform collectives)
# ---------------------------------------------------------------------------

class TestExpertParallelDispatch(unittest.TestCase):
    """Unit tests for ``ExpertParallel._token_dispatch`` with mocked collectives."""

    def setUp(self):
        """Set up ExpertParallel instance and mock device_mesh."""
        self.ep = ExpertParallel()
        self.ep_size = 2
        self.num_local_experts = 2
        self.dim = 8
        # counts[r * E + e] = tokens from rank r for local expert e
        # This rank (rank 0) receives: rank0->exp0=3, rank0->exp1=2, rank1->exp0=1, rank1->exp1=4
        self.counts_out = torch.tensor([3, 2, 1, 4])  # received token counts per (rank, expert)
        self.total_tokens = int(self.counts_out.sum())  # 10

        # input: tokens this rank sends out, same layout
        self.num_tokens_per_expert_in = torch.tensor([3, 2, 1, 4])
        self.routed_input = torch.randn(self.total_tokens, self.dim)
        self.mock_mesh = _make_mock_device_mesh(self.ep_size)

    def _call_dispatch(self, mock_platform):
        """Configure platform mock and call _token_dispatch."""
        # all_to_all_single returns (counts_out, None)
        mock_platform.all_to_all_single.return_value = (self.counts_out, None)
        # differentiable_all_to_all_single returns input unchanged (identity)
        mock_platform.differentiable_all_to_all_single.side_effect = (
            lambda inp, *_args, **_kw: inp
        )
        mock_platform.arange.side_effect = torch.arange

        return self.ep._token_dispatch(
            module=None,
            inputs=(self.routed_input, self.num_tokens_per_expert_in),
            device_mesh=self.mock_mesh,
        )

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_dispatch_returns_permuted_and_counts(self, mock_platform):
        """
        Feature: ExpertParallel._token_dispatch return values
        Description: dispatch returns (permuted_x, local_counts) with correct shapes.
        Expectation: permuted_x.shape = (10, dim), local_counts.shape = (2,)
        """
        permuted_x, local_counts = self._call_dispatch(mock_platform)
        self.assertEqual(
            permuted_x.shape, (self.total_tokens, self.dim),
            f"permuted_x.shape={permuted_x.shape}, expected ({self.total_tokens}, {self.dim})"
        )
        self.assertEqual(
            local_counts.shape, (self.num_local_experts,),
            f"local_counts.shape={local_counts.shape}, expected ({self.num_local_experts},)"
        )

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_dispatch_computes_correct_splits(self, mock_platform):
        """
        Feature: ExpertParallel._token_dispatch split computation
        Description: input_splits and output_splits are derived from token counts.
            input_splits[r] = tokens this rank sends to rank r.
            output_splits[r] = tokens this rank receives from rank r.
        Expectation: input_splits = [5, 5], output_splits = [5, 5]
        """
        self._call_dispatch(mock_platform)
        # rank0 block: 3+2=5, rank1 block: 1+4=5
        self.assertEqual(
            self.ep._input_splits, [5, 5],
            f"_input_splits={self.ep._input_splits}, expected [5, 5]"
        )
        self.assertEqual(
            self.ep._output_splits, [5, 5],
            f"_output_splits={self.ep._output_splits}, expected [5, 5]"
        )

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_dispatch_produces_expert_major_order(self, mock_platform):
        """
        Feature: ExpertParallel._token_dispatch expert-major permutation
        Description: Dispatch permutes tokens into expert-major order so that
            the first num_tok[0] rows are for local expert 0 and the remaining
            for local expert 1.
        Expectation: local_counts = [4, 6] matching counts_out column sums
        """
        _, local_counts = self._call_dispatch(mock_platform)
        # expert 0: rank0->exp0(3) + rank1->exp0(1) = 4
        # expert 1: rank0->exp1(2) + rank1->exp1(4) = 6
        self.assertEqual(int(local_counts[0]), 4, f"local_counts[0]={local_counts[0]}, expected 4")
        self.assertEqual(int(local_counts[1]), 6, f"local_counts[1]={local_counts[1]}, expected 6")

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_dispatch_saves_state_for_combine(self, mock_platform):
        """
        Feature: ExpertParallel._token_dispatch state preservation
        Description: After dispatch, _input_splits, _output_splits,
            _input_shape, and _permuted_indices are saved for use by combine.
        Expectation: all four state attributes are not None / non-empty
        """
        self._call_dispatch(mock_platform)
        self.assertIsNotNone(self.ep._input_splits, "_input_splits should be set")
        self.assertIsNotNone(self.ep._output_splits, "_output_splits should be set")
        self.assertIsNotNone(self.ep._input_shape, "_input_shape should be set")
        self.assertIsNotNone(self.ep._permuted_indices, "_permuted_indices should be set")


# ---------------------------------------------------------------------------
# C2: ExpertParallel._token_combine (mocked platform collectives)
# ---------------------------------------------------------------------------

class TestExpertParallelCombine(unittest.TestCase):
    """Unit tests for ``ExpertParallel._token_combine`` with mocked collectives."""

    def setUp(self):
        """Run dispatch first so combine has the saved state it needs."""
        self.ep = ExpertParallel()
        self.ep_size = 2
        self.num_local_experts = 2
        self.dim = 8
        self.counts_out = torch.tensor([3, 2, 1, 4])
        self.total_tokens = int(self.counts_out.sum())
        self.num_tokens_per_expert_in = torch.tensor([3, 2, 1, 4])
        self.routed_input = torch.randn(self.total_tokens, self.dim)
        self.mock_mesh = _make_mock_device_mesh(self.ep_size)

    def _run_dispatch_and_combine(self, expert_output, mock_platform):
        """Run dispatch then combine using the same platform mock."""
        mock_platform.all_to_all_single.return_value = (self.counts_out, None)
        mock_platform.arange.side_effect = torch.arange

        captured = {}

        def identity_a2a(inp, *_args, **_kw):
            captured["input"] = inp
            return inp

        mock_platform.differentiable_all_to_all_single.side_effect = identity_a2a

        self.ep._token_dispatch(
            module=None,
            inputs=(self.routed_input, self.num_tokens_per_expert_in),
            device_mesh=self.mock_mesh,
        )
        combined = self.ep._token_combine(
            module=None,
            routed_output=expert_output,
            device_mesh=self.mock_mesh,
        )
        return combined

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_combine_output_shape_matches_original(self, mock_platform):
        """
        Feature: ExpertParallel._token_combine output shape
        Description: combine restores tensor to the shape it had before dispatch.
        Expectation: combined.shape == (total_tokens, dim)
        """
        expert_output = torch.randn(self.total_tokens, self.dim)
        combined = self._run_dispatch_and_combine(expert_output, mock_platform)
        self.assertEqual(
            combined.shape, (self.total_tokens, self.dim),
            f"combined.shape={combined.shape}, expected ({self.total_tokens}, {self.dim})"
        )

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_combine_is_inverse_of_dispatch_permutation(self, mock_platform):
        """
        Feature: ExpertParallel._token_combine round-trip
        Description: dispatch then combine with identity expert computation
            restores the original routed_input tensor.
        Expectation: combined is allclose to routed_input
        """
        # Since platform a2a is identity, the token values don't change.
        # After dispatch (permute) and combine (unpermute) with identity expert,
        # we should recover routed_input.
        mock_platform.arange.side_effect = torch.arange
        _, permuted_x, perm_idx, _ = _permute(
            self.routed_input, self.counts_out, self.ep_size, self.num_local_experts
        )
        combined = self._run_dispatch_and_combine(permuted_x, mock_platform)
        self.assertTrue(
            torch.allclose(combined, self.routed_input, atol=1e-6),
            (f"Round-trip failed: "
             f"max diff={(combined - self.routed_input).abs().max():.2e}")
        )

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_combine_calls_differentiable_a2a(self, mock_platform):
        """
        Feature: ExpertParallel._token_combine uses differentiable all-to-all
        Description: _token_combine must call differentiable_all_to_all_single so
            gradients can flow back through the combine step.
        Expectation: differentiable_all_to_all_single called at least once during combine
        """
        expert_output = torch.randn(self.total_tokens, self.dim)
        self._run_dispatch_and_combine(expert_output, mock_platform)
        call_count = mock_platform.differentiable_all_to_all_single.call_count
        # dispatch calls it once, combine calls it once -> total >= 2
        self.assertGreaterEqual(
            call_count, 2,
            f"differentiable_all_to_all_single called {call_count} times, expected >= 2"
        )

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_combine_swaps_input_output_splits(self, mock_platform):
        """
        Feature: ExpertParallel._token_combine swaps splits for reverse all-to-all
        Description: combine passes (output_splits, input_splits) to the all-to-all,
            i.e. they are swapped relative to dispatch.
        Expectation: the second a2a call's input_splits == dispatch output_splits
        """
        expert_output = torch.randn(self.total_tokens, self.dim)
        self._run_dispatch_and_combine(expert_output, mock_platform)
        call_args_list = mock_platform.differentiable_all_to_all_single.call_args_list
        # call_args_list[0] = dispatch call, call_args_list[1] = combine call
        self.assertGreaterEqual(len(call_args_list), 2)
        combine_call = call_args_list[1]
        # positional args: (inp, input_splits, output_splits, group)
        combine_input_splits = combine_call.args[1]
        dispatch_output_splits = self.ep._output_splits
        self.assertEqual(
            combine_input_splits, dispatch_output_splits,
            (f"combine input_splits={combine_input_splits} "
             f"should equal dispatch output_splits={dispatch_output_splits}")
        )


# ---------------------------------------------------------------------------
# C3: TensorParallel._partition_fn (mocked DTensor helpers)
# ---------------------------------------------------------------------------

class TestTensorParallelPartition(unittest.TestCase):
    """Unit tests for ``TensorParallel._partition_fn`` with mocked DTensor helpers."""

    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_set_param")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_new_parameter")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel.distribute_tensor")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_param_source")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_iter_params")
    def test_tp_partition_uses_correct_shard_dims(
        self,
        mock_iter_params,
        mock_param_source,
        mock_distribute_tensor,
        mock_new_param,
        mock_set_param,
    ):
        """
        Feature: TensorParallel._partition_fn shard dimensions
        Description: w1 and w3 use Shard(1) (column-wise), w2 uses Shard(2) (row-wise).
        Expectation: distribute_tensor called with Shard(1) for w1/w3 and Shard(2) for w2
        """
        from hyper_parallel.core.dtensor.placement_types import Shard

        module = _make_mock_module()
        device_mesh = _make_mock_device_mesh(ep_size=2)
        tp = TensorParallel()

        params = list(module.named_parameters())
        mock_iter_params.return_value = params
        mock_param_source.side_effect = lambda p: p.data
        mock_distribute_tensor.return_value = MagicMock()
        mock_new_param.return_value = MagicMock()

        tp._partition_fn("", module, device_mesh)

        shard_by_key = {}
        for call, (key, _) in zip(mock_distribute_tensor.call_args_list, params):
            shard_by_key[key] = call.args[2]

        self.assertEqual(
            shard_by_key.get("w1"), [Shard(1)],
            f"w1 should use Shard(1), got {shard_by_key.get('w1')}"
        )
        self.assertEqual(
            shard_by_key.get("w2"), [Shard(2)],
            f"w2 should use Shard(2), got {shard_by_key.get('w2')}"
        )
        self.assertEqual(
            shard_by_key.get("w3"), [Shard(1)],
            f"w3 should use Shard(1), got {shard_by_key.get('w3')}"
        )


# ---------------------------------------------------------------------------
# C2: ExpertParallel._partition_fn (mocked DTensor helpers)
# ---------------------------------------------------------------------------

class TestExpertParallelPartition(unittest.TestCase):
    """Unit tests for ``ExpertParallel._partition_fn`` with mocked DTensor helpers."""

    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_set_param")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_new_parameter")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel.distribute_tensor")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_param_source")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_iter_params")
    def test_ep_partition_shards_all_params_on_dim0(
        self,
        mock_iter_params,
        mock_param_source,
        mock_distribute_tensor,
        mock_new_param,
        mock_set_param,
    ):
        """
        Feature: ExpertParallel._partition_fn shards on dim 0
        Description: _partition_fn calls distribute_tensor with Shard(0) for every parameter.
        Expectation: distribute_tensor called once per parameter with [Shard(0)] placement
        """
        from hyper_parallel.core.dtensor.placement_types import Shard

        module = _make_mock_module()
        device_mesh = _make_mock_device_mesh(ep_size=2)
        ep = ExpertParallel()

        # Simulate iter_params yielding w1, w2, w3
        params = list(module.named_parameters())
        mock_iter_params.return_value = params
        mock_param_source.side_effect = lambda p: p.data
        mock_distribute_tensor.return_value = MagicMock()
        mock_new_param.return_value = MagicMock()

        ep._partition_fn("", module, device_mesh)

        self.assertEqual(
            mock_distribute_tensor.call_count, len(params),
            (f"distribute_tensor should be called {len(params)} times, "
             f"got {mock_distribute_tensor.call_count}")
        )
        for call in mock_distribute_tensor.call_args_list:
            placements = call.args[2]
            self.assertEqual(
                placements, [Shard(0)],
                f"ExpertParallel should shard on Shard(0), got {placements}"
            )


# ---------------------------------------------------------------------------
# C4: ExpertTensorParallel._partition_fn (mocked DTensor helpers)
# ---------------------------------------------------------------------------

class TestExpertTensorParallelPartition(unittest.TestCase):
    """Unit tests for ``ExpertTensorParallel._partition_fn`` with mocked DTensor helpers."""

    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_set_param")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_new_parameter")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel.distribute_tensor")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_param_source")
    @patch("hyper_parallel.core.expert_parallel.expert_parallel._distribute_module_iter_params")
    def test_etp_partition_uses_2d_shard_placements(
        self,
        mock_iter_params,
        mock_param_source,
        mock_distribute_tensor,
        mock_new_param,
        mock_set_param,
    ):
        """
        Feature: ExpertTensorParallel._partition_fn 2-D shard placements
        Description: w1/w3 use [Shard(0), Shard(1)]; w2 uses [Shard(0), Shard(2)].
        Expectation: distribute_tensor called with 2-element placements list for each param
        """
        from hyper_parallel.core.dtensor.placement_types import Shard

        module = _make_mock_module()
        device_mesh = _make_mock_device_mesh(ep_size=4)
        etp = ExpertTensorParallel()

        params = list(module.named_parameters())
        mock_iter_params.return_value = params
        mock_param_source.side_effect = lambda p: p.data
        mock_distribute_tensor.return_value = MagicMock()
        mock_new_param.return_value = MagicMock()

        etp._partition_fn("", module, device_mesh)

        shard_by_key = {}
        for call, (key, _) in zip(mock_distribute_tensor.call_args_list, params):
            shard_by_key[key] = call.args[2]

        self.assertEqual(
            shard_by_key.get("w1"), [Shard(0), Shard(1)],
            f"w1 should use [Shard(0), Shard(1)], got {shard_by_key.get('w1')}"
        )
        self.assertEqual(
            shard_by_key.get("w2"), [Shard(0), Shard(2)],
            f"w2 should use [Shard(0), Shard(2)], got {shard_by_key.get('w2')}"
        )
        self.assertEqual(
            shard_by_key.get("w3"), [Shard(0), Shard(1)],
            f"w3 should use [Shard(0), Shard(1)], got {shard_by_key.get('w3')}"
        )

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_etp_dispatch_delegates_to_ep_submesh(self, mock_platform):
        """
        Feature: ExpertTensorParallel._token_dispatch delegates to EP sub-mesh
        Description: _token_dispatch should use device_mesh["ep"] for all-to-all,
            not the full 2-D mesh.
        Expectation: _token_dispatch is called with the ep sub-mesh
        """
        etp = ExpertTensorParallel()
        ep_size = 2
        num_local_experts = 2
        dim = 8
        counts_out = torch.tensor([3, 2, 1, 4])
        total_tokens = int(counts_out.sum())
        num_tokens_per_expert = torch.tensor([3, 2, 1, 4])
        routed_input = torch.randn(total_tokens, dim)

        # 2-D mesh mock: device_mesh["ep"] returns a 1-D sub-mesh mock
        ep_submesh = _make_mock_device_mesh(ep_size)
        full_mesh = MagicMock()
        full_mesh.__getitem__ = MagicMock(return_value=ep_submesh)

        mock_platform.all_to_all_single.return_value = (counts_out, None)
        mock_platform.differentiable_all_to_all_single.side_effect = (
            lambda inp, *_args, **_kw: inp
        )
        mock_platform.arange.side_effect = torch.arange

        etp._token_dispatch(
            module=None,
            inputs=(routed_input, num_tokens_per_expert),
            device_mesh=full_mesh,
        )

        # Verify ["ep"] was accessed on the 2-D mesh
        full_mesh.__getitem__.assert_called_once_with("ep")

    @patch("hyper_parallel.core.expert_parallel.expert_parallel.platform")
    def test_etp_combine_delegates_to_ep_submesh(self, mock_platform):
        """
        Feature: ExpertTensorParallel._token_combine delegates to EP sub-mesh
        Description: _token_combine should use device_mesh["ep"] for all-to-all.
        Expectation: _token_combine is called with the ep sub-mesh
        """
        etp = ExpertTensorParallel()
        ep_size = 2
        dim = 8
        counts_out = torch.tensor([3, 2, 1, 4])
        total_tokens = int(counts_out.sum())
        num_tokens_per_expert = torch.tensor([3, 2, 1, 4])
        routed_input = torch.randn(total_tokens, dim)

        ep_submesh = _make_mock_device_mesh(ep_size)
        full_mesh = MagicMock()
        full_mesh.__getitem__ = MagicMock(return_value=ep_submesh)

        mock_platform.all_to_all_single.return_value = (counts_out, None)
        mock_platform.differentiable_all_to_all_single.side_effect = (
            lambda inp, *_args, **_kw: inp
        )
        mock_platform.arange.side_effect = torch.arange

        # First dispatch to set state
        etp._token_dispatch(
            module=None,
            inputs=(routed_input, num_tokens_per_expert),
            device_mesh=full_mesh,
        )

        full_mesh.__getitem__.reset_mock()
        expert_output = torch.randn(total_tokens, dim)
        etp._token_combine(
            module=None,
            routed_output=expert_output,
            device_mesh=full_mesh,
        )

        # Verify ["ep"] was accessed on the 2-D mesh for combine too
        full_mesh.__getitem__.assert_called_once_with("ep")


if __name__ == "__main__":
    unittest.main()
