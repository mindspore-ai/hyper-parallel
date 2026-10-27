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
"""Expert output aggregation for empty ranks and repeated token routes."""

import unittest

import torch

from hyper_parallel.distributed.expert_parallel.experts import _aggregate_ep_outputs


class TestExpertAggregation(unittest.TestCase):
    """Compare the fused-aggregation contract against weighted scatter addition."""

    def test_empty_local_tokens_keep_autograd(self) -> None:
        """Feature: Empty EP source rank.

        Description: Aggregate zero returned tokens.
        Expectation: Return the requested empty shape and preserve both gradient paths.
        """
        states = torch.empty(0, 4, requires_grad=True)
        weights = torch.empty(0, requires_grad=True)
        indices = torch.empty(0, dtype=torch.long)
        result = _aggregate_ep_outputs(states, weights, indices, indices, (1, 0, 4))
        self.assertEqual(tuple(result.shape), (1, 0, 4))
        result.sum().backward()
        self.assertIsNotNone(states.grad)
        self.assertIsNotNone(weights.grad)

    def test_repeated_routes_match_scatter_gradients(self) -> None:
        """Feature: Multi-expert aggregation.

        Description: Restore a nontrivial expert order with three routes per token.
        Expectation: Output and state/router-weight gradients match scatter-add.
        """
        torch.manual_seed(71)
        states = torch.randn(6, 4, requires_grad=True)
        weights = torch.randn(6, requires_grad=True)
        expected_states = states.detach().clone().requires_grad_()
        expected_weights = weights.detach().clone().requires_grad_()
        order = torch.tensor([5, 0, 3, 1, 4, 2])
        source_indices = torch.arange(2).repeat_interleave(3)
        result = _aggregate_ep_outputs(states, weights, source_indices, order, (1, 2, 4))
        expected = torch.zeros(2, 4).index_add(
            0, source_indices[order], expected_states * expected_weights[order].unsqueeze(-1),
        ).view(1, 2, 4)
        torch.testing.assert_close(result, expected)
        result.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(states.grad, expected_states.grad)
        torch.testing.assert_close(weights.grad, expected_weights.grad)

    def test_capacity_dropped_routes_keep_source_mapping(self) -> None:
        """Feature: Capacity-limited routing.

        Description: Keep two routes for token zero and one for token two.
        Expectation: Dropped token one stays zero without inventing a fixed top-k reshape.
        """
        states = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
        weights = torch.tensor([0.2, 0.3, 0.7], requires_grad=True)
        sources = torch.tensor([0, 0, 2])
        order = torch.tensor([2, 0, 1])
        output = _aggregate_ep_outputs(states, weights, sources, order, (1, 3, 2))
        expected = torch.stack((states[1] * weights[0] + states[2] * weights[1],
                                states.new_zeros(2), states[0] * weights[2])).unsqueeze(0)
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        torch.testing.assert_close(states.grad, weights.detach()[order, None].expand_as(states))
