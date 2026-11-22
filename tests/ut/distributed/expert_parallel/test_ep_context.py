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
"""Unit tests for routed-slot expert-parallel transport."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from tests.common.mark_utils import arg_mark


def _mesh(size: int, rank: int = 0) -> MagicMock:
    """Build the process-group-free mesh surface consumed by EPContext."""
    mesh = MagicMock()
    mesh.ndim = 1
    mesh.size.return_value = size
    mesh.get_local_rank.return_value = rank
    mesh.get_group.return_value = object()
    return mesh


class TestEPContext(unittest.TestCase):
    """Verify slot metadata and the second EP permutation on CPU."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_single_rank_round_trip_keeps_slots_and_gradients(self) -> None:
        """Feature: Size-one EP transport.
        Description: Dispatch and combine already expert-major slots without collectives.
        Expectation: Counts, values and gradients return to the source slot order.
        """
        context = EPContext(_mesh(1), num_experts=4)
        hidden = torch.randn(5, 3, requires_grad=True)
        expert_ids = torch.tensor([0, 0, 1, 2, 3])
        probabilities = torch.randn(5, requires_grad=True)

        dispatched, counts, weights, send, receive, inverse = context.dispatch(
            hidden,
            expert_ids,
            probabilities,
        )
        restored = context.combine(dispatched * weights.unsqueeze(-1), send, receive, inverse)
        restored.sum().backward()

        self.assertEqual(counts.tolist(), [2, 1, 1, 1])
        self.assertEqual(send, [5])
        self.assertEqual(receive, [5])
        torch.testing.assert_close(restored, hidden.detach() * probabilities.detach().unsqueeze(-1))
        torch.testing.assert_close(hidden.grad, probabilities.detach().unsqueeze(-1).expand_as(hidden))
        torch.testing.assert_close(probabilities.grad, hidden.detach().sum(dim=-1))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_multi_rank_dispatch_groups_received_rows_and_combine_reverses_it(self) -> None:
        """Feature: EP receive-side permutation.
        Description: Emulate rank-zero receiving peer-major rows for two local experts.
        Expectation: Compute rows become local-expert-major and combine restores peer/source order.
        """
        context = EPContext(_mesh(2), num_experts=4)
        hidden = torch.tensor([[10.0], [20.0], [30.0]])
        expert_ids = torch.tensor([0, 2, 3])
        probabilities = torch.tensor([0.1, 0.2, 0.3])
        received_hidden = torch.tensor([[10.0], [41.0], [40.0]])
        received_ids = torch.tensor([[0], [1], [0]])
        received_weights = torch.tensor([[0.1], [0.4], [0.5]])
        returned_values = torch.tensor([[100.0], [200.0], [300.0]])

        with patch.object(context, "_slot_split_sizes", return_value=([1, 2], [1, 2])), patch.object(
            context,
            "_exchange",
            side_effect=(received_hidden, received_ids, received_weights, returned_values),
        ) as exchange:
            dispatched, counts, weights, send, receive, inverse = context.dispatch(
                hidden,
                expert_ids,
                probabilities,
            )
            restored = context.combine(dispatched, send, receive, inverse)

        torch.testing.assert_close(dispatched, torch.tensor([[10.0], [40.0], [41.0]]))
        self.assertEqual(counts.tolist(), [2, 1])
        torch.testing.assert_close(weights, torch.tensor([0.1, 0.5, 0.4]))
        self.assertEqual(inverse.tolist(), [0, 2, 1])
        torch.testing.assert_close(restored, returned_values)
        self.assertEqual(exchange.call_count, 4)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_invalid_expert_ids_fail_before_collectives(self) -> None:
        """Feature: EP ownership validation.
        Description: Pass a global expert ID outside the configured ownership table.
        Expectation: Device-side fixed-size counting rejects it before any exchange.
        """
        context = EPContext(_mesh(2), num_experts=4)
        with patch.object(context, "_exchange", side_effect=AssertionError("collective reached")):
            with self.assertRaises(RuntimeError):
                context.dispatch(torch.randn(1, 3), torch.tensor([4]))



if __name__ == "__main__":
    unittest.main()
