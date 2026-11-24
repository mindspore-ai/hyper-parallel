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
"""CPU-only tests for capacity-bounded collective MoonEP planning."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hyper_parallel.distributed.expert_parallel.moonep_context import MoonEPContext
from hyper_parallel.distributed.expert_parallel.moonep_planner import compile_plan, select_allocation
from tests.common.mark_utils import arg_mark


class TestMoonEPContext(unittest.TestCase):
    """Check conservation, capacity and per-invocation inverse mappings without distributed setup."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_combine_restores_peer_and_source_order_with_shared_work(self) -> None:
        """Feature: MoonEP reverse transport.
        Description: Reuse EP combine with distinct receiver and source inverses and an async shared phase.
        Expectation: Values and gradients match both inverse permutations, and shared FC2 executes once.
        """
        mesh = SimpleNamespace(ndim=1, size=lambda: 2, get_local_rank=lambda: 0)
        context = MoonEPContext(mesh, num_experts=4)
        values = torch.tensor([[2., 3.], [5., 7.], [11., 13.]], requires_grad=True)
        reference = values.detach().clone().requires_grad_()
        receiver_inverse = torch.tensor([0, 2, 1], dtype=torch.int32)
        plan = {"token_send": [1, 2], "token_recv": [2, 1], "source_inverse": torch.tensor([2, 0, 1])}
        shared = []
        with patch.object(context, "_exchange", side_effect=lambda packet, *_args, **_kwargs: packet) as exchange, \
                patch.object(context, "_overlap_ready", return_value=True), \
                patch.object(context, "_run_overlap", side_effect=lambda *_args: shared.append(values.square().sum())):
            actual = context.combine(values, plan, receiver_inverse)
        peer_major = reference.index_select(0, receiver_inverse.long())
        expected = peer_major.index_select(0, plan["source_inverse"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(shared), 1)
        self.assertEqual(exchange.call_args.args[1:], ([2, 1], [1, 2]))
        self.assertEqual(exchange.call_args.kwargs, {"async_op": True})
        upstream = torch.arange(6).reshape(3, 2)
        (actual.mul(upstream).sum() + shared[0]).backward()
        (expected.mul(upstream).sum() + reference.square().sum()).backward()
        torch.testing.assert_close(values.grad, reference.grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_allocation_conserves_routes_and_bounds_replicas(self) -> None:
        """Feature: Dynamic redundant experts.
        Description: Balance skewed, empty and uneven histograms over different EP sizes.
        Expectation: Counts are conserved, rank loads differ by at most one and default capacity suffices.
        """
        rng = np.random.default_rng(59)
        for ranks in (1, 2, 4, 8):
            for local in (1, 2, 3):
                experts = ranks * local
                cases = [np.zeros((ranks, experts), dtype=np.int64),
                         rng.integers(0, 50, size=(ranks, experts))]
                skew = np.zeros((ranks, experts), dtype=np.int64)
                skew[:, 0] = 100
                cases.append(skew)
                for counts in cases:
                    with self.subTest(ranks=ranks, local=local, total=counts.sum()):
                        allocation = select_allocation(counts)
                        np.testing.assert_array_equal(allocation.sum(1), counts.sum(0))
                        self.assertGreaterEqual(allocation.min(), 0)
                        loads = allocation.sum(0)
                        self.assertLessEqual(loads.max() - loads.min(), 1)
                        remote = np.arange(experts)[:, None] // local != np.arange(ranks)
                        self.assertTrue(np.all(((allocation > 0) & remote).sum(0) <= local))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_zero_capacity_and_invalid_configuration(self) -> None:
        """Feature: Replica capacity.
        Description: Plan balanced and skewed loads with no available replicas.
        Expectation: Balanced loads succeed; infeasible capacity and invalid configuration fail explicitly.
        """
        balanced = np.ones((2, 4), dtype=np.int64)
        np.testing.assert_array_equal(select_allocation(balanced, 0).sum(0), [4, 4])
        with self.assertRaisesRegex(RuntimeError, "replica capacity 0"):
            select_allocation(np.array([[5, 0, 0, 0], [5, 0, 0, 0]]), 0)
        mesh = SimpleNamespace(ndim=1, size=lambda: 2, get_local_rank=lambda: 0)
        for value in (-1, True, .5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "capacity"):
                MoonEPContext(mesh, num_experts=4, replica_capacity=value)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_plans_route_to_matching_logical_experts_and_reverse(self) -> None:
        """Feature: Invocation-local MoonEP transport.
        Description: Simulate all source and receiver packets for a skewed, uneven histogram.
        Expectation: Compute rows retain logical identity and reverse transport restores every source slot.
        """
        counts = np.array([[4, 2, 0, 1], [3, 0, 1, 0]])
        allocation = select_allocation(counts)
        ids = [torch.arange(4).repeat_interleave(torch.from_numpy(row)) for row in counts]
        plans = [compile_plan(counts, allocation, row, rank) for rank, row in enumerate(ids)]
        packets = [row[plan["source_order"]].split(plan["token_send"])
                   for row, plan in zip(ids, plans)]
        returned = []
        for destination, plan in enumerate(plans):
            received = torch.cat([packet[destination] for packet in packets])
            compute_ids = received[plan["receive_order"]]
            active = list(range(destination * 2, destination * 2 + 2))
            active += [e for e in range(4) if e // 2 != destination and allocation[e, destination] > 0]
            expected = torch.tensor(active).repeat_interleave(plan["counts"])
            self.assertTrue(torch.equal(compute_ids, expected))
            inverse = torch.argsort(plan["receive_order"])
            returned.append(compute_ids[inverse].split(plan["token_recv"]))
        for rank, plan in enumerate(plans):
            received = torch.cat([packet[rank] for packet in returned])
            self.assertTrue(torch.equal(received[plan["source_inverse"].long()], ids[rank]))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replica_weight_exchange_preserves_home_autograd(self) -> None:
        """Feature: Temporary expert weights.
        Description: Emulate one requested home row copied to two remote compute rows.
        Expectation: Gradients accumulate onto the original home leaves without registering any replicas.
        """
        mesh = SimpleNamespace(ndim=1, size=lambda: 2, get_local_rank=lambda: 0, get_group=lambda: None)
        context = MoonEPContext(mesh, num_experts=4)
        gate = torch.randn(2, 6, 4, requires_grad=True)
        down = torch.randn(2, 4, 3, requires_grad=True)
        plan = {"weight_exchange": True, "send_experts": torch.tensor([0, 0]),
                "weight_send": [1, 1], "weight_recv": [1, 1]}
        with patch.object(context, "_exchange", side_effect=lambda packet, *_args: packet):
            first, second = context.expert_weights(gate, down, plan)
        (first.sum() + second.sum()).backward()
        torch.testing.assert_close(gate.grad[0], torch.full_like(gate[0], 3), rtol=0, atol=0)
        torch.testing.assert_close(down.grad[0], torch.full_like(down[0], 3), rtol=0, atol=0)
        torch.testing.assert_close(gate.grad[1], torch.ones_like(gate[1]), rtol=0, atol=0)
        self.assertEqual(first.shape[0], 4)
