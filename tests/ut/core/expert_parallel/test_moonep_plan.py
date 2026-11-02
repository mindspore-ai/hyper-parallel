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
"""Invariants of the host MoonEP communication plan.

MoonEP's capacity per rank is ``NvS = S * K``, i.e. the *average* share of the
global routing, so a real router always overloads someone and the planner must
move the overflow.  Two properties have to hold for every routing, whether or
not the plan matches the reference implementation's choice of where to move
rows:

* every expert's rows are conserved (``alloc.sum(0) == total``);
* no rank receives more than ``NvS`` rows (``alloc.sum(1) <= NvS``).

``dst`` must encode a legal landing slot for every row, and ``cu_seqlens`` must
describe exactly the rows that rank receives.  A plan that violates either
would send a row to an expert that never computes it.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from hyper_parallel.distributed.expert_parallel import moonep_ep
from hyper_parallel.distributed.expert_parallel.moonep_plan import build_moonep_plan


def _rank_rows(rank_counts) -> torch.Tensor:
    """Return the expert id of every row one rank presents, grouped by expert."""
    return torch.cat([
        torch.full((int(count),), expert, dtype=torch.long)
        for expert, count in enumerate(rank_counts)
    ])


def _plan(counts_all, rank: int, epn: int, nvs: int):
    """Build the plan for one rank, with that rank's own row ids."""
    route_ids = _rank_rows(counts_all[rank])
    local_counts = torch.bincount(route_ids, minlength=counts_all.shape[1])
    return build_moonep_plan(
        counts_all.to(torch.int32), local_counts.to(torch.int32),
        route_ids.to(torch.int32), rank, epn, nvs,
    )


def _counts(counts_per_expert) -> torch.Tensor:
    """Stack per-rank expert counts into the planner's ``[R, E]`` table."""
    return torch.stack([torch.as_tensor(row, dtype=torch.int64) for row in counts_per_expert])


class TestPlanCacheConsensus(unittest.TestCase):
    """Only reuse a cached plan when every EP rank has the same routing."""

    def _cache_decision(self, local_hit: bool, peer_hit: bool) -> bool:
        """Evaluate one rank's vote while simulating the other rank."""
        expected_group = object()

        def all_reduce(hit_count, group=None):
            self.assertIs(group, expected_group)
            hit_count.add_(int(peer_hit))

        with mock.patch.object(moonep_ep.dist, "all_reduce", side_effect=all_reduce):
            return moonep_ep._all_ranks_plan_cache_hit(
                local_hit, torch.device("cpu"), expected_group, world_size=2,
            )

    def test_a_local_hit_is_rejected_when_a_peer_misses(self):
        """A cache hit on only one rank must force all ranks to replan."""
        self.assertFalse(self._cache_decision(local_hit=True, peer_hit=False))

    def test_cache_is_reused_when_every_rank_hits(self):
        """SAC recompute still reuses the plan when the full group matches."""
        self.assertTrue(self._cache_decision(local_hit=True, peer_hit=True))

    def test_a_local_miss_rejects_the_cache_even_when_a_peer_hits(self):
        """A changed local route invalidates the group plan on every rank."""
        self.assertFalse(self._cache_decision(local_hit=False, peer_hit=True))

    def test_partial_group_hit_rebuilds_instead_of_reusing(self):
        """A local cache match must still rebuild after the group vote misses."""
        route_ids = torch.tensor([0, 1], dtype=torch.int32)
        old_state = SimpleNamespace(layout=[])
        new_state = SimpleNamespace(layout=[(0, 0, 2)])
        old_plan, new_plan = object(), object()
        transport = moonep_ep.MoonEPExpertParallel.__new__(
            moonep_ep.MoonEPExpertParallel)
        transport._plan_cache_on = True
        transport._plan_cache = (route_ids.numpy().tobytes(), old_plan, old_state)
        transport.group = object()
        transport.ep_size = 2
        transport.num_experts = 2
        transport.hidden = 2
        transport.rank = 0
        transport._make_plan = mock.Mock(return_value=(new_plan, new_state))

        rows = torch.ones((2, 2), dtype=torch.bfloat16)
        weights = torch.ones(2)
        with mock.patch.dict(os.environ, {
                "HP_MOONEP_ROUTE_WEIGHTS": "0", "HP_MOONEP_GROUPED": "0"}), \
                mock.patch.object(moonep_ep, "_all_ranks_plan_cache_hit",
                                  return_value=False) as hit_vote, \
                mock.patch.object(moonep_ep._MoonEPDispatch, "apply",
                                  side_effect=lambda x, _w, _s: (x, None)), \
                mock.patch.object(moonep_ep._MoonEPCombine, "apply",
                                  side_effect=lambda x, _s: x), \
                mock.patch.object(moonep_ep, "_prof_start", return_value=None), \
                mock.patch.object(moonep_ep, "_prof_mark"), \
                mock.patch.object(moonep_ep, "_prof_report"):
            transport.forward(
                rows, route_ids, weights,
                expert_forward=lambda segment, _expert, _weights: segment,
            )

        hit_vote.assert_called_once()
        transport._make_plan.assert_called_once()
        self.assertIs(transport._plan_cache[1], new_plan)


class TestScratchIsolation(unittest.TestCase):
    """Sharing requires compatible group geometry and device storage."""

    def test_equal_rows_different_widths_do_not_alias(self):
        """Two layers with different hidden widths retain correctly sized rows."""
        group = object()
        with mock.patch.dict(os.environ, {"HP_MOONEP_SHARED_SCRATCH": "1"}), \
                mock.patch.object(moonep_ep.dist, "get_world_size", return_value=2), \
                mock.patch.object(moonep_ep.dist, "get_rank", return_value=0), \
                mock.patch.dict(moonep_ep._SHARED_ROW_SCRATCH, clear=True), \
                mock.patch.dict(moonep_ep._SHARED_ROW_TURN, clear=True):
            outputs = []
            for hidden in (2, 3):
                transport = moonep_ep.MoonEPExpertParallel(group, 4, hidden)
                rows = torch.arange(2 * hidden, dtype=torch.float32).reshape(2, hidden)
                state = SimpleNamespace(plan=SimpleNamespace(N=2), received_rows=2)
                buffer = SimpleNamespace(dispatch=lambda _x, _rows=rows, **_kw: (_rows, None, None, None))
                with mock.patch.object(transport, "_buffer_for", return_value=buffer):
                    output, _ = transport._dispatch_raw(rows, None, state)
                self.assertEqual(tuple(output.shape), (2, hidden))
                self.assertTrue(torch.equal(output, rows))
                outputs.append(output)
            self.assertNotEqual(outputs[0].data_ptr(), outputs[1].data_ptr())


class TestRequiredRouteWeights(unittest.TestCase):
    """Callbacks consuming scores must not be overridden by the M3 flag."""

    def test_required_weights_reach_callback_when_global_flag_is_off(self):
        """The explicit callback contract preserves score tensors and gradients."""
        transport = moonep_ep.MoonEPExpertParallel.__new__(moonep_ep.MoonEPExpertParallel)
        transport._plan_cache_on = False
        transport.hidden = 2
        transport.rank = 0
        transport.num_experts = 2
        state = SimpleNamespace(layout=[(0, 0, 2)])
        transport._make_plan = mock.Mock(return_value=(object(), state))
        rows = torch.ones(2, 2, dtype=torch.bfloat16)
        weights = torch.tensor([0.25, 0.75], requires_grad=True)
        received = []

        def callback(segment, _expert, scores):
            received.append(scores)
            return segment * scores[:, None]

        with mock.patch.dict(os.environ, {"HP_MOONEP_ROUTE_WEIGHTS": "0", "HP_MOONEP_GROUPED": "0"}), \
                mock.patch.object(moonep_ep._MoonEPDispatch, "apply", side_effect=lambda x, w, _s: (x, w)), \
                mock.patch.object(moonep_ep._MoonEPCombine, "apply", side_effect=lambda x, _s: x), \
                mock.patch.object(moonep_ep, "_prof_start", return_value=None), \
                mock.patch.object(moonep_ep, "_prof_mark"), \
                mock.patch.object(moonep_ep, "_prof_report"):
            result = transport.forward(rows, torch.tensor([0, 1]), weights,
                                       expert_forward=callback, requires_route_weights=True)
        result.float().sum().backward()
        self.assertTrue(torch.equal(received[0], weights))
        self.assertTrue(torch.equal(weights.grad, torch.tensor([2.0, 2.0])))


class TestMoonEPPlanInvariants(unittest.TestCase):
    """Row conservation, capacity and a legal slot for every row."""

    def _assert_invariants(self, counts_all, epn: int, nvs: int) -> None:
        """Check every invariant that must hold for any legal plan."""
        world = counts_all.shape[0]
        total = counts_all.sum(dim=0)
        for rank in range(world):
            fields = _plan(counts_all, rank, epn, nvs)
            route_ids = _rank_rows(counts_all[rank])
            alloc = fields["alloc"]
            self.assertEqual(
                alloc.sum(dim=0).tolist(), total.tolist(),
                f"rank {rank}: expert rows are not conserved: "
                f"{alloc.sum(dim=0).tolist()} != {total.tolist()}",
            )
            self.assertLessEqual(
                int(alloc.sum(dim=1).max()), nvs,
                f"rank {rank}: a destination exceeds capacity {nvs}: "
                f"{alloc.sum(dim=1).tolist()}",
            )
            self.assertEqual(
                fields["dst"].numel(), route_ids.numel(),
                f"rank {rank}: one slot per row is required",
            )
            self.assertTrue(
                bool(((fields["dst"] >= 0) & (fields["dst"] < world * nvs)).all()),
                f"rank {rank}: a slot falls outside the buffer: {fields['dst'].tolist()[:8]}",
            )
            self.assertEqual(
                int(fields["cu_seqlens"][-1]), int(alloc[rank].sum()),
                f"rank {rank}: cu_seqlens must end at the rows this rank receives",
            )
            self.assertEqual(
                tuple(fields["experts_to_copy"].shape), (world, epn),
                f"rank {rank}: one prefetch slot per local expert is required",
            )

    def test_balanced_routing_keeps_every_row_on_its_owner(self):
        """When no rank exceeds capacity the plan does not move anything."""
        epn, nvs = 2, 12
        counts_all = _counts([
            [6, 6, 0, 0, 0, 0, 0, 0],
            [0, 0, 6, 6, 0, 0, 0, 0],
            [0, 0, 0, 0, 6, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 6, 6],
        ])

        self._assert_invariants(counts_all, epn, nvs)

        owner = torch.arange(8) // epn
        for rank in range(4):
            route_ids = _rank_rows(counts_all[rank])
            fields = _plan(counts_all, rank, epn, nvs)
            self.assertTrue(
                bool((fields["dst"] // nvs == owner[route_ids]).all()),
                f"rank {rank}: a balanced plan must keep each row on its owner: "
                f"{(fields['dst'] // nvs).tolist()} vs {owner[route_ids].tolist()}",
            )
            self.assertTrue(
                bool((fields["experts_to_copy"] < 0).all()),
                f"rank {rank}: a balanced plan must not prefetch: "
                f"{fields['experts_to_copy'].tolist()}",
            )

    def test_skewed_routing_moves_the_overflow_and_conserves_rows(self):
        """All rows on one owner overflow it; the planner sheds them elsewhere."""
        epn, nvs = 2, 12
        counts_all = _counts([[nvs, 0, 0, 0, 0, 0, 0, 0]] * 4)

        self._assert_invariants(counts_all, epn, nvs)

        fields = _plan(counts_all, rank=0, epn=epn, nvs=nvs)
        alloc = fields["alloc"]
        self.assertEqual(
            int(alloc.sum(dim=0)[0]), nvs * 4,
            f"every row must still be assigned: {alloc.sum(dim=0).tolist()}",
        )
        self.assertLess(
            int(alloc[0, 0]), nvs * 4,
            f"the overloaded owner must shed rows, got {int(alloc[0, 0])}",
        )
        self.assertTrue(
            bool((alloc.sum(dim=1) > 0).all()),
            f"every rank has spare capacity, so all of them take rows: "
            f"{alloc.sum(dim=1).tolist()}",
        )
        self.assertTrue(
            bool((fields["experts_to_copy"][1:] >= 0).any()),
            f"receiving ranks must name the prefetched experts: "
            f"{fields['experts_to_copy'].tolist()}",
        )

    def test_the_plan_is_deterministic(self):
        """The same routing must produce the same plan (checkpointable training)."""
        epn, nvs = 2, 12
        counts_all = _counts([
            [4, 4, 1, 1, 1, 1, 0, 0],
            [5, 3, 1, 1, 1, 1, 0, 0],
            [6, 2, 1, 1, 1, 1, 0, 0],
            [2, 6, 1, 1, 1, 1, 0, 0],
        ])

        self._assert_invariants(counts_all, epn, nvs)

        first = _plan(counts_all, rank=2, epn=epn, nvs=nvs)
        second = _plan(counts_all, rank=2, epn=epn, nvs=nvs)
        for name in ("dst", "cu_seqlens", "experts_to_copy", "alloc"):
            with self.subTest(field=name):
                self.assertTrue(
                    torch.equal(first[name], second[name]),
                    f"{name} changed between identical plans",
                )


class TestMoonEPPlanValidation(unittest.TestCase):
    """Malformed routing inputs fail with a message, not an encoded bad plan."""

    def _build(self, **overrides):
        """Call the planner with a valid 2-rank / 4-expert plan, then patch it."""
        counts_all = _counts([[2, 2, 0, 0], [0, 0, 2, 2]])
        route_ids = _rank_rows(counts_all[0]).to(torch.int32)
        local_counts = torch.bincount(route_ids.long(), minlength=4).to(torch.int32)
        kwargs = {
            "counts_all": counts_all.to(torch.int32),
            "local_counts": local_counts,
            "route_ids": route_ids,
            "rank": 0,
            "epn": 2,
            "nvs": 4,
        }
        kwargs.update(overrides)
        return build_moonep_plan(**kwargs)

    def test_a_row_count_mismatch_is_rejected(self):
        """One id per routed row, exactly the capacity."""
        with self.assertRaisesRegex(ValueError, "one id per routed row"):
            self._build(route_ids=torch.zeros(3, dtype=torch.int32))

    def test_an_expert_count_that_does_not_partition_the_ranks_is_rejected(self):
        """E must equal R * epn, or the layout has no owner for some experts."""
        with self.assertRaisesRegex(ValueError, r"must equal R \* epn"):
            self._build(epn=3)

    def test_an_out_of_range_route_is_rejected(self):
        """A route id past the last expert would be encoded outside the buffer."""
        route_ids = torch.tensor([0, 0, 4, 4], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, r"route_ids must name experts"):
            self._build(route_ids=route_ids)

    def test_local_counts_must_describe_the_presented_rows(self):
        """A count table that disagrees with the rows shifts every slot."""
        with self.assertRaisesRegex(ValueError, "rows but the capacity"):
            self._build(local_counts=torch.tensor([2, 2, 0, 1], dtype=torch.int32))

    def test_remote_stats_count_every_prefetch(self):
        """Two experts lent by one rank are two prefetches, not one.

        The counter is accumulated per destination, and a rank can lend more
        than one expert to the same destination.  A fancy-index ``+=`` keeps
        only one of those updates, which reported half the real prefetch load.
        """
        epn, nvs = 4, 16
        # Two ranks, four experts each.  Rank 0 owns experts 0-3 and receives 24
        # of their rows, so it sheds 8: six rows of expert 0 and two of expert 1
        # land on rank 1 -- two prefetches of the same owner.
        counts_all = _counts([
            [6, 6, 2, 2, 0, 0, 0, 0],
            [0, 0, 4, 4, 4, 4, 0, 0],
        ])

        plans = [_plan(counts_all, rank, epn, nvs) for rank in range(2)]
        prefetches_per_dest = [int(fields["remote_stats"][0]) for fields in plans]
        lent_by_owner = [int(fields["remote_stats"][1]) for fields in plans]

        self.assertEqual(
            prefetches_per_dest, [0, 2],
            f"the spare rank prefetches both hot experts, got {prefetches_per_dest}",
        )
        self.assertEqual(
            lent_by_owner, [2, 0],
            f"rank 0 lends both experts once each, got {lent_by_owner}",
        )

    def test_needed_remote_covers_unprefetched_remote_experts(self):
        """Remote experts left in regular groups must still be in needed_remote.

        R=3, epn=1: rank 2 receives rows of both expert 0 and expert 1, but
        only prefetches one of them -- the other stays in its regular group.
        Weight gathering keyed only on the prefetch picks would miss it.
        """
        epn, nvs = 1, 4
        counts_all = _counts([
            [4, 0, 0],
            [2, 2, 0],
            [0, 4, 0],
        ])

        plans = [_plan(counts_all, rank, epn, nvs) for rank in range(3)]
        needed = set(plans[0]["needed_remote"].tolist())
        for rank, fields in enumerate(plans):
            alloc_col = fields["alloc"][rank]
            eligible = {
                e for e in range(3)
                if int(alloc_col[e]) > 0 and e // epn != rank
            }
            self.assertTrue(
                eligible <= needed,
                f"rank {rank}'s remote experts {eligible} must all be in "
                f"needed_remote {needed}",
            )


if __name__ == "__main__":
    unittest.main()
