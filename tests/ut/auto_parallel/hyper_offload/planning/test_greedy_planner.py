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
"""Unit tests for GreedyResidencyPlanner."""

from __future__ import annotations

import unittest

from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyActionType,
    ResidencySchedule,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
    StorageAccess,
    TraceOp,
)
from hyper_parallel.auto_parallel.hyper_offload.planning.greedy import (
    GreedyResidencyPlanner,
)


def _make_trace(
    ops: list[TraceOp],
    storage_sizes: dict[int, int] | None = None,
    memory_limit_bytes: int | None = None,
    retained_sids: set[int] | None = None,
) -> ActivationTrace:
    """Helper to build an ActivationTrace."""
    return ActivationTrace(
        ops=ops,
        storage_sizes=storage_sizes or {},
        memory_limit_bytes=memory_limit_bytes,
        retained_sids=retained_sids or set(),
    )


class TestGreedyResidencyPlanner(unittest.TestCase):
    """Unit tests for GreedyResidencyPlanner."""

    def setUp(self) -> None:
        self.planner = GreedyResidencyPlanner()

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_trace_returns_empty_schedule(self) -> None:
        """An empty trace (no ops) should produce an empty schedule."""
        trace = _make_trace([])
        sched = self.planner.build(trace)
        self.assertIsInstance(sched, ResidencySchedule)
        self.assertEqual(len(sched.pre), 0)
        self.assertEqual(len(sched.post), 0)

    def test_trace_with_no_storage_sizes_returns_empty_schedule(self) -> None:
        """When no storage sizes are provided, no eviction should be scheduled."""
        trace = _make_trace(
            [TraceOp(name="op", accesses=[StorageAccess(0, 1, AccessKind.WRITE)])],
            memory_limit_bytes=1,
        )
        sched = self.planner.build(trace)
        self.assertEqual(len(sched.pre), 0)
        self.assertEqual(len(sched.post), 0)

    def test_infinite_memory_limit_returns_empty_schedule(self) -> None:
        """With no memory limit (None), no eviction actions should be emitted."""
        trace = _make_trace(
            [TraceOp(name="op", accesses=[StorageAccess(0, 1, AccessKind.WRITE)])],
            storage_sizes={1: 1024},
            memory_limit_bytes=None,
        )
        sched = self.planner.build(trace)
        self.assertEqual(len(sched.pre), 0)

    # ------------------------------------------------------------------
    # Single storage, single access
    # ------------------------------------------------------------------

    def test_single_storage_within_budget_no_eviction(self) -> None:
        """If the single storage fits in budget, no eviction is needed."""
        trace = _make_trace(
            [TraceOp(name="write", accesses=[StorageAccess(0, 1, AccessKind.WRITE)])],
            storage_sizes={1: 1024},
            memory_limit_bytes=2048,
        )
        sched = self.planner.build(trace)
        # No pre or post actions because the single access fits
        self.assertEqual(len(sched.pre), 0)

    def test_single_storage_exceeds_budget_but_no_gap(self) -> None:
        """A single access to storage larger than budget — no gap to evict."""
        trace = _make_trace(
            [TraceOp(name="write", accesses=[StorageAccess(0, 1, AccessKind.WRITE)])],
            storage_sizes={1: 1024},
            memory_limit_bytes=512,
        )
        sched = self.planner.build(trace)
        # No candidate gap exists (only one access) → no pre actions
        self.assertEqual(len(sched.pre), 0)

    # ------------------------------------------------------------------
    # Two accesses with a gap
    # ------------------------------------------------------------------

    def test_evicts_storage_with_gap_when_over_budget(self) -> None:
        """Storage with two accesses spanning ops should be evicted in the gap."""
        trace = _make_trace(
            [
                TraceOp(
                    name="produce",
                    accesses=[StorageAccess(0, 1, AccessKind.WRITE)],
                ),
                TraceOp(
                    name="consume",
                    accesses=[StorageAccess(5, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 1024},
            memory_limit_bytes=512,
        )
        sched = self.planner.build(trace)
        # Should have a COPY_D2H after op 0 and COPY_H2D before op 5
        post_0 = sched.post_actions(0)
        pre_5 = sched.pre_actions(5)

        self.assertTrue(
            any(a.kind == ResidencyActionType.COPY_D2H and a.storage_id == 1 for a in post_0),
            msg="Expected COPY_D2H for sid=1 after op 0",
        )
        self.assertTrue(
            any(a.kind == ResidencyActionType.COPY_H2D and a.storage_id == 1 for a in pre_5),
            msg="Expected COPY_H2D for sid=1 before op 5",
        )

    def test_last_access_gets_release_device(self) -> None:
        """After the last access, RELEASE_DEVICE should be emitted."""
        trace = _make_trace(
            [
                TraceOp(
                    name="produce",
                    accesses=[StorageAccess(0, 1, AccessKind.WRITE)],
                ),
                TraceOp(
                    name="last_read",
                    accesses=[StorageAccess(1, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 1024},
            memory_limit_bytes=512,
        )
        sched = self.planner.build(trace)

        post_1 = sched.post_actions(1)
        self.assertTrue(
            any(
                a.kind == ResidencyActionType.RELEASE_DEVICE and a.storage_id == 1
                for a in post_1
            ),
            msg="Expected RELEASE_DEVICE for sid=1 after last access (op 1)",
        )

    def test_release_host_emitted_when_evicted(self) -> None:
        """An evicted storage should also get RELEASE_HOST after last access."""
        trace = _make_trace(
            [
                TraceOp(
                    name="produce",
                    accesses=[StorageAccess(0, 1, AccessKind.WRITE)],
                ),
                TraceOp(
                    name="last_read",
                    accesses=[StorageAccess(5, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 1024},
            memory_limit_bytes=512,
        )
        sched = self.planner.build(trace)

        post_5 = sched.post_actions(5)
        self.assertTrue(
            any(
                a.kind == ResidencyActionType.RELEASE_HOST and a.storage_id == 1
                for a in post_5
            ),
            msg="Expected RELEASE_HOST for sid=1 after last access (evicted)",
        )

    def test_release_host_not_emitted_when_not_evicted(self) -> None:
        """A storage that was never evicted should not get RELEASE_HOST."""
        trace = _make_trace(
            [
                TraceOp(
                    name="produce",
                    accesses=[StorageAccess(0, 1, AccessKind.WRITE)],
                ),
                TraceOp(
                    name="read",
                    accesses=[StorageAccess(1, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 256},
            memory_limit_bytes=2048,
        )
        sched = self.planner.build(trace)

        # May have RELEASE_DEVICE but should NOT have RELEASE_HOST
        post_1 = sched.post_actions(1)
        release_hosts = [
            a for a in post_1 if a.kind == ResidencyActionType.RELEASE_HOST
        ]
        self.assertEqual(len(release_hosts), 0)

    # ------------------------------------------------------------------
    # Multiple storages — greedy selection
    # ------------------------------------------------------------------

    def test_greedy_selects_longest_gap_first(self) -> None:
        """When multiple storages exceed budget, the planner picks the one with the longest gap."""
        # Storage 1: gap of 5 ops (op 0 → op 6)
        # Storage 2: gap of 2 ops (op 0 → op 3)
        trace = _make_trace(
            [
                TraceOp(
                    name="produce_both",
                    accesses=[
                        StorageAccess(0, 1, AccessKind.WRITE),
                        StorageAccess(0, 2, AccessKind.WRITE),
                    ],
                ),
                TraceOp(name="no_access"),
                TraceOp(name="no_access"),
                TraceOp(
                    name="read_sid2",
                    accesses=[StorageAccess(3, 2, AccessKind.READ)],
                ),
                TraceOp(name="no_access"),
                TraceOp(name="no_access"),
                TraceOp(
                    name="read_sid1",
                    accesses=[StorageAccess(6, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 1024, 2: 1024},
            memory_limit_bytes=1024,  # only one storage can stay
        )
        sched = self.planner.build(trace)

        # sid=1 has longer gap (0→6 vs 0→3) so should be evicted.
        post_0 = sched.post_actions(0)
        evicted_sid1 = any(
            a.kind == ResidencyActionType.COPY_D2H and a.storage_id == 1
            for a in post_0
        )
        evicted_sid2 = any(
            a.kind == ResidencyActionType.COPY_D2H and a.storage_id == 2
            for a in post_0
        )
        self.assertTrue(
            evicted_sid1,
            msg="Expected sid=1 (longer gap) to be evicted",
        )
        self.assertFalse(
            evicted_sid2,
            msg="Expected sid=2 (shorter gap) NOT to be evicted",
        )

    def test_stops_evicting_once_budget_met(self) -> None:
        """Once the budget is satisfied, remaining candidates are skipped."""
        # Three storages, each 1024 bytes. Budget = 2048 → need to evict only one.
        trace = _make_trace(
            [
                TraceOp(
                    name="produce",
                    accesses=[
                        StorageAccess(0, 1, AccessKind.WRITE),
                        StorageAccess(0, 2, AccessKind.WRITE),
                        StorageAccess(0, 3, AccessKind.WRITE),
                    ],
                ),
                TraceOp(name="read1", accesses=[StorageAccess(10, 1, AccessKind.READ)]),
                TraceOp(name="read2", accesses=[StorageAccess(10, 2, AccessKind.READ)]),
                TraceOp(name="read3", accesses=[StorageAccess(10, 3, AccessKind.READ)]),
            ],
            storage_sizes={1: 1024, 2: 1024, 3: 1024},
            memory_limit_bytes=2048,
        )
        sched = self.planner.build(trace)

        # At most 1 storage should be evicted (evicting one frees 1024, resident=2048=budget)
        copy_d2h_count = sum(
            1
            for actions in sched.post.values()
            for a in actions
            if a.kind == ResidencyActionType.COPY_D2H
        )
        self.assertLessEqual(copy_d2h_count, 1)

    # ------------------------------------------------------------------
    # retained_sids behaviour
    # ------------------------------------------------------------------

    def test_retained_sids_prevent_release(self) -> None:
        """Storages in retained_sids should not get RELEASE_DEVICE or RELEASE_HOST."""
        trace = _make_trace(
            [
                TraceOp(
                    name="produce",
                    accesses=[StorageAccess(0, 1, AccessKind.WRITE)],
                ),
                TraceOp(
                    name="last_read",
                    accesses=[StorageAccess(1, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 1024},
            memory_limit_bytes=512,
            retained_sids={1},
        )
        sched = self.planner.build(trace)

        # sid=1 is retained → no RELEASE_DEVICE or RELEASE_HOST
        for actions in sched.post.values():
            for a in actions:
                self.assertNotEqual(
                    a.kind,
                    ResidencyActionType.RELEASE_DEVICE,
                    msg="retained sid should not get RELEASE_DEVICE",
                )
                self.assertNotEqual(
                    a.kind,
                    ResidencyActionType.RELEASE_HOST,
                    msg="retained sid should not get RELEASE_HOST",
                )

    # ------------------------------------------------------------------
    # D2H copy-start fine-tuning
    # ------------------------------------------------------------------

    def test_copy_d2h_moves_before_read_only_accesses(self) -> None:
        """D2H can be scheduled at the earliest safe point (after last write)."""
        trace = _make_trace(
            [
                TraceOp(
                    name="write",
                    accesses=[StorageAccess(0, 1, AccessKind.WRITE)],
                ),
                TraceOp(
                    name="read1",
                    accesses=[StorageAccess(1, 1, AccessKind.READ)],
                ),
                TraceOp(
                    name="read2",
                    accesses=[StorageAccess(4, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 1024},
            memory_limit_bytes=512,
        )
        sched = self.planner.build(trace)

        # COPY_D2H should be after the first (write) access, not after read1
        post_0 = sched.post_actions(0)
        self.assertTrue(
            any(a.kind == ResidencyActionType.COPY_D2H and a.storage_id == 1 for a in post_0),
            msg="COPY_D2H should be scheduled right after the write op",
        )

    # ------------------------------------------------------------------
    # Plan with zero budget
    # ------------------------------------------------------------------

    def test_zero_budget_evicts_aggressively(self) -> None:
        """With memory_limit_bytes=0, the planner should evict as much as possible."""
        trace = _make_trace(
            [
                TraceOp(
                    name="write",
                    accesses=[StorageAccess(0, 1, AccessKind.WRITE)],
                ),
                TraceOp(
                    name="read",
                    accesses=[StorageAccess(2, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 1024},
            memory_limit_bytes=0,
        )
        sched = self.planner.build(trace)

        post_0 = sched.post_actions(0)
        self.assertTrue(
            any(a.kind == ResidencyActionType.COPY_D2H for a in post_0),
            msg="Expected eviction with zero budget",
        )

    # ------------------------------------------------------------------
    # Multiple storages with shared op
    # ------------------------------------------------------------------

    def test_multiple_storages_in_one_op(self) -> None:
        """Multiple storages accessed in the same op should all be tracked."""
        trace = _make_trace(
            [
                TraceOp(
                    name="big_op",
                    accesses=[
                        StorageAccess(0, 1, AccessKind.WRITE),
                        StorageAccess(0, 2, AccessKind.WRITE),
                        StorageAccess(0, 3, AccessKind.READ),
                    ],
                ),
                TraceOp(
                    name="later",
                    accesses=[
                        StorageAccess(5, 1, AccessKind.READ),
                        StorageAccess(5, 2, AccessKind.READ),
                    ],
                ),
            ],
            storage_sizes={1: 512, 2: 512, 3: 512},
            memory_limit_bytes=768,
        )
        sched = self.planner.build(trace)
        # At least one storage should be evicted
        copy_actions = [
            a
            for actions in sched.post.values()
            for a in actions
            if a.kind == ResidencyActionType.COPY_D2H
        ]
        self.assertGreaterEqual(len(copy_actions), 1)

    # ------------------------------------------------------------------
    # Release actions for non-evicted storages
    # ------------------------------------------------------------------

    def test_non_evicted_storage_still_gets_release_device(self) -> None:
        """A storage that fits within budget should still have device released after last use."""
        trace = _make_trace(
            [
                TraceOp(
                    name="write",
                    accesses=[StorageAccess(0, 1, AccessKind.WRITE)],
                ),
                TraceOp(
                    name="last_read",
                    accesses=[StorageAccess(1, 1, AccessKind.READ)],
                ),
            ],
            storage_sizes={1: 256},
            memory_limit_bytes=512,
        )
        sched = self.planner.build(trace)

        post_1 = sched.post_actions(1)
        self.assertTrue(
            any(
                a.kind == ResidencyActionType.RELEASE_DEVICE and a.storage_id == 1
                for a in post_1
            ),
            msg="Expected RELEASE_DEVICE after last use even when not evicted",
        )
