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
"""Peak selection and forward-copy/backward-prefetch scheduling contracts."""

from __future__ import annotations

import copy
import random
import unittest
from unittest.mock import patch

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyAction,
    ResidencySchedule,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyActionType as Action,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
    StorageAccess,
    TraceOp,
)
from hyper_parallel.auto_parallel.hyper_offload.planning import (
    GreedyResidencyPlanner,
    PeakResidencyPlanner,
)
from hyper_parallel.auto_parallel.hyper_offload.planning._peak_model import (
    EvictionGap,
    build_profile,
)
from hyper_parallel.auto_parallel.hyper_offload.planning.communication_mask import (
    limit_prefetch_advance,
    mask_communications,
)
from hyper_parallel.auto_parallel.hyper_offload.planning.peak import ScoreWeights, _choose_gap
from tests.common.mark_utils import arg_mark


def make_trace(
    rows: list, sizes: dict, limit: int | None, retained: set | None = None
) -> ActivationTrace:
    """Create a trace with the same contiguous operator ids as warmup."""
    return ActivationTrace(
        ops=[
            TraceOp(
                f"op{op}", accesses=[StorageAccess(op, sid, kind) for sid, kind in row]
            )
            for op, row in enumerate(rows)
        ],
        storage_sizes=sizes,
        retained_sids=retained or set(),
        memory_limit_bytes=limit,
    )


def positions(schedule: ResidencySchedule, kind: Action, sid: int) -> list[int]:
    """Extract action positions without relying on internal planner state."""
    return sorted(
        action.op_id
        for phase in (schedule.pre, schedule.post)
        for actions in phase.values()
        for action in actions
        if action.kind == kind and action.storage_id == sid
    )


def simulate(
    test: unittest.TestCase, trace: ActivationTrace, schedule: ResidencySchedule, *, complete_copies: bool = False
) -> list[int]:
    """Replay emitted actions against a versioned physical-storage state machine."""
    versions, device, host, resident = {}, {}, {}, []
    retired_early = set()

    def apply(action: ResidencyAction) -> None:
        """Apply one scheduled physical transition and verify its dependencies."""
        sid = action.storage_id
        if action.kind == Action.COPY_D2H:
            test.assertIn(sid, device, "D2H needs a device source")
            test.assertNotIn(sid, host, "D2H must not silently reuse an old snapshot")
            host[sid] = device[sid]
        elif action.kind == Action.COPY_H2D:
            test.assertNotIn(sid, device, "H2D must restore an evicted storage")
            test.assertIn(sid, host, "H2D needs a host source")
            device[sid] = host[sid]
        elif action.kind == Action.RELEASE_HOST:
            test.assertIn(sid, host)
            test.assertIn(sid, device)
            del host[sid]
        elif action.kind == Action.RELEASE_DEVICE:
            if sid in retired_early:
                retired_early.remove(sid)
                test.assertNotIn(sid, device)
                return
            test.assertIn(sid, device, "release must have a live device allocation")
            del device[sid]
        elif action.kind == Action.TRY_RELEASE_DEVICE:
            test.assertIn(sid, host)
            if complete_copies and sid in device:
                del device[sid]
                retired_early.add(sid)

    for op_id, op in enumerate(trace.ops):
        for action in schedule.pre_actions(op_id):
            apply(action)
        by_storage = {}
        for access in op.accesses:
            by_storage.setdefault(access.storage_id, set()).add(access.kind)
        for sid, kinds in by_storage.items():
            if sid in versions:
                test.assertIn(sid, device, "consumer must not demand-page")
                test.assertEqual(
                    device[sid], versions[sid], "consumer must see the current version"
                )
            else:
                versions[sid] = device[sid] = 0
            if AccessKind.WRITE in kinds:
                test.assertNotIn(
                    sid, host, "an outstanding D2H must not race a mutation"
                )
                versions[sid] += 1
                device[sid] = versions[sid]
        resident.append(sum(trace.storage_sizes[sid] for sid in device))
        for action in schedule.post_actions(op_id):
            apply(action)
    test.assertFalse(host, "all snapshots must be retired by the end of replay")
    test.assertEqual(set(device), trace.retained_sids)
    return resident


class TestPeakResidencyPlanner(unittest.TestCase):
    """Validate selection separately from communication overlap and replay."""

    @staticmethod
    def forward_backward_trace() -> ActivationTrace:
        """Two saved activations cross a forward peak before backward consumers."""
        rows = [
            [(1, AccessKind.WRITE)],
            [(2, AccessKind.WRITE)],
            [],
            [],
            [],
            [(3, AccessKind.WRITE)],
            [(3, AccessKind.READ)],
            [],
            [(1, AccessKind.READ)],
            [(2, AccessKind.READ)],
        ]
        return make_trace(rows, {1: 60, 2: 60, 3: 130}, 180)

    def test_prefetch_uses_headroom_before_delaying_source_release(self):
        resident = [100, 100, 40, 40, 40, 40, 40, 40, 100]
        gap = EvictionGap(1, 60, 1, 8, 0, 0, 8)

        placement = mask_communications([gap], resident, 100)[0]

        self.assertEqual(placement.prefetch_before, 2)
        self.assertEqual(placement.release_after, 1)
        self.assertLessEqual(max(resident), 100)

    def test_default_score_favors_bytes_over_tiny_long_lived_candidates(self):
        """Cover the peak with a large interval before scheduling many tiny copies."""
        small = EvictionGap(1, 1, 0, 20, 0, 0, 20)
        large = EvictionGap(2, 100, 8, 12, 8, 8, 12)

        chosen = _choose_gap([small, large], 10, [], ScoreWeights())

        self.assertEqual(chosen, large, f"chosen={chosen}, expected={large}")

    def test_timed_prefetch_reduces_idle_destinations_and_keeps_transfer_lead(self):
        """Avoid holding a restored buffer throughout a long idle backward interval."""
        resident = [100, 100] + [40] * 97 + [100]
        gap = EvictionGap(1, 60, 1, 99, 0, 0, 99)
        placements = mask_communications([gap], resident, 100)
        before = list(resident)
        bandwidth = 60 * 1000 / (2 * 1024**3)

        limit_prefetch_advance(placements, resident, [1.0] * 100, bandwidth)

        restore = placements[0].prefetch_before
        self.assertGreaterEqual(restore, 90, f"restore={restore}, earliest_use=99")
        self.assertLessEqual(restore, 97, f"restore={restore}, minimum_copy_lead=2")
        self.assertEqual(resident[10], 40, f"resident={resident[10]}, expected=40")
        self.assertTrue(all(new <= old for new, old in zip(resident, before)))

    def test_timed_prefetch_accounts_for_serial_copy_queue(self):
        """Overlapping consumers need lead for all copies on the shared stream."""
        resident = [120, 120] + [0] * 97 + [120]
        gaps = [EvictionGap(sid, 60, 1, 99, 0, 0, 99) for sid in (1, 2)]
        placements = mask_communications(gaps, resident, 120)
        bandwidth = 60 * 1000 / (2 * 1024**3)

        limit_prefetch_advance(placements, resident, [1.0] * 100, bandwidth)

        lead = sorted(99 - row.prefetch_before for row in placements)
        self.assertGreaterEqual(lead[1] - lead[0], 8, f"lead={lead}, minimum_spacing=8")
        self.assertLessEqual(max(resident), 120, f"peak={max(resident)}, budget=120")

    def test_saved_gap_preference_preserves_budget_and_general_fallback(self):
        """Prefer a saved activation while retaining other gaps when memory requires them."""
        rows = [
            [(1, AccessKind.WRITE)], [(2, AccessKind.WRITE)], [],
            [(3, AccessKind.WRITE)], [(3, AccessKind.READ)], [], [], [],
            [(1, AccessKind.READ)], [(2, AccessKind.READ)],
        ]
        for budget, expected in ((200, {1}), (150, {1, 2})):
            with self.subTest(budget=budget):
                trace = make_trace(rows, {1: 60, 2: 100, 3: 100}, budget)
                trace.preferred_eviction_gaps = [(1, 0, 8)]
                schedule = PeakResidencyPlanner().build(trace)
                selected = {action.storage_id for actions in schedule.post.values()
                            for action in actions if action.kind == Action.COPY_D2H}
                self.assertEqual(selected, expected, f"selected={selected}, expected={expected}")
                peak = max(simulate(self, trace, schedule))
                self.assertLessEqual(peak, budget, f"peak={peak}, budget={budget}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_framework_workspace_reserve_uses_additional_native_intervals(self):
        """Feature: Framework workspace reserve.

        Description: Plan native saved intervals with and without framework workspace metadata.
        Expectation: Extra offload leaves workspace headroom without modifying the trace budget.
        """
        trace = self.forward_backward_trace()
        trace.memory_limit_bytes = 250
        trace.preferred_eviction_gaps = [(1, 0, 8), (2, 1, 9)]
        planner = PeakResidencyPlanner()
        baseline = planner.build(trace)
        self.assertEqual(positions(baseline, Action.COPY_D2H, 1), [])
        self.assertEqual(positions(baseline, Action.COPY_D2H, 2), [])
        trace.framework_allocation_bytes = 40
        schedule = planner.build(trace)
        self.assertEqual(len(positions(schedule, Action.COPY_D2H, 1)), 1)
        self.assertEqual(len(positions(schedule, Action.COPY_D2H, 2)), 1)
        self.assertLessEqual(max(simulate(self, trace, schedule)), 170)
        self.assertEqual(trace.memory_limit_bytes, 250)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_workspace_slack_does_not_force_non_native_extra_intervals(self):
        """Feature: Native replay eligibility.

        Description: Provide workspace metadata with only one eligible native saved interval.
        Expectation: Extra offload preserves native replay and the original activation budget.
        """
        trace = self.forward_backward_trace()
        trace.memory_limit_bytes = 250
        trace.framework_allocation_bytes = 40
        trace.preferred_eviction_gaps = [(1, 0, 8)]
        schedule = PeakResidencyPlanner().build(trace)
        self.assertEqual(len(positions(schedule, Action.COPY_D2H, 1)), 1)
        self.assertEqual(positions(schedule, Action.COPY_D2H, 2), [])
        self.assertLessEqual(max(simulate(self, trace, schedule)), trace.memory_limit_bytes)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_invalid_framework_workspace_is_rejected(self):
        """Feature: Framework workspace validation.

        Description: Supply negative, boolean and non-integer workspace metadata.
        Expectation: Reject each invalid value before returning a schedule.
        """
        for value in (-1, True, 1.5):
            with self.subTest(value=value):
                trace = self.forward_backward_trace()
                trace.framework_allocation_bytes = value
                with self.assertRaisesRegex(ValueError, "framework_allocation_bytes"):
                    PeakResidencyPlanner().build(trace)

    def test_forward_delayed_release_and_backward_prefetch(self):
        trace = self.forward_backward_trace()
        schedule = PeakResidencyPlanner().build(trace)
        for sid, produced, consumed in ((1, 0, 8), (2, 1, 9)):
            self.assertEqual(positions(schedule, Action.COPY_D2H, sid), [produced])
            self.assertEqual(
                positions(schedule, Action.RELEASE_DEVICE, sid), [4, consumed]
            )
            self.assertEqual(positions(schedule, Action.COPY_H2D, sid), [7])
            self.assertEqual(positions(schedule, Action.RELEASE_HOST, sid), [7])
            self.assertLess(produced, 4, "forward D2H must precede delayed release")
            self.assertLess(7, consumed, "backward H2D must precede its consumer")
        self.assertLessEqual(max(simulate(self, trace, schedule)), 180)

    def test_measured_copy_window_checks_before_mandatory_memory_deadline(self):
        trace = self.forward_backward_trace()
        for op in trace.ops:
            op.duration_ms = 1.0
        trace.d2h_bandwidth_gbps = 60 * 1000 / (2 * 1024**3)
        schedule = PeakResidencyPlanner().build(trace)
        self.assertEqual(positions(schedule, Action.TRY_RELEASE_DEVICE, 1), [2])
        self.assertEqual(positions(schedule, Action.TRY_RELEASE_DEVICE, 2), [])
        self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 1), [4, 8])
        self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 2), [4, 9])
        for sid in (1, 2):
            self.assertEqual(positions(schedule, Action.COPY_H2D, sid), [sid + 6])
        self.assertLessEqual(max(simulate(self, trace, schedule)), 180)
        self.assertLessEqual(max(simulate(self, trace, schedule, complete_copies=True)), 180)

    def test_optional_checks_do_not_require_elidable_replay_operators(self):
        for name in ("detach.default", "framework_read:fsdp"):
            with self.subTest(name=name):
                trace = self.forward_backward_trace()
                for op in trace.ops:
                    op.duration_ms = 1.0
                trace.ops[2].name = name
                trace.d2h_bandwidth_gbps = 60 * 1000 / (2 * 1024**3)
                schedule = PeakResidencyPlanner().build(trace)
                self.assertEqual(positions(schedule, Action.TRY_RELEASE_DEVICE, 1), [])
                self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 1), [4, 8])

    def test_slow_copy_window_keeps_memory_deadline(self):
        trace = self.forward_backward_trace()
        for op in trace.ops:
            op.duration_ms = 1.0
        trace.d2h_bandwidth_gbps = 60 * 1000 / (100 * 1024**3)
        schedule = PeakResidencyPlanner().build(trace)
        for sid, consumed in ((1, 8), (2, 9)):
            self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, sid), [4, consumed])
        self.assertLessEqual(max(simulate(self, trace, schedule)), 180)

    def test_memory_fence_uses_prior_native_boundary_without_crossing_last_use(self):
        trace = self.forward_backward_trace()
        for index in (3, 4):
            trace.ops[index].name = "detach.default"
        schedule = PeakResidencyPlanner().build(trace)
        for sid in (1, 2):
            self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, sid)[0], 2)
        self.assertLessEqual(max(simulate(self, trace, schedule)), 180)

        trace.ops[2].name = "framework_read:fsdp"
        trace.ops[1].name = "detach.default"
        schedule = PeakResidencyPlanner().build(trace)
        self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 1)[0], 0)
        self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 2)[0], 1)
        self.assertLessEqual(max(simulate(self, trace, schedule)), 180)

    def test_unusable_transfer_measurements_keep_original_mask(self):
        for bandwidth in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(bandwidth=bandwidth):
                trace = self.forward_backward_trace()
                for op in trace.ops:
                    op.duration_ms = 1.0
                trace.d2h_bandwidth_gbps = bandwidth
                schedule = PeakResidencyPlanner().build(trace)
                self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 1), [4, 8])
                self.assertLessEqual(max(simulate(self, trace, schedule)), 180)

    def test_invalid_operator_timing_keeps_original_mask(self):
        for duration in (-1.0, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                trace = self.forward_backward_trace()
                for op in trace.ops:
                    op.duration_ms = 1.0
                trace.ops[3].duration_ms = duration
                schedule = PeakResidencyPlanner().build(trace)
                self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 1), [4, 8])
                self.assertLessEqual(max(simulate(self, trace, schedule)), 180)

    def test_mask_toggle_preserves_selection(self):
        trace = self.forward_backward_trace()
        masked = PeakResidencyPlanner().build(trace)
        immediate = PeakResidencyPlanner(enable_communication_masking=False).build(
            trace
        )
        for sid, produced, consumed in ((1, 0, 8), (2, 1, 9)):
            self.assertEqual(positions(immediate, Action.COPY_D2H, sid), [produced])
            self.assertEqual(
                positions(immediate, Action.RELEASE_DEVICE, sid), [produced, consumed]
            )
            self.assertEqual(positions(immediate, Action.COPY_H2D, sid), [consumed])
        self.assertLessEqual(
            max(simulate(self, trace, masked)), trace.memory_limit_bytes
        )
        self.assertLessEqual(
            max(simulate(self, trace, immediate)), trace.memory_limit_bytes
        )

    def test_selects_peak_covering_gap(self):
        trace = make_trace(
            [[(1, AccessKind.WRITE)], [(2, AccessKind.WRITE)], [(1, AccessKind.READ)]],
            {1: 100, 2: 150},
            200,
        )
        schedule = PeakResidencyPlanner().build(trace)
        self.assertEqual(positions(schedule, Action.COPY_D2H, 1), [0])
        self.assertEqual(max(simulate(self, trace, schedule)), 150)

    def test_unavoidable_peak_does_not_hide_later_reducible_peak(self):
        trace = make_trace(
            [
                [(9, AccessKind.WRITE)],
                [(1, AccessKind.WRITE)],
                [(2, AccessKind.WRITE)],
                [(1, AccessKind.READ)],
            ],
            {9: 1000, 1: 100, 2: 150},
            200,
        )
        with self.assertLogs(
            "hyper_parallel.auto_parallel.hyper_offload.planning.peak", level="WARNING"
        ):
            schedule = PeakResidencyPlanner().build(trace)
        resident = simulate(self, trace, schedule)
        self.assertEqual(resident[0], 1000)
        self.assertLessEqual(max(resident[1:]), 200)
        self.assertEqual(positions(schedule, Action.COPY_D2H, 1), [1])

    def test_multiple_gaps_refresh_snapshot_after_write(self):
        trace = make_trace(
            [
                [(1, AccessKind.WRITE)],
                [],
                [(1, AccessKind.READ), (1, AccessKind.WRITE)],
                [],
                [(1, AccessKind.READ)],
                [],
                [(1, AccessKind.READ)],
            ],
            {1: 8},
            0,
        )
        for masking in (False, True):
            with self.subTest(masking=masking):
                schedule = PeakResidencyPlanner(
                    enable_communication_masking=masking
                ).build(trace)
                self.assertEqual(positions(schedule, Action.COPY_D2H, 1), [0, 2, 4])
                self.assertEqual(simulate(self, trace, schedule), [8, 0, 8, 0, 8, 0, 8])

    def test_read_chain_allows_early_d2h_but_not_early_release(self):
        trace = make_trace(
            [
                [(1, AccessKind.WRITE)],
                [(1, AccessKind.READ)],
                [],
                [(2, AccessKind.WRITE)],
                [],
                [(1, AccessKind.READ)],
            ],
            {1: 8, 2: 16},
            16,
        )
        schedule = PeakResidencyPlanner().build(trace)
        self.assertEqual(positions(schedule, Action.COPY_D2H, 1), [0])
        self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 1), [2, 5])
        self.assertEqual(positions(schedule, Action.COPY_H2D, 1), [4])
        self.assertLessEqual(max(simulate(self, trace, schedule)), 16)

    def test_retained_storage_counts_through_trace_end_and_is_not_evicted(self):
        trace = make_trace(
            [
                [(1, AccessKind.WRITE)],
                [],
                [(1, AccessKind.READ)],
                [(2, AccessKind.WRITE)],
            ],
            {1: 8, 2: 8},
            8,
            {1},
        )
        self.assertEqual(build_profile(trace).resident_bytes, [8, 8, 8, 16])
        schedule = PeakResidencyPlanner().build(trace)
        self.assertEqual(positions(schedule, Action.COPY_D2H, 1), [])
        self.assertEqual(positions(schedule, Action.RELEASE_DEVICE, 1), [])
        simulate(self, trace, schedule)

    def test_empty_unlimited_and_sufficient_budget(self):
        self.assertEqual(PeakResidencyPlanner().build(ActivationTrace()).pre, {})
        trace = make_trace(
            [[(1, AccessKind.WRITE)], [], [(1, AccessKind.READ)]], {1: 8}, None
        )
        self.assertEqual(PeakResidencyPlanner().build(trace).post, {})
        trace.memory_limit_bytes = 8
        schedule = PeakResidencyPlanner().build(trace)
        self.assertEqual(positions(schedule, Action.COPY_D2H, 1), [])
        self.assertEqual(simulate(self, trace, schedule), [8, 8, 8])

    def test_invalid_input(self):
        for budget in (-1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                PeakResidencyPlanner().build(make_trace([], {}, budget))
        for size in (-1, None, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                PeakResidencyPlanner().build(
                    make_trace([[(1, AccessKind.WRITE)]], {1: size}, 0)
                )
        with self.assertRaises(ValueError):
            PeakResidencyPlanner(weights=ScoreWeights(size=float("nan")))
        with self.assertRaises(TypeError):
            PeakResidencyPlanner(enable_communication_masking=1)
        trace = make_trace([[(1, AccessKind.WRITE)]], {1: 8}, 0)
        trace.ops[0].accesses[0].op_id = 1
        with self.assertRaises(ValueError):
            PeakResidencyPlanner().build(trace)

    def test_default_session_and_explicit_planner(self):
        class FalseyPlanner(GreedyResidencyPlanner):
            def __bool__(self) -> bool:
                """Exercise explicit planner selection independently of truthiness."""
                return False

        with patch(
            "hyper_parallel.auto_parallel.hyper_offload.api.session.WarmupExecutor"
        ):
            session = OffloadSession(OffloadConfig())
            self.assertIsInstance(session.planner, PeakResidencyPlanner)
            self.assertTrue(session.planner.enable_communication_masking)
            custom = FalseyPlanner()
            self.assertIs(OffloadSession(OffloadConfig(planner=custom)).planner, custom)

    def test_generated_schedules_preserve_versions_budget_and_input_trace(self):
        rng = random.Random(20261109)
        for case in range(150):
            rows, sizes, retained = [[] for _ in range(24)], {}, set()
            for sid in range(8):
                sizes[sid] = rng.randint(1, 20)
                access_ops = sorted(rng.sample(range(24), rng.randint(2, 8)))
                for index, op in enumerate(access_ops):
                    kind = (
                        AccessKind.WRITE
                        if index == 0 or rng.random() < 0.2
                        else AccessKind.READ
                    )
                    rows[op].append((sid, kind))
                if rng.random() < 0.1:
                    retained.add(sid)
            trace = make_trace(rows, sizes, rng.randint(10, 65), retained)
            saved = copy.deepcopy(trace)
            immediate = PeakResidencyPlanner(enable_communication_masking=False).build(
                trace
            )
            floor = simulate(self, trace, immediate)
            masked = PeakResidencyPlanner().build(trace)
            actual = simulate(self, trace, masked)
            with self.subTest(case=case):
                self.assertTrue(
                    all(
                        value <= max(trace.memory_limit_bytes, floor[slot])
                        for slot, value in enumerate(actual)
                    )
                )
                self.assertEqual(trace, saved)
                self.assertEqual(PeakResidencyPlanner().build(trace), masked)
                for op in trace.ops:
                    op.duration_ms = rng.uniform(0.0, 0.01)
                trace.d2h_bandwidth_gbps = rng.uniform(0.001, 0.01)
                timed = PeakResidencyPlanner().build(trace)
                timed_resident = simulate(self, trace, timed)
                completed_resident = simulate(self, trace, timed, complete_copies=True)
                self.assertTrue(all(early <= late for early, late in zip(completed_resident, timed_resident)))
                self.assertTrue(all(early <= late for early, late in zip(timed_resident, actual)))
                for sid in sizes:
                    self.assertEqual(positions(timed, Action.COPY_D2H, sid), positions(masked, Action.COPY_D2H, sid))
                    for kind in (Action.COPY_H2D, Action.RELEASE_HOST):
                        before = positions(masked, kind, sid)
                        after = positions(timed, kind, sid)
                        self.assertEqual(len(after), len(before), f"after={after}, before={before}")
                        self.assertTrue(all(new >= old for new, old in zip(after, before)))
