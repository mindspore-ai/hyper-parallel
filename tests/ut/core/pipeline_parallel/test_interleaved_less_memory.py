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
"""Unit tests for the ``less_memory`` (shallow-warmup) Interleaved 1F1B variant.

The classic interleaved warmup keeps up to ``2*P - 2*i - 1`` micro-batch
activations in flight on stage ``i``; ``less_memory=True`` caps the per-stage
stagger at the plain-1F1B depth (``P - 1 - i`` extra ops instead of
``2*(P - 1 - i)``).  These tests exercise only schedule *ordering*
(``construct_exec_order``), so they need no distributed runtime.  Schedules
are built via ``object.__new__`` + manual attribute assignment, mirroring the
existing ``test_order.py`` pattern.

Beyond pinning warmup depths, both modes are run through a strict rendezvous
execution simulator that checks:

* every FWD/BWD is emitted exactly once on its owning rank;
* every cross-rank consumer has its RECV earlier in the rank's list and every
  SEND follows its producing compute;
* per adjacent rank pair, the two P2P op subsequences are positionally
  complementary (queue-order send<->recv matching, the HCCL invariant);
* the whole schedule executes to completion under blocking (rendezvous) P2P
  with all data dependencies satisfied at execution time.
"""
import unittest
from collections import Counter

from hyper_parallel.core.pipeline_parallel.mpipe import ScheduleMPipeTranspose
from hyper_parallel.core.pipeline_parallel.scheduler import (
    MetaStepType,
    Schedule1F1B,
    ScheduleInterleaved1F1B,
)

_SEND_TYPES = frozenset({
    MetaStepType.FWD_SEND, MetaStepType.BWD_SEND, MetaStepType.DATA_SEND,
})
_RECV_TYPES = frozenset({
    MetaStepType.FWD_RECV, MetaStepType.BWD_RECV, MetaStepType.DATA_RECV,
})
_COMM_TYPES = _SEND_TYPES | _RECV_TYPES


def _set_interleaved_attrs(schedule, p, v, m, less_memory, overlap_b_f=False):
    """Populate the pure-ordering attributes shared by both schedule classes."""
    # pylint: disable=protected-access
    schedule.real_stage_num = p
    schedule._stage_num = p * v
    schedule.n_local_stages = v
    schedule.micro_batch_num = m
    schedule._overlap_b_f = overlap_b_f
    schedule._less_memory = less_memory
    schedule.exec_order = {}
    schedule._init_round_layout()


def _build_interleaved(p, v, m, less_memory=False, overlap_b_f=False):
    """Construct an interleaved schedule through the pure-ordering path."""
    schedule = object.__new__(ScheduleInterleaved1F1B)
    _set_interleaved_attrs(schedule, p, v, m, less_memory, overlap_b_f)
    ScheduleInterleaved1F1B.construct_exec_order(schedule)
    return schedule


class _StubMPipeStage:
    """Minimal stage stand-in for the mpipe DATA routing tables."""

    dst_stage = 0
    src_stage = 0


def _build_mpipe(p, v, m, less_memory=False, trainable=False, overflow_mode="min"):
    """Construct a ScheduleMPipeTranspose order through the pure-ordering path.

    ``overflow_mode`` defaults to ``"min"`` (the legacy layout the pinned
    orders were authored against); the composition matrix runs both modes.
    """
    # pylint: disable=protected-access
    schedule = object.__new__(ScheduleMPipeTranspose)
    _set_interleaved_attrs(schedule, p, v, m, less_memory)
    schedule._num_transpose_layers = 2
    schedule._num_visual_layers = None if trainable else 2
    schedule._has_trainable_preprocess = trainable
    schedule._explicit_nontransposed_backward = False
    schedule._kwargs_batch_dim = {}
    schedule.stages = [_StubMPipeStage()]
    schedule._owner_backward = False
    schedule._overflow_mode = overflow_mode
    schedule.construct_exec_order()
    return schedule


def _build_1f1b(p, m):
    schedule = object.__new__(Schedule1F1B)
    schedule.real_stage_num = p
    schedule.micro_batch_num = m
    schedule.exec_order = {}
    Schedule1F1B.construct_exec_order(schedule)
    return schedule


def _flatten(order):
    """Leaf steps of a rank's order, skipping bubbles and expanding OVERLAP."""
    for step in order:
        if step is None:
            continue
        if step.sub_steps:
            yield from step.sub_steps
        else:
            yield step


def _compute_projection(order):
    """The (kind, stage, micro) sequence of FWD/BWD steps in a rank's order."""
    proj = []
    for step in _flatten(order):
        if step.type == MetaStepType.FWD:
            proj.append(("F", step.stage_index, step.micro_index))
        elif step.type == MetaStepType.BWD:
            proj.append(("B", step.stage_index, step.micro_index))
    return proj


def _comm_peer(step, p, stage_num):
    """(peer_rank, counterpart (type, stage, micro)) for a P2P step, else None."""
    s, m = step.stage_index, step.micro_index
    if step.type in (MetaStepType.FWD_SEND, MetaStepType.DATA_SEND):
        if s + 1 >= stage_num:
            return None
        recv = (MetaStepType.FWD_RECV if step.type == MetaStepType.FWD_SEND
                else MetaStepType.DATA_RECV)
        return (s + 1) % p, (recv, s + 1, m)
    if step.type in (MetaStepType.FWD_RECV, MetaStepType.DATA_RECV):
        send = (MetaStepType.FWD_SEND if step.type == MetaStepType.FWD_RECV
                else MetaStepType.DATA_SEND)
        return (s - 1) % p, (send, s - 1, m)
    if step.type == MetaStepType.BWD_SEND:
        if s - 1 < 0:
            return None
        return (s - 1) % p, (MetaStepType.BWD_RECV, s - 1, m)
    if step.type == MetaStepType.BWD_RECV:
        return (s + 1) % p, (MetaStepType.BWD_SEND, s + 1, m)
    return None


class _ScheduleChecker:
    """Structural + executional validation of a built schedule."""

    def __init__(self, test: unittest.TestCase, schedule: ScheduleInterleaved1F1B) -> None:
        """Snapshot the schedule's emitted per-rank order and dimensions."""
        self.test = test
        self.p = schedule.real_stage_num
        self.v = schedule.n_local_stages
        self.m = schedule.micro_batch_num
        self.stage_num = self.p * self.v
        self.order = {r: list(schedule.exec_order[r]) for r in range(self.p)}

    def check_all(self) -> None:
        """Run every structural check plus the rendezvous execution simulation."""
        self.check_compute_coverage()
        self.check_recv_before_consumer_send_after_producer()
        self.check_pairwise_fifo()
        self.check_rendezvous_execution()

    def check_structural(self) -> None:
        """Coverage + queue-order pairing only — for the overlap_b_f emission,
        whose recv waits are driven inside the OVERLAP callback rather than by
        list position, so the strict recv-order/rendezvous model does not apply.
        """
        self.check_compute_coverage()
        self.check_pairwise_fifo()

    # -- structural checks -------------------------------------------------

    def check_compute_coverage(self) -> None:
        """Every (stage, micro) FWD and BWD appears exactly once, on its rank."""
        for kind in (MetaStepType.FWD, MetaStepType.BWD):
            seen = {}
            for rank in range(self.p):
                for step in _flatten(self.order[rank]):
                    if step.type != kind:
                        continue
                    key = (step.stage_index, step.micro_index)
                    self.test.assertNotIn(key, seen, f"duplicate {kind} {key}")
                    self.test.assertEqual(step.stage_index % self.p, rank)
                    seen[key] = rank
            expected = {(s, mi) for s in range(self.stage_num) for mi in range(self.m)}
            self.test.assertEqual(set(seen), expected, f"missing/extra {kind}")

    def check_recv_before_consumer_send_after_producer(self) -> None:
        """Cross-rank consumers must follow their RECV; SENDs their compute."""
        for rank in range(self.p):
            received = set()
            computed = set()
            for step in _flatten(self.order[rank]):
                s, mi = step.stage_index, step.micro_index
                if step.type in _RECV_TYPES:
                    received.add((step.type, s, mi))
                elif step.type in _SEND_TYPES:
                    if step.type != MetaStepType.DATA_SEND:
                        producer = (MetaStepType.FWD
                                    if step.type == MetaStepType.FWD_SEND
                                    else MetaStepType.BWD)
                        self.test.assertIn(
                            (producer, s, mi), computed,
                            f"rank {rank}: {step.type} ({s},{mi}) before its compute")
                elif step.type == MetaStepType.FWD:
                    if s > 0 and (s - 1) % self.p != rank:
                        self.test.assertIn(
                            (MetaStepType.FWD_RECV, s, mi), received,
                            f"rank {rank}: FWD ({s},{mi}) before its FWD_RECV")
                    computed.add((step.type, s, mi))
                elif step.type == MetaStepType.BWD:
                    if s < self.stage_num - 1 and (s + 1) % self.p != rank:
                        self.test.assertIn(
                            (MetaStepType.BWD_RECV, s, mi), received,
                            f"rank {rank}: BWD ({s},{mi}) before its BWD_RECV")
                    computed.add((step.type, s, mi))

    def check_pairwise_fifo(self) -> None:
        """Per rank pair, the P2P subsequences must match positionally."""
        per_pair = {}
        for rank in range(self.p):
            for step in _flatten(self.order[rank]):
                if step.type not in _COMM_TYPES:
                    continue
                peer_info = _comm_peer(step, self.p, self.stage_num)
                self.test.assertIsNotNone(peer_info, f"boundary comm {step}")
                peer, _ = peer_info
                pair = (min(rank, peer), max(rank, peer))
                per_pair.setdefault(pair, {rank: [], peer: []})
                per_pair[pair].setdefault(rank, [])
                per_pair[pair][rank].append(step)
        for pair, sides in per_pair.items():
            a, b = pair
            ops_a, ops_b = sides.get(a, []), sides.get(b, [])
            self.test.assertEqual(
                len(ops_a), len(ops_b), f"pair {pair}: unbalanced P2P counts")
            for k, (sa, sb) in enumerate(zip(ops_a, ops_b)):
                expected = _comm_peer(sa, self.p, self.stage_num)[1]
                actual = (sb.type, sb.stage_index, sb.micro_index)
                self.test.assertEqual(
                    actual, expected,
                    f"pair {pair} slot {k}: {sa.type}({sa.stage_index},"
                    f"{sa.micro_index}) vs {actual} — queue heads must match")

    # -- execution simulation ---------------------------------------------

    def check_rendezvous_execution(self) -> None:
        """Simulate blocking P2P; assert completion and dependency order."""
        flat = {r: list(_flatten(self.order[r])) for r in range(self.p)}
        ptr = {r: 0 for r in range(self.p)}
        done = set()

        def _deps_ok(step):
            s, mi = step.stage_index, step.micro_index
            if step.type == MetaStepType.FWD and s > 0:
                return (MetaStepType.FWD, s - 1, mi) in done
            if step.type == MetaStepType.BWD and s < self.stage_num - 1:
                return (MetaStepType.BWD, s + 1, mi) in done
            return True

        def _advance(rank):
            moved = False
            while ptr[rank] < len(flat[rank]):
                step = flat[rank][ptr[rank]]
                if step.type in _COMM_TYPES:
                    peer, counterpart = _comm_peer(step, self.p, self.stage_num)
                    if ptr[peer] >= len(flat[peer]):
                        return moved
                    peer_step = flat[peer][ptr[peer]]
                    peer_key = (peer_step.type, peer_step.stage_index,
                                peer_step.micro_index)
                    if peer_key != counterpart:
                        return moved
                    ptr[rank] += 1
                    ptr[peer] += 1
                    moved = True
                    continue
                if step.type in (MetaStepType.FWD, MetaStepType.BWD):
                    self.test.assertTrue(
                        _deps_ok(step),
                        f"rank {rank}: {step.type} ({step.stage_index},"
                        f"{step.micro_index}) ran before its producer")
                    done.add((step.type, step.stage_index, step.micro_index))
                ptr[rank] += 1
                moved = True
            return moved

        while any(ptr[r] < len(flat[r]) for r in range(self.p)):
            if not any(_advance(r) for r in range(self.p)):
                stuck = {r: (flat[r][ptr[r]].type, flat[r][ptr[r]].stage_index,
                             flat[r][ptr[r]].micro_index)
                         for r in range(self.p) if ptr[r] < len(flat[r])}
                self.test.fail(f"schedule deadlocked; blocked ranks: {stuck}")
        self.test.assertEqual(
            len(done), 2 * self.stage_num * self.m, "not all compute executed")


_CASES = [
    # (p, v, m)
    (2, 2, 4),
    (3, 2, 6),
    (3, 3, 9),
    (4, 1, 8),
    (4, 2, 8),
    (4, 2, 12),
    (4, 2, 10),   # non-divisible micro count
    (4, 3, 8),
    (4, 4, 8),
    (5, 2, 10),
    (6, 2, 12),
    (8, 2, 16),
    (8, 3, 24),
    (4, 2, 2),    # short-micro regime (m < p)
    (4, 1, 3),    # short-micro, single chunk
]

# Shapes on which the CLASSIC (Megatron-depth) emission is fully well-formed
# today.  On the excluded deep-warmup shapes the classic order emits the
# steady-state wraparound BWD (last rank, chunk row below) BEFORE its
# BWD_RECV — a pre-existing base misordering (silent: the torch stage reads
# the pre-allocated grad buffer without blocking), documented by
# ``test_classic_wraparound_bwd_recv_misordered`` and tracked separately.
# The less-memory emission is validated on ALL shapes above.
_CASES_CLASSIC_VALID = list(_CASES)


class TestInterleavedLessMemory(unittest.TestCase):
    """Validation of both warmup modes and the less-memory depth guarantee."""

    def test_classic_mode_valid(self):
        """The classic warmup passes on every shape, including the deep-warmup
        ones the zero-width DATA_LOAD splice fixed (vpp >= 2, M > PP)."""
        for p, v, m in _CASES_CLASSIC_VALID:
            with self.subTest(p=p, v=v, m=m):
                schedule = _build_interleaved(p, v, m, less_memory=False)
                _ScheduleChecker(self, schedule).check_all()

    def test_less_memory_mode_valid(self):
        """The shallow warmup produces a structurally valid schedule."""
        for p, v, m in _CASES:
            with self.subTest(p=p, v=v, m=m):
                schedule = _build_interleaved(p, v, m, less_memory=True)
                _ScheduleChecker(self, schedule).check_all()

    def test_overlap_b_f_structural_both_modes(self):
        """overlap_b_f stays structurally valid in both warmup modes."""
        for less_memory in (False, True):
            for p, v, m in [(4, 2, 8), (4, 2, 12), (2, 2, 4)]:
                with self.subTest(less_memory=less_memory, p=p, v=v, m=m):
                    schedule = _build_interleaved(
                        p, v, m, less_memory=less_memory, overlap_b_f=True)
                    _ScheduleChecker(self, schedule).check_structural()

    def test_warmup_never_deeper_than_classic(self):
        """The shallow warmup is <= classic everywhere, < on early stages."""
        for p, v, m in _CASES:
            classic = _build_interleaved(p, v, m, less_memory=False)
            shallow = _build_interleaved(p, v, m, less_memory=True)
            for rank in range(p):
                w_classic = self._warmup_len(classic, rank)
                w_shallow = self._warmup_len(shallow, rank)
                self.assertLessEqual(
                    w_shallow, w_classic, f"p={p} v={v} m={m} rank={rank}")
            if m >= p:
                self.assertLess(
                    self._warmup_len(shallow, 0), self._warmup_len(classic, 0),
                    f"p={p} v={v} m={m}: rank 0 depth should shrink")

    @staticmethod
    def _warmup_len(schedule, rank):
        """Number of FWDs before the first BWD in a rank's compute order.

        This is ``warmup_ops + 1`` when a 1F1B steady state exists (the first
        1F1B op emits its F before its B), and ``warmup_ops`` when warmup is
        capped at all ``V*M`` forwards.
        """
        proj = _compute_projection(schedule.exec_order[rank])
        for idx, (kind, _, _) in enumerate(proj):
            if kind == "B":
                return idx
        return len(proj)

    @staticmethod
    def _expected_f_before_b(warmup, total_fwd):
        return warmup + 1 if warmup < total_fwd else warmup

    def test_less_memory_warmup_formula(self):
        """Pinned warmup counts: (V-1)*R + (P-1-i), capped at V*M."""
        for p, v, m in _CASES:
            schedule = _build_interleaved(p, v, m, less_memory=True)
            r0 = schedule.n_microbatch_per_round[0]
            for rank in range(p):
                warmup = min((v - 1) * r0 + (p - 1 - rank), v * m)
                self.assertEqual(
                    self._warmup_len(schedule, rank),
                    self._expected_f_before_b(warmup, v * m),
                    f"p={p} v={v} m={m} rank={rank}")

    def test_vpp1_less_memory_matches_plain_1f1b(self):
        """At V=1 the shallow warmup reproduces plain 1F1B compute order."""
        for p, m in [(2, 4), (4, 8), (4, 10), (8, 16), (4, 3)]:
            with self.subTest(p=p, m=m):
                inter = _build_interleaved(p, 1, m, less_memory=True)
                plain = _build_1f1b(p, m)
                for rank in range(p):
                    self.assertEqual(
                        _compute_projection(inter.exec_order[rank]),
                        _compute_projection(plain.exec_order[rank]),
                        f"rank {rank}")

    def test_default_is_classic(self):
        """Omitting the flag (and the pure-construction path) keeps the classic warmup."""
        schedule = _build_interleaved(4, 2, 8)
        r0 = schedule.n_microbatch_per_round[0]
        for rank in range(4):
            warmup = (2 - 1) * r0 + 2 * (4 - 1 - rank)
            self.assertEqual(
                self._warmup_len(schedule, rank),
                self._expected_f_before_b(warmup, 2 * 8))


class TestMPipeLessMemoryComposition(unittest.TestCase):
    """less_memory x ScheduleMPipeTranspose: the mpipe layering is depth-agnostic.

    The mpipe passes (DATA_LOAD strip/keep, transpose prefix, rank-0 inline
    steps) key on step type/stage/micro only, so the shallow warmup must change
    neither the per-rank step multiset nor the MPIPE prefix, and must add no
    recv-after-consumer ordering the plain interleaved base does not already
    have in the same mode.
    """

    _MATRIX = [(p, v, m, trainable, mode)
               for p in (2, 4) for v in (1, 2) for m in (p, 2 * p)
               for trainable in (False, True)
               for mode in ("min", "full")]

    @staticmethod
    def _key(step):
        return (step.type, step.stage_index, step.micro_index)

    @classmethod
    def _prefix(cls, order):
        out = []
        for step in order:
            if step is None or step.stage_index != -1:
                break
            out.append(cls._key(step))
        return out

    @classmethod
    def _recv_after_consumer(cls, exec_order, p):
        """(rank, recv-key) pairs where a recv lists after its consuming compute."""
        out = set()
        for rank in range(p):
            pos = {}
            for i, step in enumerate(exec_order[rank]):
                if step is not None:
                    pos.setdefault(cls._key(step), i)
            for (kind, s, mi), i in pos.items():
                if kind not in (MetaStepType.FWD, MetaStepType.BWD):
                    continue
                recv = (MetaStepType.FWD_RECV if kind == MetaStepType.FWD
                        else MetaStepType.BWD_RECV)
                j = pos.get((recv, s, mi))
                if j is not None and j > i:
                    out.add((rank, recv, s, mi))
        return out

    def test_multiset_prefix_and_ordering_invariant(self):
        """The shallow warmup must not disturb the MPipe layering."""
        for p, v, m, trainable, mode in self._MATRIX:
            with self.subTest(p=p, v=v, m=m, trainable=trainable, mode=mode):
                classic = _build_mpipe(p, v, m, less_memory=False,
                                       trainable=trainable, overflow_mode=mode)
                shallow = _build_mpipe(p, v, m, less_memory=True,
                                       trainable=trainable, overflow_mode=mode)
                for rank in range(p):
                    oc = classic.exec_order[rank]
                    os_ = shallow.exec_order[rank]
                    self.assertEqual(
                        Counter(map(self._key, oc)), Counter(map(self._key, os_)),
                        f"rank {rank}: step multiset differs")
                    self.assertEqual(
                        self._prefix(oc), self._prefix(os_),
                        f"rank {rank}: MPIPE prefix differs")
                for less_memory, sched in ((False, classic), (True, shallow)):
                    base = _build_interleaved(p, v, m, less_memory=less_memory)
                    new = (self._recv_after_consumer(sched.exec_order, p)
                           - self._recv_after_consumer(base.exec_order, p))
                    self.assertFalse(
                        new, f"mpipe layering added recv-after-consumer: {sorted(new)}")

    def test_shallow_warmup_engages_under_mpipe(self):
        """The knob reaches the MPipe body, not just the plain interleaved one."""
        classic = _build_mpipe(4, 2, 8, less_memory=False)
        shallow = _build_mpipe(4, 2, 8, less_memory=True)
        self.assertLess(shallow.warmup_ops(0), classic.warmup_ops(0))


if __name__ == "__main__":
    unittest.main()
