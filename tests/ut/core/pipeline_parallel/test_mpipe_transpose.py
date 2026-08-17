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
"""Unit tests for the ScheduleMPipeTranspose execution-order construction.

These tests exercise only the schedule *ordering* (``construct_exec_order``),
which depends solely on the stage / micro-batch counts, so they need no
distributed runtime, GPU, or preprocess module.  Schedules are built via
``object.__new__`` + manual attribute assignment, mirroring the existing
``test_order.py`` pattern for ``ScheduleInterleaved1F1B``.
"""
import itertools
import unittest

from hyper_parallel.core.pipeline_parallel.scheduler import (
    MetaStep,
    MetaStepType,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.core.pipeline_parallel.mpipe import (
    MpipeStepType,
    ScheduleMPipeTranspose,
)

_MPIPE_STEP_TYPES = frozenset({
    MpipeStepType.MPIPE_PARAM_BROADCAST,
    MpipeStepType.MPIPE_TRANSPOSE_FWD,
    MpipeStepType.MPIPE_FWD_SEND,
    MpipeStepType.MPIPE_FWD_RECV,
    MpipeStepType.MPIPE_GRAPH_SEND,
    MpipeStepType.MPIPE_GRAPH_RECV,
    MpipeStepType.MPIPE_TRANSPOSE_BWD,
    # In-schedule dataload steps (added in the DATA_LOAD/SEND/RECV feature):
    # MPipe emits DATA_LOAD in the prefix on each owning rank and strips it
    # from the body; DATA_SEND/RECV in the prefix ship raw inputs to stage 0
    # and in the body ride the per-stage chain. All three are part of the
    # MPipe layering for these ordering tests.
    MetaStepType.DATA_LOAD,
    MetaStepType.DATA_SEND,
    MetaStepType.DATA_RECV,
})


class _StubStage:
    """Placeholder for ``schedule.stages[0]`` in the ordering-only test path.

    The mpipe ``construct_exec_order`` reads ``dst_stage`` / ``src_stage`` to
    populate the body's DATA_SEND/RECV routing table; the tests only care
    about the ordered list of steps, not the routing targets themselves.
    """

    dst_stage = 0
    src_stage = 0


def _set_interleaved_attrs(schedule, real_stage_num, n_local_stages, micro_batch_num):
    """Populate the attributes ``construct_exec_order`` reads, mirroring
    ``ScheduleInterleaved1F1B.__init__`` (which we bypass to avoid the
    distributed stage setup)."""
    # pylint: disable=protected-access
    schedule.real_stage_num = real_stage_num
    schedule._stage_num = real_stage_num * n_local_stages
    schedule.n_local_stages = n_local_stages
    schedule.micro_batch_num = micro_batch_num
    schedule._overlap_b_f = False
    n_rounds = max(1, micro_batch_num // real_stage_num)
    if micro_batch_num < real_stage_num:
        base, remainder = micro_batch_num - real_stage_num, 0
    else:
        n_extra = micro_batch_num % real_stage_num
        base, remainder = n_extra // n_rounds, n_extra % n_rounds
    schedule.n_rounds = n_rounds
    schedule.n_microbatch_per_round = [
        real_stage_num + base + 1 if i < remainder else real_stage_num + base
        for i in range(n_rounds)
    ]
    accu = [x * n_local_stages for x in itertools.accumulate(schedule.n_microbatch_per_round)]
    accu.insert(0, 0)
    schedule.n_microbatch_per_round_accu = accu
    schedule.exec_order = {}


def _build_mpipe(real_stage_num, n_local_stages, micro_batch_num,
                 num_transpose_layers=2,
                 owner_backward=False, num_visual_layers=None, has_trainable=None,
                 kwargs_batch_dim=None, overflow_mode="min"):
    """Build a ScheduleMPipeTranspose with ordering-only state populated.

    ``owner_backward`` selects the owner-does-backward ship-back ordering (only
    effective for a trainable preprocess).
    ``num_visual_layers`` mirrors the ctor param (``None`` = no visual tower,
    like the real constructor's default and the text-style cases here).
    ``has_trainable`` overrides the T>0 derivation so frozen-with-T>0 (the
    real frozen visual tower) is constructible; ``kwargs_batch_dim`` sets the
    kwarg spec ``_DATA_KEYS`` is built from (default empty).
    ``overflow_mode`` gates the ``M > NT`` distribution: ``"min"`` (default
    here to match the legacy ordering expectations) keeps a single owner per
    rank and inline-loads the overflow on rank 0; ``"full"`` round-robins
    every micro across the owner ranks.
    """
    schedule = object.__new__(ScheduleMPipeTranspose)
    _set_interleaved_attrs(schedule, real_stage_num, n_local_stages, micro_batch_num)
    # pylint: disable=protected-access
    schedule._num_transpose_layers = num_transpose_layers
    schedule._num_visual_layers = num_visual_layers
    # For the text-style cases the ordering tests cover, "has trainable preprocess"
    # is equivalent to T>0 (the transposed layers are trainable; T=0 is identity).
    schedule._has_trainable_preprocess = (
        num_transpose_layers > 0 if has_trainable is None else has_trainable)
    schedule._overflow_mode = overflow_mode
    # ``__init__`` is bypassed via ``object.__new__``, so set the ``_kwargs_batch_dim``
    # that ``construct_exec_order`` reads to build ``_DATA_KEYS``.
    schedule._kwargs_batch_dim = kwargs_batch_dim if kwargs_batch_dim is not None else {}
    # ``construct_exec_order`` reads ``stages[0].dst_stage`` / ``src_stage`` for the
    # DATA_SEND/RECV routing, which these ordering tests never assert on.
    schedule.stages = [_StubStage()]
    schedule._owner_backward = owner_backward and schedule._has_trainable_preprocess
    schedule.construct_exec_order()
    return schedule


def _build_interleaved_body(real_stage_num, n_local_stages, micro_batch_num):
    """Build the reference (plain Interleaved 1F1B) body order for comparison."""
    schedule = object.__new__(ScheduleInterleaved1F1B)
    _set_interleaved_attrs(schedule, real_stage_num, n_local_stages, micro_batch_num)
    schedule.construct_exec_order()
    return schedule.exec_order


def _strip_mpipe(order):
    """Drop every MPIPE_* + DATA_* step **and** ``None`` bubbles.

    MPipe's internal body filter (mpipe/schedule.py:264) also drops ``None``
    bubbles when it lays down its layered order, so the comparison with the
    plain Interleaved 1F1B reference body has to strip ``None`` symmetrically
    or the two sides diverge by the bubble count.
    """
    return [step for step in order if step is not None and step.type not in _MPIPE_STEP_TYPES]


def _index_of(order, target):
    for index, step in enumerate(order):
        if step is not None and step == target:
            return index
    raise AssertionError(f"step {target} not found in order")


def _count(order, step_type, stage_index=None):
    return sum(
        1 for step in order
        if step is not None and step.type == step_type
        and (stage_index is None or step.stage_index == stage_index)
    )


class TestMPipeTransposePrefix(unittest.TestCase):
    """The transpose-phase prefix prepended to each rank's body order."""

    def test_full_transpose_prefix_per_rank(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: PP=4, MB=4 (NT=4) so every rank is transposed.
        Expectation: stage 0's prefix loads its own micro, receives the raw
            inputs + output + recompute input of the three other transposed
            micro-batches; each other rank loads + ships its own.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=1, micro_batch_num=4)

        # stage_index == -1 flags "MPipe prefix" (not a real body stage); the
        # BROADCAST / DATA_LOAD carry the emitting rank as their micro_index.
        expected_rank0 = [
            MetaStep(0, MpipeStepType.MPIPE_PARAM_BROADCAST, -1),
            MetaStep(0, MetaStepType.DATA_LOAD, -1),
            MetaStep(0, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1),
            MetaStep(1, MetaStepType.DATA_RECV, -1),
            MetaStep(2, MetaStepType.DATA_RECV, -1),
            MetaStep(3, MetaStepType.DATA_RECV, -1),
            MetaStep(1, MpipeStepType.MPIPE_FWD_RECV, -1),
            MetaStep(2, MpipeStepType.MPIPE_FWD_RECV, -1),
            MetaStep(3, MpipeStepType.MPIPE_FWD_RECV, -1),
            MetaStep(1, MpipeStepType.MPIPE_GRAPH_RECV, -1),
            MetaStep(2, MpipeStepType.MPIPE_GRAPH_RECV, -1),
            MetaStep(3, MpipeStepType.MPIPE_GRAPH_RECV, -1),
        ]
        assert schedule.exec_order[0][:len(expected_rank0)] == expected_rank0, \
            (f"rank 0 prefix mismatch: "
             f"expected={expected_rank0}, got={schedule.exec_order[0][:len(expected_rank0)]}")

        for rank in (1, 2, 3):
            expected = [
                MetaStep(rank, MpipeStepType.MPIPE_PARAM_BROADCAST, -1),
                MetaStep(rank, MetaStepType.DATA_LOAD, -1),
                MetaStep(rank, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1),
                MetaStep(rank, MetaStepType.DATA_SEND, -1),
                MetaStep(rank, MpipeStepType.MPIPE_FWD_SEND, -1),
                MetaStep(rank, MpipeStepType.MPIPE_GRAPH_SEND, -1),
            ]
            assert schedule.exec_order[rank][:len(expected)] == expected, \
                (f"rank {rank} prefix mismatch: "
                 f"expected={expected}, got={schedule.exec_order[rank][:len(expected)]}")

    def test_short_micro_prefix(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: PP=4, MB=2 (NT=2) so only ranks 0,1 are transposed.
        Expectation: ranks 2,3 only participate in the parameter broadcast and
            emit no DATA_LOAD / transpose forward / send.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=1, micro_batch_num=2)

        expected_rank0 = [
            MetaStep(0, MpipeStepType.MPIPE_PARAM_BROADCAST, -1),
            MetaStep(0, MetaStepType.DATA_LOAD, -1),
            MetaStep(0, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1),
            MetaStep(1, MetaStepType.DATA_RECV, -1),
            MetaStep(1, MpipeStepType.MPIPE_FWD_RECV, -1),
            MetaStep(1, MpipeStepType.MPIPE_GRAPH_RECV, -1),
        ]
        assert schedule.exec_order[0][:len(expected_rank0)] == expected_rank0, \
            (f"rank 0 short-micro prefix mismatch: "
             f"expected={expected_rank0}, got={schedule.exec_order[0][:len(expected_rank0)]}")

        expected_rank1 = [
            MetaStep(1, MpipeStepType.MPIPE_PARAM_BROADCAST, -1),
            MetaStep(1, MetaStepType.DATA_LOAD, -1),
            MetaStep(1, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1),
            MetaStep(1, MetaStepType.DATA_SEND, -1),
            MetaStep(1, MpipeStepType.MPIPE_FWD_SEND, -1),
            MetaStep(1, MpipeStepType.MPIPE_GRAPH_SEND, -1),
        ]
        assert schedule.exec_order[1][:len(expected_rank1)] == expected_rank1, \
            (f"rank 1 short-micro prefix mismatch: "
             f"expected={expected_rank1}, got={schedule.exec_order[1][:len(expected_rank1)]}")

        for rank in (2, 3):
            assert schedule.exec_order[rank][0] == MetaStep(rank, MpipeStepType.MPIPE_PARAM_BROADCAST, -1), \
                (f"rank {rank} should start with a broadcast, "
                 f"got {schedule.exec_order[rank][0]}")
            # The body's 1F1B chain still emits DATA_SEND / FWD_SEND from stage 0, so
            # absence can only be asserted on the prefix layer (stage_index=-1).
            for absent in (MetaStepType.DATA_LOAD,
                           MpipeStepType.MPIPE_TRANSPOSE_FWD,
                           MetaStepType.DATA_SEND,
                           MpipeStepType.MPIPE_FWD_SEND):
                assert _count(schedule.exec_order[rank], absent, stage_index=-1) == 0, \
                    f"rank {rank} should emit no {absent} in the MPipe prefix when not transposed"

    def test_param_broadcast_once_per_rank(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Build the schedule for PP=4, n_local=2, MB=8.
        Expectation: exactly one parameter broadcast leads every rank's order.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=2, micro_batch_num=8)
        for rank in range(4):
            assert _count(schedule.exec_order[rank], MpipeStepType.MPIPE_PARAM_BROADCAST) == 1, \
                (f"rank {rank} should have exactly one broadcast, got "
                 f"{_count(schedule.exec_order[rank], MpipeStepType.MPIPE_PARAM_BROADCAST)}")

    def test_send_recv_counts_balanced(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Count the prefix send/recv steps for PP=4, MB=4.
        Expectation: stage 0 receives one output+input per other transposed
            micro-batch, and each sender rank sends exactly one of each.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=1, micro_batch_num=4)
        num_transpose = 4
        assert _count(schedule.exec_order[0], MpipeStepType.MPIPE_FWD_RECV) == num_transpose - 1, \
            "stage 0 must receive one preprocess output per other transposed micro-batch"
        assert _count(schedule.exec_order[0], MpipeStepType.MPIPE_GRAPH_RECV) == num_transpose - 1, \
            "stage 0 must receive one preprocess input per other transposed micro-batch"
        total_fwd_send = sum(_count(schedule.exec_order[r], MpipeStepType.MPIPE_FWD_SEND) for r in range(4))
        total_graph_send = sum(_count(schedule.exec_order[r], MpipeStepType.MPIPE_GRAPH_SEND) for r in range(4))
        assert total_fwd_send == num_transpose - 1, \
            f"expected {num_transpose - 1} MPIPE_FWD_SEND across ranks, got {total_fwd_send}"
        assert total_graph_send == num_transpose - 1, \
            f"expected {num_transpose - 1} MPIPE_GRAPH_SEND across ranks, got {total_graph_send}"


class TestMPipeTransposeRank0Patches(unittest.TestCase):
    """Inline preprocess forward (non-transposed) and stage-0 backward
    (transposed) inserted into stage 0's body order."""

    def test_transpose_bwd_follows_stage0_bwd(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Inspect stage 0's body order for PP=4, n_local=2, MB=8.
        Expectation: every micro-batch gets an MPIPE_TRANSPOSE_BWD immediately
            after its stage-0 body backward — under the round-robin design the
            preprocess output is always detached, so a trainable preprocess
            pays one stage-0 backward per micro (transposed and overflow alike).
        """
        micro_batch_num = 8
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=2, micro_batch_num=micro_batch_num)
        order = schedule.exec_order[0]
        for micro in range(micro_batch_num):
            bwd_index = _index_of(order, MetaStep(micro, MetaStepType.BWD, 0))
            following = order[bwd_index + 1]
            assert following == MetaStep(micro, MpipeStepType.MPIPE_TRANSPOSE_BWD, -1), \
                (f"micro {micro}: expected MPIPE_TRANSPOSE_BWD right after BWD(stage 0), "
                 f"got {following}")
        assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD) == micro_batch_num, \
            (f"expected {micro_batch_num} transpose backwards on stage 0, got "
             f"{_count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD)}")

    def test_inline_transpose_fwd_precedes_nontransposed_stage0_fwd(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Inspect stage 0's body order for PP=4, n_local=2, MB=8.
        Expectation: each non-transposed micro-batch gets an inline
            MPIPE_TRANSPOSE_FWD just before its stage-0 forward; transposed
            micro-batches do not.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=2, micro_batch_num=8)
        order = schedule.exec_order[0]
        num_transpose = 4

        for micro in range(num_transpose, 8):
            fwd_index = _index_of(order, MetaStep(micro, MetaStepType.FWD, 0))
            preceding = order[fwd_index - 1]
            assert preceding == MetaStep(micro, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1), \
                (f"micro {micro}: expected inline MPIPE_TRANSPOSE_FWD right before "
                 f"FWD(stage 0), got {preceding}")

        for micro in range(num_transpose):
            fwd_index = _index_of(order, MetaStep(micro, MetaStepType.FWD, 0))
            preceding = order[fwd_index - 1]
            assert preceding != MetaStep(micro, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1), \
                (f"transposed micro {micro} must not get an inline transpose forward "
                 f"before its stage-0 body forward")

        # One inline transpose forward per non-transposed micro-batch, plus the
        # single prefix transpose forward for stage 0's own transposed micro 0.
        assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_FWD) == (8 - num_transpose) + 1, \
            (f"unexpected transpose-forward count on stage 0: "
             f"got {_count(order, MpipeStepType.MPIPE_TRANSPOSE_FWD)}")

    def test_no_inline_forward_when_all_transposed(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Build the schedule with MB == PP (=4) so every micro is transposed.
        Expectation: no inline preprocess forward is inserted into the body
            (only the single prefix transpose forward), with one backward per micro.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=1, micro_batch_num=4)
        order = schedule.exec_order[0]
        assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_FWD) == 1, \
            (f"all-transposed case should have a single (prefix) transpose forward, "
             f"got {_count(order, MpipeStepType.MPIPE_TRANSPOSE_FWD)}")
        assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD) == 4, \
            (f"all-transposed case should have one transpose backward per micro, "
             f"got {_count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD)}")


class TestMPipeTransposeStage0Backward(unittest.TestCase):
    """Trainable preprocess: the stage-0 backward fires after every stage-0
    body backward.

    Under the round-robin design the preprocess output is always detached
    (no graph-connected non-transposed path), so a trainable preprocess needs
    one ``MPIPE_TRANSPOSE_BWD`` per micro-batch on stage 0 — regardless of
    which rank transposed the micro or which overflow mode distributed it.
    """

    def test_transpose_bwd_emitted_for_all_micros(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (trainable path).
        Description: Build the default (min-mode, T=2 trainable) schedule, MB=8.
        Expectation: stage 0 gets one MPIPE_TRANSPOSE_BWD immediately after
            every body-0 backward — all 8 micros, transposed and overflow alike.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=2, micro_batch_num=8)
        order = schedule.exec_order[0]
        for micro in range(8):
            bwd_index = _index_of(order, MetaStep(micro, MetaStepType.BWD, 0))
            following = order[bwd_index + 1]
            assert following == MetaStep(micro, MpipeStepType.MPIPE_TRANSPOSE_BWD, -1), \
                (f"micro {micro}: expected MPIPE_TRANSPOSE_BWD after BWD(stage 0), "
                 f"got {following}")
        assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD) == 8, \
            (f"trainable preprocess should emit one stage-0 backward per micro-batch, got "
             f"{_count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD)}")

    def test_full_mode_also_emits_bwd_for_all_micros(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (trainable, full mode).
        Description: Same topology under overflow_mode="full".
        Expectation: the stage-0 backward count is identical to min mode —
            ownership distribution moves the *forward*, not the backward.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=2, micro_batch_num=8,
            overflow_mode="full")
        assert _count(schedule.exec_order[0], MpipeStepType.MPIPE_TRANSPOSE_BWD) == 8, \
            "full mode must emit the same per-micro stage-0 backward as min mode"


class TestMPipeTransposeDataloadOnly(unittest.TestCase):
    """``T == 0``: only the dataload is transposed — no parameter broadcast,
    no stage-0-backward input transport, and no stage-0 backward."""

    def test_no_preprocess_steps_emitted(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (T=0 dataload-only).
        Description: Build the schedule with num_transpose_layers=0.
        Expectation: no param broadcast, transpose forward, encoded-output
            transport, stage-0-backward input transport, or transpose backward
            is emitted on any rank — only the DATA_LOAD/SEND/RECV chain remains.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=4, num_transpose_layers=0)
        for rank in range(4):
            order = schedule.exec_order[rank]
            for absent in (
                MpipeStepType.MPIPE_PARAM_BROADCAST,
                MpipeStepType.MPIPE_TRANSPOSE_FWD,
                MpipeStepType.MPIPE_FWD_SEND,
                MpipeStepType.MPIPE_FWD_RECV,
                MpipeStepType.MPIPE_GRAPH_SEND,
                MpipeStepType.MPIPE_GRAPH_RECV,
                MpipeStepType.MPIPE_TRANSPOSE_BWD,
            ):
                assert _count(order, absent) == 0, \
                    f"T=0 rank {rank} should emit no {absent}, got {_count(order, absent)}"

    def test_dataload_transport_still_present(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (T=0 dataload-only).
        Description: Build the schedule with num_transpose_layers=0.
        Expectation: each transposed rank loads its micro-batch and ships the
            raw inputs to stage 0 via DATA_SEND; stage 0 loads its own micro
            and receives the other transposed micro-batches' inputs via
            DATA_RECV. No MPIPE_TRANSPOSE_FWD / MPIPE_FWD_SEND/RECV — stage 0
            runs the visual tower itself during body forward.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=4, num_transpose_layers=0)

        expected_rank0 = [
            MetaStep(0, MetaStepType.DATA_LOAD, -1),
            MetaStep(1, MetaStepType.DATA_RECV, -1),
            MetaStep(2, MetaStepType.DATA_RECV, -1),
            MetaStep(3, MetaStepType.DATA_RECV, -1),
        ]
        assert schedule.exec_order[0][:len(expected_rank0)] == expected_rank0, \
            (f"T=0 rank 0 prefix mismatch: "
             f"expected={expected_rank0}, got={schedule.exec_order[0][:len(expected_rank0)]}")

        for rank in (1, 2, 3):
            expected = [
                MetaStep(rank, MetaStepType.DATA_LOAD, -1),
                MetaStep(rank, MetaStepType.DATA_SEND, -1),
            ]
            assert schedule.exec_order[rank][:len(expected)] == expected, \
                (f"T=0 rank {rank} prefix mismatch: "
                 f"expected={expected}, got={schedule.exec_order[rank][:len(expected)]}")

    def test_min_overflow_inlines_data_load_not_transpose_fwd(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (T=0 dataload-only, "min" overflow).
        Description: PP=4, M=8, NT=4, overflow_mode="min", num_transpose_layers=0.
            Stage 0 still needs the inline DATA_LOAD for its overflow micros
            (4..7) so the body FWD has data to consume, but T=0 suppresses
            MPIPE_TRANSPOSE_FWD everywhere: emitting it would run the param-free
            identity preprocess and re-route the output through it.
        Expectation: rank 0's body has one inline DATA_LOAD immediately before
            each overflow micro's stage-0 FWD, and zero MPIPE_TRANSPOSE_FWD.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            num_transpose_layers=0, overflow_mode="min")
        order = schedule.exec_order[0]
        assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_FWD) == 0, \
            "T=0 must suppress MPIPE_TRANSPOSE_FWD even for min-mode overflow micros"
        for overflow_micro in (4, 5, 6, 7):
            fwd_index = _index_of(order, MetaStep(overflow_micro, MetaStepType.FWD, 0))
            assert order[fwd_index - 1] == MetaStep(overflow_micro, MetaStepType.DATA_LOAD, -1), \
                f"micro {overflow_micro}: expected an inline DATA_LOAD before its stage-0 FWD"

    def test_body_preserved(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (T=0 dataload-only).
        Description: Strip the MPIPE_* + DATA_LOAD/SEND/RECV steps from the
            T=0 schedule per rank; strip DATA_LOAD/SEND/RECV from the
            reference body (both are stripped via ``_strip_mpipe`` since the
            in-schedule dataload steps are part of the MPipe layering).
        Expectation: the remaining FWD/BWD/SEND/RECV body order matches the
            plain Interleaved 1F1B body.
        """
        mpipe = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=4, num_transpose_layers=0)
        body = _build_interleaved_body(real_stage_num=4, n_local_stages=1, micro_batch_num=4)
        for rank in range(4):
            assert _strip_mpipe(mpipe.exec_order[rank]) == _strip_mpipe(body[rank]), \
                f"T=0 rank {rank}: body order changed after MPipe layering"


class TestMPipeTransposeBodyPreserved(unittest.TestCase):
    """Stripping the MPIPE_* steps must recover the plain Interleaved 1F1B body
    exactly — MPipe only layers steps around an unchanged body schedule."""

    def _assert_body_matches(self, real_stage_num, n_local_stages, micro_batch_num):
        mpipe = _build_mpipe(real_stage_num, n_local_stages, micro_batch_num)
        body = _build_interleaved_body(real_stage_num, n_local_stages, micro_batch_num)
        for rank in range(real_stage_num):
            stripped = _strip_mpipe(mpipe.exec_order[rank])
            # Both sides carry DATA_LOAD/SEND/RECV, so strip both before
            # comparing.
            expected = _strip_mpipe(body[rank])
            assert stripped == expected, \
                (f"rank {rank}: body order changed after MPipe layering "
                 f"(PP={real_stage_num}, n_local={n_local_stages}, MB={micro_batch_num})")

    def test_body_preserved_plain(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Plain (n_local=1) PP=4, MB=4 schedule.
        Expectation: stripping MPIPE_* steps recovers the Interleaved 1F1B body exactly.
        """
        self._assert_body_matches(real_stage_num=4, n_local_stages=1, micro_batch_num=4)

    def test_body_preserved_interleaved(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Interleaved (n_local=2) PP=4, MB=8 schedule.
        Expectation: stripping MPIPE_* steps recovers the Interleaved 1F1B body exactly.
        """
        self._assert_body_matches(real_stage_num=4, n_local_stages=2, micro_batch_num=8)

    def test_body_preserved_short_micro(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Short-micro (MB=2 < PP=4) schedule.
        Expectation: stripping MPIPE_* steps recovers the Interleaved 1F1B body exactly.
        """
        self._assert_body_matches(real_stage_num=4, n_local_stages=1, micro_batch_num=2)


class TestMPipeTransposeValidation(unittest.TestCase):
    """Constructor argument validation that runs before any distributed setup."""

    def test_swap_rejected_before_other_constructor_work(self):
        """MPipe rejects swap before inspecting stages or preprocess state."""
        with self.assertRaisesRegex(ValueError, "activation swap is not yet supported"):
            ScheduleMPipeTranspose(
                stages=None,
                micro_batch_num=0,
                preprocess_module=None,
                num_transpose_layers=0,
                swap=True,
            )

    def test_negative_transpose_layers_rejected(self):
        """
        Feature: MPipe Transpose constructor validation.
        Description: Construct with num_transpose_layers=-1.
        Expectation: a ValueError is raised before any distributed setup.
        """
        with self.assertRaises(ValueError):
            ScheduleMPipeTranspose(
                stages=None,
                micro_batch_num=4,
                preprocess_module=None,
                num_transpose_layers=-1,
            )

    def test_non_int_transpose_layers_rejected(self):
        """
        Feature: MPipe Transpose constructor validation.
        Description: Construct with a non-integer num_transpose_layers (1.5).
        Expectation: a ValueError is raised before any distributed setup.
        """
        with self.assertRaises(ValueError):
            ScheduleMPipeTranspose(
                stages=None,
                micro_batch_num=4,
                preprocess_module=None,
                num_transpose_layers=1.5,
            )

    def test_invalid_overflow_mode_rejected(self):
        """
        Feature: MPipe Transpose constructor validation.
        Description: Construct with overflow_mode='bogus'.
        Expectation: a ValueError is raised before any distributed setup.
        """
        with self.assertRaises(ValueError):
            ScheduleMPipeTranspose(
                stages=None,
                micro_batch_num=4,
                preprocess_module=None,
                num_transpose_layers=2,
                overflow_mode="bogus",
            )


# ---------------------------------------------------------------------------
# Round-robin overflow (``overflow_mode="full"``)
# ---------------------------------------------------------------------------


def _prefix_ops(exec_order, rank):
    """Slice of a rank's exec_order that lives in the MPipe prefix (stage_index=-1)."""
    return [op for op in exec_order[rank]
            if op is not None and op.stage_index == -1]


class TestMPipeTransposeFullOverflow(unittest.TestCase):
    """``overflow_mode="full"``: distribute the ``M > NT`` overflow micros
    round-robin across the owner ranks instead of dumping them on rank 0.

    At ``PP=4, M=8`` each rank runs ``M/NT = 2`` transpose forwards; the
    ViT phase is balanced.  Under ``"min"`` rank 0 would inline-load the
    four overflow micros and run 5 transpose forwards while ranks 1..3
    ran only 1.
    """

    def test_round_robin_ownership(self):
        """
        Feature: MPipe Transpose round-robin overflow.
        Description: PP=4, MB=8 (NT=4), overflow_mode="full".
        Expectation: rank ``i`` owns ``{m : m % NT == i}`` (each owner
            emits DATA_LOAD + MPIPE_TRANSPOSE_FWD for exactly those micros).
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="full")
        for rank in range(4):
            expected_owned = {m for m in range(8) if m % 4 == rank}
            prefix = _prefix_ops(schedule.exec_order, rank)
            owned_loads = {op.micro_index for op in prefix
                           if op.type == MetaStepType.DATA_LOAD}
            assert owned_loads == expected_owned, (
                f"rank {rank}: DATA_LOAD micros mismatch, "
                f"expected {expected_owned}, got {owned_loads}")
            owned_fwds = {op.micro_index for op in prefix
                          if op.type == MpipeStepType.MPIPE_TRANSPOSE_FWD}
            assert owned_fwds == expected_owned, (
                f"rank {rank}: MPIPE_TRANSPOSE_FWD micros mismatch, "
                f"expected {expected_owned}, got {owned_fwds}")

    def test_balanced_transpose_forward_count(self):
        """
        Feature: MPipe Transpose round-robin overflow.
        Description: PP=4, MB=8 (NT=4), overflow_mode="full".
        Expectation: every owner rank emits exactly M/NT=2
            MPIPE_TRANSPOSE_FWD steps — the ViT phase is balanced.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="full")
        for rank in range(4):
            n_transpose_fwd = _count(
                schedule.exec_order[rank],
                MpipeStepType.MPIPE_TRANSPOSE_FWD,
                stage_index=-1)
            assert n_transpose_fwd == 2, (
                f"rank {rank}: expected 2 prefix MPIPE_TRANSPOSE_FWD "
                f"(balanced ViT), got {n_transpose_fwd}")

    def test_type_major_send_ordering_on_owners(self):
        """
        Feature: MPipe Transpose round-robin overflow (wire ordering).
        Description: PP=4, MB=8 (NT=4), overflow_mode="full". Owning rank>0.
        Expectation: the prefix sends are **type-major** — all DATA_SEND
            steps, then all MPIPE_FWD_SEND steps, then all
            MPIPE_GRAPH_SEND steps. Micro-major sends would drift the
            per-(sender, rank-0) HCCL FIFO past send_object_list bytes on
            the receive side and unpickle a garbage shape.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="full")
        for rank in (1, 2, 3):
            prefix = _prefix_ops(schedule.exec_order, rank)
            send_types = [op.type for op in prefix if op.type in (
                MetaStepType.DATA_SEND,
                MpipeStepType.MPIPE_FWD_SEND,
                MpipeStepType.MPIPE_GRAPH_SEND,
            )]
            # Expected shape: [DATA_SEND × k, MPIPE_FWD_SEND × k, MPIPE_GRAPH_SEND × k]
            n_owned = 2  # M/NT for PP=4, M=8
            expected = ([MetaStepType.DATA_SEND] * n_owned
                        + [MpipeStepType.MPIPE_FWD_SEND] * n_owned
                        + [MpipeStepType.MPIPE_GRAPH_SEND] * n_owned)
            assert send_types == expected, (
                f"rank {rank}: send ordering not type-major, "
                f"expected {expected}, got {send_types}")

    def test_type_major_recv_ordering_on_rank0(self):
        """
        Feature: MPipe Transpose round-robin overflow (wire ordering).
        Description: PP=4, MB=8 (NT=4), overflow_mode="full". Rank 0.
        Expectation: rank 0's prefix receives are **type-major** — all
            DATA_RECV, then all MPIPE_FWD_RECV, then all MPIPE_GRAPH_RECV
            — matching the senders' type-major order so the HCCL FIFO
            stays consistent.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="full")
        prefix = _prefix_ops(schedule.exec_order, 0)
        recv_types = [op.type for op in prefix if op.type in (
            MetaStepType.DATA_RECV,
            MpipeStepType.MPIPE_FWD_RECV,
            MpipeStepType.MPIPE_GRAPH_RECV,
        )]
        # Rank 0 owns {0, 4}; non-owned set is {1, 2, 3, 5, 6, 7} — 6 micros.
        non_owned = 6
        expected = ([MetaStepType.DATA_RECV] * non_owned
                    + [MpipeStepType.MPIPE_FWD_RECV] * non_owned
                    + [MpipeStepType.MPIPE_GRAPH_RECV] * non_owned)
        assert recv_types == expected, (
            f"rank 0: recv ordering not type-major, "
            f"expected {expected}, got {recv_types}")

    def test_no_inline_preprocess_on_rank0(self):
        """
        Feature: MPipe Transpose round-robin overflow.
        Description: PP=4, MB=8 (NT=4), overflow_mode="full".
        Expectation: rank 0's body has **no** inline MPIPE_TRANSPOSE_FWD
            (every overflow micro is transposed on some owner rank via
            round-robin), unlike "min" mode which inlines the overflow
            micros' preprocess on rank 0 alongside their body FWD.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="full")
        # No inline preprocess forward with stage_index=0 (body) — every
        # MPIPE_TRANSPOSE_FWD is in the prefix (stage_index=-1).
        inline_fwds = [op for op in schedule.exec_order[0]
                       if op is not None
                       and op.type == MpipeStepType.MPIPE_TRANSPOSE_FWD
                       and op.stage_index != -1]
        assert inline_fwds == [], (
            f"full-mode rank 0 should have no inline MPIPE_TRANSPOSE_FWD, "
            f"got {inline_fwds}")

    def test_min_and_full_collapse_when_m_eq_nt(self):
        """
        Feature: MPipe Transpose round-robin overflow.
        Description: PP=4, MB=4 (NT=4) — no overflow.
        Expectation: overflow_mode="full" and overflow_mode="min" emit
            byte-identical exec_orders per rank.
        """
        full = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=4,
            overflow_mode="full")
        minm = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=4,
            overflow_mode="min")
        for rank in range(4):
            assert full.exec_order[rank] == minm.exec_order[rank], (
                f"rank {rank}: full/min diverge when M=NT (should be identity), "
                f"full={full.exec_order[rank]}, min={minm.exec_order[rank]}")


class TestMPipeTransposeMinOverflow(unittest.TestCase):
    """``overflow_mode="min"`` (legacy): overflow micros ``NT..M-1`` load
    inline on rank 0's body; ranks ``1..NT-1`` each own only their single
    transposed micro. Saves inter-rank P2P at the cost of a longer rank-0
    ViT phase."""

    def test_min_inline_preprocess_on_overflow_micros(self):
        """
        Feature: MPipe Transpose min-overflow mode.
        Description: PP=4, MB=8 (NT=4), overflow_mode="min".
        Expectation: rank 0's body inlines DATA_LOAD + MPIPE_TRANSPOSE_FWD
            for every overflow micro ``NT..M-1``, immediately preceding
            that micro's stage-0 body FWD.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="min")
        order = schedule.exec_order[0]
        for micro in range(4, 8):  # overflow micros NT..M-1
            fwd_index = _index_of(order, MetaStep(micro, MetaStepType.FWD, 0))
            # The two steps immediately preceding stage-0 body FWD should
            # be DATA_LOAD then MPIPE_TRANSPOSE_FWD (both with stage_index=-1).
            assert order[fwd_index - 2] == MetaStep(
                micro, MetaStepType.DATA_LOAD, -1), (
                f"micro {micro}: expected inline DATA_LOAD at fwd_index-2, "
                f"got {order[fwd_index - 2]}")
            assert order[fwd_index - 1] == MetaStep(
                micro, MpipeStepType.MPIPE_TRANSPOSE_FWD, -1), (
                f"micro {micro}: expected inline MPIPE_TRANSPOSE_FWD at "
                f"fwd_index-1, got {order[fwd_index - 1]}")

    def test_min_owners_ship_only_their_own_micro(self):
        """
        Feature: MPipe Transpose min-overflow mode.
        Description: PP=4, MB=8 (NT=4), overflow_mode="min".
        Expectation: non-stage-0 owning ranks ``1..NT-1`` emit exactly one
            prefix DATA_LOAD / MPIPE_TRANSPOSE_FWD / DATA_SEND /
            MPIPE_FWD_SEND for their single transposed micro (rank i owns
            {i}) — no round-robin overflow ships.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="min")
        for rank in (1, 2, 3):
            prefix = _prefix_ops(schedule.exec_order, rank)
            for step_type in (MetaStepType.DATA_LOAD,
                              MpipeStepType.MPIPE_TRANSPOSE_FWD,
                              MetaStepType.DATA_SEND,
                              MpipeStepType.MPIPE_FWD_SEND):
                count = sum(1 for op in prefix if op.type == step_type)
                assert count == 1, (
                    f"min-mode rank {rank}: expected exactly one prefix "
                    f"{step_type}, got {count}")


class TestMPipeTransposeDataRoutingTable(unittest.TestCase):
    """``_data_dst`` / ``_data_src`` populated by ``construct_exec_order``
    for the DATA_SEND/RECV handlers. Under the round-robin change the
    tables are indexed by full micro range (length ``M``) so the body's
    per-stage DATA_SEND from an overflow micro doesn't ``IndexError`` when
    stage 0 forwards it to stage 1."""

    def test_dst_table_length_is_m_not_nt(self):
        """
        Feature: MPipe Transpose routing table.
        Description: PP=4, MB=8 (NT=4), overflow_mode="full".
        Expectation: every ``_data_dst[stage_index]`` entry has length M
            (not NT), so body-level DATA_SEND for overflow micros indexes
            without going out of range.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="full")
        # pylint: disable=protected-access
        for stage_index, dst in schedule._data_dst.items():
            assert len(dst) == 8, (
                f"_data_dst[{stage_index}] length {len(dst)} != M=8")
        for stage_index, src in schedule._data_src.items():
            assert len(list(src)) == 8, (
                f"_data_src[{stage_index}] length {len(list(src))} != M=8")

    def test_full_mode_prefix_source_is_round_robin(self):
        """
        Feature: MPipe Transpose routing table (full mode).
        Description: PP=4, MB=8 (NT=4), overflow_mode="full".
        Expectation: ``_data_src[-1][m] == m % NT`` for every micro
            (rank 0 receives micro m from the round-robin owner).
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="full")
        # pylint: disable=protected-access
        src_table = list(schedule._data_src[-1])
        expected = [m % 4 for m in range(8)]
        assert src_table == expected, (
            f"full-mode prefix _data_src mismatch: "
            f"expected {expected}, got {src_table}")

    def test_min_mode_prefix_source_folds_overflow_to_rank0(self):
        """
        Feature: MPipe Transpose routing table (min mode).
        Description: PP=4, MB=8 (NT=4), overflow_mode="min".
        Expectation: ``_data_src[-1][m] == m`` for m < NT, and ``0`` for
            m >= NT — overflow micros come from rank 0's inline load, so
            they aren't received from any other rank.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=8,
            overflow_mode="min")
        # pylint: disable=protected-access
        src_table = list(schedule._data_src[-1])
        expected = [m if m < 4 else 0 for m in range(8)]
        assert src_table == expected, (
            f"min-mode prefix _data_src mismatch: "
            f"expected {expected}, got {src_table}")


class TestMPipeTransposeOwnerBackwardOrder(unittest.TestCase):
    """``construct_exec_order`` under owner-backward: the feature-grad ship-back
    replaces the stage-0-backward input transport + stage-0 backward."""

    def test_owner_backward_step_layout(self):
        """
        Feature: MPipe owner-backward full schedule construction.
        Description: Build a trainable PP=4 / MB=4 schedule with owner_backward and
            inspect the per-rank MPIPE_* layout.
        Expectation: no GRAPH_SEND/GRAPH_RECV/TRANSPOSE_BWD anywhere; a GRAD_SEND
            appears; owner ranks 1..NT-1 end with GRAD_RECV_WITH_BACKWARD then GRAD_REDUCE; every
            rank ends with exactly one GRAD_REDUCE; stage 0 has no GRAD_RECV_WITH_BACKWARD suffix.
        """
        sched = _build_mpipe(4, 1, 4, num_transpose_layers=2, owner_backward=True)
        types = [s.type for order in sched.exec_order.values() for s in order if s is not None]
        for absent in (MpipeStepType.MPIPE_GRAPH_SEND,
                       MpipeStepType.MPIPE_GRAPH_RECV,
                       MpipeStepType.MPIPE_TRANSPOSE_BWD):
            assert absent not in types, f"{absent} should be gone under owner-backward"
        assert MpipeStepType.MPIPE_GRAD_SEND in types
        for rank in (1, 2, 3):  # owners of transposed micros 1..3
            tail = [s.type for s in sched.exec_order[rank][-2:]]
            assert tail == [MpipeStepType.MPIPE_GRAD_RECV_WITH_BACKWARD, MpipeStepType.MPIPE_GRAD_REDUCE], \
                f"rank {rank} cooldown tail = {tail}"
        for order in sched.exec_order.values():
            reduces = [s for s in order if s is not None and s.type == MpipeStepType.MPIPE_GRAD_REDUCE]
            assert len(reduces) == 1
        assert sched.exec_order[0][-1].type == MpipeStepType.MPIPE_GRAD_REDUCE
        rank0_types = [s.type for s in sched.exec_order[0] if s is not None]
        assert MpipeStepType.MPIPE_GRAD_RECV_WITH_BACKWARD not in rank0_types

    def test_owner_backward_off_keeps_stage0_backward_order(self):
        """
        Feature: MPipe owner-backward opt-in.
        Description: With owner_backward off, the schedule keeps the stage-0-backward
            ordering (GRAPH_SEND/RECV + TRANSPOSE_BWD, no GRAD_* steps).
        Expectation: stage-0-backward steps present; owner-backward steps absent.
        """
        sched = _build_mpipe(4, 1, 4, num_transpose_layers=2, owner_backward=False)
        types = [s.type for order in sched.exec_order.values() for s in order if s is not None]
        assert MpipeStepType.MPIPE_TRANSPOSE_BWD in types
        assert MpipeStepType.MPIPE_GRAPH_SEND in types
        for absent in (MpipeStepType.MPIPE_GRAD_SEND, MpipeStepType.MPIPE_GRAD_RECV_WITH_BACKWARD,
                       MpipeStepType.MPIPE_GRAD_REDUCE):
            assert absent not in types


class TestMPipeTransposeDataKeys(unittest.TestCase):
    """``_DATA_KEYS`` pixels policy: ``pixel_values`` leaves the wire only for a
    frozen, fully-transposed visual tower."""

    _VL_SPEC = {"attention_mask": 0, "pixel_values": 0, "targets": 0}

    def _keys(self, num_transpose_layers, num_visual_layers, has_trainable):
        sched = _build_mpipe(4, 1, 4, num_transpose_layers=num_transpose_layers,
                             num_visual_layers=num_visual_layers,
                             has_trainable=has_trainable,
                             kwargs_batch_dim=self._VL_SPEC)
        return sched._DATA_KEYS  # pylint: disable=protected-access

    def test_frozen_fully_transposed_drops_pixels(self):
        """
        Feature: MPipe DATA wire schema.
        Description: Frozen tower with T == num_visual_layers (the shipped
            ``pp_mpipe_transpose_layers: visual`` + freeze default) -- pixels are
            consumed on the owning rank, stage 0 only receives features.
        Expectation: ``pixel_values`` leaves ``_DATA_KEYS``; the other keys stay.
        """
        keys = self._keys(4, 4, has_trainable=False)
        assert "pixel_values" not in keys
        assert "input_ids" in keys and "attention_mask" in keys and "targets" in keys

    def test_trainable_fully_transposed_keeps_pixels(self):
        """
        Feature: MPipe DATA wire schema.
        Description: TRAINABLE tower with T == num_visual_layers -- the stage-0
            backward recomputes the tower forward from the ctx kwargs, which
            DATA_RECV fills under ``data.load: single``.
        Expectation: ``pixel_values`` stays in ``_DATA_KEYS`` (this PR's fix).
        """
        assert "pixel_values" in self._keys(4, 4, has_trainable=True)

    def test_dataload_only_keeps_pixels(self):
        """
        Feature: MPipe DATA wire schema.
        Description: Dataload-only mode (T == 0) -- stage 0 runs the visual
            tower itself on the received micros.
        Expectation: ``pixel_values`` stays in ``_DATA_KEYS``.
        """
        assert "pixel_values" in self._keys(0, 4, has_trainable=False)

    def test_unknown_visual_depth_keeps_pixels(self):
        """
        Feature: MPipe DATA wire schema.
        Description: ``num_visual_layers`` is ``None`` (schedule built without
            the param, e.g. a non-VL model).
        Expectation: conservative -- ``pixel_values`` stays in ``_DATA_KEYS``.
        """
        assert "pixel_values" in self._keys(2, None, has_trainable=False)


if __name__ == "__main__":
    unittest.main()
