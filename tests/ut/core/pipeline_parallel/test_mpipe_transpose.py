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
})


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
                 num_transpose_layers=2, explicit_nontransposed_backward=False):
    """Build a ScheduleMPipeTranspose with ordering-only state populated.

    ``explicit_nontransposed_backward`` mirrors the MindSpore path (recompute
    backward for every micro-batch); the default ``False`` is the torch path.
    """
    schedule = object.__new__(ScheduleMPipeTranspose)
    _set_interleaved_attrs(schedule, real_stage_num, n_local_stages, micro_batch_num)
    # pylint: disable=protected-access
    schedule._num_transpose_layers = num_transpose_layers
    # For the text-style cases the ordering tests cover, "has trainable preprocess"
    # is equivalent to T>0 (the transposed layers are trainable; T=0 is identity).
    schedule._has_trainable_preprocess = num_transpose_layers > 0
    schedule._explicit_nontransposed_backward = explicit_nontransposed_backward
    schedule.construct_exec_order()
    return schedule


def _build_interleaved_body(real_stage_num, n_local_stages, micro_batch_num):
    """Build the reference (plain Interleaved 1F1B) body order for comparison."""
    schedule = object.__new__(ScheduleInterleaved1F1B)
    _set_interleaved_attrs(schedule, real_stage_num, n_local_stages, micro_batch_num)
    schedule.construct_exec_order()
    return schedule.exec_order


def _strip_mpipe(order):
    """Drop every MPIPE_* step, leaving the underlying body order (bubbles kept)."""
    return [step for step in order if step is None or step.type not in _MPIPE_STEP_TYPES]


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
        Expectation: stage 0's prefix receives the output + input of the three
            other transposed micro-batches; each other rank ships its own.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=1, micro_batch_num=4)

        expected_rank0 = [
            MetaStep(None, MpipeStepType.MPIPE_PARAM_BROADCAST, 0),
            MetaStep(0, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0),
            MetaStep(1, MpipeStepType.MPIPE_FWD_RECV, 0),
            MetaStep(2, MpipeStepType.MPIPE_FWD_RECV, 0),
            MetaStep(3, MpipeStepType.MPIPE_FWD_RECV, 0),
            MetaStep(1, MpipeStepType.MPIPE_GRAPH_RECV, 0),
            MetaStep(2, MpipeStepType.MPIPE_GRAPH_RECV, 0),
            MetaStep(3, MpipeStepType.MPIPE_GRAPH_RECV, 0),
        ]
        assert schedule.exec_order[0][:len(expected_rank0)] == expected_rank0, \
            (f"rank 0 prefix mismatch: "
             f"expected={expected_rank0}, got={schedule.exec_order[0][:len(expected_rank0)]}")

        for rank in (1, 2, 3):
            expected = [
                MetaStep(None, MpipeStepType.MPIPE_PARAM_BROADCAST, 0),
                MetaStep(rank, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0),
                MetaStep(rank, MpipeStepType.MPIPE_FWD_SEND, 0),
                MetaStep(rank, MpipeStepType.MPIPE_GRAPH_SEND, 0),
            ]
            assert schedule.exec_order[rank][:len(expected)] == expected, \
                (f"rank {rank} prefix mismatch: "
                 f"expected={expected}, got={schedule.exec_order[rank][:len(expected)]}")

    def test_short_micro_prefix(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: PP=4, MB=2 (NT=2) so only ranks 0,1 are transposed.
        Expectation: ranks 2,3 only participate in the parameter broadcast and
            emit no transpose forward / send.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=1, micro_batch_num=2)

        expected_rank0 = [
            MetaStep(None, MpipeStepType.MPIPE_PARAM_BROADCAST, 0),
            MetaStep(0, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0),
            MetaStep(1, MpipeStepType.MPIPE_FWD_RECV, 0),
            MetaStep(1, MpipeStepType.MPIPE_GRAPH_RECV, 0),
        ]
        assert schedule.exec_order[0][:len(expected_rank0)] == expected_rank0, \
            (f"rank 0 short-micro prefix mismatch: "
             f"expected={expected_rank0}, got={schedule.exec_order[0][:len(expected_rank0)]}")

        expected_rank1 = [
            MetaStep(None, MpipeStepType.MPIPE_PARAM_BROADCAST, 0),
            MetaStep(1, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0),
            MetaStep(1, MpipeStepType.MPIPE_FWD_SEND, 0),
            MetaStep(1, MpipeStepType.MPIPE_GRAPH_SEND, 0),
        ]
        assert schedule.exec_order[1][:len(expected_rank1)] == expected_rank1, \
            (f"rank 1 short-micro prefix mismatch: "
             f"expected={expected_rank1}, got={schedule.exec_order[1][:len(expected_rank1)]}")

        for rank in (2, 3):
            assert schedule.exec_order[rank][0] == MetaStep(None, MpipeStepType.MPIPE_PARAM_BROADCAST, 0), \
                (f"rank {rank} should start with a broadcast, "
                 f"got {schedule.exec_order[rank][0]}")
            # Non-transposed ranks contribute no transpose forward / send.
            assert _count(schedule.exec_order[rank], MpipeStepType.MPIPE_TRANSPOSE_FWD) == 0, \
                f"rank {rank} should run no transpose forward when not transposed"
            assert _count(schedule.exec_order[rank], MpipeStepType.MPIPE_FWD_SEND) == 0, \
                f"rank {rank} should issue no MPIPE_FWD_SEND when not transposed"

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
    """Inline preprocess forward (non-transposed) and recompute backward
    (transposed) inserted into stage 0's body order."""

    def test_transpose_bwd_follows_stage0_bwd(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering.
        Description: Inspect stage 0's body order for PP=4, n_local=2, MB=8.
        Expectation: each transposed micro-batch gets an MPIPE_TRANSPOSE_BWD
            immediately after its stage-0 body backward.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=2, micro_batch_num=8)
        order = schedule.exec_order[0]
        num_transpose = 4
        for micro in range(num_transpose):
            bwd_index = _index_of(order, MetaStep(micro, MetaStepType.BWD, 0))
            following = order[bwd_index + 1]
            assert following == MetaStep(micro, MpipeStepType.MPIPE_TRANSPOSE_BWD, 0), \
                (f"micro {micro}: expected MPIPE_TRANSPOSE_BWD right after BWD(stage 0), "
                 f"got {following}")
        assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD) == num_transpose, \
            (f"expected {num_transpose} transpose backwards on stage 0, got "
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
            assert preceding == MetaStep(micro, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0), \
                (f"micro {micro}: expected inline MPIPE_TRANSPOSE_FWD right before "
                 f"FWD(stage 0), got {preceding}")

        for micro in range(num_transpose):
            fwd_index = _index_of(order, MetaStep(micro, MetaStepType.FWD, 0))
            preceding = order[fwd_index - 1]
            assert preceding != MetaStep(micro, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0), \
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


class TestMPipeTransposeExplicitBackward(unittest.TestCase):
    """MindSpore-style path: an explicit recompute backward for every micro-batch."""

    def test_transpose_bwd_emitted_for_all_micros(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (MindSpore path).
        Description: Build the schedule with explicit_nontransposed_backward=True.
        Expectation: stage 0 gets one MPIPE_TRANSPOSE_BWD after every body-0
            backward, including the non-transposed micro-batches.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=2, micro_batch_num=8,
            explicit_nontransposed_backward=True)
        order = schedule.exec_order[0]
        for micro in range(8):
            bwd_index = _index_of(order, MetaStep(micro, MetaStepType.BWD, 0))
            following = order[bwd_index + 1]
            assert following == MetaStep(micro, MpipeStepType.MPIPE_TRANSPOSE_BWD, 0), \
                (f"micro {micro}: expected MPIPE_TRANSPOSE_BWD after BWD(stage 0) under "
                 f"explicit-backward, got {following}")
        assert _count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD) == 8, \
            (f"explicit-backward should emit one transpose backward per micro-batch, got "
             f"{_count(order, MpipeStepType.MPIPE_TRANSPOSE_BWD)}")

    def test_torch_path_unaffected(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (torch path).
        Description: Build the default schedule (no explicit backward) for MB=8.
        Expectation: transpose backward is emitted only for the NT transposed
            micro-batches.
        """
        schedule = _build_mpipe(real_stage_num=4, n_local_stages=2, micro_batch_num=8)
        assert _count(schedule.exec_order[0], MpipeStepType.MPIPE_TRANSPOSE_BWD) == 4, \
            "torch path must emit transpose backward only for the NT transposed micro-batches"


class TestMPipeTransposeDataloadOnly(unittest.TestCase):
    """``T == 0``: only the dataload is transposed — no parameter broadcast,
    no recompute-input transport, and no recompute backward."""

    def test_no_preprocess_steps_emitted(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (T=0 dataload-only).
        Description: Build the schedule with num_transpose_layers=0.
        Expectation: no param broadcast, recompute-input transport, or transpose
            backward is emitted on any rank.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=4, num_transpose_layers=0)
        for rank in range(4):
            order = schedule.exec_order[rank]
            for absent in (
                MpipeStepType.MPIPE_PARAM_BROADCAST,
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
        Expectation: each transposed rank still loads and ships its micro-batch
            and stage 0 receives the other transposed micro-batches' inputs.
        """
        schedule = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=4, num_transpose_layers=0)
        # Each transposed rank loads + ships its micro-batch; stage 0 receives
        # the three other micro-batches' inputs.
        assert _count(schedule.exec_order[0], MpipeStepType.MPIPE_FWD_RECV) == 3, \
            "T=0 stage 0 should receive one input per other transposed micro-batch"
        assert schedule.exec_order[0][0] == MetaStep(0, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0), \
            (f"T=0 rank 0 should start with the dataload transpose forward, "
             f"got {schedule.exec_order[0][0]}")
        for rank in (1, 2, 3):
            expected = [
                MetaStep(rank, MpipeStepType.MPIPE_TRANSPOSE_FWD, 0),
                MetaStep(rank, MpipeStepType.MPIPE_FWD_SEND, 0),
            ]
            assert schedule.exec_order[rank][:len(expected)] == expected, \
                (f"T=0 rank {rank} prefix mismatch: "
                 f"expected={expected}, got={schedule.exec_order[rank][:len(expected)]}")

    def test_body_preserved(self):
        """
        Feature: MPipe Transpose pipeline schedule ordering (T=0 dataload-only).
        Description: Strip the MPIPE_* steps from the T=0 schedule per rank.
        Expectation: the remaining body order matches the plain Interleaved 1F1B body.
        """
        mpipe = _build_mpipe(
            real_stage_num=4, n_local_stages=1, micro_batch_num=4, num_transpose_layers=0)
        body = _build_interleaved_body(real_stage_num=4, n_local_stages=1, micro_batch_num=4)
        for rank in range(4):
            assert _strip_mpipe(mpipe.exec_order[rank]) == body[rank], \
                f"T=0 rank {rank}: body order changed after MPipe layering"


class TestMPipeTransposeBodyPreserved(unittest.TestCase):
    """Stripping the MPIPE_* steps must recover the plain Interleaved 1F1B body
    exactly — MPipe only layers steps around an unchanged body schedule."""

    def _assert_body_matches(self, real_stage_num, n_local_stages, micro_batch_num):
        mpipe = _build_mpipe(real_stage_num, n_local_stages, micro_batch_num)
        body = _build_interleaved_body(real_stage_num, n_local_stages, micro_batch_num)
        for rank in range(real_stage_num):
            stripped = _strip_mpipe(mpipe.exec_order[rank])
            assert stripped == body[rank], \
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


if __name__ == "__main__":
    unittest.main()
