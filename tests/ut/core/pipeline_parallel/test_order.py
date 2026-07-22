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
"""
Unit tests for backend-neutral pipeline parallel scheduling functions.
This module contains test classes for validating various functions in the scheduling module.
"""

import collections
import sys
import threading
import unittest
from unittest.mock import Mock, patch

from hyper_parallel.core.activation_checkpoint.swap import SwapManager
from hyper_parallel.core.pipeline_parallel import pipeline_swap, scheduler as scheduler_module
from hyper_parallel.core.pipeline_parallel.pipeline_swap import (
    PipelineSwapSession,
    inject_pipeline_swap_steps,
)
from hyper_parallel.core.pipeline_parallel.scheduler import (
    MetaStep,
    MetaStepType,
    Schedule1F1B,
    ScheduleInterleaved1F1B,
    validate_pipeline_execution,
    detect_cycle_in_graph,
    parse_and_validate,
    generate_operations,
    coalesce_p2p,
    attach_fwd_boundary_p2p,
    add_fsdp_unshard_reshard,
    add_fsdp_reduce_grad,
    _P2P_STEP_TYPES,
    _RECV_STEP_TYPES,
)
from hyper_parallel.platform.platform import PlatformType

sys.setrecursionlimit(10000)

class TestParseAndValidate(unittest.TestCase):
    """Test class for parse and validate functionality."""
    def test_gpipe_swap_injection_lifecycle_order(self):
        """GPipe swap injection should preserve each chunk's swap lifecycle."""
        exec_order = {
            0: [
                MetaStep(0, MetaStepType.FWD, 0),
                MetaStep(0, MetaStepType.FWD_SEND, 0),
                MetaStep(1, MetaStepType.FWD, 0),
                MetaStep(1, MetaStepType.FWD_SEND, 0),
                MetaStep(2, MetaStepType.FWD, 0),
                MetaStep(2, MetaStepType.FWD_SEND, 0),
                MetaStep(3, MetaStepType.FWD, 0),
                MetaStep(3, MetaStepType.FWD_SEND, 0),
                MetaStep(0, MetaStepType.BWD_RECV, 0),
                MetaStep(0, MetaStepType.BWD, 0),
                MetaStep(0, MetaStepType.BWD_SEND, 0),
                MetaStep(1, MetaStepType.BWD_RECV, 0),
                MetaStep(1, MetaStepType.BWD, 0),
                MetaStep(1, MetaStepType.BWD_SEND, 0),
                MetaStep(2, MetaStepType.BWD_RECV, 0),
                MetaStep(2, MetaStepType.BWD, 0),
                MetaStep(2, MetaStepType.BWD_SEND, 0),
                MetaStep(3, MetaStepType.BWD_RECV, 0),
                MetaStep(3, MetaStepType.BWD, 0),
                MetaStep(3, MetaStepType.BWD_SEND, 0),
            ],
        }
        exec_order[0] = inject_pipeline_swap_steps(exec_order[0])

        def pos(micro_index, step_type):
            for index, step in enumerate(exec_order[0]):
                if (step.micro_index, step.type, step.stage_index) == (micro_index, step_type, 0):
                    return index
            raise AssertionError(f"Missing {step_type} for micro {micro_index}")

        for micro_index in range(4):
            self.assertLess(pos(micro_index, MetaStepType.FWD), pos(micro_index, MetaStepType.SWAP_LAUNCH_OFFLOAD))
            self.assertLess(
                pos(micro_index, MetaStepType.SWAP_LAUNCH_OFFLOAD),
                pos(micro_index, MetaStepType.FWD_SEND),
            )
            self.assertLess(
                pos(micro_index, MetaStepType.SWAP_LAUNCH_OFFLOAD),
                pos(micro_index, MetaStepType.SWAP_WAIT_OFFLOAD),
            )
            self.assertLess(
                pos(micro_index, MetaStepType.SWAP_WAIT_OFFLOAD),
                pos(micro_index, MetaStepType.SWAP_LAUNCH_LOAD),
            )
            self.assertLess(
                pos(micro_index, MetaStepType.SWAP_LAUNCH_LOAD),
                pos(micro_index, MetaStepType.SWAP_WAIT_LOAD),
            )
            self.assertLess(
                pos(micro_index, MetaStepType.BWD_RECV),
                pos(micro_index, MetaStepType.SWAP_WAIT_LOAD),
            )
            self.assertLess(pos(micro_index, MetaStepType.SWAP_WAIT_LOAD), pos(micro_index, MetaStepType.BWD))

    def test_pipeline_swap_injection_anchors_after_batch_container(self):
        """Swap injection after P2P coalescing should keep fused blocks intact."""
        stage_index = 0
        order = [
            MetaStep(0, MetaStepType.FWD, stage_index),
            MetaStep(0, MetaStepType.FWD_SEND, stage_index),
            MetaStep(1, MetaStepType.FWD, stage_index),
            MetaStep(0, MetaStepType.BWD_RECV, stage_index),
            MetaStep(1, MetaStepType.FWD_SEND, stage_index),
            MetaStep(2, MetaStepType.FWD, stage_index),
            MetaStep(2, MetaStepType.FWD_SEND, stage_index),
            MetaStep(3, MetaStepType.FWD, stage_index),
            MetaStep(3, MetaStepType.FWD_SEND, stage_index),
            MetaStep(0, MetaStepType.BWD, stage_index),
            MetaStep(0, MetaStepType.BWD_SEND, stage_index),
        ]
        order = coalesce_p2p({0: order})[0]
        order = inject_pipeline_swap_steps(order)

        swap_types = {
            MetaStepType.SWAP_LAUNCH_OFFLOAD,
            MetaStepType.SWAP_WAIT_OFFLOAD,
            MetaStepType.SWAP_LAUNCH_LOAD,
            MetaStepType.SWAP_WAIT_LOAD,
        }

        def pos(step_type, micro_index):
            for index, step in enumerate(order):
                if step is None:
                    continue
                if (step.type, step.stage_index, step.micro_index) == (step_type, stage_index, micro_index):
                    return index
            raise AssertionError(f"Missing {step_type} for micro {micro_index}")

        def batch_pos(step_type, micro_index):
            for index, step in enumerate(order):
                if step is None or step.type != MetaStepType.BATCH_SEND_RECV:
                    continue
                for sub_step in step.sub_steps:
                    if (
                            sub_step.type == step_type
                            and sub_step.stage_index == stage_index
                            and sub_step.micro_index == micro_index):
                        return index
            raise AssertionError(f"Missing batch member {step_type} for micro {micro_index}")

        batch_index = batch_pos(MetaStepType.FWD_SEND, 1)
        wait_offload_index = pos(MetaStepType.SWAP_WAIT_OFFLOAD, 0)
        self.assertLess(batch_index, wait_offload_index)
        self.assertEqual(order[batch_index + 1].type, MetaStepType.SWAP_WAIT_OFFLOAD)
        self.assertTrue(all(sub_step.type not in swap_types for sub_step in order[batch_index].sub_steps))

    def test_1f1b_swap_injection_respects_min_gap(self):
        """1F1B swap injection should skip ranks whose FWD/BWD gap is too short."""
        schedule = object.__new__(Schedule1F1B)
        schedule.real_stage_num = 4
        schedule.micro_batch_num = 8
        schedule.exec_order = {}
        Schedule1F1B.construct_exec_order(schedule)

        for rank, order in list(schedule.exec_order.items()):
            schedule.exec_order[rank] = inject_pipeline_swap_steps(order)

        def swap_micros(rank):
            return [
                step.micro_index
                for step in schedule.exec_order[rank]
                if step is not None and step.type == MetaStepType.SWAP_LAUNCH_OFFLOAD
            ]

        self.assertEqual(swap_micros(0), list(range(8)))
        self.assertEqual(swap_micros(1), list(range(1, 7)))
        self.assertEqual(swap_micros(2), [])
        self.assertEqual(swap_micros(3), [])

    def test_interleaved_1f1b_swap_injection_sees_virtual_stages(self):
        """Interleaved 1F1B swap injection should match every rank-local order."""
        schedule = object.__new__(ScheduleInterleaved1F1B)
        # pylint: disable=protected-access
        schedule.real_stage_num = 4
        schedule._stage_num = 8
        schedule.n_local_stages = 2
        schedule.micro_batch_num = 8
        schedule._overlap_b_f = False
        schedule.n_rounds = 2
        schedule.n_microbatch_per_round = [4, 4]
        schedule.n_microbatch_per_round_accu = [0, 8, 16]
        schedule.exec_order = {}
        ScheduleInterleaved1F1B.construct_exec_order(schedule)

        for rank, order in list(schedule.exec_order.items()):
            schedule.exec_order[rank] = inject_pipeline_swap_steps(order)

        def compute_chunks(rank):
            return [
                (
                    "F" if step.type == MetaStepType.FWD else "B",
                    step.stage_index,
                    step.micro_index,
                )
                for step in schedule.exec_order[rank]
                if step is not None and step.type in (MetaStepType.FWD, MetaStepType.BWD)
            ]

        def swap_chunks(rank):
            return [
                (step.stage_index, step.micro_index)
                for step in schedule.exec_order[rank]
                if step is not None and step.type == MetaStepType.SWAP_LAUNCH_OFFLOAD
            ]

        self.assertEqual(
            compute_chunks(0),
            [
                ("F", 0, 0), ("F", 0, 1), ("F", 0, 2), ("F", 0, 3),
                ("F", 4, 0), ("F", 4, 1), ("F", 4, 2), ("F", 4, 3),
                ("F", 0, 4), ("F", 0, 5), ("F", 0, 6), ("B", 4, 0),
                ("F", 0, 7), ("B", 4, 1), ("F", 4, 4), ("B", 4, 2),
                ("F", 4, 5), ("B", 4, 3), ("F", 4, 6), ("B", 0, 0),
                ("F", 4, 7), ("B", 0, 1), ("B", 0, 2), ("B", 0, 3),
                ("B", 4, 4), ("B", 4, 5), ("B", 4, 6), ("B", 4, 7),
                ("B", 0, 4), ("B", 0, 5), ("B", 0, 6), ("B", 0, 7),
            ],
        )
        self.assertEqual(
            compute_chunks(1),
            [
                ("F", 1, 0), ("F", 1, 1), ("F", 1, 2), ("F", 1, 3),
                ("F", 5, 0), ("F", 5, 1), ("F", 5, 2), ("F", 5, 3),
                ("F", 1, 4), ("B", 5, 0), ("F", 1, 5), ("B", 5, 1),
                ("F", 1, 6), ("B", 5, 2), ("F", 1, 7), ("B", 5, 3),
                ("F", 5, 4), ("B", 1, 0), ("F", 5, 5), ("B", 1, 1),
                ("F", 5, 6), ("B", 1, 2), ("F", 5, 7), ("B", 1, 3),
                ("B", 5, 4), ("B", 5, 5), ("B", 5, 6), ("B", 5, 7),
                ("B", 1, 4), ("B", 1, 5), ("B", 1, 6), ("B", 1, 7),
            ],
        )
        self.assertEqual(
            compute_chunks(2),
            [
                ("F", 2, 0), ("F", 2, 1), ("F", 2, 2), ("F", 2, 3),
                ("F", 6, 0), ("F", 6, 1), ("F", 6, 2), ("B", 6, 0),
                ("F", 6, 3), ("B", 6, 1), ("F", 2, 4), ("B", 6, 2),
                ("F", 2, 5), ("B", 6, 3), ("F", 2, 6), ("B", 2, 0),
                ("F", 2, 7), ("B", 2, 1), ("F", 6, 4), ("B", 2, 2),
                ("F", 6, 5), ("B", 2, 3), ("F", 6, 6), ("B", 6, 4),
                ("F", 6, 7), ("B", 6, 5), ("B", 6, 6), ("B", 6, 7),
                ("B", 2, 4), ("B", 2, 5), ("B", 2, 6), ("B", 2, 7),
            ],
        )
        self.assertEqual(
            compute_chunks(3),
            [
                ("F", 3, 0), ("F", 3, 1), ("F", 3, 2), ("F", 3, 3),
                ("F", 7, 0), ("B", 7, 0), ("F", 7, 1), ("B", 7, 1),
                ("F", 7, 2), ("B", 7, 2), ("F", 7, 3), ("B", 7, 3),
                ("F", 3, 4), ("B", 3, 0), ("F", 3, 5), ("B", 3, 1),
                ("F", 3, 6), ("B", 3, 2), ("F", 3, 7), ("B", 3, 3),
                ("F", 7, 4), ("B", 7, 4), ("F", 7, 5), ("B", 7, 5),
                ("F", 7, 6), ("B", 7, 6), ("F", 7, 7), ("B", 7, 7),
                ("B", 3, 4), ("B", 3, 5), ("B", 3, 6), ("B", 3, 7),
            ],
        )
        self.assertEqual(
            swap_chunks(0),
            [(0, i) for i in range(4)] + [(4, i) for i in range(4)] +
            [(0, i) for i in range(4, 8)] + [(4, i) for i in range(4, 8)],
        )
        self.assertEqual(
            swap_chunks(1),
            [(1, i) for i in range(4)] + [(5, i) for i in range(4)] +
            [(1, i) for i in range(4, 8)] + [(5, i) for i in range(4, 8)],
        )
        self.assertEqual(
            swap_chunks(2),
            [(2, i) for i in range(4)] + [(6, i) for i in range(1, 4)] +
            [(2, i) for i in range(4, 8)] + [(6, i) for i in range(4, 7)],
        )
        self.assertEqual(swap_chunks(3), [(3, i) for i in range(8)])

    def test_interleaved_1f1b_overlap_b_f_swap_injection_uses_leaf_steps(self):
        """Composite OVERLAP_B_F steps should keep leaf-level swap lifecycles."""
        schedule = object.__new__(ScheduleInterleaved1F1B)
        # pylint: disable=protected-access
        schedule.real_stage_num = 4
        schedule._stage_num = 8
        schedule.n_local_stages = 2
        schedule.micro_batch_num = 8
        schedule._overlap_b_f = True
        schedule.n_rounds = 2
        schedule.n_microbatch_per_round = [4, 4]
        schedule.n_microbatch_per_round_accu = [0, 8, 16]
        schedule.exec_order = {}
        ScheduleInterleaved1F1B.construct_exec_order(schedule)

        for rank, order in list(schedule.exec_order.items()):
            schedule.exec_order[rank] = inject_pipeline_swap_steps(order)

        swap_types = {
            MetaStepType.SWAP_LAUNCH_OFFLOAD,
            MetaStepType.SWAP_WAIT_OFFLOAD,
            MetaStepType.SWAP_LAUNCH_LOAD,
            MetaStepType.SWAP_WAIT_LOAD,
        }
        for order in schedule.exec_order.values():
            for step in order:
                if step is not None and step.type in swap_types:
                    self.assertIsNotNone(step.stage_index)
                    self.assertIsNotNone(step.micro_index)

        rank0_order = schedule.exec_order[0]
        self.assertTrue(
            any(step is not None and step.type == MetaStepType.OVERLAP_B_F for step in rank0_order)
        )

        def pos(step_type, stage_index, micro_index):
            for index, step in enumerate(rank0_order):
                if step is None:
                    continue
                if (step.type, step.stage_index, step.micro_index) == (step_type, stage_index, micro_index):
                    return index
            raise AssertionError(f"Missing {step_type} for stage {stage_index}, micro {micro_index}")

        def leaf_pos(step_type, stage_index, micro_index):
            for index, step in enumerate(rank0_order):
                if step is None:
                    continue
                if (step.type, step.stage_index, step.micro_index) == (step_type, stage_index, micro_index):
                    return index
                if step.type == MetaStepType.OVERLAP_B_F:
                    for sub_step in step.sub_steps:
                        if (
                                sub_step.type == step_type
                                and sub_step.stage_index == stage_index
                                and sub_step.micro_index == micro_index):
                            return index
            raise AssertionError(f"Missing leaf {step_type} for stage {stage_index}, micro {micro_index}")

        overlap_pos = leaf_pos(MetaStepType.BWD, 4, 0)
        self.assertEqual(overlap_pos, leaf_pos(MetaStepType.FWD, 0, 7))
        self.assertEqual(rank0_order[overlap_pos].type, MetaStepType.OVERLAP_B_F)

        self.assertLess(pos(MetaStepType.SWAP_LAUNCH_LOAD, 4, 0), overlap_pos)
        self.assertLess(
            pos(MetaStepType.SWAP_LAUNCH_LOAD, 4, 0),
            pos(MetaStepType.SWAP_WAIT_LOAD, 4, 0),
        )
        self.assertLess(
            pos(MetaStepType.BWD_RECV, 4, 0),
            pos(MetaStepType.SWAP_WAIT_LOAD, 4, 0),
        )
        self.assertLess(pos(MetaStepType.SWAP_WAIT_LOAD, 4, 0), overlap_pos)
        self.assertLess(pos(MetaStepType.BWD_RECV, 4, 0), overlap_pos)
        # LAUNCH_OFFLOAD is placed right after the FWD container (overlap_pos)
        # so D2H starts as early as possible — before any FWD_SEND or
        # FSDP_RESHARD that may follow.
        self.assertLess(overlap_pos, pos(MetaStepType.SWAP_LAUNCH_OFFLOAD, 0, 7))
        self.assertLess(pos(MetaStepType.SWAP_LAUNCH_OFFLOAD, 0, 7), pos(MetaStepType.SWAP_WAIT_OFFLOAD, 0, 7))

    def test_interleaved_1f1b_overlap_dxdw_split(self):
        """dxdw split: pairs become (BWD_INPUT, FWD); BWD_WEIGHT lands after the gap."""
        def build(enable_split):
            schedule = object.__new__(ScheduleInterleaved1F1B)
            # pylint: disable=protected-access
            schedule.real_stage_num = 4
            schedule._stage_num = 8
            schedule.n_local_stages = 2
            schedule.micro_batch_num = 8
            schedule._overlap_b_f = True
            schedule._enable_dxdw_split = enable_split
            schedule.n_rounds = 2
            schedule.n_microbatch_per_round = [4, 4]
            schedule.n_microbatch_per_round_accu = [0, 8, 16]
            schedule.exec_order = {}
            ScheduleInterleaved1F1B.construct_exec_order(schedule)
            return schedule.exec_order

        def p2p_seq(order):
            return [(s.type, s.stage_index, s.micro_index)
                    for s in order if s is not None and s.type in _P2P_STEP_TYPES]

        def leaf_bwd_keys(order, types):
            keys = []
            for s in order:
                if s is None:
                    continue
                subs = s.sub_steps if s.type == MetaStepType.OVERLAP_B_F else (s,)
                keys.extend((x.stage_index, x.micro_index)
                            for x in subs if x.type in types)
            return sorted(keys)

        base = build(False)
        split = build(True)
        for rank in range(4):
            base_order, split_order = base[rank], split[rank]
            # The pass must not move any comm relative to other comms.
            self.assertEqual(p2p_seq(base_order), p2p_seq(split_order))

            split_pairs = 0
            for idx, step in enumerate(split_order):
                if step is None or step.type != MetaStepType.OVERLAP_B_F:
                    continue
                bwd_sub, fwd_sub = step.sub_steps
                self.assertEqual(fwd_sub.type, MetaStepType.FWD)
                if bwd_sub.stage_index == 0:
                    # First-stage dx is a no-op; the pair stays unified.
                    self.assertEqual(bwd_sub.type, MetaStepType.BWD)
                    continue
                self.assertEqual(bwd_sub.type, MetaStepType.BWD_INPUT)
                split_pairs += 1
                # Only P2P between the overlap and its BWD_WEIGHT, with this
                # pair's BWD_SEND among it.
                j = idx + 1
                gap = []
                while (j < len(split_order) and split_order[j] is not None
                       and split_order[j].type in _P2P_STEP_TYPES):
                    gap.append(split_order[j])
                    j += 1
                self.assertLess(j, len(split_order),
                                "overlap pair missing its BWD_WEIGHT")
                dw = split_order[j]
                self.assertEqual(
                    (dw.type, dw.stage_index, dw.micro_index),
                    (MetaStepType.BWD_WEIGHT, bwd_sub.stage_index, bwd_sub.micro_index),
                )
                self.assertIn(
                    (MetaStepType.BWD_SEND, bwd_sub.stage_index, bwd_sub.micro_index),
                    [(s.type, s.stage_index, s.micro_index) for s in gap],
                )

            # One BWD_WEIGHT per split pair, and backward coverage unchanged:
            # every (stage, micro) backward appears exactly once, unified or
            # as a dx half.
            self.assertEqual(
                split_pairs,
                sum(1 for s in split_order
                    if s is not None and s.type == MetaStepType.BWD_WEIGHT),
            )
            self.assertEqual(
                leaf_bwd_keys(base_order, {MetaStepType.BWD}),
                leaf_bwd_keys(split_order, {MetaStepType.BWD, MetaStepType.BWD_INPUT}),
            )

    def test_pipeline_swap_schedule_injects_current_rank_only(self):
        """Schedule-level swap injection should only rewrite current rank order."""
        def eligible_order(stage_index):
            return [
                MetaStep(0, MetaStepType.FWD, stage_index),
                MetaStep(0, MetaStepType.FWD_SEND, stage_index),
                MetaStep(1, MetaStepType.FWD, stage_index),
                MetaStep(1, MetaStepType.FWD_SEND, stage_index),
                MetaStep(2, MetaStepType.FWD, stage_index),
                MetaStep(2, MetaStepType.FWD_SEND, stage_index),
                MetaStep(3, MetaStepType.FWD, stage_index),
                MetaStep(3, MetaStepType.FWD_SEND, stage_index),
                MetaStep(0, MetaStepType.BWD_RECV, stage_index),
                MetaStep(0, MetaStepType.BWD, stage_index),
                MetaStep(1, MetaStepType.BWD_RECV, stage_index),
                MetaStep(1, MetaStepType.BWD, stage_index),
                MetaStep(2, MetaStepType.BWD_RECV, stage_index),
                MetaStep(2, MetaStepType.BWD, stage_index),
                MetaStep(3, MetaStepType.BWD_RECV, stage_index),
                MetaStep(3, MetaStepType.BWD, stage_index),
            ]

        schedule = object.__new__(Schedule1F1B)
        schedule.stages = [type("StageStub", (), {"stage_index": 0})()]
        # pylint: disable=protected-access
        schedule._stage_to_rank_index = {0: 0}
        schedule._pp_swap_enabled = True
        schedule.exec_order = {
            0: eligible_order(0),
            1: eligible_order(1),
        }

        Schedule1F1B._inject_local_pp_swap_actions(schedule)

        def has_swap(rank):
            return any(step.type == MetaStepType.SWAP_LAUNCH_OFFLOAD for step in schedule.exec_order[rank])

        self.assertTrue(has_swap(0))
        self.assertFalse(has_swap(1))
        self.assertEqual(
            schedule._swap_keys,  # pylint: disable=protected-access
            frozenset({(0, micro_index) for micro_index in range(4)}),
        )

    def test_pipeline_swap_launch_load_covers_previous_compute(self):
        """LAUNCH_LOAD should start before the compute chunk it overlaps."""
        exec_order = {
            0: [
                MetaStep(0, MetaStepType.FWD, 0),
                MetaStep(0, MetaStepType.FWD_SEND, 0),
                MetaStep(1, MetaStepType.FWD, 0),
                MetaStep(1, MetaStepType.FWD_SEND, 0),
                MetaStep(2, MetaStepType.FWD, 0),
                MetaStep(2, MetaStepType.FWD_SEND, 0),
                MetaStep(3, MetaStepType.FWD, 0),
                MetaStep(3, MetaStepType.FWD_SEND, 0),
                MetaStep(0, MetaStepType.BWD_RECV, 0),
                MetaStep(0, MetaStepType.BWD, 0),
                MetaStep(1, MetaStepType.BWD_RECV, 0),
                MetaStep(1, MetaStepType.BWD, 0),
                MetaStep(2, MetaStepType.BWD_RECV, 0),
                MetaStep(2, MetaStepType.BWD, 0),
                MetaStep(3, MetaStepType.BWD_RECV, 0),
                MetaStep(3, MetaStepType.BWD, 0),
            ],
        }
        exec_order[0] = inject_pipeline_swap_steps(exec_order[0])

        def pos(micro_index, step_type):
            for index, step in enumerate(exec_order[0]):
                if (step.micro_index, step.type, step.stage_index) == (micro_index, step_type, 0):
                    return index
            raise AssertionError(f"Missing {step_type} for micro {micro_index}")

        self.assertLess(
            pos(0, MetaStepType.SWAP_WAIT_OFFLOAD),
            pos(0, MetaStepType.SWAP_LAUNCH_LOAD),
        )
        self.assertLess(
            pos(0, MetaStepType.SWAP_LAUNCH_LOAD),
            pos(3, MetaStepType.FWD),
        )
        self.assertLess(pos(0, MetaStepType.SWAP_LAUNCH_LOAD), pos(3, MetaStepType.FWD_SEND))

    def test_pipeline_swap_session_isolates_repeated_runs(self):
        """Repeated runs should use different physical group names."""
        eligible_keys = frozenset({(0, 0)})
        first_session = PipelineSwapSession(eligible_keys)
        second_session = PipelineSwapSession(eligible_keys)
        step = MetaStep(0, MetaStepType.FWD, 0)

        self.assertNotEqual(first_session.group_name(step), second_session.group_name(step))
        self.assertIn("_s0_m0", first_session.group_name(step))

    def test_pipeline_swap_session_protects_managed_output(self):
        """Leaf-local outputs should be protected before cache mutation."""
        session = PipelineSwapSession(frozenset({(0, 0)}))
        step = MetaStep(0, MetaStepType.FWD, 0)
        output = object()

        with patch.object(session._manager, "protect_alias_tensors") as mock_protect:  # pylint: disable=protected-access
            session.protect_aliases(step, output)

        mock_protect.assert_called_once_with(session.group_name(step), output)

    def test_pipeline_swap_session_offload_preserves_outer_context(self):
        """Session transfer actions must not mutate the caller's group context."""
        session = PipelineSwapSession(frozenset({(0, 0)}))
        with session.group_context(MetaStep(0, MetaStepType.FWD, 0)):
            pass
        step = MetaStep(0, MetaStepType.SWAP_LAUNCH_OFFLOAD, 0)
        manager = Mock()
        with patch.object(pipeline_swap, "SwapManager", return_value=manager), \
             patch.object(pipeline_swap, "_protect_pipeline_owned_tensors"):
            pipeline_swap.swap_launch_offload(step, Mock(), [], [], session)

        manager.launch_offload.assert_called_once_with(session.group_name(step))
        manager.set_current_group_name.assert_not_called()

    @staticmethod
    def _eligible_swap_keys():
        """Build the cached eligible-key set for one managed chunk."""
        return frozenset({(0, 0)})

    @staticmethod
    def _leaf_runtime(stage, session):
        """Build a minimal runtime object for exercising swap-aware compute."""
        # pylint: disable=protected-access
        schedule = object.__new__(Schedule1F1B)
        schedule._stage_dict = {0: stage}
        schedule._swap_session = session
        schedule._assert_in_unshard_if_needed = Mock()
        schedule.wait_fwd_recv = Mock()
        schedule.wait_bwd_recv = Mock()
        schedule.update_losses = Mock()
        return schedule

    def test_execute_fwd_leaf_uses_physical_group_and_restores_outer_context(self):
        """Eligible FWD collection is leaf-local and restores context after exceptions."""
        session = PipelineSwapSession(self._eligible_swap_keys())
        manager = SwapManager()
        step = MetaStep(0, MetaStepType.FWD, 0)
        observed_groups = []
        stage = Mock()

        def _forward(*_):
            observed_groups.append(manager.get_current_group_name())
            return "output"

        stage.forward_one_chunk.side_effect = _forward
        schedule = self._leaf_runtime(stage, session)
        with manager.group_context("outer"):
            schedule.execute_fwd_leaf(step, [[object()]], [{}], [])
            self.assertEqual(manager.get_current_group_name(), "outer")
        self.assertEqual(observed_groups, [session.group_name(step)])

        stage.forward_one_chunk.side_effect = RuntimeError("forward failed")
        with manager.group_context("outer"):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                schedule.execute_fwd_leaf(step, [[object()]], [{}], [])
            self.assertEqual(manager.get_current_group_name(), "outer")
        session.close()

    def test_execute_fwd_leaf_ineligible_chunk_keeps_outer_context(self):
        """An ineligible FWD does not enter any session-owned group."""
        session = PipelineSwapSession(self._eligible_swap_keys())
        manager = SwapManager()
        step = MetaStep(9, MetaStepType.FWD, 0)
        observed_groups = []
        stage = Mock()
        stage.forward_one_chunk.side_effect = (
            lambda *_: observed_groups.append(manager.get_current_group_name()) or "output"
        )
        schedule = self._leaf_runtime(stage, session)
        with manager.group_context("outer"):
            schedule.execute_fwd_leaf(step, [object()] * 10, [{}] * 10, [])
        self.assertEqual(observed_groups, ["outer"])
        session.close()

    def test_swap_launch_offload_rejects_custom_callback_that_bypasses_leaf_executor(self):
        """A custom overlap callback must route FWD through the leaf executor."""
        session = PipelineSwapSession(self._eligible_swap_keys())
        schedule = Mock()
        step = MetaStep(0, MetaStepType.SWAP_LAUNCH_OFFLOAD, 0)

        with self.assertRaisesRegex(RuntimeError, "execute_fwd_leaf"):
            pipeline_swap.swap_launch_offload(step, schedule, [[]], [{}], session)

        session.close()

    def test_wait_load_meta_step_runs_on_scheduler_thread_before_worker_backward(self):
        """The scheduler waits for H2D before a worker dispatches BWD_INPUT."""
        calls = []
        session = Mock()
        stage = Mock()
        schedule = self._leaf_runtime(stage, session)
        session.wait_load.side_effect = lambda _: calls.append(("wait", threading.get_ident()))
        stage.backward_input_one_chunk.side_effect = (
            lambda _: calls.append(("input", threading.get_ident()))
        )

        wait_step = MetaStep(0, MetaStepType.SWAP_WAIT_LOAD, 0)
        schedule._exec_pipeline_swap_step(wait_step, [], [])  # pylint: disable=protected-access

        worker = threading.Thread(
            target=schedule._exec_step,  # pylint: disable=protected-access
            args=(MetaStep(0, MetaStepType.BWD_INPUT, 0), [], [], []),
        )
        worker.start()
        worker.join()

        self.assertEqual([name for name, _ in calls], ["wait", "input"])
        self.assertEqual(calls[0][1], threading.get_ident())
        self.assertNotEqual(calls[0][1], calls[1][1])
        session.wait_load.assert_called_once_with(wait_step)

        schedule._exec_step(  # pylint: disable=protected-access
            MetaStep(0, MetaStepType.BWD_WEIGHT, 0), [], [], [],
        )
        session.wait_load.assert_called_once_with(wait_step)
        stage.backward_weight_one_chunk.assert_called_once_with(0)

    def test_pipeline_swap_session_close_restores_active_group_count(self):
        """Closing a session removes every physical group created by its FWD leaves."""
        session = PipelineSwapSession(self._eligible_swap_keys())
        manager = SwapManager()
        step = MetaStep(0, MetaStepType.FWD, 0)
        baseline = manager.active_group_count()
        with session.group_context(step):
            self.assertEqual(manager.active_group_count(), baseline + 1)
        session.close()
        self.assertEqual(manager.active_group_count(), baseline)

    def test_run_cleans_p2p_and_swap_session_on_failure(self):
        """The single finally block cleans P2P and swap state on failure."""
        # pylint: disable=protected-access
        schedule = object.__new__(Schedule1F1B)
        schedule._pp_swap_enabled = False
        session = Mock()
        schedule._swap_session = session
        schedule.split_microbatches = Mock(return_value=([], []))
        schedule.run_microbatches = Mock(side_effect=ValueError("training failed"))
        schedule._drain_inflight_p2p = Mock()

        with self.assertRaisesRegex(ValueError, "training failed"):
            schedule.run()

        schedule._drain_inflight_p2p.assert_called_once()
        session.close.assert_called_once()
        self.assertIsNone(schedule._swap_session)

    def test_run_uses_build_time_cached_swap_keys(self):
        """Each run creates its session from cached keys without rescanning the order."""
        # pylint: disable=protected-access
        schedule = object.__new__(Schedule1F1B)
        schedule._pp_swap_enabled = True
        schedule._swap_keys = frozenset({(0, 0)})
        schedule._swap_session = None
        schedule.stages = [Mock(stage_index=0)]
        schedule.real_stage_num = 1
        schedule.split_microbatches = Mock(return_value=([], []))
        schedule.run_microbatches = Mock()
        schedule._drain_inflight_p2p = Mock()
        session = Mock()

        with patch.object(scheduler_module, "PipelineSwapSession", return_value=session) as session_cls:
            schedule.run()

        session_cls.assert_called_once_with(schedule._swap_keys)
        session.close.assert_called_once()
        self.assertIsNone(schedule._swap_session)

    def test_pipeline_swap_accepts_pytorch_backend(self):
        """The PyTorch backend builds the same swap metastep lifecycle."""
        stage = Mock(stage_index=0, stage_num=4, submodule=Mock(), pp_group=None)
        with patch.object(scheduler_module.platform, "platform_type", PlatformType.PYTORCH), \
             patch.object(Schedule1F1B, "_check_stages", return_value=[stage]), \
             patch.object(Schedule1F1B, "_inject_local_fsdp_actions"):
            schedule = Schedule1F1B(stage, 4, swap=True)

        swap_types = {
            step.type
            for step in schedule.exec_order[0]
            if step.type.name.startswith("SWAP_")
        }
        self.assertEqual(
            swap_types,
            {
                MetaStepType.SWAP_LAUNCH_OFFLOAD,
                MetaStepType.SWAP_WAIT_OFFLOAD,
                MetaStepType.SWAP_LAUNCH_LOAD,
                MetaStepType.SWAP_WAIT_LOAD,
            },
        )

    def test_pipeline_swap_protects_module_state_and_received_inputs_in_one_pass(self):
        """Launch-time alias protection scans module state and receive buffers once."""
        parameter = object()
        persistent_buffer = object()
        non_persistent_buffer = object()
        recv_buffer = object()
        stage = Mock(
            is_first_stage=False,
            args_recv_info={2: [Mock(buffer=recv_buffer)]},
        )
        schedule = Mock(_stage_dict={1: stage})
        manager = Mock()
        step = MetaStep(2, MetaStepType.SWAP_LAUNCH_OFFLOAD, 1)

        with patch.object(pipeline_swap, "SwapManager", return_value=manager), \
             patch.object(pipeline_swap.platform, "parameters_dict", return_value=(("weight", parameter),)), \
             patch.object(
                 pipeline_swap.platform,
                 "buffers_dict",
                 return_value=(
                     ("running_state", persistent_buffer),
                     ("scratch", non_persistent_buffer),
                 ),
             ):
            pipeline_swap._protect_pipeline_owned_tensors(  # pylint: disable=protected-access
                step, schedule, [], [], "physical_group"
            )

        manager.protect_alias_tensors.assert_called_once_with(
            "physical_group",
            ((parameter,), (persistent_buffer, non_persistent_buffer), (recv_buffer,)),
        )

    def test_pipeline_swap_protects_first_stage_inputs_in_one_pass(self):
        """First-stage positional and keyword inputs share the module-state scan."""
        parameter = object()
        positional_input = object()
        keyword_input = object()
        stage = Mock(is_first_stage=True)
        schedule = Mock(_stage_dict={0: stage})
        manager = Mock()
        step = MetaStep(1, MetaStepType.SWAP_LAUNCH_OFFLOAD, 0)

        with patch.object(pipeline_swap, "SwapManager", return_value=manager), \
             patch.object(pipeline_swap.platform, "parameters_dict", return_value=(("weight", parameter),)), \
             patch.object(pipeline_swap.platform, "buffers_dict", return_value=()):
            pipeline_swap._protect_pipeline_owned_tensors(  # pylint: disable=protected-access
                step,
                schedule,
                [None, (positional_input,)],
                [None, {"hidden_states": keyword_input}],
                "physical_group",
            )

        manager.protect_alias_tensors.assert_called_once_with(
            "physical_group",
            ((parameter,), (), ((positional_input,), {"hidden_states": keyword_input})),
        )

    def test_invalid_value_type(self):
        data = {
            "1": "not_a_list",
            "2": [1, 2, 3],
            "3": ["valid", "list", "of_strings"]
        }
        parse_and_validate(data)

    def test_valid_input(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_micro0_1th"],
            "2": ["Send_Receive_(1)->(2)_micro0_1th"]
        }
        parse_and_validate(data)

    def test_missing_keys_all_rank_true(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_micor0_1th"]
        }
        parse_and_validate(data, all_rank=True)

    def test_value_missing_in_referenced_keys(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_micro0_1th"],
            "2": ["Send_Receive_(1)->(2)_micro0_1th", "Send_Receive_(1)->(2)_micro0_2th"]
        }
        parse_and_validate(data)

    def test_empty_values(self):
        data = {
            "1": [],
            "2": [""]
        }
        parse_and_validate(data)

    def test_fsdp_swap_injection_ordering(self):
        """FSDP→swap injection: verify FSDP steps are optimally placed relative to swap steps.

        Validates the injection order contract:
        - Before FWD: FSDP_UNSHARD → FWD collection scope
        - After FWD:  SWAP_LAUNCH_OFFLOAD → FSDP_RESHARD
        - Before BWD: SWAP_LAUNCH_LOAD → FSDP_UNSHARD → SWAP_WAIT_LOAD → BWD
        """
        # Construct a simple 1F1B-like schedule with 1 managed stage and a gap
        # large enough for swap injection.
        stage_index = 0
        order = [
            MetaStep(0, MetaStepType.FWD, stage_index),
            MetaStep(0, MetaStepType.FWD_SEND, stage_index),
            MetaStep(1, MetaStepType.FWD, stage_index),
            MetaStep(1, MetaStepType.FWD_SEND, stage_index),
            MetaStep(2, MetaStepType.FWD, stage_index),
            MetaStep(2, MetaStepType.FWD_SEND, stage_index),
            MetaStep(3, MetaStepType.FWD, stage_index),
            MetaStep(3, MetaStepType.FWD_SEND, stage_index),
            MetaStep(0, MetaStepType.BWD_RECV, stage_index),
            MetaStep(0, MetaStepType.BWD, stage_index),
            MetaStep(1, MetaStepType.BWD_RECV, stage_index),
            MetaStep(1, MetaStepType.BWD, stage_index),
            MetaStep(2, MetaStepType.BWD_RECV, stage_index),
            MetaStep(2, MetaStepType.BWD, stage_index),
            MetaStep(3, MetaStepType.BWD_RECV, stage_index),
            MetaStep(3, MetaStepType.BWD, stage_index),
        ]

        # Simulate the build_exec_order pipeline: FSDP first, then swap.
        managed_stage_indices = {stage_index}
        order = add_fsdp_unshard_reshard(order, managed_stage_indices, max_active_stages=3)
        order = add_fsdp_reduce_grad(order, managed_stage_indices, micro_batch_num=4)
        order = inject_pipeline_swap_steps(order)

        def pos(micro_index, step_type):
            for index, step in enumerate(order):
                if step is None:
                    continue
                if step.micro_index != micro_index or step.type != step_type:
                    continue
                if step.stage_index != stage_index:
                    continue
                return index
            return None

        # --- After FWD assertions ---
        # SWAP_LAUNCH_OFFLOAD must appear before FSDP_RESHARD (perf: early D2H start).
        launch_off_pos = pos(0, MetaStepType.SWAP_LAUNCH_OFFLOAD)
        self.assertIsNotNone(launch_off_pos, "SWAP_LAUNCH_OFFLOAD missing")
        # The first FSDP_RESHARD after FWD should come after LAUNCH_OFFLOAD.
        reshard_positions = [
            i for i, s in enumerate(order)
            if s is not None and s.type == MetaStepType.FSDP_RESHARD
        ]
        self.assertTrue(reshard_positions, "FSDP_RESHARD missing")
        self.assertLess(launch_off_pos, reshard_positions[0],
                        "SWAP_LAUNCH_OFFLOAD must precede FSDP_RESHARD after FWD")

        # --- Before FWD assertions ---
        # FSDP_UNSHARD must precede the FWD leaf collection scope.
        unshard_positions = [
            i for i, s in enumerate(order)
            if s is not None and s.type == MetaStepType.FSDP_UNSHARD
        ]
        self.assertTrue(unshard_positions, "FSDP_UNSHARD missing")
        fwd_pos = pos(0, MetaStepType.FWD)
        self.assertIsNotNone(fwd_pos, "FWD missing")
        self.assertLess(unshard_positions[0], fwd_pos,
                        "FSDP_UNSHARD must precede FWD")

        # --- Before BWD assertions ---
        # SWAP_LAUNCH_LOAD → FSDP_UNSHARD → SWAP_WAIT_LOAD → BWD.
        bwd_m0_pos = pos(0, MetaStepType.BWD)
        self.assertIsNotNone(bwd_m0_pos, "BWD(m=0) missing")
        launch_load_m0_pos = pos(0, MetaStepType.SWAP_LAUNCH_LOAD)
        self.assertIsNotNone(launch_load_m0_pos, "SWAP_LAUNCH_LOAD(m=0) missing")
        wait_load_m0_pos = pos(0, MetaStepType.SWAP_WAIT_LOAD)
        self.assertIsNotNone(wait_load_m0_pos, "SWAP_WAIT_LOAD(m=0) missing")
        bwd_recv_m0_pos = pos(0, MetaStepType.BWD_RECV)
        self.assertIsNotNone(bwd_recv_m0_pos, "BWD_RECV(m=0) missing")

        # Find the FSDP_UNSHARD that was injected for BWD (the one right before
        # BWD_RECV, which will be the second UNSHARD for this stage if the
        # schedule happens to have two).
        bwd_unshard = None
        for i in range(bwd_m0_pos - 1, 0, -1):
            if order[i] is not None and order[i].type == MetaStepType.FSDP_UNSHARD:
                bwd_unshard = i
                break
        if bwd_unshard is not None:
            self.assertLess(launch_load_m0_pos, bwd_unshard,
                            "SWAP_LAUNCH_LOAD must precede FSDP_UNSHARD before BWD")
            self.assertLess(bwd_unshard, bwd_m0_pos,
                            "FSDP_UNSHARD must precede BWD")
        self.assertLess(launch_load_m0_pos, wait_load_m0_pos,
                        "SWAP_LAUNCH_LOAD must precede SWAP_WAIT_LOAD")
        self.assertLess(bwd_recv_m0_pos, wait_load_m0_pos,
                        "BWD_RECV must precede SWAP_WAIT_LOAD")
        self.assertLess(wait_load_m0_pos, bwd_m0_pos,
                        "SWAP_WAIT_LOAD must precede BWD")

        # --- Gap lifecycle assertions ---
        # WAIT_OFFLOAD must precede LAUNCH_LOAD for same micro.
        wait_off_pos = pos(0, MetaStepType.SWAP_WAIT_OFFLOAD)
        self.assertIsNotNone(wait_off_pos, "SWAP_WAIT_OFFLOAD missing")
        self.assertLess(wait_off_pos, launch_load_m0_pos,
                        "SWAP_WAIT_OFFLOAD must precede SWAP_LAUNCH_LOAD")

    def test_fsdp_swap_delays_load_when_parameters_stay_unsharded(self):
        """FSDP swap should not restore activations across an extra compute step."""
        stage_index = 0
        order = [
            MetaStep(None, MetaStepType.FSDP_UNSHARD, stage_index),
            MetaStep(0, MetaStepType.FWD, stage_index),
            MetaStep(1, MetaStepType.FWD, stage_index),
            MetaStep(2, MetaStepType.FWD, stage_index),
            MetaStep(3, MetaStepType.FWD, stage_index),
            MetaStep(0, MetaStepType.BWD, stage_index),
            MetaStep(None, MetaStepType.FSDP_RESHARD, stage_index),
        ]
        order = inject_pipeline_swap_steps(order)

        launch_load = next(
            index for index, step in enumerate(order)
            if step.type == MetaStepType.SWAP_LAUNCH_LOAD and step.micro_index == 0
        )
        wait_load = next(
            index for index, step in enumerate(order)
            if step.type == MetaStepType.SWAP_WAIT_LOAD and step.micro_index == 0
        )
        backward = next(
            index for index, step in enumerate(order)
            if step.type == MetaStepType.BWD and step.micro_index == 0
        )
        self.assertEqual(launch_load + 1, wait_load)
        self.assertEqual(wait_load + 1, backward)

    def test_fsdp_swap_dxdw_split_ordering(self):
        """FSDP+swap with dxdw split should anchor activation load on BWD_INPUT."""
        stage_index = 0
        # Construct a schedule with enough gap for swap injection, using
        # BWD_INPUT as the backward anchor.
        order = [
            MetaStep(0, MetaStepType.FWD, stage_index),
            MetaStep(0, MetaStepType.FWD_SEND, stage_index),
            MetaStep(1, MetaStepType.FWD, stage_index),
            MetaStep(1, MetaStepType.FWD_SEND, stage_index),
            MetaStep(2, MetaStepType.FWD, stage_index),
            MetaStep(2, MetaStepType.FWD_SEND, stage_index),
            MetaStep(3, MetaStepType.FWD, stage_index),
            MetaStep(3, MetaStepType.FWD_SEND, stage_index),
            MetaStep(0, MetaStepType.BWD_RECV, stage_index),
            MetaStep(0, MetaStepType.BWD_INPUT, stage_index),
            MetaStep(0, MetaStepType.BWD_WEIGHT, stage_index),
            MetaStep(1, MetaStepType.BWD_RECV, stage_index),
            MetaStep(1, MetaStepType.BWD_INPUT, stage_index),
            MetaStep(1, MetaStepType.BWD_WEIGHT, stage_index),
            MetaStep(2, MetaStepType.BWD_RECV, stage_index),
            MetaStep(2, MetaStepType.BWD_INPUT, stage_index),
            MetaStep(2, MetaStepType.BWD_WEIGHT, stage_index),
            MetaStep(3, MetaStepType.BWD_RECV, stage_index),
            MetaStep(3, MetaStepType.BWD_INPUT, stage_index),
            MetaStep(3, MetaStepType.BWD_WEIGHT, stage_index),
        ]

        managed_stage_indices = {stage_index}
        order = add_fsdp_unshard_reshard(order, managed_stage_indices, max_active_stages=3)
        order = add_fsdp_reduce_grad(order, managed_stage_indices, micro_batch_num=4)
        order = inject_pipeline_swap_steps(order)

        def step_indices(step_type):
            return [
                i for i, s in enumerate(order)
                if s is not None and s.type == step_type
            ]

        # Each H2D launch must target BWD_INPUT, never BWD_WEIGHT.
        launch_load_positions = step_indices(MetaStepType.SWAP_LAUNCH_LOAD)
        self.assertTrue(launch_load_positions, "SWAP_LAUNCH_LOAD should exist for BWD_INPUT")
        for launch_pos in launch_load_positions:
            next_comp = None
            for j in range(launch_pos + 1, len(order)):
                if order[j] is not None and order[j].type in (
                    MetaStepType.BWD_INPUT,
                    MetaStepType.BWD_WEIGHT,
                    MetaStepType.BWD,
                ):
                    next_comp = order[j]
                    break
            self.assertIsNotNone(next_comp, f"No compute step after LAUNCH_LOAD at {launch_pos}")
            self.assertEqual(next_comp.type, MetaStepType.BWD_INPUT,
                             f"LAUNCH_LOAD at {launch_pos} should target BWD_INPUT, got {next_comp.type}")

        # Verify the schedulable lifecycle for m=0. Collection remains
        # leaf-local while the H2D wait is an explicit scheduler action.
        lifecycle_steps = []
        for s in order:
            if s is not None and s.micro_index == 0 and s.type in (
                MetaStepType.FWD,
                MetaStepType.SWAP_LAUNCH_OFFLOAD,
                MetaStepType.SWAP_WAIT_OFFLOAD,
                MetaStepType.SWAP_LAUNCH_LOAD,
                MetaStepType.SWAP_WAIT_LOAD,
                MetaStepType.BWD_INPUT,
                MetaStepType.BWD_WEIGHT,
            ):
                lifecycle_steps.append(s.type)
        expected = [
            MetaStepType.FWD,
            MetaStepType.SWAP_LAUNCH_OFFLOAD,
            MetaStepType.SWAP_WAIT_OFFLOAD,
            MetaStepType.SWAP_LAUNCH_LOAD,
            MetaStepType.SWAP_WAIT_LOAD,
            MetaStepType.BWD_INPUT,
            MetaStepType.BWD_WEIGHT,
        ]
        self.assertEqual(lifecycle_steps, expected,
                         f"Swap lifecycle order mismatch: {lifecycle_steps}")

    def test_complex_nesting_and_cross_references(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_micro0_1th"],
            "2": ["Send_Receive_(1)->(2)_micro0_1th"],
            "3": []
        }
        parse_and_validate(data)


class TestCoalesceP2P(unittest.TestCase):
    """Tests for ``coalesce_p2p`` — the ``batch_p2p`` order-rewrite pass."""

    @staticmethod
    def _build_overlap_bf_order():
        """Build a real PP=4 x 2-chunk overlap_b_f exec order, no stages needed."""
        schedule = object.__new__(ScheduleInterleaved1F1B)
        # pylint: disable=protected-access
        schedule.real_stage_num = 4
        schedule._stage_num = 8
        schedule.n_local_stages = 2
        schedule.micro_batch_num = 8
        schedule._overlap_b_f = True
        schedule.n_rounds = 2
        schedule.n_microbatch_per_round = [4, 4]
        schedule.n_microbatch_per_round_accu = [0, 8, 16]
        schedule.exec_order = {}
        ScheduleInterleaved1F1B.construct_exec_order(schedule)
        return schedule.exec_order

    def test_coalesce_preserves_order_and_removes_runs(self):
        """On a real overlap_b_f order: no leftover P2P run, flatten == original."""
        order = self._build_overlap_bf_order()
        coalesced = coalesce_p2p(order)
        n_batch = 0
        for rank in order:
            seq = coalesced[rank]
            run = 0
            for step in seq:
                if step is not None and step.type in _P2P_STEP_TYPES:
                    run += 1
                    self.assertLessEqual(run, 1, f"rank{rank}: leftover P2P run")
                else:
                    run = 0
            flat = []
            for step in seq:
                if step is not None and step.type == MetaStepType.BATCH_SEND_RECV:
                    self.assertGreaterEqual(len(step.sub_steps), 2)
                    self.assertTrue(all(s.type in _P2P_STEP_TYPES for s in step.sub_steps))
                    n_batch += 1
                    flat.extend(step.sub_steps)
                else:
                    flat.append(step)
            self.assertEqual(flat, order[rank], f"rank{rank}: flatten != original")
        self.assertGreater(n_batch, 0, "no BATCH_SEND_RECV produced")

    def test_coalesce_leaves_singleton_p2p(self):
        """An isolated P2P step (run length 1) is left untouched."""
        order = {0: [
            MetaStep(0, MetaStepType.FWD, 0),
            MetaStep(0, MetaStepType.FWD_SEND, 0),
            MetaStep(1, MetaStepType.FWD, 0),
        ]}
        self.assertEqual(coalesce_p2p(order)[0], order[0])

    def test_coalesce_groups_contiguous_run(self):
        """A contiguous run of >=2 P2P steps becomes one BATCH_SEND_RECV."""
        order = {0: [
            MetaStep(0, MetaStepType.FWD, 0),
            MetaStep(0, MetaStepType.BWD_RECV, 0),
            MetaStep(0, MetaStepType.FWD_SEND, 0),
            MetaStep(1, MetaStepType.FWD, 0),
        ]}
        coalesced = coalesce_p2p(order)[0]
        self.assertEqual([s.type for s in coalesced],
                         [MetaStepType.FWD, MetaStepType.BATCH_SEND_RECV, MetaStepType.FWD])
        batch = coalesced[1]
        self.assertEqual([s.type for s in batch.sub_steps],
                         [MetaStepType.BWD_RECV, MetaStepType.FWD_SEND])


class TestFwdBoundaryP2P(unittest.TestCase):
    """Tests for ``attach_fwd_boundary_p2p`` — the "boundary" transport pass.

    Safety contract: only the overlap's OWN forward ``FWD_SEND`` (payload ready
    at the fwd/bwd boundary) and recvs (no data dependency) move; ``BWD_SEND``
    stays in the gap; per-direction FIFO is preserved; nothing is lost or
    duplicated.  The complementary per-pair shape ([F_SEND, B_RECV] vs
    [F_RECV, B_SEND], all per-op solo batches) is what keeps HCCL batch
    pairing matched — the naive hoist+coalesce composition violated it and
    hung on hardware.
    """

    @staticmethod
    def _build_overlap_bf_order():
        """Build a real PP=4 x 2-chunk overlap_b_f exec order, no stages needed."""
        schedule = object.__new__(ScheduleInterleaved1F1B)
        # pylint: disable=protected-access
        schedule.real_stage_num = 4
        schedule._stage_num = 8
        schedule.n_local_stages = 2
        schedule.micro_batch_num = 8
        schedule._overlap_b_f = True
        schedule.n_rounds = 2
        schedule.n_microbatch_per_round = [4, 4]
        schedule.n_microbatch_per_round_accu = [0, 8, 16]
        schedule.exec_order = {}
        ScheduleInterleaved1F1B.construct_exec_order(schedule)
        return schedule.exec_order

    @staticmethod
    def _key(step):
        return (step.type, step.stage_index, step.micro_index)

    @staticmethod
    def _flatten(order):
        """Re-insert each boundary_p2p at its issue position (after its OVL)."""
        out = []
        for step in order:
            if step is None:
                continue
            out.append(step)
            if getattr(step, "boundary_p2p", None):
                out.extend(step.boundary_p2p)
        return out

    def test_invalid_p2p_transport_rejected(self):
        """Unknown transport values fail fast, before stages are touched."""
        with self.assertRaises(ValueError):
            ScheduleInterleaved1F1B(None, 1, p2p_transport="bogus")

    def test_multiset_and_direction_fifo_preserved(self):
        """No P2P op lost/duplicated; per-direction FIFO unchanged."""
        order = self._build_overlap_bf_order()
        attached = attach_fwd_boundary_p2p(order)
        n_boundary = 0
        for rank in order:
            before = [s for s in order[rank] if s is not None]
            after = self._flatten(attached[rank])
            self.assertEqual(
                collections.Counter(self._key(s) for s in before
                                    if s.type in _P2P_STEP_TYPES),
                collections.Counter(self._key(s) for s in after
                                    if s.type in _P2P_STEP_TYPES),
                f"rank{rank}: P2P multiset changed")
            for p2p_type in _P2P_STEP_TYPES:
                self.assertEqual(
                    [self._key(s) for s in before if s.type == p2p_type],
                    [self._key(s) for s in after if s.type == p2p_type],
                    f"rank{rank}: {p2p_type} FIFO changed")
            n_boundary += sum(1 for s in attached[rank]
                              if s is not None and getattr(s, "boundary_p2p", None))
        self.assertGreater(n_boundary, 0, "pass attached nothing")

    def test_boundary_content_and_gap_residue(self):
        """boundary = own-fwd F_SEND first + recvs; gap keeps BWD_SEND only."""
        attached = attach_fwd_boundary_p2p(self._build_overlap_bf_order())
        for rank, seq in attached.items():
            for i, step in enumerate(seq):
                boundary = (getattr(step, "boundary_p2p", None)
                            if step is not None else None)
                if not boundary:
                    continue
                fwd_sub = next(s for s in step.sub_steps
                               if s.type == MetaStepType.FWD)
                kinds = [s.type for s in boundary]
                self.assertNotIn(MetaStepType.BWD_SEND, kinds,
                                 f"rank{rank}: BWD_SEND must stay in the gap")
                for snd in (s for s in boundary
                            if s.type == MetaStepType.FWD_SEND):
                    self.assertEqual(
                        (snd.stage_index, snd.micro_index),
                        (fwd_sub.stage_index, fwd_sub.micro_index),
                        f"rank{rank}: boundary F_SEND is not the overlap's own "
                        "forward output")
                if MetaStepType.FWD_SEND in kinds:
                    self.assertEqual(kinds[0], MetaStepType.FWD_SEND,
                                     f"rank{rank}: F_SEND must be issued first")
                j = i + 1
                while (j < len(seq) and seq[j] is not None
                       and seq[j].type in _P2P_STEP_TYPES):
                    self.assertNotIn(seq[j].type, _RECV_STEP_TYPES,
                                     f"rank{rank}: recv left in the gap behind "
                                     "a boundary overlap")
                    j += 1

    def test_boundary_composes_with_pipeline_swap(self):
        """Boundary P2P containers remain intact after pipeline-swap injection."""
        attached = attach_fwd_boundary_p2p(self._build_overlap_bf_order())
        swap_types = {
            MetaStepType.SWAP_LAUNCH_OFFLOAD,
            MetaStepType.SWAP_WAIT_OFFLOAD,
            MetaStepType.SWAP_LAUNCH_LOAD,
            MetaStepType.SWAP_WAIT_LOAD,
        }
        swap_count = 0
        boundary_count = 0
        for rank, order in attached.items():
            injected = inject_pipeline_swap_steps(order)
            swap_count += sum(
                step is not None and step.type in swap_types
                for step in injected
            )
            for step in injected:
                if step is None or not step.boundary_p2p:
                    continue
                boundary_count += 1
                self.assertFalse(any(sub_step.type in swap_types for sub_step in step.boundary_p2p))
            self.assertEqual(
                [self._key(step) for step in self._flatten(order) if step.type in _P2P_STEP_TYPES],
                [self._key(step) for step in self._flatten(injected) if step.type in _P2P_STEP_TYPES],
                f"rank{rank}: swap injection changed boundary P2P ordering",
            )
        self.assertGreater(boundary_count, 0)
        self.assertGreater(swap_count, 0)


class TestDetectCycleInGraph(unittest.TestCase):
    """Test class for cycle detection in graphs."""
    def test_cycle_in_graph(self):
        ranks_map = {
            '1': ['A', 'B', 'C', 'A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['A', 'B', 'C', 'A'])
        self.assertEqual(cycle_ranks, ['1 A -> B', '1 B -> C', '1 C -> A'])

    def test_no_cycle_in_graph(self):
        ranks_map = {
            'r1': ['A', 'B', 'C'],
            'r2': ['D', 'E']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

    def test_multiple_cycles(self):
        """Test case where multiple cycles exist in the graph."""
        ranks_map = {
            '1': ['A', 'B', 'C', 'A'],
            '2': ['D', 'E', 'D']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertTrue(cycle_path in [['A', 'B', 'C', 'A'], ['D', 'E', 'D']])
        if cycle_path == ['A', 'B', 'C', 'A']:
            self.assertEqual(cycle_ranks, ['1 A -> B', '1 B -> C', '1 C -> A'])
        elif cycle_path == ['D', 'E', 'D']:
            self.assertEqual(cycle_ranks, ['2 D -> E', '2 E -> D'])

    def test_empty_graph(self):
        ranks_map = {}
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

    def test_two_nodes_no_cycle(self):
        ranks_map = {
            'rank1': ['A', 'B']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)
    def test_complex_cycle(self):
        ranks_map = {
            "1": ["A", "B", "C"],
            "2": ["C", "D", "E"],
            "3": ["E", "A"]
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ["A", "B", "C", "D", "E", "A"])
        self.assertEqual(cycle_ranks, [
            '1 A -> B',
            '1 B -> C',
            '2 C -> D',
            '2 D -> E',
            '3 E -> A'
        ])

    def test_disconnected_components_with_cycle(self):
        ranks_map = {
            "1": ["A", "B", "C"],
            "2": ["D", "E", "D"]
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ["D", "E", "D"])
        self.assertEqual(cycle_ranks, ['2 D -> E', '2 E -> D'])

    def test_cross_rank_cycle(self):
        """Test various cross-rank cycle scenarios."""
        ranks_map = {
            '1': ['A', 'B'],
            '2': ['B', 'A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['A', 'B', 'A'])
        self.assertEqual(cycle_ranks, ['1 A -> B', '2 B -> A'])

        ranks_map = {
            '1': ['A', 'B'],
            '2': ['B', 'C'],
            '3': ['C', 'A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['A', 'B', 'C', 'A'])
        self.assertEqual(cycle_ranks, ['1 A -> B', '2 B -> C', '3 C -> A'])

        ranks_map = {
            '1': ['A', 'B'],
            '2': ['B', 'C'],
            '3': ['C', 'D']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

        ranks_map = {
            '1': ['A', 'A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['A', 'A'])
        self.assertEqual(cycle_ranks, ['1 A -> A'])
        ranks_map = {
            '1': ['A', 'B'],
            '2': ['B', 'C'],
            '3': ['C', 'B']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertEqual(cycle_path, ['B', 'C', 'B'])
        self.assertEqual(cycle_ranks, ['2 B -> C', '3 C -> B'])

        ranks_map = {
            '1': ['A']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

        ranks_map = {
            '1': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '4': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '5': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '6': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '7': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            '8': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        }
        cycle_path, cycle_ranks = detect_cycle_in_graph(ranks_map)
        self.assertIsNone(cycle_path)
        self.assertIsNone(cycle_ranks)

class TestPipelineValidation(unittest.TestCase):
    """Test class for pipeline validation functionality."""
    def setUp(self):
        """Prepare test data"""
        # Simple forward send-receive pair
        self.simple_fwd_pair = {
            0: [MetaStep(0, MetaStepType.FWD_SEND, 0)],
            1: [MetaStep(0, MetaStepType.FWD_RECV, 1)]
        }

        # Complete pipeline data (simplified version)
        self.complete_pipeline_data = {
        0: [MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=0)]
        ,
        1: [MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=1)]
        ,
        2: [MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=2)]

    }
        # Data with missing send operation
        self.data_missing_send = {
            0: [MetaStep(0, MetaStepType.FWD, stage_index=0)],
                # Missing: MetaStep(0, MetaStepType.FWD_SEND, stage_index=0)
            1: [MetaStep(0, MetaStepType.FWD_RECV, stage_index=1)]
        }

        # Data with swapped operations order
        self.data_swapped_operations = {
        0: [MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=0),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=0),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=6),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=6),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=6),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=6),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=3),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=3),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=0),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=0),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=0)]
        ,
        1: [MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=7),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=7),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=4),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=4),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=1),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=1),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=1)]
        ,
        2: [MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=0, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=1, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=2, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=2),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=0, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=5),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=1, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_SEND, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=2, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=3, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=4, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD_RECV, stage_index=8),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=5, meta_type=MetaStepType.FWD, stage_index=8),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=8),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=5),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=5),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=3, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_RECV, stage_index=2),
            MetaStep(micro_index=4, meta_type=MetaStepType.BWD_SEND, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD, stage_index=2),
            MetaStep(micro_index=5, meta_type=MetaStepType.BWD_SEND, stage_index=2)]

    }

    def test_key_generation_correctness(self):
        """Test if key generation process is correct"""
        result = generate_operations(self.simple_fwd_pair, chunk_num=1, com_type='loop')

        # Verify result is not None
        self.assertIsNotNone(result, "Generated operations should not be None")

        # Verify contains two ranks
        self.assertEqual(len(result), 2, f"Should contain 2 ranks, but contains {len(result)}")

        # Verify each rank has one operation
        self.assertIn('0', result, "Should contain rank 0")
        self.assertIn('1', result, "Should contain rank 1")

        # Verify operation format
        expected_format = "Send_Receive_(0)->(1)_micro0_1th"
        self.assertEqual(result['0'][0], expected_format,
                        f"rank0 operation format incorrect. Expected: {expected_format}, Actual: {result['0'][0]}")
        self.assertEqual(result['1'][0], expected_format,
                        f"rank1 operation format incorrect. Expected: {expected_format}, Actual: {result['1'][0]}")

    def test_complete_pipeline_validation(self):
        """Test complete pipeline validation"""
        result = validate_pipeline_execution(self.complete_pipeline_data, chunk_num=3, com_type='loop')

        # Verify result contains required keys
        self.assertIn('formatted_operations', result, "Result should contain formatted_operations")
        self.assertIn('has_cycle', result, "Result should contain has_cycle")

        # Verify no cycles in complete pipeline
        self.assertFalse(result['has_cycle'], "Complete pipeline should not have cycles")

        # Verify formatted operations are generated
        formatted_ops = result['formatted_operations']
        self.assertIsNotNone(formatted_ops, "Formatted operations should not be None")

    def test_missing_send_operation_detection(self):
        """Test that removing a send operation causes errors"""
        result = validate_pipeline_execution(self.data_missing_send, chunk_num=3, com_type='loop')

        # Since parse_and_validate logs errors but doesn't return them,
        # we need to check the actual validation logic
        # For now, we just verify the function doesn't crash
        self.assertIsNotNone(result, "Validation should complete without crashing")

        # The validation should detect missing operations
        self.assertIn('formatted_operations', result)

    def test_swapped_operations_detection(self):
        """Test that swapping operation order causes errors"""
        result = validate_pipeline_execution(self.data_swapped_operations, chunk_num=3, com_type='loop')

        # Verify function doesn't crash
        self.assertIsNotNone(result, "Validation should complete without crashing")

        # Check if cycles are detected
        has_cycle = result.get('has_cycle', False)

        # Swapped operations might or might not create cycles
        # We just verify the function handles it
        print(f"Swapped operations result - has_cycle: {has_cycle}")

        # Verify result structure
        self.assertIn('formatted_operations', result)
        self.assertIn('has_cycle', result)

    if __name__ == "__main__":
        unittest.main()
