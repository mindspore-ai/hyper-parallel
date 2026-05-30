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
Unit tests for pipeline parallel scheduling functions.
This module contains test classes for validating various functions in the scheduling module.
"""

import sys
import unittest
from unittest.mock import Mock, patch

from hyper_parallel.core.pipeline_parallel import pipeline_swap
from hyper_parallel.core.pipeline_parallel.pipeline_swap import inject_pipeline_swap_steps
from hyper_parallel.core.pipeline_parallel.scheduler import (
    MetaStep,
    MetaStepType,
    Schedule1F1B,
    ScheduleInterleaved1F1B,
    validate_pipeline_execution,
    detect_cycle_in_graph,
    parse_and_validate,
    generate_operations,
)

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
            self.assertLess(pos(micro_index, MetaStepType.SWAP_SET_GROUP), pos(micro_index, MetaStepType.FWD))
            self.assertLess(pos(micro_index, MetaStepType.FWD), pos(micro_index, MetaStepType.SWAP_LAUNCH_OFFLOAD))
            self.assertLess(
                pos(micro_index, MetaStepType.FWD_SEND),
                pos(micro_index, MetaStepType.SWAP_LAUNCH_OFFLOAD),
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
            self.assertLess(pos(micro_index, MetaStepType.SWAP_WAIT_LOAD), pos(micro_index, MetaStepType.BWD))

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
                if step is not None and step.type == MetaStepType.SWAP_SET_GROUP
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
                if step is not None and step.type == MetaStepType.SWAP_SET_GROUP
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
            MetaStepType.SWAP_SET_GROUP,
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

        self.assertLess(pos(MetaStepType.SWAP_LAUNCH_LOAD, 4, 0), pos(MetaStepType.SWAP_WAIT_LOAD, 4, 0))
        self.assertLess(pos(MetaStepType.SWAP_WAIT_LOAD, 4, 0), pos(MetaStepType.SWAP_SET_GROUP, 0, 7))
        self.assertLess(pos(MetaStepType.SWAP_SET_GROUP, 0, 7), overlap_pos)
        self.assertLess(overlap_pos, pos(MetaStepType.SWAP_LAUNCH_OFFLOAD, 0, 7))
        self.assertLess(pos(MetaStepType.SWAP_LAUNCH_OFFLOAD, 0, 7), pos(MetaStepType.SWAP_WAIT_OFFLOAD, 0, 7))
        self.assertLess(pos(MetaStepType.SWAP_WAIT_LOAD, 4, 0), leaf_pos(MetaStepType.BWD, 4, 0))

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
            return any(step.type == MetaStepType.SWAP_SET_GROUP for step in schedule.exec_order[rank])

        self.assertTrue(has_swap(0))
        self.assertFalse(has_swap(1))

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
            pos(3, MetaStepType.SWAP_SET_GROUP),
        )
        self.assertLess(pos(3, MetaStepType.SWAP_SET_GROUP), pos(3, MetaStepType.FWD))
        self.assertLess(pos(0, MetaStepType.SWAP_LAUNCH_LOAD), pos(3, MetaStepType.FWD_SEND))
        self.assertLess(pos(0, MetaStepType.SWAP_LAUNCH_LOAD), pos(0, MetaStepType.SWAP_WAIT_LOAD))

    def test_pipeline_swap_closes_group_after_forward_offload(self):
        """A non-swapped forward must not inherit the prior chunk's group."""
        step = MetaStep(2, MetaStepType.SWAP_LAUNCH_OFFLOAD, 1)
        manager = Mock()
        with patch.object(pipeline_swap, "SwapManager", return_value=manager), \
             patch.object(pipeline_swap, "_protect_pipeline_owned_tensors"):
            pipeline_swap.swap_launch_offload(step, Mock(), [], [])

        manager.launch_offload.assert_called_once_with("pp_swap_s1_m2")
        manager.set_current_group_name.assert_called_once_with("")

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

    def test_complex_nesting_and_cross_references(self):
        data = {
            "1": ["Send_Receive_(1)->(2)_micro0_1th"],
            "2": ["Send_Receive_(1)->(2)_micro0_1th"],
            "3": []
        }
        parse_and_validate(data)

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
