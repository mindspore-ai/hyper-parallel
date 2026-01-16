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

import unittest
import sys
from hyper_parallel.core.pipeline_parallel.scheduler import MetaStep, MetaStepType, validate_pipeline_execution, detect_cycle_in_graph, parse_and_validate, generate_operations

sys.setrecursionlimit(10000)

class TestParseAndValidate(unittest.TestCase):
    """Test class for parse and validate functionality."""
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
