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
"""Unit tests for Pipeline Parallelism modeling — pp_result and stage_partition."""

import unittest
from typing import List, Tuple

from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_structs import (
    PPBOutput,
    PPStrategyResult,
    RecomputeType,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import YamlOptimizationConfig
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_balancer import PPBalancer
from hyper_parallel.auto_parallel.sapp_ppb.utils.recompute import TYPE as _RecomputeType

_StagePartitionType = List[List[Tuple[int, _RecomputeType]]]


class _StagePartition:
    """Stage partition manager for pipeline parallelism (test-local copy).

    When ``num_body_layers > 0``, HEAD (layer_id 0) is always placed in
    the first stage and TAIL (layer_id ``num_layers - 1``) in the last
    stage.  Only BODY layers (layer_id 1 … ``num_layers - 2``) are
    uniformly distributed across stages.

    When ``num_body_layers == 0`` (legacy mode), all ``num_layers`` are
    distributed uniformly with no HEAD/TAIL pinning.

    **Empty intermediate stages.**  By default
    (``allow_empty_stages=False``), every stage must contain at
    least one layer.  When ``allow_empty_stages=True``, intermediate
    stages may be empty (zero layers).  An empty intermediate stage
    means that the corresponding device performs no computation in the
    pipeline.  Use with caution.

    Args:
        num_layers: Total number of layers (HEAD + body layers + TAIL)
            when ``num_body_layers > 0``; otherwise just the total
            number of layers to distribute.
        pp_degree: Number of pipeline stages.
        num_body_layers: Number of BODY layers.  When positive,
            ``num_layers`` must equal ``num_body_layers + 2`` (HEAD + BODY
            + TAIL).  When zero, legacy uniform-partition mode is used.
    """

    def __init__(
        self,
        num_layers: int,
        pp_degree: int,
        num_body_layers: int = 0,
    ) -> None:
        if pp_degree <= 0:
            raise ValueError(
                f"For StagePartition, pp_degree should be positive, but got {pp_degree}."
            )
        if num_layers <= 0:
            raise ValueError(
                f"For StagePartition, num_layers should be positive, but got {num_layers}."
            )
        if num_body_layers < 0:
            raise ValueError(
                f"For StagePartition, num_body_layers should be non-negative, got {num_body_layers}."
            )
        if num_body_layers > 0 and num_layers != num_body_layers + 2:
            raise ValueError(
                f"For StagePartition, when num_body_layers ({num_body_layers}) > 0, "
                f"num_layers ({num_layers}) must equal num_body_layers + 2 "
                f"({num_body_layers + 2})."
            )
        if pp_degree > num_layers:
            raise ValueError(
                f"For StagePartition, pp_degree ({pp_degree}) should not exceed "
                f"number of layers ({num_layers})."
            )

        self.pp_degree = pp_degree
        self.num_layers = num_layers
        self.num_body_layers = num_body_layers

    def uniform_partition(self) -> _StagePartitionType:
        """Create uniform stage partition."""
        if self.num_body_layers <= 0:
            return self._uniform_partition_all()
        return self._uniform_partition_body_only()

    def _uniform_partition_all(self) -> _StagePartitionType:
        """Uniformly distribute all layers (legacy mode)."""
        stages: _StagePartitionType = []
        layers_per_stage = self.num_layers // self.pp_degree
        remainder = self.num_layers % self.pp_degree

        current_idx = 0
        for stage_id in range(self.pp_degree):
            stage_size = layers_per_stage
            if stage_id < remainder:
                stage_size += 1

            stage_layers = [
                (lid, _RecomputeType.NONE)
                for lid in range(current_idx, current_idx + stage_size)
            ]
            stages.append(stage_layers)
            current_idx += stage_size

        return stages

    def _uniform_partition_body_only(self) -> _StagePartitionType:
        """Uniformly distribute BODY layers; pin HEAD to stage 0, TAIL to last stage."""
        stages: _StagePartitionType = [
            [] for _ in range(self.pp_degree)
        ]

        body_per_stage = self.num_body_layers // self.pp_degree
        remainder = self.num_body_layers % self.pp_degree

        body_start = 1
        for stage_id in range(self.pp_degree):
            stage_body = body_per_stage
            if stage_id < remainder:
                stage_body += 1

            for lid in range(body_start, body_start + stage_body):
                stages[stage_id].append((lid, _RecomputeType.NONE))
            body_start += stage_body

        stages[0].insert(0, (0, _RecomputeType.NONE))
        stages[-1].append((self.num_layers - 1, _RecomputeType.NONE))

        return stages

    @staticmethod
    def _collect_layer_ids(
        stage_partition: _StagePartitionType,
        num_layers: int,
        allow_empty_stages: bool,
    ) -> Tuple[set, str]:
        """Collect all layer IDs and validate per-entry constraints."""
        all_layer_ids = set()
        for stage_id, stage_layers in enumerate(stage_partition):
            if not stage_layers and not allow_empty_stages:
                return set(), f"Stage {stage_id} is empty."
            for entry in stage_layers:
                layer_id = entry[0]
                if layer_id < 0 or layer_id >= num_layers:
                    return set(), f"Invalid layer ID {layer_id} in stage {stage_id}."
                if layer_id in all_layer_ids:
                    return set(), f"Layer {layer_id} appears in multiple stages."
                all_layer_ids.add(layer_id)
        return all_layer_ids, ""

    @staticmethod
    def _check_stage_ordering(stage_partition: _StagePartitionType) -> str:
        """Check that non-empty stages are ordered by layer ID."""
        prev_max: int = -1
        prev_nonempty_stage: int = -1
        for stage_id, stage_layers in enumerate(stage_partition):
            if not stage_layers:
                continue
            cur_min = min(e[0] for e in stage_layers)
            cur_max = max(e[0] for e in stage_layers)
            if cur_min <= prev_max:
                return (
                    f"Stage {stage_id} and {prev_nonempty_stage} are not "
                    f"ordered correctly."
                )
            prev_max = cur_max
            prev_nonempty_stage = stage_id
        return ""

    @staticmethod
    def _check_head_tail_placement(
        stage_partition: _StagePartitionType,
        num_layers: int,
    ) -> str:
        """Check that HEAD is in stage 0 and TAIL in last stage with NONE recompute."""
        if not stage_partition or not stage_partition[0]:
            return "Stage 0 is empty; cannot contain HEAD (layer_id 0)."

        head_ids = [e[0] for e in stage_partition[0]]
        if 0 not in head_ids:
            return "HEAD (layer_id 0) must be in stage 0."

        for entry in stage_partition[0]:
            if entry[0] == 0 and entry[1] != _RecomputeType.NONE:
                return "HEAD (layer_id 0) must have RecomputeType.NONE."

        if not stage_partition[-1]:
            return f"Last stage is empty; cannot contain TAIL (layer_id {num_layers - 1})."

        tail_ids = [e[0] for e in stage_partition[-1]]
        if (num_layers - 1) not in tail_ids:
            return f"TAIL (layer_id {num_layers - 1}) must be in the last stage."

        for entry in stage_partition[-1]:
            if entry[0] == num_layers - 1 and entry[1] != _RecomputeType.NONE:
                return "TAIL must have RecomputeType.NONE."

        return ""

    def validate_partition(
        self,
        stage_partition: _StagePartitionType,
        allow_empty_stages: bool = False,
    ) -> Tuple[bool, str]:
        """Validate stage partition."""
        if len(stage_partition) != self.pp_degree:
            return (
                False,
                f"Expected {self.pp_degree} stages, got {len(stage_partition)}.",
            )

        all_layer_ids, err = self._collect_layer_ids(
            stage_partition, self.num_layers, allow_empty_stages,
        )
        if err:
            return False, err

        expected_layers = set(range(self.num_layers))
        if all_layer_ids != expected_layers:
            missing = expected_layers - all_layer_ids
            return (
                False,
                f"Partition does not cover all layers. Missing: {missing}.",
            )

        if self.num_body_layers > 0:
            ht_err = self._check_head_tail_placement(
                stage_partition, self.num_layers,
            )
            if ht_err:
                return False, ht_err

        ordering_err = self._check_stage_ordering(stage_partition)
        if ordering_err:
            return False, ordering_err

        return True, ""


StagePartition = _StagePartition


class TestRecomputeType(unittest.TestCase):
    """Test RecomputeType enum."""

    def test_enum_values(self) -> None:
        """Test all five recompute type integer values."""
        self.assertEqual(RecomputeType.NONE.value, 0)
        self.assertEqual(RecomputeType.SLCT.value, 1)
        self.assertEqual(RecomputeType.COMM.value, 2)
        self.assertEqual(RecomputeType.BOTH.value, 3)
        self.assertEqual(RecomputeType.FULL.value, 4)

    def test_enum_count(self) -> None:
        """Test that exactly five recompute types exist."""
        self.assertEqual(len(RecomputeType), 5)


class TestPPStrategyResultStrategyFields(unittest.TestCase):
    """Test PPStrategyResult strategy-related fields."""

    def test_default_strategy_fields(self) -> None:
        """Test PPStrategyResult with default strategy fields."""
        result = PPStrategyResult()
        self.assertEqual(result.pp_degree, 0)
        self.assertEqual(result.micro_batch_num, 1)
        self.assertEqual(result.num_of_interleave, 1)
        self.assertFalse(result.vpp_less_memory)
        self.assertEqual(result.stage_partition, [])
        self.assertEqual(result.layer_offset, {})

    def test_custom_strategy_fields(self) -> None:
        """Test PPStrategyResult with all strategy fields populated."""
        result = PPStrategyResult(
            pp_degree=2,
            micro_batch_num=4,
            stage_partition=[
                [(0, RecomputeType.NONE), (1, RecomputeType.NONE), (2, RecomputeType.NONE)],
                [(3, RecomputeType.NONE), (4, RecomputeType.NONE), (5, RecomputeType.NONE)],
            ],
            layer_offset={"layer_group_1": [[-1]]},
        )
        self.assertEqual(result.pp_degree, 2)
        self.assertEqual(result.micro_batch_num, 4)
        self.assertEqual(len(result.stage_partition), 2)
        self.assertEqual(result.layer_offset, {"layer_group_1": [[-1]]})


class TestPPStrategyResult(unittest.TestCase):
    """Test PPStrategyResult dataclass."""

    def test_default_result(self) -> None:
        """Test PPStrategyResult with default fields."""
        result = PPStrategyResult()
        self.assertIsNone(result.pipeline_bubble)
        self.assertEqual(result.simulator_end_time, 0.0)
        self.assertEqual(result.simulation_status, "not_run")
        self.assertIsNone(result.simulation_error)
        self.assertEqual(result.simulator_bubbles, {})
        self.assertEqual(result.simulator_peak_memory, [])
        self.assertFalse(result.is_successful)


class TestPPBOutput(unittest.TestCase):
    """Test PPBOutput dataclass."""

    def test_default_output(self) -> None:
        """Test PPBOutput defaults."""
        output = PPBOutput()
        self.assertEqual(output.stage_partition, [])
        self.assertEqual(output.layer_offset, {})
        self.assertFalse(output.is_feasible)
        self.assertFalse(output.is_successful)
        self.assertEqual(output.infeasibility_details, {})
        self.assertEqual(output.simulator_end_time, 0.0)
        self.assertEqual(output.simulator_bubbles, {})
        self.assertEqual(output.simulation_status, "not_run")
        self.assertIsNone(output.simulation_error)

    def test_simulator_fields(self) -> None:
        """Test PPBOutput simulator fields can be populated."""
        output = PPBOutput(
            simulator_end_time=1234.5,
            simulator_bubbles={"real": 0.15, "ideal": 0.12},
        )
        self.assertEqual(output.simulator_end_time, 1234.5)
        self.assertEqual(output.simulator_bubbles["real"], 0.15)

    def test_simulation_status_failed(self) -> None:
        """Test PPBOutput with failed simulation status."""
        output = PPBOutput(
            is_feasible=True,
            simulation_status="failed",
            simulation_error="micro_batch_num < pp_degree",
        )
        self.assertTrue(output.is_feasible)
        self.assertEqual(output.simulation_status, "failed")
        self.assertEqual(output.simulation_error, "micro_batch_num < pp_degree")

    def test_simulation_status_success(self) -> None:
        """Test PPBOutput with successful simulation status."""
        output = PPBOutput(
            simulation_status="success",
            simulator_end_time=100.0,
        )
        self.assertEqual(output.simulation_status, "success")
        self.assertIsNone(output.simulation_error)


class TestStagePartition(unittest.TestCase):
    """Test StagePartition class."""

    def test_uniform_partition_legacy_mode(self) -> None:
        """Test legacy uniform partition (no HEAD/TAIL pinning)."""
        partition = StagePartition(num_layers=8, pp_degree=2)
        stages = partition.uniform_partition()
        self.assertEqual(len(stages), 2)
        self.assertEqual(len(stages[0]), 4)
        self.assertEqual(len(stages[1]), 4)

    def test_uniform_partition_with_remainder_legacy(self) -> None:
        """Test legacy uniform partition with remainder."""
        partition = StagePartition(num_layers=8, pp_degree=3)
        stages = partition.uniform_partition()
        self.assertEqual(len(stages), 3)
        self.assertEqual(len(stages[0]), 3)
        self.assertEqual(len(stages[1]), 3)
        self.assertEqual(len(stages[2]), 2)

    def test_uniform_partition_body_only(self) -> None:
        """Test BODY-only uniform partition with HEAD/TAIL pinning."""
        partition = StagePartition(num_layers=10, pp_degree=2, num_body_layers=8)
        stages = partition.uniform_partition()
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0][0], (0, RecomputeType.NONE))
        self.assertEqual(stages[-1][-1], (9, RecomputeType.NONE))
        body_in_s0 = len(stages[0]) - 1
        body_in_s1 = len(stages[1]) - 1
        self.assertEqual(body_in_s0, 4)
        self.assertEqual(body_in_s1, 4)

    def test_uniform_partition_body_only_with_remainder(self) -> None:
        """Test BODY-only partition with remainder: 8 BODY, PP=3."""
        partition = StagePartition(num_layers=10, pp_degree=3, num_body_layers=8)
        stages = partition.uniform_partition()
        self.assertEqual(len(stages), 3)
        self.assertEqual(stages[0][0], (0, RecomputeType.NONE))
        self.assertEqual(stages[-1][-1], (9, RecomputeType.NONE))
        body_counts = [len(s) - (1 if i == 0 else 0) - (1 if i == len(stages) - 1 else 0)
                       for i, s in enumerate(stages)]
        self.assertEqual(body_counts, [3, 3, 2])

    def test_uniform_partition_32_body_pp4(self) -> None:
        """Test 32 BODY + HEAD + TAIL, PP=4 → 8/8/8/8 BODY layers."""
        partition = StagePartition(num_layers=34, pp_degree=4, num_body_layers=32)
        stages = partition.uniform_partition()
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[0][0], (0, RecomputeType.NONE))
        self.assertEqual(stages[-1][-1], (33, RecomputeType.NONE))
        body_counts = [len(s) - (1 if i == 0 else 0) - (1 if i == len(stages) - 1 else 0)
                       for i, s in enumerate(stages)]
        self.assertEqual(body_counts, [8, 8, 8, 8])

    def test_validate_partition_success(self) -> None:
        """Test valid partition validation (legacy mode)."""
        partition = StagePartition(num_layers=8, pp_degree=2)
        stages = [
            [(0, RecomputeType.NONE), (1, RecomputeType.NONE),
             (2, RecomputeType.NONE), (3, RecomputeType.NONE)],
            [(4, RecomputeType.NONE), (5, RecomputeType.NONE),
             (6, RecomputeType.NONE), (7, RecomputeType.NONE)],
        ]
        is_valid, msg = partition.validate_partition(stages)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_validate_partition_with_head_tail(self) -> None:
        """Test valid partition with HEAD/TAIL pinning."""
        partition = StagePartition(num_layers=10, pp_degree=2, num_body_layers=8)
        stages = [
            [(0, RecomputeType.NONE), (1, RecomputeType.NONE),
             (2, RecomputeType.NONE), (3, RecomputeType.NONE),
             (4, RecomputeType.NONE)],
            [(5, RecomputeType.NONE), (6, RecomputeType.NONE),
             (7, RecomputeType.NONE), (8, RecomputeType.NONE),
             (9, RecomputeType.NONE)],
        ]
        is_valid, msg = partition.validate_partition(stages)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_validate_partition_incomplete_coverage(self) -> None:
        """Test incomplete coverage validation."""
        partition = StagePartition(num_layers=8, pp_degree=2)
        stages = [
            [(0, RecomputeType.NONE), (1, RecomputeType.NONE), (2, RecomputeType.NONE)],
            [(4, RecomputeType.NONE), (5, RecomputeType.NONE), (6, RecomputeType.NONE)],
        ]
        is_valid, msg = partition.validate_partition(stages)
        self.assertFalse(is_valid)
        self.assertIn("Missing", msg)

    def test_validate_head_not_in_first_stage(self) -> None:
        """Test that HEAD not in stage 0 is rejected."""
        partition = StagePartition(num_layers=10, pp_degree=2, num_body_layers=8)
        stages = [
            [(1, RecomputeType.NONE), (2, RecomputeType.NONE),
             (3, RecomputeType.NONE), (4, RecomputeType.NONE)],
            [(0, RecomputeType.NONE), (5, RecomputeType.NONE),
             (6, RecomputeType.NONE), (7, RecomputeType.NONE),
             (8, RecomputeType.NONE), (9, RecomputeType.NONE)],
        ]
        is_valid, msg = partition.validate_partition(stages)
        self.assertFalse(is_valid)
        self.assertIn("HEAD", msg)

    def test_validate_tail_not_in_last_stage(self) -> None:
        """Test that TAIL not in last stage is rejected."""
        partition = StagePartition(num_layers=10, pp_degree=2, num_body_layers=8)
        stages = [
            [(0, RecomputeType.NONE), (1, RecomputeType.NONE),
             (2, RecomputeType.NONE), (3, RecomputeType.NONE),
             (9, RecomputeType.NONE)],
            [(4, RecomputeType.NONE), (5, RecomputeType.NONE),
             (6, RecomputeType.NONE), (7, RecomputeType.NONE),
             (8, RecomputeType.NONE)],
        ]
        is_valid, msg = partition.validate_partition(stages)
        self.assertFalse(is_valid)
        self.assertIn("TAIL", msg)

    def test_validate_head_with_wrong_recompute(self) -> None:
        """Test that HEAD with non-NONE recompute is rejected."""
        partition = StagePartition(num_layers=10, pp_degree=2, num_body_layers=8)
        stages = [
            [(0, RecomputeType.SLCT), (1, RecomputeType.NONE),
             (2, RecomputeType.NONE), (3, RecomputeType.NONE),
             (4, RecomputeType.NONE)],
            [(5, RecomputeType.NONE), (6, RecomputeType.NONE),
             (7, RecomputeType.NONE), (8, RecomputeType.NONE),
             (9, RecomputeType.NONE)],
        ]
        is_valid, msg = partition.validate_partition(stages)
        self.assertFalse(is_valid)
        self.assertIn("HEAD", msg)

    def test_num_body_layers_mismatch_raises(self) -> None:
        """Test that num_body_layers + 2 != num_layers raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            StagePartition(num_layers=10, pp_degree=2, num_body_layers=7)
        self.assertIn("must equal num_body_layers + 2", str(ctx.exception))

    def test_pp_degree_exceeds_layers(self) -> None:
        """Test pp_degree exceeds number of layers."""
        with self.assertRaises(ValueError) as ctx:
            StagePartition(num_layers=8, pp_degree=10)
        self.assertIn("should not exceed", str(ctx.exception))

    def test_zero_pp_degree_raises(self) -> None:
        """Test pp_degree=0 raises ValueError."""
        with self.assertRaises(ValueError):
            StagePartition(num_layers=8, pp_degree=0)

    def test_zero_num_layers_raises(self) -> None:
        """Test num_layers=0 raises ValueError."""
        with self.assertRaises(ValueError):
            StagePartition(num_layers=0, pp_degree=2)

    def test_negative_num_body_layers_raises(self) -> None:
        """Test negative num_body_layers raises ValueError."""
        with self.assertRaises(ValueError):
            StagePartition(num_layers=8, pp_degree=2, num_body_layers=-1)

    def test_validate_ordering_across_empty_stages(self) -> None:
        """Cross-empty-stage layer disorder should be detected."""
        partition = StagePartition(num_layers=6, pp_degree=4)
        stages = [
            [(4, RecomputeType.NONE), (5, RecomputeType.NONE)],
            [],
            [(0, RecomputeType.NONE), (1, RecomputeType.NONE)],
            [(2, RecomputeType.NONE), (3, RecomputeType.NONE)],
        ]
        is_valid, msg = partition.validate_partition(stages, allow_empty_stages=True)
        self.assertFalse(is_valid)
        self.assertIn("not ordered correctly", msg)

    def test_validate_ordering_with_empty_stage_valid(self) -> None:
        """Properly ordered stages with an empty stage in between should pass."""
        partition = StagePartition(num_layers=4, pp_degree=4)
        stages = [
            [(0, RecomputeType.NONE)],
            [],
            [(1, RecomputeType.NONE), (2, RecomputeType.NONE)],
            [(3, RecomputeType.NONE)],
        ]
        is_valid, msg = partition.validate_partition(stages, allow_empty_stages=True)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")


class TestPulpHasFeasibleSolution(unittest.TestCase):
    """Tests for PPBalancer._pulp_has_feasible_solution static method."""

    def test_sol_status_optimal_returns_true(self) -> None:
        """sol_status=LpSolutionOptimal should return True."""
        from unittest.mock import MagicMock
        from pulp import LpSolutionOptimal  # pylint: disable=C0415
        problem = MagicMock()
        problem.sol_status = LpSolutionOptimal
        self.assertTrue(
            PPBalancer._pulp_has_feasible_solution(problem)
        )

    def test_sol_status_integer_feasible_returns_true(self) -> None:
        """sol_status=LpSolutionIntegerFeasible should return True."""
        from unittest.mock import MagicMock
        from pulp import LpSolutionIntegerFeasible  # pylint: disable=C0415
        problem = MagicMock()
        problem.sol_status = LpSolutionIntegerFeasible
        self.assertTrue(
            PPBalancer._pulp_has_feasible_solution(problem)
        )

    def test_sol_status_no_solution_found_returns_false(self) -> None:
        """sol_status=LpSolutionNoSolutionFound should return False."""
        from unittest.mock import MagicMock
        from pulp import LpSolutionNoSolutionFound  # pylint: disable=C0415
        problem = MagicMock()
        problem.sol_status = LpSolutionNoSolutionFound
        self.assertFalse(
            PPBalancer._pulp_has_feasible_solution(problem)
        )

    def test_sol_status_infeasible_returns_false(self) -> None:
        """sol_status=LpSolutionInfeasible should return False."""
        from unittest.mock import MagicMock
        from pulp import LpSolutionInfeasible  # pylint: disable=C0415
        problem = MagicMock()
        problem.sol_status = LpSolutionInfeasible
        self.assertFalse(
            PPBalancer._pulp_has_feasible_solution(problem)
        )

    def test_exception_returns_false(self) -> None:
        """Problem without sol_status attribute should return False."""
        from unittest.mock import MagicMock
        problem = MagicMock(spec=[])
        self.assertFalse(
            PPBalancer._pulp_has_feasible_solution(problem)
        )

    def test_real_ilp_timeout_no_incumbent(self) -> None:
        """Real PuLP problem with timeLimit=0 should not be mistaken as feasible.

        When CBC times out without finding an incumbent, sol_status is
        ``LpSolutionNoSolutionFound`` even though all variables have
        default varValue assigned.  The old varValue-based check
        incorrectly returned True in this case.
        """
        import pulp  # pylint: disable=C0415
        prob = pulp.LpProblem("timeout_test", pulp.LpMinimize)
        xs = [pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer") for i in range(100)]
        prob += pulp.lpSum(xs)
        for i in range(99):
            prob += xs[i] + xs[i + 1] >= 1
        prob += xs[0] >= 500
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=0))
        self.assertFalse(
            PPBalancer._pulp_has_feasible_solution(prob)
        )


class TestCheckIlpSolveStatus(unittest.TestCase):
    """Tests for PPBalancer._check_ilp_solve_status via mock pipeline."""

    @staticmethod
    def _make_balancer_with_mock_pipeline(pulp_status: int,
                                          has_feasible: bool = True) -> PPBalancer:
        """Build a minimal balancer with a mock pipeline/problem tree.

        Bypasses ``__init__`` to avoid the sapp_ppb availability check
        and layer construction.
        """
        from unittest.mock import MagicMock
        from pulp import (  # pylint: disable=C0415
            LpSolutionIntegerFeasible,
            LpSolutionNoSolutionFound,
        )

        mock_pulp = MagicMock()
        mock_pulp.status = pulp_status
        if has_feasible:
            mock_pulp.sol_status = LpSolutionIntegerFeasible
        else:
            mock_pulp.sol_status = LpSolutionNoSolutionFound

        mock_solver = MagicMock()
        mock_solver.problem_ = mock_pulp

        mock_pipeline = MagicMock()
        mock_pipeline.problem_ = mock_solver

        balancer = object.__new__(PPBalancer)
        balancer._layer_builder = MagicMock()
        balancer._pipeline = mock_pipeline
        balancer._is_successful = False
        return balancer

    def test_optimal_status_passes(self) -> None:
        """Optimal status (1) should return None and set _is_successful to True."""
        balancer = self._make_balancer_with_mock_pipeline(pulp_status=1)
        result = balancer._check_ilp_solve_status()
        self.assertIsNone(result)
        self.assertTrue(balancer._is_successful)

    def test_not_solved_with_feasible_passes(self) -> None:
        """Not Solved (0) with feasible incumbent should return None."""
        balancer = self._make_balancer_with_mock_pipeline(
            pulp_status=0, has_feasible=True,
        )
        result = balancer._check_ilp_solve_status()
        self.assertIsNone(result)
        self.assertTrue(balancer._is_successful)

    def test_not_solved_without_feasible_fails(self) -> None:
        """Not Solved (0) without feasible incumbent should return infeasible output."""
        balancer = self._make_balancer_with_mock_pipeline(
            pulp_status=0, has_feasible=False,
        )
        result = balancer._check_ilp_solve_status()
        self.assertIsNotNone(result)
        self.assertFalse(result.is_feasible)
        self.assertFalse(result.is_successful)

    def test_infeasible_status_fails(self) -> None:
        """Infeasible status (-1) should return infeasible output."""
        balancer = self._make_balancer_with_mock_pipeline(pulp_status=-1)
        result = balancer._check_ilp_solve_status()
        self.assertIsNotNone(result)
        self.assertFalse(result.is_feasible)
        self.assertFalse(result.is_successful)

    def test_undefined_status_fails(self) -> None:
        """Undefined status (-3) should return infeasible output."""
        balancer = self._make_balancer_with_mock_pipeline(pulp_status=-3)
        result = balancer._check_ilp_solve_status()
        self.assertIsNotNone(result)
        self.assertFalse(result.is_feasible)
        self.assertFalse(result.is_successful)


if __name__ == "__main__":
    unittest.main()
