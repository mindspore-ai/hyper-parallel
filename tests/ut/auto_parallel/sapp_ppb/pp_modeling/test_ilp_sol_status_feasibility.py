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
"""Tests for sol_status-based feasibility detection (Issue 1 fix).

Verifies that ``_pulp_has_feasible_solution`` and ``_check_ilp_solve_status``
use PuLP's ``sol_status`` (``LpSolutionOptimal`` / ``LpSolutionIntegerFeasible``)
instead of merely checking whether variables have ``varValue``.

Covers:
- Infeasible problem where variables still have varValue
- Not-Solved status with no incumbent
- Optimal and Integer-Feasible incumbent statuses
- Empty problem
- End-to-end infeasible result via ``balance_with_ilp``
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pulp

from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_structs import (
    PPBOutput,
)
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.yaml_parser import YamlOptimizationConfig
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE
from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_balancer import PPBalancer
from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import LayerBuilder

_DEMO_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixture_profile_32layers.json"
)


def _make_infeasible_problem() -> pulp.LpProblem:
    """Create an infeasible problem: x >= 5 AND x <= 1."""
    prob = pulp.LpProblem("infeasible_test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, cat="Integer")
    prob += x
    prob += x >= 5
    prob += x <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob


def _make_feasible_problem() -> pulp.LpProblem:
    """Create a simple feasible problem: minimize x subject to x <= 3."""
    prob = pulp.LpProblem("feasible_test", pulp.LpMinimize)
    x = pulp.LpVariable("y", lowBound=0, cat="Integer")
    prob += x
    prob += x <= 3
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob


class TestPulpHasFeasibleSolution(unittest.TestCase):
    """Test ``PPBalancer._pulp_has_feasible_solution`` uses sol_status."""

    def test_infeasible_var_value_exists_sol_status_rejects(self) -> None:
        """Variables have varValue even when problem is infeasible; sol_status must reject."""
        prob = _make_infeasible_problem()
        x = prob.variables()[0]
        self.assertIsNotNone(
            x.varValue,
            "Bug premise: PuLP assigns varValue even for infeasible problems",
        )
        self.assertEqual(prob.status, pulp.LpStatusInfeasible)
        self.assertEqual(prob.sol_status, pulp.LpSolutionInfeasible)
        self.assertFalse(
            PPBalancer._pulp_has_feasible_solution(prob),
            "Infeasible problem must NOT be reported as feasible",
        )

    def test_not_solved_no_incumbent_returns_infeasible(self) -> None:
        """Not-Solved with LpSolutionNoSolutionFound has no incumbent → not feasible."""
        prob = pulp.LpProblem("mock_not_solved", pulp.LpMinimize)
        prob.status = pulp.LpStatusNotSolved
        prob.sol_status = pulp.LpSolutionNoSolutionFound
        self.assertFalse(
            PPBalancer._pulp_has_feasible_solution(prob),
            "Not-Solved without incumbent must NOT be reported as feasible",
        )

    def test_optimal_problem_has_feasible_solution(self) -> None:
        """Optimal sol_status should be reported as feasible."""
        prob = _make_feasible_problem()
        self.assertEqual(prob.status, pulp.LpStatusOptimal)
        self.assertEqual(prob.sol_status, pulp.LpSolutionOptimal)
        self.assertTrue(
            PPBalancer._pulp_has_feasible_solution(prob),
            "Optimal problem should be reported as feasible",
        )

    def test_integer_feasible_incumbent_has_feasible_solution(self) -> None:
        """Integer-feasible incumbent (CBC timeout with incumbent) → feasible."""
        prob = pulp.LpProblem("mock_timeout_incumbent", pulp.LpMinimize)
        prob.status = pulp.LpStatusNotSolved
        prob.sol_status = pulp.LpSolutionIntegerFeasible
        self.assertTrue(
            PPBalancer._pulp_has_feasible_solution(prob),
            "Integer-feasible incumbent should be reported as feasible",
        )

    def test_empty_problem_detected_correctly(self) -> None:
        """Empty problem (no vars/constraints) gets OPTIMAL from PuLP → feasible."""
        prob = pulp.LpProblem("empty_test", pulp.LpMinimize)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        self.assertEqual(prob.status, pulp.LpStatusOptimal)
        self.assertTrue(
            PPBalancer._pulp_has_feasible_solution(prob),
            "Empty problem with OPTIMAL sol_status should be feasible",
        )

    def test_missing_sol_status_attribute_returns_false(self) -> None:
        """Object without sol_status should return False (no crash)."""
        obj = SimpleNamespace()
        self.assertFalse(
            PPBalancer._pulp_has_feasible_solution(obj),
            "Missing sol_status should safely return False",
        )


class TestCheckIlpSolveStatus(unittest.TestCase):
    """Test ``PPBalancer._check_ilp_solve_status`` with mocked pipeline."""

    def _make_balancer_with_mock_pipeline(self, pulp_problem: pulp.LpProblem) -> PPBalancer:
        """Create a minimal balancer and wire in a mock pipeline with the given PuLP problem."""
        mock_solver = MagicMock()
        mock_solver.problem_ = pulp_problem

        mock_pipeline = MagicMock()
        mock_pipeline.problem_ = mock_solver

        balancer = object.__new__(PPBalancer)
        balancer._layer_builder = MagicMock()
        balancer._pipeline = mock_pipeline
        balancer._is_successful = False
        return balancer

    def test_infeasible_problem_check_ilp_solve_status_returns_infeasible(self) -> None:
        """Infeasible problem → _check_ilp_solve_status returns is_feasible=False."""
        prob = _make_infeasible_problem()
        balancer = self._make_balancer_with_mock_pipeline(prob)
        result = balancer._check_ilp_solve_status()
        self.assertIsNotNone(result)
        self.assertFalse(result.is_feasible)
        self.assertFalse(result.is_successful)
        self.assertIn("infeasible", result.infeasibility_details.get("reason", "").lower())

    def test_optimal_problem_check_ilp_solve_status_returns_none(self) -> None:
        """Optimal problem → _check_ilp_solve_status returns None (proceed to extract)."""
        prob = _make_feasible_problem()
        balancer = self._make_balancer_with_mock_pipeline(prob)
        result = balancer._check_ilp_solve_status()
        self.assertIsNone(result, "Optimal problem should return None to proceed")

    def test_not_solved_no_incumbent_check_returns_not_solved(self) -> None:
        """Not-Solved without incumbent → infeasible output with is_successful=False."""
        prob = pulp.LpProblem("mock_not_solved", pulp.LpMinimize)
        prob.status = pulp.LpStatusNotSolved
        prob.sol_status = pulp.LpSolutionNoSolutionFound
        balancer = self._make_balancer_with_mock_pipeline(prob)
        result = balancer._check_ilp_solve_status()
        self.assertIsNotNone(result)
        self.assertFalse(result.is_feasible)
        self.assertFalse(result.is_successful)

    def test_not_solved_with_incumbent_check_returns_none(self) -> None:
        """Not-Solved with integer-feasible incumbent → None (proceed, _is_successful=True)."""
        prob = pulp.LpProblem("mock_timeout_incumbent", pulp.LpMinimize)
        prob.status = pulp.LpStatusNotSolved
        prob.sol_status = pulp.LpSolutionIntegerFeasible
        balancer = self._make_balancer_with_mock_pipeline(prob)
        result = balancer._check_ilp_solve_status()
        self.assertIsNone(result)
        self.assertTrue(balancer._is_successful)


@unittest.skipUnless(SAPP_PPB_AVAILABLE, "sapp_ppb not available")
class TestInfeasibleEndToEnd(unittest.TestCase):
    """End-to-end: balance_with_ilp returns infeasible for impossible constraints."""

    def test_infeasible_balancer_end_to_end(self) -> None:
        """Impossibly small memory_limit → ILP infeasible, is_feasible=False."""
        yaml_config = YamlOptimizationConfig(
            num_layer=32, pp_degree=2, micro_batch_num=4, memory_limit=1,
        )
        layer_builder = LayerBuilder(yaml_config, _DEMO_JSON)
        balancer = PPBalancer(layer_builder)
        output = balancer.balance_with_ilp(time_limit=30, solver="pulp")
        self.assertFalse(
            output.is_feasible,
            f"Should be infeasible with memory_limit=1, got: {output.infeasibility_details}",
        )
        self.assertFalse(output.is_successful)


if __name__ == "__main__":
    unittest.main()
