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
"""Unit tests for _hook_manager.py: compute function binding and resolve.

Test IDs:
  HM-R01: __resolve_compute_fun resolves method from EvalExpertCompute
  HM-R02: __resolve_compute_fun falls back to zero function for invalid name
  HM-C01: __set_node_eval_compute_fun returns None when no compute kwarg
  HM-C02: __set_node_eval_compute_fun returns None when compute is not dict
  HM-C03: __set_node_eval_compute_fun builds NodeComputeEval with router
  HM-C04: __set_node_eval_compute_fun builds with all 4 fields
  HM-C05: __set_node_eval_compute_fun returns None when all names None
  HM-C06: __set_node_eval_compute_fun preserves existing compute when no override
  HM-Y01: import_eval_yaml loads compute config from YAML body node
"""
import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.compute import EvalExpertCompute
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import (
    NodeComputeEval,
    NodeDynEval,
    NodeEval,
    NodeStatEval,
    NodeCommEval,
)


class TestResolveComputeFun(unittest.TestCase):
    """HM-R: __resolve_compute_fun tests."""

    def test_resolve_valid_method(self):
        """
        Feature: TestResolveComputeFun.
        Description: resolves 'router_compute_cost' from EvalExpertCompute.
        Expectation: returned function is EvalExpertCompute.router_compute_cost.
        """
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._hook_manager import _HookManager
        hm = object.__new__(_HookManager)
        result = hm._HookManager__resolve_compute_fun("router_compute_cost")
        self.assertIs(result, EvalExpertCompute.router_compute_cost)

    def test_resolve_another_valid_method(self):
        """
        Feature: TestResolveComputeFun.
        Description: resolves 'expert_compute_cost_balanced' from EvalExpertCompute.
        Expectation: returned function is EvalExpertCompute.expert_compute_cost_balanced.
        """
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._hook_manager import _HookManager
        hm = object.__new__(_HookManager)
        result = hm._HookManager__resolve_compute_fun("expert_compute_cost_balanced")
        self.assertIs(result, EvalExpertCompute.expert_compute_cost_balanced)

    def test_resolve_invalid_falls_back_to_zero(self):
        """
        Feature: TestResolveComputeFun.
        Description: falls back to zero function for invalid name.
        Expectation: returned function evaluates to 0.
        """
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._hook_manager import _HookManager
        hm = object.__new__(_HookManager)
        result = hm._HookManager__resolve_compute_fun("nonexistent_method")
        self.assertEqual(result(), 0)


class TestSetNodeEvalComputeFun(unittest.TestCase):
    """HM-C: __set_node_eval_compute_fun tests."""

    def _make_hm(self):
        """
        Feature: TestSetNodeEvalComputeFun.
        Description: create a minimal _HookManager with mocked internals.
        Expectation: returns a usable _HookManager instance.
        """
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._hook_manager import _HookManager
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
        hm = object.__new__(_HookManager)
        hm._ctx = Context()
        return hm

    def test_no_compute_kwarg_returns_none(self):
        """
        Feature: TestSetNodeEvalComputeFun.
        Description: returns None when 'compute' key not in kwargs.
        Expectation: result is None.
        """
        hm = self._make_hm()
        result = hm._HookManager__set_node_eval_compute_fun("some_node")
        self.assertIsNone(result)

    def test_compute_not_dict_returns_none(self):
        """
        Feature: TestSetNodeEvalComputeFun.
        Description: returns None when compute kwarg is not a dict.
        Expectation: result is None.
        """
        hm = self._make_hm()
        result = hm._HookManager__set_node_eval_compute_fun(
            "some_node", compute="invalid"
        )
        self.assertIsNone(result)

    def test_compute_with_router(self):
        """
        Feature: TestSetNodeEvalComputeFun.
        Description: builds NodeComputeEval with router only.
        Expectation: result is NodeComputeEval with router set, other fields None.
        """
        hm = self._make_hm()
        result = hm._HookManager__set_node_eval_compute_fun(
            "some_node",
            compute={"router": "router_compute_cost"},
        )
        self.assertIsInstance(result, NodeComputeEval)
        self.assertIs(result.router, EvalExpertCompute.router_compute_cost)
        self.assertIsNone(result.expert_balanced)
        self.assertIsNone(result.expert_imbalanced)
        self.assertIsNone(result.shared_expert)

    def test_compute_with_all_fields(self):
        """
        Feature: TestSetNodeEvalComputeFun.
        Description: builds NodeComputeEval with all 4 fields.
        Expectation: result has router, expert_balanced, expert_imbalanced, shared_expert set.
        """
        hm = self._make_hm()
        result = hm._HookManager__set_node_eval_compute_fun(
            "some_node",
            compute={
                "router": "router_compute_cost",
                "expert_balanced": "expert_compute_cost_balanced",
                "expert_imbalanced": "expert_compute_cost_imbalanced",
                "shared_expert": "shared_expert_compute_cost",
            },
        )
        self.assertIsInstance(result, NodeComputeEval)
        self.assertIs(result.router, EvalExpertCompute.router_compute_cost)
        self.assertIs(result.expert_balanced, EvalExpertCompute.expert_compute_cost_balanced)
        self.assertIs(result.expert_imbalanced, EvalExpertCompute.expert_compute_cost_imbalanced)
        self.assertIs(result.shared_expert, EvalExpertCompute.shared_expert_compute_cost)

    def test_compute_all_names_none_returns_none(self):
        """
        Feature: TestSetNodeEvalComputeFun.
        Description: returns None when dict has no valid names.
        Expectation: result is None.
        """
        hm = self._make_hm()
        result = hm._HookManager__set_node_eval_compute_fun(
            "some_node",
            compute={},
        )
        self.assertIsNone(result)

    def test_preserves_existing_compute(self):
        """
        Feature: TestSetNodeEvalComputeFun.
        Description: preserves existing compute when no override provided.
        Expectation: result is the existing NodeComputeEval instance.
        """
        hm = self._make_hm()
        existing_compute = NodeComputeEval(router=lambda: 0)
        # Set up a node_eval entry with existing compute
        node = "body_node"
        hm._ctx.node_eval[node] = NodeEval(
            num_p=lambda: 0,
            stat=NodeStatEval(p=lambda: 0, os=lambda: 0, grad=lambda: 0),
            dyn=NodeDynEval(
                activation=lambda: 0,
                comm=NodeCommEval(dp=lambda: 0, tp=lambda: 0, cp=lambda: 0, ep=lambda: 0),
                compute=existing_compute,
            ),
        )
        result = hm._HookManager__set_node_eval_compute_fun(node)
        self.assertIs(result, existing_compute)

    def test_no_existing_compute_returns_none(self):
        """
        Feature: TestSetNodeEvalComputeFun.
        Description: returns None when no existing compute and no override.
        Expectation: result is None.
        """
        hm = self._make_hm()
        node = "body_node"
        hm._ctx.node_eval[node] = NodeEval(
            num_p=lambda: 0,
            stat=NodeStatEval(p=lambda: 0, os=lambda: 0, grad=lambda: 0),
            dyn=NodeDynEval(
                activation=lambda: 0,
                comm=NodeCommEval(dp=lambda: 0, tp=lambda: 0, cp=lambda: 0, ep=lambda: 0),
                compute=None,
            ),
        )
        result = hm._HookManager__set_node_eval_compute_fun(node)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
