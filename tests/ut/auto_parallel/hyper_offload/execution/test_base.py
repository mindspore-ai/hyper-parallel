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
"""Unit tests for BaseExecutor."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from hyper_parallel.auto_parallel.hyper_offload.execution.base import BaseExecutor
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import PhysicalBuffer, ResidencyManager


class _ConcreteExecutor(BaseExecutor):
    """Concrete subclass for testing BaseExecutor."""

    def __init__(self, residency_manager: ResidencyManager) -> None:
        super().__init__(residency_manager)
        self.on_op_end_called_with: list = []
        self.apply_shadows_called = False

    def on_op_end(self, result) -> object:
        """Implement abstract method."""
        self.on_op_end_called_with.append(result)
        return self.apply_shadows(result, {})


class _FakeResidencyManager:
    """Minimal fake that satisfies the ResidencyManager interface used by BaseExecutor."""

    def __init__(self) -> None:
        self._residency: dict[int, PhysicalBuffer] = {}

    def bind(self, sid: int, tensor: torch.Tensor) -> PhysicalBuffer:
        buf = PhysicalBuffer(device=tensor.device)
        self._residency[sid] = buf
        return buf

    def clear_runtime(self) -> None:
        self._residency.clear()


class TestBaseExecutor(unittest.TestCase):
    """BaseExecutor lifecycle hooks and orchestration."""

    def setUp(self) -> None:
        self.manager = _FakeResidencyManager()
        self.executor = _ConcreteExecutor(self.manager)

    def test_initial_state(self) -> None:
        self.assertEqual(self.executor.op_idx, -1)
        self.assertEqual(self.executor._opaque_depth, 0)  # pylint: disable=protected-access
        self.assertFalse(self.executor.in_opaque_region)

    def test_on_op_begin_increments_op_idx(self) -> None:
        self.executor.on_op_begin(MagicMock(), (), {})
        self.assertEqual(self.executor.op_idx, 0)
        self.assertEqual(self.executor._last_func is not None, True)

    def test_on_op_begin_caches_args(self) -> None:
        func = MagicMock()
        args = (1, 2)
        kwargs = {"a": 3}
        self.executor.on_op_begin(func, args, kwargs)
        self.assertIs(self.executor._last_func, func)
        self.assertIs(self.executor._last_args, args)
        self.assertIs(self.executor._last_kwargs, kwargs)

    def test_dispatch_calls_on_op_begin_and_on_op_end(self) -> None:
        result = self.executor.dispatch(lambda x: x + 1, (torch.tensor(1.0),), {})
        self.assertEqual(self.executor.op_idx, 0)
        self.assertEqual(len(self.executor.on_op_end_called_with), 1)
        self.assertIsNotNone(result)

    def test_dispatch_inside_opaque_region_skips_hooks(self) -> None:
        self.executor.enter_opaque_region()
        self.assertTrue(self.executor.in_opaque_region)

        result = self.executor.dispatch(lambda x: x + 1, (torch.tensor(1.0),), {})
        # Hooks should NOT have been called
        self.assertEqual(self.executor.op_idx, -1)
        self.assertEqual(len(self.executor.on_op_end_called_with), 0)
        self.assertIsNotNone(result)

    def test_opaque_region_depth_tracking(self) -> None:
        """Nested opaque region depth tracking works correctly."""
        self.assertEqual(self.executor._opaque_depth, 0)  # pylint: disable=protected-access
        self.executor.enter_opaque_region()
        self.assertEqual(self.executor._opaque_depth, 1)
        self.assertTrue(self.executor.in_opaque_region)
        self.executor.enter_opaque_region()
        self.assertEqual(self.executor._opaque_depth, 2)
        self.executor.exit_opaque_region()
        self.assertEqual(self.executor._opaque_depth, 1)
        self.assertTrue(self.executor.in_opaque_region)
        self.executor.exit_opaque_region()
        self.assertEqual(self.executor._opaque_depth, 0)
        self.assertFalse(self.executor.in_opaque_region)

    def test_exit_opaque_region_below_zero_does_not_raise(self) -> None:
        """Calling exit_opaque_region when not in region should not crash."""
        self.executor.exit_opaque_region()
        self.assertEqual(self.executor._opaque_depth, -1)

    def test_retained_sids_empty_initially(self) -> None:
        self.assertEqual(self.executor.retained_sids, set())

    def test_make_shadow_creates_shadow(self) -> None:
        t = torch.randn(2, 3)
        shadow = self.executor.make_shadow(42, t)
        from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
        self.assertIsInstance(shadow, ShadowTensor)
        self.assertEqual(shadow.storage_id, 42)
        # Should be tracked in _alive_shadows
        self.assertIn(42, self.executor._alive_shadows)  # pylint: disable=protected-access

    def test_make_shadow_on_existing_shadow_returns_same(self) -> None:
        t = torch.randn(2, 3)
        shadow1 = self.executor.make_shadow(42, t)
        # make_shadow on a ShadowTensor returns the same instance
        shadow2 = self.executor.make_shadow(42, shadow1)
        self.assertIs(shadow2, shadow1)

    def test_retained_sids_after_make_shadow(self) -> None:
        t = torch.randn(2, 3)
        shadow = self.executor.make_shadow(1, t)  # keep reference
        self.assertIn(1, self.executor.retained_sids)
        # Once shadow reference is dropped, sid should be removed from retained_sids
        del shadow
        self.assertNotIn(1, self.executor.retained_sids)

    def test_apply_shadows_wraps_bindings(self) -> None:
        leaves = [torch.randn(2), torch.randn(3), torch.randn(4)]
        result = self.executor.apply_shadows(leaves, {0: 10, 2: 20})
        from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
        self.assertIsInstance(result[0], ShadowTensor)
        self.assertNotIsInstance(result[1], ShadowTensor)
        self.assertIsInstance(result[2], ShadowTensor)
        self.assertEqual(result[0].storage_id, 10)
        self.assertEqual(result[2].storage_id, 20)

    def test_reset_clears_state(self) -> None:
        t = torch.randn(2, 3)
        self.executor.on_op_begin(MagicMock(), (), {})
        self.executor.make_shadow(1, t)
        self.executor.reset()
        self.assertEqual(self.executor.op_idx, -1)
        self.assertEqual(len(self.executor._alive_shadows), 0)  # pylint: disable=protected-access
        self.assertEqual(self.executor.retained_sids, set())

    def test_execute_opaque_op_calls_hooks_and_returns_result(self) -> None:
        """execute_opaque_op should call on_op_begin/on_op_end and return the function result."""
        def my_fn(x):
            return x + 1

        result = self.executor.execute_opaque_op("my_fn", my_fn, (torch.tensor(5.0),), {})
        # op_idx should be 0 (one op tracked)
        self.assertEqual(self.executor.op_idx, 0)
        # result should be the original function output
        self.assertEqual(result, torch.tensor(6.0))

    def test_execute_opaque_op_inside_opaque_region_bypasses(self) -> None:
        """When already inside an opaque region, execute_opaque_op should call the function directly."""
        self.executor.enter_opaque_region()

        def my_fn(x):
            return x + 1

        result = self.executor.execute_opaque_op("my_fn", my_fn, (torch.tensor(5.0),), {})
        self.assertEqual(result, torch.tensor(6.0))
        # No hooks should have been called
        self.assertEqual(self.executor.op_idx, -1)

    def test_execute_opaque_op_preserves_func_name(self) -> None:
        """The func_name should be used in the wrapper names (fwd/bwd)."""
        def my_fn(x):
            return x.sin()

        result = self.executor.execute_opaque_op("custom_name", my_fn, (torch.tensor(0.5),), {})
        # Should not crash, and names should contain "custom_name_fwd" etc.
        self.assertAlmostEqual(result.item(), 0.479, places=3)
