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
"""Unit tests for WarmupExecutor."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.auto_parallel.hyper_offload.execution.warmup import WarmupExecutor
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    ActivationTrace,
    TraceOp,
)
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import PhysicalBuffer


class _FakeResidencyManager:
    """Recording fake for WarmupExecutor tests.

    Tracks device-resident sizes in a dict and records ``copy_d2h`` /
    ``release_device`` calls so that tests can verify eviction behaviour
    without needing accelerator hardware.
    """

    def __init__(self) -> None:
        self._resident: dict[int, int] = {}
        self.calls: list[str] = []

    @property
    def resident_bytes(self) -> int:
        return sum(self._resident.values())

    def device_resident_size(self, sid: int) -> int | None:
        return self._resident.get(sid)

    def copy_d2h(self, sid: int) -> None:
        self.calls.append(f"copy_d2h({sid})")

    def release_device(self, sid: int) -> None:
        self.calls.append(f"release_device({sid})")
        self._resident.pop(sid, None)

    def bind(self, sid: int, tensor: torch.Tensor) -> PhysicalBuffer:
        buf = PhysicalBuffer(device=tensor.device)
        self._resident[sid] = tensor.untyped_storage().size()
        return buf

    def clear_runtime(self) -> None:
        self._resident.clear()
        self.calls.clear()


class TestWarmupExecutor(unittest.TestCase):
    """WarmupExecutor lifecycle and trace recording."""

    def setUp(self) -> None:
        # Patch DeviceTimer so that start/stop are no-ops returning 0.0 ms.
        self._timer_patch = patch(
            "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.DeviceTimer"
        )
        self._mock_timer_cls = self._timer_patch.start()
        self._mock_timer_cls.return_value.start.return_value = None
        self._mock_timer_cls.return_value.stop.return_value = 0.0

        self.manager = _FakeResidencyManager()
        self.executor = WarmupExecutor(
            residency_manager=self.manager,
            memory_limit_bytes=1024 * 1024,  # large enough to avoid eviction
        )

    def tearDown(self) -> None:
        self._timer_patch.stop()

    def test_initial_state(self) -> None:
        self.assertEqual(self.executor.op_idx, -1)
        self.assertEqual(len(self.executor._ops), 0)  # pylint: disable=protected-access
        self.assertEqual(len(self.executor._guide), 0)  # pylint: disable=protected-access

    def test_dispatch_records_trace_op(self) -> None:
        """dispatch should record a TraceOp after execution."""
        def my_func(x):
            return x + 1

        result = self.executor.dispatch(my_func, (torch.tensor(1.0),), {})
        self.assertEqual(result, torch.tensor(2.0))
        self.assertEqual(len(self.executor._ops), 1)  # pylint: disable=protected-access
        self.assertEqual(self.executor._ops[0].name, "my_func")  # pylint: disable=protected-access

    def test_dispatch_records_guide(self) -> None:
        """Each dispatch should create an OpGuide entry."""
        self.executor.dispatch(lambda x: x + 1, (torch.tensor(1.0),), {})
        self.assertEqual(len(self.executor._guide), 1)  # pylint: disable=protected-access

    def test_dispatch_output_leaf_count(self) -> None:
        """Output leaf count should match the number of tensor leaves in the result."""
        def my_func(x):
            return (x + 1, x + 2)

        result = self.executor.dispatch(my_func, (torch.tensor(1.0),), {})
        self.assertIsInstance(result, tuple)
        guide = self.executor._guide[0]  # pylint: disable=protected-access
        # The result tuple has 2 tensors
        self.assertEqual(guide.output_leaf_count, 2)

    def test_dispatch_records_guide_leaf_count(self) -> None:
        """Dispatch should record the correct output_leaf_count in the guide."""
        self.executor.dispatch(lambda x: x.sin(), (torch.tensor(0.5),), {})
        guide = self.executor._guide[0]  # pylint: disable=protected-access
        # Output leaf count should be 1 (single tensor output)
        self.assertEqual(guide.output_leaf_count, 1)
        # Output bindings may be empty for CPU tensors (not activations)
        self.assertIsInstance(guide.output_bindings, dict)

    @patch(
        "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
        return_value=(16.0, 16.0),
    )
    def test_finish_returns_trace_and_guide(self, mock_bandwidth: object) -> None:  # pylint: disable=unused-argument
        """finish() should return an ActivationTrace and the guide list."""
        self.executor.dispatch(lambda x: x + 1, (torch.tensor(1.0),), {})
        trace, guide = self.executor.finish()
        self.assertIsInstance(trace, ActivationTrace)
        self.assertEqual(len(trace.ops), 1)
        self.assertEqual(len(guide), 1)
        # finish should reset the executor
        self.assertEqual(self.executor.op_idx, -1)
        self.assertEqual(len(self.executor._ops), 0)  # pylint: disable=protected-access

    def test_multiple_dispatches_create_multiple_ops(self) -> None:
        """Multiple dispatch calls should create multiple trace ops."""
        for i in range(3):
            self.executor.dispatch(lambda x, val=i: x + val, (torch.tensor(0.0),), {})
        self.assertEqual(len(self.executor._ops), 3)  # pylint: disable=protected-access

    def test_reset_clears_all_state(self) -> None:
        """reset() should clear ops, guide, tracker, and call super().reset()."""
        self.executor.dispatch(lambda x: x + 1, (torch.tensor(1.0),), {})
        self.executor.reset()
        self.assertEqual(len(self.executor._ops), 0)  # pylint: disable=protected-access
        self.assertEqual(len(self.executor._guide), 0)  # pylint: disable=protected-access
        self.assertEqual(self.executor.op_idx, -1)
        self.assertEqual(len(self.executor._sid_produced_at_op), 0)  # pylint: disable=protected-access

    def test_enforce_budget_noop_when_under_budget(self) -> None:
        """When resident_bytes <= memory_limit, _enforce_budget should do nothing."""
        old_calls = list(self.manager.calls)
        self.executor._enforce_budget(protected_sids=set())  # pylint: disable=protected-access
        self.assertEqual(self.manager.calls, old_calls)

    def test_enforce_budget_evicts_when_over_budget(self) -> None:
        """With a tight budget, _enforce_budget should evict storage."""
        manager = _FakeResidencyManager()
        executor = WarmupExecutor(
            residency_manager=manager,
            memory_limit_bytes=16,  # very small
        )

        # Register two storages via make_shadow
        t1 = torch.randn(100)  # 400 bytes
        t2 = torch.randn(100)  # 400 bytes
        executor.make_shadow(1, t1)
        executor.make_shadow(2, t2)

        # Manually set _sid_produced_at_op so _enforce_budget can find them
        executor._sid_produced_at_op[1] = 0  # pylint: disable=protected-access
        executor._sid_produced_at_op[2] = 1  # pylint: disable=protected-access

        self.assertGreater(manager.resident_bytes, 16)
        executor._enforce_budget(protected_sids=set())  # pylint: disable=protected-access

        # After eviction, resident_bytes should be at or under budget
        self.assertLessEqual(manager.resident_bytes, 16)

    def test_enforce_budget_protects_sids_raises_when_over_budget(self) -> None:
        """When all SIDs are protected and over budget, should raise RuntimeError."""
        manager = _FakeResidencyManager()
        executor = WarmupExecutor(
            residency_manager=manager,
            memory_limit_bytes=16,
        )

        t = torch.randn(200)  # 800 bytes
        executor.make_shadow(1, t)
        executor._sid_produced_at_op[1] = 0  # pylint: disable=protected-access

        self.assertGreater(manager.resident_bytes, 16)
        with self.assertRaises(RuntimeError):
            executor._enforce_budget(protected_sids={1})  # pylint: disable=protected-access

        # Sid 1 should still be on device (not evicted, but budget violation surfaced)
        self.assertIsNotNone(manager.device_resident_size(1))

    def test_enforce_budget_prefers_oldest_first(self) -> None:
        """Eviction should pick the oldest (smallest produced_at_op) sid first."""
        manager = _FakeResidencyManager()
        executor = WarmupExecutor(
            residency_manager=manager,
            memory_limit_bytes=900,
        )

        t1 = torch.randn(200)  # 800 bytes
        t2 = torch.randn(200)  # 800 bytes
        executor.make_shadow(1, t1)
        executor.make_shadow(2, t2)

        # sid 1 produced at op 5 (newer), sid 2 at op 0 (older)
        executor._sid_produced_at_op[1] = 5  # pylint: disable=protected-access
        executor._sid_produced_at_op[2] = 0  # pylint: disable=protected-access

        executor._enforce_budget(protected_sids=set())  # pylint: disable=protected-access

        # sid 2 (produced at op 0, older) should be evicted.
        self.assertIsNone(manager.device_resident_size(2),
                          msg="sid 2 (oldest, produced_at_op=0) should have been evicted")
        # sid 1 (newer) should remain resident — evicting only sid 2
        # frees 800 bytes, bringing resident_bytes to 800 which is <= 900.
        self.assertIsNotNone(manager.device_resident_size(1),
                             msg="sid 1 (newer, produced_at_op=5) should NOT have been evicted")

    def test_dispatch_inside_opaque_region_skips_recording(self) -> None:
        """When in opaque region, dispatch should not record trace ops."""
        self.executor.enter_opaque_region()
        self.executor.dispatch(lambda x: x + 1, (torch.tensor(1.0),), {})
        self.assertEqual(len(self.executor._ops), 0)  # pylint: disable=protected-access
        self.assertEqual(self.executor.op_idx, -1)

    def test_trace_op_has_duration_ms(self) -> None:
        """Each recorded trace op should have a positive (or zero) duration_ms."""
        self.executor.dispatch(lambda x: x + 1, (torch.tensor(1.0),), {})
        op = self.executor._ops[0]  # pylint: disable=protected-access
        self.assertGreaterEqual(op.duration_ms, 0.0)

    def test_enforce_budget_raises_when_all_protected(self) -> None:
        """_enforce_budget should raise RuntimeError when all sids are protected."""
        manager = _FakeResidencyManager()
        executor = WarmupExecutor(
            residency_manager=manager,
            memory_limit_bytes=16,
        )
        t = torch.randn(200)
        executor.make_shadow(1, t)
        executor._sid_produced_at_op[1] = 0  # pylint: disable=protected-access

        self.assertGreater(manager.resident_bytes, 16)
        with self.assertRaisesRegex(RuntimeError, "no evictable activation"):
            executor._enforce_budget(protected_sids={1})  # pylint: disable=protected-access

    def test_mutated_input_detected_as_write(self) -> None:
        """An in-place op (add_) should mark its mutated input as WRITE access."""
        from hyper_parallel.auto_parallel.hyper_offload.execution.warmup.tracker import ActivationTracker

        op = torch.ops.aten.add_.Tensor

        # Mock get_activation_sid to return non-None so the WRITE access is recorded.
        # Without this, CPU tensors always return None and no access entries are created.
        with patch.object(ActivationTracker, "get_activation_sid", return_value=1):
            ex = WarmupExecutor(
                residency_manager=_FakeResidencyManager(),
                memory_limit_bytes=1024 * 1024,
            )
            ex.dispatch(op, (torch.randn(4), torch.randn(4)), {})
            recorded_op = ex._ops[0]  # pylint: disable=protected-access
            writes = [a for a in recorded_op.accesses if a.kind.name == "WRITE"]
            self.assertGreater(len(writes), 0,
                              msg="add_ should produce at least one WRITE access for the mutated input")

    def test_output_binding_tracks_activation_sid(self) -> None:
        """When output is a tracked activation, output_bindings should be populated."""
        t = torch.randn(4)
        # Manually track the output sid so get_activation_sid returns non-None
        sid = self.executor._tracker._ensure_id(t)  # pylint: disable=protected-access
        self.executor._tracker._activation_sids.add(sid)  # pylint: disable=protected-access

        self.executor.dispatch(lambda x: x + 1, (t,), {})
        guide = self.executor._guide[0]  # pylint: disable=protected-access
        # The output tensor (x+1) is a new tensor with a different storage,
        # so even though t was tracked, the output won't match._
        # To verify the binding logic, use a mock on get_activation_sid.
        from hyper_parallel.auto_parallel.hyper_offload.execution.warmup.tracker import ActivationTracker
        with patch.object(ActivationTracker, "get_activation_sid", return_value=99):
            self.executor2 = WarmupExecutor(
                residency_manager=_FakeResidencyManager(),
                memory_limit_bytes=1024 * 1024,
            )
            self.executor2.dispatch(lambda x: x + 1, (torch.tensor(1.0),), {})
            guide2 = self.executor2._guide[0]  # pylint: disable=protected-access
            self.assertIn(0, guide2.output_bindings,
                          msg="leaf_index=0 should be in output_bindings when sid is tracked")
            self.assertEqual(guide2.output_bindings[0], 99)

    def test_output_binding_with_non_tensor_leaf_skipped(self) -> None:
        """Non-tensor leaves in the output should be skipped in output_bindings."""
        from hyper_parallel.auto_parallel.hyper_offload.execution.warmup.tracker import ActivationTracker

        def mixed_output(x):
            return (x + 1, "string_leaf", 42)

        with patch.object(ActivationTracker, "get_activation_sid", return_value=42):
            ex = WarmupExecutor(
                residency_manager=_FakeResidencyManager(),
                memory_limit_bytes=1024 * 1024,
            )
            ex.dispatch(mixed_output, (torch.tensor(1.0),), {})
            guide = ex._guide[0]  # pylint: disable=protected-access
            self.assertEqual(guide.output_leaf_count, 3)
            # Leaf 0 is a tensor -> should be bound
            self.assertIn(0, guide.output_bindings,
                          msg="tensor leaf should be in output_bindings")
            # Leaves 1 (str) and 2 (int) -> should NOT be bound
            self.assertNotIn(1, guide.output_bindings)
            self.assertNotIn(2, guide.output_bindings)
