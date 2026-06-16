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
"""Unit tests for IR data structures (trace, replay, schedule)."""

from __future__ import annotations

import copy
import unittest

from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyAction,
    ResidencyActionType,
    ResidencySchedule,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
    StorageAccess,
    TraceOp,
)


class TestAccessKind(unittest.TestCase):
    """AccessKind enum behaviour."""

    def test_read_value(self) -> None:
        self.assertIsInstance(AccessKind.READ, AccessKind)

    def test_write_value(self) -> None:
        self.assertIsInstance(AccessKind.WRITE, AccessKind)

    def test_read_not_write(self) -> None:
        self.assertIsNot(AccessKind.READ, AccessKind.WRITE)

    def test_auto_values(self) -> None:
        """Both values should be auto-assigned (no duplicate values)."""
        self.assertNotEqual(AccessKind.READ.value, AccessKind.WRITE.value)


class TestStorageAccess(unittest.TestCase):
    """StorageAccess dataclass."""

    def test_creation(self) -> None:
        acc = StorageAccess(op_id=5, storage_id=42, kind=AccessKind.READ)
        self.assertEqual(acc.op_id, 5)
        self.assertEqual(acc.storage_id, 42)
        self.assertEqual(acc.kind, AccessKind.READ)

    def test_defaults(self) -> None:
        acc = StorageAccess(op_id=0, storage_id=1, kind=AccessKind.WRITE)
        self.assertEqual(acc.op_id, 0)

    def test_equality(self) -> None:
        a1 = StorageAccess(1, 10, AccessKind.READ)
        a2 = StorageAccess(1, 10, AccessKind.READ)
        self.assertEqual(a1, a2)

    def test_inequality(self) -> None:
        a1 = StorageAccess(1, 10, AccessKind.READ)
        a2 = StorageAccess(2, 10, AccessKind.READ)
        self.assertNotEqual(a1, a2)

    def test_repr(self) -> None:
        acc = StorageAccess(3, 7, AccessKind.WRITE)
        r = repr(acc)
        self.assertIn("op_id=3", r)
        self.assertIn("storage_id=7", r)
        self.assertIn("WRITE", r)


class TestTraceOp(unittest.TestCase):
    """TraceOp dataclass."""

    def test_creation(self) -> None:
        op = TraceOp(name="test_op", duration_ms=1.5)
        self.assertEqual(op.name, "test_op")
        self.assertEqual(op.duration_ms, 1.5)
        self.assertEqual(op.accesses, [])

    def test_default_duration(self) -> None:
        op = TraceOp(name="no_duration")
        self.assertEqual(op.duration_ms, 0.0)

    def test_accesses_mutable(self) -> None:
        op = TraceOp(name="op")
        acc = StorageAccess(0, 1, AccessKind.READ)
        op.accesses.append(acc)
        self.assertEqual(len(op.accesses), 1)
        self.assertIs(op.accesses[0], acc)

    def test_equality(self) -> None:
        op1 = TraceOp("a", 1.0, [StorageAccess(0, 1, AccessKind.READ)])
        op2 = TraceOp("a", 1.0, [StorageAccess(0, 1, AccessKind.READ)])
        self.assertEqual(op1, op2)


class TestActivationTrace(unittest.TestCase):
    """ActivationTrace dataclass."""

    def test_creation(self) -> None:
        trace = ActivationTrace(memory_limit_bytes=1024)
        self.assertEqual(trace.ops, [])
        self.assertEqual(trace.storage_sizes, {})
        self.assertEqual(trace.retained_sids, set())
        self.assertEqual(trace.memory_limit_bytes, 1024)
        self.assertEqual(trace.d2h_bandwidth_gbps, 16.0)
        self.assertEqual(trace.h2d_bandwidth_gbps, 16.0)

    def test_default_memory_limit(self) -> None:
        trace = ActivationTrace()
        self.assertIsNone(trace.memory_limit_bytes)

    def test_with_ops(self) -> None:
        op = TraceOp("test")
        trace = ActivationTrace(ops=[op], storage_sizes={1: 256}, retained_sids={1})
        self.assertEqual(len(trace.ops), 1)
        self.assertEqual(trace.storage_sizes[1], 256)
        self.assertIn(1, trace.retained_sids)

    def test_copy_semantics(self) -> None:
        """ActivationTrace should be a regular dataclass (no defensive copy)."""
        trace = ActivationTrace()
        trace.ops.append(TraceOp("a"))
        self.assertEqual(len(trace.ops), 1)


class TestOpGuide(unittest.TestCase):
    """OpGuide dataclass."""

    def test_creation(self) -> None:
        guide = OpGuide(name="test", output_leaf_count=3, output_bindings={0: 1, 2: 5})
        self.assertEqual(guide.name, "test")
        self.assertEqual(guide.output_leaf_count, 3)
        self.assertDictEqual(guide.output_bindings, {0: 1, 2: 5})

    def test_defaults(self) -> None:
        guide = OpGuide(name="empty")
        self.assertEqual(guide.output_leaf_count, 0)
        self.assertEqual(guide.output_bindings, {})

    def test_deepcopy(self) -> None:
        guide = OpGuide(name="g", output_leaf_count=2, output_bindings={0: 10})
        copied = copy.deepcopy(guide)
        self.assertEqual(copied.name, "g")
        self.assertEqual(copied.output_bindings, {0: 10})


class TestResidencyActionType(unittest.TestCase):
    """ResidencyActionType enum."""

    def test_all_members(self) -> None:
        self.assertIsInstance(ResidencyActionType.COPY_D2H, ResidencyActionType)
        self.assertIsInstance(ResidencyActionType.COPY_H2D, ResidencyActionType)
        self.assertIsInstance(ResidencyActionType.RELEASE_DEVICE, ResidencyActionType)
        self.assertIsInstance(ResidencyActionType.RELEASE_HOST, ResidencyActionType)

    def test_members_unique(self) -> None:
        names = [m.name for m in ResidencyActionType]
        self.assertEqual(len(names), len(set(names)))

    def test_str(self) -> None:
        self.assertEqual(str(ResidencyActionType.COPY_D2H), "ResidencyActionType.COPY_D2H")


class TestResidencyAction(unittest.TestCase):
    """ResidencyAction dataclass."""

    def test_creation(self) -> None:
        action = ResidencyAction(op_id=3, storage_id=7, kind=ResidencyActionType.COPY_D2H)
        self.assertEqual(action.op_id, 3)
        self.assertEqual(action.storage_id, 7)
        self.assertEqual(action.kind, ResidencyActionType.COPY_D2H)

    def test_frozen(self) -> None:
        action = ResidencyAction(0, 1, ResidencyActionType.COPY_D2H)
        with self.assertRaises(AttributeError):
            action.op_id = 99

    def test_hashable(self) -> None:
        a1 = ResidencyAction(0, 1, ResidencyActionType.COPY_D2H)
        a2 = ResidencyAction(0, 1, ResidencyActionType.COPY_D2H)
        self.assertEqual(hash(a1), hash(a2))

    def test_equality(self) -> None:
        a1 = ResidencyAction(0, 1, ResidencyActionType.COPY_D2H)
        a2 = ResidencyAction(0, 1, ResidencyActionType.COPY_D2H)
        self.assertEqual(a1, a2)


class TestResidencySchedule(unittest.TestCase):
    """ResidencySchedule behaviour."""

    def test_empty_schedule(self) -> None:
        sched = ResidencySchedule()
        self.assertEqual(sched.pre_actions(0), [])
        self.assertEqual(sched.post_actions(0), [])

    def test_add_pre(self) -> None:
        sched = ResidencySchedule()
        sched.add_pre(0, ResidencyActionType.COPY_H2D, 1)
        actions = sched.pre_actions(0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].op_id, 0)
        self.assertEqual(actions[0].storage_id, 1)
        self.assertEqual(actions[0].kind, ResidencyActionType.COPY_H2D)

    def test_add_post(self) -> None:
        sched = ResidencySchedule()
        sched.add_post(1, ResidencyActionType.RELEASE_DEVICE, 2)
        actions = sched.post_actions(1)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, ResidencyActionType.RELEASE_DEVICE)

    def test_multiple_actions_same_op(self) -> None:
        sched = ResidencySchedule()
        sched.add_post(0, ResidencyActionType.COPY_D2H, 1)
        sched.add_post(0, ResidencyActionType.RELEASE_DEVICE, 1)
        actions = sched.post_actions(0)
        self.assertEqual(len(actions), 2)

    def test_actions_per_op_isolated(self) -> None:
        sched = ResidencySchedule()
        sched.add_pre(0, ResidencyActionType.COPY_H2D, 1)
        sched.add_pre(2, ResidencyActionType.COPY_H2D, 2)
        self.assertEqual(len(sched.pre_actions(0)), 1)
        self.assertEqual(len(sched.pre_actions(1)), 0)
        self.assertEqual(len(sched.pre_actions(2)), 1)

    def test_no_key_error_for_missing_op(self) -> None:
        sched = ResidencySchedule()
        self.assertEqual(sched.pre_actions(999), [])
        self.assertEqual(sched.post_actions(999), [])

    def test_pre_actions_returns_list_ref(self) -> None:
        """pre_actions returns a reference to the internal list."""
        sched = ResidencySchedule()
        sched.add_pre(0, ResidencyActionType.COPY_H2D, 1)
        actions = sched.pre_actions(0)
        # The internal list is shared; modifying the returned list modifies the schedule.
        self.assertEqual(len(actions), 1)
        actions.append(ResidencyAction(0, 2, ResidencyActionType.COPY_H2D))
        self.assertEqual(len(sched.pre_actions(0)), 2)
