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
"""Unit tests for ReplayExecutor."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.auto_parallel.hyper_offload.execution.replay import ReplayExecutor
from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyActionType,
    ResidencySchedule,
)
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import PhysicalBuffer
from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor




class _RecordingFake:
    """Fake ResidencyManager that records method calls for verification.

    Does **not** inherit from :class:`ResidencyManager`, so the replay
    executor tests never need real accelerator or pinned-memory setup.
    """

    def __init__(self) -> None:
        self._residency: dict[int, PhysicalBuffer] = {}
        self.calls: list[str] = []

    def copy_d2h(self, sid: int) -> None:
        self.calls.append(f"copy_d2h({sid})")
        if sid not in self._residency:
            raise RuntimeError(f"copy_d2h sid={sid}: no physical buffer registered")

    def copy_h2d(self, sid: int) -> None:
        self.calls.append(f"copy_h2d({sid})")
        if sid not in self._residency:
            raise RuntimeError(f"copy_h2d sid={sid}: no physical buffer registered")

    def release_device(self, sid: int) -> None:
        self.calls.append(f"release_device({sid})")

    def release_host(self, sid: int) -> None:
        self.calls.append(f"release_host({sid})")

    def bind(self, sid: int, tensor: torch.Tensor) -> PhysicalBuffer:
        if sid not in self._residency:
            self._residency[sid] = PhysicalBuffer(device=tensor.device)
        return self._residency[sid]

    def clear_runtime(self) -> None:
        self._residency.clear()

    def wait_for_transfers(self) -> None:
        pass

    def sync_all_transfers(self) -> None:
        pass

    @property
    def resident_bytes(self) -> int:
        return 0

    def device_resident_size(self, sid: int) -> int | None:
        buf = self._residency.get(sid)
        if buf is None or buf.device_buffer is None:
            return None
        return buf.device_buffer.numel()


class TestReplayExecutor(unittest.TestCase):
    """ReplayExecutor: schedule execution and output validation."""

    def setUp(self) -> None:
        self.manager = _RecordingFake()
        self.schedule = ResidencySchedule()
        self.guide = [
            OpGuide(name="op0", output_leaf_count=1, output_bindings={0: 10}),
        ]
        self.executor = ReplayExecutor(self.manager, self.schedule, self.guide)

    # ------------------------------------------------------------------
    # Basic lifecycle
    # ------------------------------------------------------------------

    def test_initial_state(self) -> None:
        self.assertEqual(self.executor.op_idx, -1)

    def test_on_op_begin_increments_op_idx(self) -> None:
        self.executor.on_op_begin(lambda x: x, (torch.tensor(1.0),), {})
        self.assertEqual(self.executor.op_idx, 0)

    def test_on_op_end_returns_result(self) -> None:
        self.executor.on_op_begin(lambda x: x, (torch.tensor(1.0),), {})
        result = self.executor.on_op_end(torch.tensor(2.0))
        self.assertIsNotNone(result)

    # ------------------------------------------------------------------
    # Guide validation
    # ------------------------------------------------------------------

    def test_op_idx_exceeds_guide_raises(self) -> None:
        """When op_idx goes beyond the guide length, on_op_begin should raise."""
        executor = ReplayExecutor(self.manager, self.schedule, self.guide)
        executor.on_op_begin(lambda x: x, (), {})  # op_idx = 0, ok
        with self.assertRaisesRegex(RuntimeError, "replay op count exceeds warmup trace"):
            executor.on_op_begin(lambda x: x, (), {})  # op_idx = 1, but guide has only 1 entry

    def test_guide_bound_exceeded_raises_in_on_op_begin(self) -> None:
        """When on_op_begin is called beyond guide length, it should raise."""
        guide = [OpGuide(name="only_op", output_leaf_count=1)]
        executor = ReplayExecutor(self.manager, self.schedule, guide)
        executor.on_op_begin(lambda x: x, (), {})  # idx 0
        with self.assertRaisesRegex(RuntimeError, "replay op count exceeds warmup trace"):
            executor.on_op_begin(lambda x: x, (), {})  # idx 1 - no guide

    def test_output_leaf_count_mismatch_raises(self) -> None:
        """If the number of output leaves changes, on_op_end should raise."""
        self.executor.on_op_begin(lambda x: x, (), {})
        with self.assertRaisesRegex(RuntimeError, "replay output structure differs"):
            self.executor.on_op_end((torch.tensor(1.0), torch.tensor(2.0)))  # 2 leaves, expected 1

    def test_output_leaf_count_match_passes(self) -> None:
        """When output leaf count matches, no error."""
        self.executor.on_op_begin(lambda x: x, (torch.tensor(1.0),), {})
        result = self.executor.on_op_end(torch.tensor(2.0))  # 1 leaf
        self.assertIsNotNone(result)

    # ------------------------------------------------------------------
    # Pre-actions
    # ------------------------------------------------------------------

    def test_on_op_begin_executes_pre_actions(self) -> None:
        """Pre-actions should be executed before the op."""
        self.schedule.add_pre(0, ResidencyActionType.COPY_H2D, 1)

        # Register storage with host data for H2D
        host_data = torch.arange(8, dtype=torch.float32)
        host_buf = host_data.view(dtype=torch.uint8).clone()
        self.manager._residency[1] = PhysicalBuffer(  # pylint: disable=protected-access
            device=torch.device("cpu"),
            host_buffer=host_buf,
        )

        self.executor.on_op_begin(lambda x: x, (), {})
        self.assertIn("copy_h2d(1)", self.manager.calls)

    def test_unsupported_pre_action_raises(self) -> None:
        """An unsupported pre-action kind should raise."""
        self.schedule.add_pre(0, ResidencyActionType.RELEASE_DEVICE, 1)
        with self.assertRaisesRegex(RuntimeError, "unsupported pre action"):
            self.executor.on_op_begin(lambda x: x, (), {})

    # ------------------------------------------------------------------
    # Post-actions
    # ------------------------------------------------------------------

    def test_on_op_end_executes_post_actions(self) -> None:
        """Post-actions should be executed after the op."""
        self.schedule.add_post(0, ResidencyActionType.RELEASE_DEVICE, 1)

        self.manager.bind(1, torch.randn(4, 4))
        self.manager.copy_d2h(1)

        self.executor.on_op_begin(lambda x: x, (), {})
        self.executor.on_op_end(torch.tensor(1.0))
        self.assertIn("release_device(1)", self.manager.calls)

    def test_post_action_copy_d2h(self) -> None:
        """COPY_D2H post-action should call copy_d2h on the manager."""
        self.schedule.add_post(0, ResidencyActionType.COPY_D2H, 1)
        self.manager.bind(1, torch.randn(4, 4))

        self.executor.on_op_begin(lambda x: x, (), {})
        self.executor.on_op_end(torch.tensor(1.0))
        self.assertIn("copy_d2h(1)", self.manager.calls)

    def test_post_action_release_host(self) -> None:
        """RELEASE_HOST post-action should call release_host."""
        self.schedule.add_post(0, ResidencyActionType.RELEASE_HOST, 1)
        self.manager.bind(1, torch.randn(4, 4))
        self.manager.copy_d2h(1)

        self.executor.on_op_begin(lambda x: x, (), {})
        self.executor.on_op_end(torch.tensor(1.0))
        self.assertIn("release_host(1)", self.manager.calls)

    def test_unsupported_post_action_raises(self) -> None:
        """An unsupported post-action kind should raise."""
        # Create a fake action type
        from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencyAction
        self.schedule.post[0] = [
            ResidencyAction(0, 1, ResidencyActionType.COPY_H2D),  # H2D is not valid as post-action
        ]
        self.executor.on_op_begin(lambda x: x, (), {})
        with self.assertRaisesRegex(RuntimeError, "unsupported post action"):
            self.executor.on_op_end(torch.tensor(1.0))

    # ------------------------------------------------------------------
    # Output binding
    # ------------------------------------------------------------------

    def test_output_binding_by_leaf_index(self) -> None:
        """Output binding should use warmup leaf index for shadow creation."""
        guide = [
            OpGuide(
                name="dummy",
                output_leaf_count=2,
                output_bindings={1: 7},
            )
        ]
        executor = ReplayExecutor(self.manager, self.schedule, guide)

        result = (torch.randn(2, 2), torch.randn(2))

        executor.on_op_begin(lambda: None, (), {})
        wrapped = executor.on_op_end(result)

        self.assertNotIsInstance(wrapped[0], ShadowTensor)
        self.assertIsInstance(wrapped[1], ShadowTensor)
        self.assertEqual(wrapped[1].storage_id, 7)

    def test_copy_d2h_on_unregistered_sid_raises(self) -> None:
        """copy_d2h on a storage with no physical buffer should raise RuntimeError."""
        schedule = ResidencySchedule()
        schedule.add_post(0, ResidencyActionType.COPY_D2H, 1)
        guide = [OpGuide(name="dummy", output_leaf_count=1)]
        ex = ReplayExecutor(self.manager, schedule, guide)

        ex.on_op_begin(lambda: torch.randn(2), (), {})
        with self.assertRaises(RuntimeError):
            ex.on_op_end(torch.randn(2))


class TestReplayExecutorWithFakeManager(unittest.TestCase):
    """ReplayExecutor with a minimal fake manager for isolated testing."""

    def test_output_binding_uses_manager_bind(self) -> None:
        """Bind should be called on the manager for each bound output."""

        class FakeManager:
            """Minimal fake for testing output binding."""

            def __init__(self):
                self.bound = []

            def bind(self, sid, tensor):
                self.bound.append((sid, tensor))
                return PhysicalBuffer(device=tensor.device)

            def clear_runtime(self):
                pass

            def copy_d2h(self, sid):
                pass

            def copy_h2d(self, sid):
                pass

            def release_device(self, sid):
                pass

            def release_host(self, sid):
                pass

            def wait_for_transfers(self):
                pass

            def sync_all_transfers(self):
                pass

            @property
            def resident_bytes(self):
                return 0

            def device_resident_size(self, sid):
                return None

        manager = FakeManager()
        guide = [
            OpGuide(
                name="dummy",
                output_leaf_count=2,
                output_bindings={1: 7},
            )
        ]
        ex = ReplayExecutor(manager, ResidencySchedule(), guide)

        result = (torch.randn(2, 2), torch.randn(2))

        ex.on_op_begin(lambda: None, (), {})
        wrapped = ex.on_op_end(result)

        self.assertEqual(manager.bound, [(7, result[1])])
        self.assertIn(7, ex.retained_sids)

    def test_output_structure_mismatch_raises(self) -> None:
        """Replay should fail fast when output leaf count differs from warmup."""

        class FakeManager:
            """Minimal fake."""

            def clear_runtime(self):
                pass

        guide = [
            OpGuide(name="dummy", output_leaf_count=2, output_bindings={1: 7})
        ]
        ex = ReplayExecutor(FakeManager(), ResidencySchedule(), guide)

        ex.on_op_begin(lambda: None, (), {})
        with self.assertRaisesRegex(RuntimeError, "replay output structure differs"):
            ex.on_op_end(torch.randn(2))
