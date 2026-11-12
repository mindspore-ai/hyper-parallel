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
"""Real byte-copy contracts for autograd saved-activation replay."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import MagicMock, Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession
from hyper_parallel.auto_parallel.hyper_offload.execution.saved_tensors import (
    SavedNative, SavedSlot, SavedTensorReplay, tensor_metadata,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencyActionType as Action
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencySchedule
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import AccessKind, ActivationTrace, StorageAccess, TraceOp
from tests.common.mark_utils import arg_mark
from tests.ut.auto_parallel.hyper_offload.execution.test_selective_replay import BytePool, CPUActivationTracker


class _CopySavedActivation:
    """Select a real produced activation independently of saved-slot indices."""

    def __init__(self) -> None:
        self.sid = None

    def build(self, trace: ActivationTrace) -> ResidencySchedule:
        """Copy the sine output once with backward restore."""
        op_id = next(index for index, op in enumerate(trace.ops) if op.name == "sin.default")
        self.sid = next(a.storage_id for a in trace.ops[op_id].accesses if a.kind == AccessKind.WRITE)
        copy_id = next((index for index, op in enumerate(trace.ops) if op.name == "add_.Tensor"), op_id)
        schedule = ResidencySchedule()
        schedule.add_post(copy_id, Action.COPY_D2H, self.sid)
        schedule.add_post(copy_id + 1, Action.RELEASE_DEVICE, self.sid)
        backward_id = next(index for index, op in enumerate(trace.ops) if op.name == "ones_like.default")
        restore_id = next(index for index, op in enumerate(trace.ops) if index > backward_id
                          and op.name in ("mul.Tensor", "pow.Tensor_Scalar")
                          and any(a.storage_id == self.sid for a in op.accesses))
        schedule.add_pre(restore_id, Action.COPY_H2D, self.sid)
        schedule.add_post(len(trace.ops) - 1, Action.RELEASE_DEVICE, self.sid)
        schedule.add_post(len(trace.ops) - 1, Action.RELEASE_HOST, self.sid)
        return schedule


class _Observer(TorchDispatchMode):
    """Keep lower modes observable during native replay."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return func(*args, **(kwargs or {}))


class TestSavedTensorReplay(unittest.TestCase):
    """Exercise production sessions with real tensors and mocked device queues."""

    def setUp(self) -> None:
        """Replace accelerator queues while preserving real tensor storage copies."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("torch.accelerator.current_stream", "torch.Event", "torch.Tensor.record_stream"):
            self.stack.enter_context(patch(name))
        self.stack.enter_context(patch(
            "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
            return_value=(10.0, 10.0),
        ))
        self.data = torch.linspace(-2, 2, 16)

    def _session(self) -> OffloadSession:
        planner = _CopySavedActivation()
        session = OffloadSession(OffloadConfig(planner=planner))
        manager = session.residency_manager
        manager._host_pool = BytePool()
        manager._copy_stream = MagicMock()
        manager._copy_stream.device = torch.device("cpu")
        manager.wait_for_transfers = lambda: None
        manager.sync_all_transfers = Mock()
        session.executor._timer = Mock()
        session.executor._timer.stop.return_value = 0.0
        session.executor._tracker = CPUActivationTracker()
        return session

    @staticmethod
    def _compute(value: torch.Tensor, mutate: bool = False) -> torch.Tensor:
        activation = value.sin()
        if mutate:
            activation.add_(1)
        view = activation.view(4, 4).transpose(0, 1)
        loss = view.square().sum()
        loss.backward()
        return loss

    def _warmup(self, session: OffloadSession, mutate: bool = False) -> None:
        value = self.data.clone().requires_grad_()
        with session:
            self._compute(value, mutate)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_real_transfers_exact_gradients_and_lower_mode_without_global_dispatch(self) -> None:
        """Feature: Automatic native saved activation replay.

        Description: Replay a noncontiguous saved alias with real CPU byte copies.
        Expectation: Automatically preserve gradients, copies and observer mode without global dispatch.
        """
        reference = self.data.clone().requires_grad_()
        expected = self._compute(reference)
        session = self._session()
        self._warmup(session)
        self.assertIsNotNone(session._saved_replay)
        manager = session.residency_manager
        for _ in range(2):
            value = self.data.clone().requires_grad_()
            observer = _Observer()
            with (observer, patch.object(manager, "copy_d2h", wraps=manager.copy_d2h) as d2h,
                  patch.object(manager, "copy_h2d", wraps=manager.copy_h2d) as h2d,
                  patch("hyper_parallel.auto_parallel.hyper_offload.api.session.ActivationDispatchMode",
                        side_effect=RuntimeError("native replay installed a global mode"))):
                with session:
                    actual = self._compute(value)
            self.assertEqual(d2h.call_count, 1)
            self.assertEqual(h2d.call_count, 1)
            self.assertGreater(observer.calls, 0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_mutated_activation_plan_uses_operator_replay(self) -> None:
        """Feature: Native replay eligibility.

        Description: Record a produced activation with a subsequent in-place update.
        Expectation: Keep operator replay and preserve exact gradients.
        """
        session = self._session()
        self._warmup(session, mutate=True)
        self.assertIsNone(session._saved_replay)
        reference = self.data.clone().requires_grad_()
        expected = self._compute(reference, mutate=True)
        value = self.data.clone().requires_grad_()
        with session:
            actual = self._compute(value, mutate=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_changed_selected_metadata_fails_and_session_recovers(self) -> None:
        """Feature: Replay validation and exception cleanup.

        Description: Save a selected activation with a different shape, then retry the original step.
        Expectation: Reject the mismatch, clear in-flight resources and preserve later gradients.
        """
        session = self._session()
        self._warmup(session)
        value = torch.ones(32, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "metadata differs"):
            with session:
                value.sin().square().sum().backward()
        self.assertIsNone(OffloadSession.get_active())
        session.residency_manager.sync_all_transfers.assert_called()
        self.assertFalse(session.residency_manager._residency)
        self._warmup(session)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_truncated_step_fails_and_clears_transfers(self) -> None:
        """Feature: Complete-step replay contract.

        Description: Exit before saving or consuming all recorded activations.
        Expectation: Reject truncation and clear the residency table and active session.
        """
        session = self._session()
        self._warmup(session)
        with self.assertRaisesRegex(RuntimeError, "step differs"):
            with session:
                pass
        self.assertFalse(session.residency_manager._residency)
        self.assertIsNone(OffloadSession.get_active())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_unselected_native_saved_value_keeps_version_check(self) -> None:
        """Feature: Saved native tensor mutation checking.

        Description: Modify an unrelated saved input after its forward use.
        Expectation: Raise the saved-for-backward mutation error.
        """
        value = self.data.clone()
        replay = SavedTensorReplay(MagicMock(), [], set())
        replay._prefetched = True
        packed = SavedNative(value.detach(), value._version)
        value.add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified in place"):
            replay.unpack(packed)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_moved_saved_value_does_not_own_original_storage(self) -> None:
        """Feature: Activation source retirement.

        Description: Save the only planned tensor and run a completed-copy retirement check.
        Expectation: Keep metadata only and retain the host bytes with no device source.
        """
        session = self._session()
        value = self.data.clone()
        sid = 1
        replay = SavedTensorReplay(session.residency_manager, [SavedSlot(sid, tensor_metadata(value), 0)], {sid})
        packed = replay.pack(value)
        replay._retire_sources()
        buffer = session.residency_manager._residency[sid]
        self.assertIsNone(buffer.device_buffer)
        self.assertIsNotNone(buffer.host_buffer)
        self.assertFalse(hasattr(packed, "tensor"))
        restored = replay.unpack(packed)
        torch.testing.assert_close(restored, value, rtol=0, atol=0)
        replay.finish()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_overlapping_saved_aliases_with_offsets_keep_exact_gradients(self) -> None:
        """Feature: Shared activation storage and saved views.

        Description: Save two overlapping offset views of the same immutable activation.
        Expectation: Transfer once and reproduce every gradient exactly across two replays.
        """
        def compute(value: torch.Tensor) -> torch.Tensor:
            """Save overlapping views and differentiate their combined loss."""
            activation = value.sin()
            left = activation[1:9].view(2, 4).t()
            right = activation[4:12]
            loss = left.square().sum() + right.cos().sum()
            loss.backward()
            return loss

        reference = self.data.clone().requires_grad_()
        expected = compute(reference)
        session = self._session()
        with session:
            compute(self.data.clone().requires_grad_())
        self.assertEqual(len(session._saved_replay.slots), 2)
        for _ in range(2):
            value = self.data.clone().requires_grad_()
            with session:
                actual = compute(value)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
            self.assertFalse(session.residency_manager._residency)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_transfer_failure_cleans_up_and_allows_retry(self) -> None:
        """Feature: Native replay transfer exception cleanup.

        Description: Raise after the actual source bind and D2H submission in a saved hook.
        Expectation: Drain queues, remove session state and successfully retry the recorded step.
        """
        session = self._session()
        self._warmup(session)
        manager = session.residency_manager
        original_copy = manager.copy_d2h

        def fail_copy(sid: int) -> None:
            """Raise after submitting the real copy to exercise exception draining."""
            original_copy(sid)
            raise RuntimeError("injected transfer failure")

        with patch.object(manager, "copy_d2h", side_effect=fail_copy):
            with self.assertRaisesRegex(RuntimeError, "injected transfer failure"):
                self._warmup(session)
        self.assertIsNone(OffloadSession.get_active())
        self.assertFalse(manager._residency)
        manager.sync_all_transfers.assert_called()
        self._warmup(session)

    @staticmethod
    def _recorded_plan() -> tuple:
        value = torch.ones(4)
        slots = [SavedSlot(sid, tensor_metadata(value), sid) for sid in range(1, 4)]
        recorder = SimpleNamespace(slots=slots, unpacks=[(2, 10), (1, 20), (0, 30)])
        trace = ActivationTrace(ops=[TraceOp("empty.memory_format") for _ in range(31)])
        for sid in range(1, 4):
            trace.ops[sid].accesses = [StorageAccess(sid, sid, AccessKind.WRITE)]
        schedule = ResidencySchedule()
        for sid, restore in ((1, 25), (2, 15), (3, 10)):
            schedule.add_post(sid, Action.COPY_D2H, sid)
            schedule.add_pre(restore, Action.COPY_H2D, sid)
        return recorder, trace, schedule

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_three_activations_restore_at_separate_compiled_anchors(self) -> None:
        """Feature: Multiple activation prefetch scheduling.

        Description: Replay three immutable saves with different backward restore anchors.
        Expectation: No activation-count cap and no premature bulk restore at backward entry.
        """
        recorder, trace, schedule = self._recorded_plan()
        session = self._session()
        replay = SavedTensorReplay.from_trace(session.residency_manager, recorder, trace, schedule)
        self.assertIsNotNone(replay)
        self.assertEqual(replay.prefetch_at, {2: [1], 1: [2], 0: [3]})
        values = [torch.arange(4, dtype=torch.float32) + sid for sid in range(1, 4)]
        packed = [replay.pack(value) for value in values]
        replay._retire_sources()
        for sid in (3, 2, 1):
            restored = replay.unpack(packed[sid - 1])
            torch.testing.assert_close(restored, values[sid - 1], rtol=0, atol=0)
            self.assertEqual(replay._restored, set(range(sid, 4)))
        replay.finish()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_ineligible_retained_mutable_and_repeated_use_plans_fall_back(self) -> None:
        """Feature: Native replay fallback.

        Description: Check retained storage, multiple intervals, repeated uses and late saves.
        Expectation: Reject each unsupported plan before enabling native replay.
        """
        for case in ("retained", "interval", "repeated", "late_save", "mutation", "forward_restore"):
            with self.subTest(case=case):
                recorder, trace, schedule = self._recorded_plan()
                if case == "retained":
                    trace.retained_sids.add(1)
                elif case == "interval":
                    schedule.add_post(5, Action.COPY_D2H, 1)
                elif case == "repeated":
                    recorder.unpacks.append((0, 30))
                elif case == "late_save":
                    recorder.slots[0] = SavedSlot(1, recorder.slots[0].metadata, 10)
                elif case == "forward_restore":
                    schedule.pre[25][0] = type(schedule.pre[25][0])(5, 1, Action.COPY_H2D)
                else:
                    trace.ops[5].accesses = [StorageAccess(5, 1, AccessKind.WRITE)]
                    trace.ops[5].name = "add_.Tensor"
                self.assertIsNone(SavedTensorReplay.from_trace(MagicMock(), recorder, trace, schedule))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_source_storage_can_be_destroyed_before_backward(self) -> None:
        """Feature: Physical source memory release.

        Description: Drop the caller's last source tensor after a completed D2H copy.
        Expectation: The source storage expires while the packed metadata remains usable.
        """
        manager = self._session().residency_manager
        value = self.data.clone()
        expected = value.clone()
        storage = value.untyped_storage()
        reference = weakref.ref(storage)
        replay = SavedTensorReplay(manager, [SavedSlot(1, tensor_metadata(value), 0)], {1})
        packed = replay.pack(value)
        replay._retire_sources()
        del value, storage
        gc.collect()
        self.assertIsNone(reference())
        restored = replay.unpack(packed)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        replay.finish()


if __name__ == "__main__":
    unittest.main()
