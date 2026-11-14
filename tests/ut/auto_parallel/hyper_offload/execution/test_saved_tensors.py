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
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils._python_dispatch import TorchDispatchMode

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession, skip_offload
from hyper_parallel.auto_parallel.hyper_offload.execution.saved_tensors import (
    SavedNative, SavedSlot, SavedTensorReplay, tensor_metadata,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencyActionType as Action
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencySchedule
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import AccessKind, ActivationTrace, StorageAccess, TraceOp
from hyper_parallel.core.activation_memory.api import checkpoint
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
        self.assertEqual(session.executor._guide, [])
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
    def test_initialized_activation_uses_native_replay(self) -> None:
        """Feature: Native replay eligibility.

        Description: Initialize an activation in place before autograd saves it.
        Expectation: Use native replay and preserve exact gradients of the initialized value.
        """
        session = self._session()
        self._warmup(session, mutate=True)
        self.assertIsNotNone(session._saved_replay)
        reference = self.data.clone().requires_grad_()
        expected = self._compute(reference, mutate=True)
        value = self.data.clone().requires_grad_()
        with session:
            actual = self._compute(value, mutate=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_split_and_unbind_aliases_use_native_replay(self) -> None:
        """Feature: Native replay for multiple saved views.

        Description: Split and unbind an immutable activation before saving several views.
        Expectation: Transfer its storage once and preserve exact loss and gradients.
        """
        def compute(value: torch.Tensor) -> torch.Tensor:
            """Save multiple alias views produced by list-returning operators."""
            first, second = value.sin().split(8)
            left, right = first.view(2, 4).unbind(0)
            loss = left.square().sum() + right.square().sum() + second.square().sum()
            loss.backward()
            return loss

        reference = self.data.clone().requires_grad_()
        expected = compute(reference)
        session = self._session()
        with session:
            compute(self.data.clone().requires_grad_())
        self.assertIsNotNone(session._saved_replay)
        value = self.data.clone().requires_grad_()
        manager = session.residency_manager
        with (patch.object(manager, "copy_d2h", wraps=manager.copy_d2h) as d2h,
              patch.object(manager, "copy_h2d", wraps=manager.copy_h2d) as h2d):
            with session:
                actual = compute(value)
        self.assertEqual(d2h.call_count, 1)
        self.assertEqual(h2d.call_count, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_restore_allocates_on_copy_stream_and_fences_consumer(self) -> None:
        """Feature: Restore buffer allocation stream.

        Description: Restore actual bytes on a mocked copy stream and access them from the consumer.
        Expectation: Allocate in the copy stream context and wait for its event before reading.
        """
        manager = self._session().residency_manager
        source = self.data.clone()
        manager.bind(1, source)
        manager.copy_d2h(1)
        manager.release_device(1)
        copy_stream = manager._copy_stream
        active = []
        copy_stream.__enter__.side_effect = lambda: active.append(True)
        copy_stream.__exit__.side_effect = lambda *args: active.pop() and False
        original_empty = torch.empty

        def allocate(*args: Any, **kwargs: Any) -> torch.Tensor:
            """Verify the allocator observes the stream that owns the H2D copy."""
            self.assertEqual(active, [True])
            return original_empty(*args, **kwargs)

        with patch("torch.empty", side_effect=allocate) as allocation:
            manager.copy_h2d(1)
        self.assertEqual(allocation.call_count, 1)
        event = manager._residency[1].device_event
        event.wait.reset_mock()
        storage = manager._residency[1].device_storage()
        event.wait.assert_called_once()
        restored = torch.empty(0, dtype=source.dtype).set_(storage, 0, source.size(), source.stride())
        torch.testing.assert_close(restored, source, rtol=0, atol=0)

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
    def test_native_saved_alias_keeps_version_after_original_tensor_dies(self) -> None:
        """Feature: Native saved-view ownership and mutation validation.

        Description: Drop the original saved tensor, then mutate a surviving alias.
        Expectation: Keep the exact view and shared version counter without its autograd graph.
        """
        value = self.data.clone().requires_grad_()
        activation = value.sin()
        view = activation.view(4, 4).t()[1:]
        expected = view.detach().clone()
        reference = weakref.ref(view)
        replay = SavedTensorReplay(MagicMock(), [SavedSlot(None, tensor_metadata(view), 0)], set())
        packed = replay.pack(view)
        replay._prefetched = True
        self.assertIsNone(packed.tensor.grad_fn)
        self.assertFalse(packed.tensor.requires_grad)
        self.assertEqual(packed.tensor.stride(), view.stride())
        self.assertEqual(packed.tensor.storage_offset(), view.storage_offset())
        del view
        self.assertIsNone(reference())
        torch.testing.assert_close(replay.unpack(packed), expected, rtol=0, atol=0)
        with torch.no_grad():
            activation.add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified in place"):
            replay.unpack(packed)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_native_saved_output_does_not_keep_backward_graph_alive(self) -> None:
        """Feature: Acyclic saved tensors.

        Description: Save a differentiable output whose backward node owns the packed value.
        Expectation: Release its input and saved output without requiring cyclic garbage collection.
        """
        value = self.data.clone().requires_grad_()
        replay = SavedTensorReplay(MagicMock(), [SavedSlot(None, tensor_metadata(value), 0)], set())
        value_ref = weakref.ref(value)
        with torch.autograd.graph.saved_tensors_hooks(replay.pack, replay.unpack):
            output = value.exp()
        output_ref = weakref.ref(output)
        del output, value
        self.assertIsNone(output_ref())
        self.assertIsNone(value_ref())

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
    def test_planning_hints_cover_only_immutable_used_forward_saves(self) -> None:
        """Feature: Peak planning preferences.

        Description: Record immutable, mutable, retained and unused saved storage.
        Expectation: Prefer only valid forward-to-backward activation gaps.
        """
        recorder, trace, _ = self._recorded_plan()
        for sid, op_id in ((1, 30), (2, 20), (3, 10)):
            trace.ops[op_id].accesses = [StorageAccess(op_id, sid, AccessKind.READ)]
        SavedTensorReplay.add_planning_hints(recorder, trace)
        self.assertEqual(set(trace.preferred_eviction_gaps), {(1, 1, 30), (2, 2, 20), (3, 3, 10)})
        trace.preferred_eviction_gaps.clear()
        trace.ops[6].name = "add_.Tensor"
        trace.ops[6].accesses = [StorageAccess(6, 1, AccessKind.WRITE)]
        trace.retained_sids.add(2)
        recorder.unpacks = [(1, 20), (0, 30)]
        SavedTensorReplay.add_planning_hints(recorder, trace)
        self.assertEqual(trace.preferred_eviction_gaps, [])

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
    def test_ineligible_retained_mutable_and_unused_plans_fall_back(self) -> None:
        """Feature: Native replay fallback.

        Description: Check retained storage, multiple intervals, unused values and late first saves.
        Expectation: Reject each unsupported plan before enabling native replay.
        """
        for case in ("retained", "interval", "unused", "late_save", "mutation", "forward_restore"):
            with self.subTest(case=case):
                recorder, trace, schedule = self._recorded_plan()
                if case == "retained":
                    trace.retained_sids.add(1)
                elif case == "interval":
                    schedule.add_post(5, Action.COPY_D2H, 1)
                elif case == "unused":
                    recorder.unpacks.pop()
                elif case == "late_save":
                    recorder.slots[0] = SavedSlot(1, recorder.slots[0].metadata, 10)
                elif case == "forward_restore":
                    schedule.pre[25][0] = type(schedule.pre[25][0])(5, 1, Action.COPY_H2D)
                else:
                    trace.ops[5].accesses = [StorageAccess(5, 1, AccessKind.WRITE)]
                    trace.ops[5].name = "add_.Tensor"
                self.assertIsNone(SavedTensorReplay.from_trace(MagicMock(), recorder, trace, schedule))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_unused_saved_alias_does_not_retain_restored_storage(self) -> None:
        """Feature: Unused saves sharing a used storage.

        Description: Record an extra saved alias that is never unpacked.
        Expectation: Compile native replay and release after the final actual use.
        """
        recorder, trace, schedule = self._recorded_plan()
        recorder.slots.append(recorder.slots[0])
        manager = self._session().residency_manager
        replay = SavedTensorReplay.from_trace(manager, recorder, trace, schedule)
        self.assertIsNotNone(replay)
        values = [torch.arange(4, dtype=torch.float32) + sid for sid in range(1, 4)]
        packed = [replay.pack(value) for value in values]
        replay.pack(values[0])
        replay._retire_sources()
        for sid in (3, 2, 1):
            restored = replay.unpack(packed[sid - 1])
            torch.testing.assert_close(restored, values[sid - 1], rtol=0, atol=0)
            self.assertNotIn(sid, manager._residency)
        replay.finish()

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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_checkpoint_repeated_inputs_use_native_replay_and_exact_gradients(self) -> None:
        """Feature: Checkpoint and native offload replay composition.

        Description: Share an offloaded input between two checkpoint arguments.
        Expectation: Replay both checkpoint implementations without global dispatch and with exact gradients.
        """
        for implementation in (checkpoint, torch_checkpoint):
            with self.subTest(implementation=implementation.__module__):
                calls = []

                def region(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
                    """Save both aliases and count forward and recomputed executions."""
                    calls.append(None)
                    return left.square() + right.cos()

                def compute(value: torch.Tensor) -> torch.Tensor:
                    """Differentiate through a checkpoint with repeated arguments."""
                    activation = value.sin()
                    result = implementation(region, activation, activation, use_reentrant=False)
                    loss = result.sum()
                    loss.backward()
                    return loss

                reference = self.data.clone().requires_grad_()
                expected = compute(reference)
                session = self._session()
                with session:
                    compute(self.data.clone().requires_grad_())
                self.assertIsNotNone(session._saved_replay)
                for _ in range(2):
                    value = self.data.clone().requires_grad_()
                    calls.clear()
                    with patch("hyper_parallel.auto_parallel.hyper_offload.api.session.ActivationDispatchMode",
                               side_effect=RuntimeError("native replay installed a global mode")):
                        with session:
                            actual = compute(value)
                    self.assertEqual(len(calls), 2)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
                    self.assertFalse(session.residency_manager._residency)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_reentrant_checkpoint_backward_saves_use_native_replay(self) -> None:
        """Feature: Backward-time activation saves during checkpoint recomputation.

        Description: Reentrant checkpointing saves a new graph while backward is active.
        Expectation: Preserve native replay, real transfers, and exact repeated-step gradients.
        """
        def compute(value: torch.Tensor) -> torch.Tensor:
            """Create backward-time saves through a reentrant checkpoint."""
            activation = value.sin()
            result = torch_checkpoint(lambda tensor: tensor.square().cos(), activation, use_reentrant=True)
            loss = result.sum()
            loss.backward()
            return loss

        reference = self.data.clone().requires_grad_()
        expected = compute(reference)
        session = self._session()
        with session:
            compute(self.data.clone().requires_grad_())
        self.assertIsNotNone(session._saved_replay)
        for _ in range(2):
            value = self.data.clone().requires_grad_()
            with patch("hyper_parallel.auto_parallel.hyper_offload.api.session.ActivationDispatchMode",
                       side_effect=RuntimeError("native replay installed a global mode")):
                with session:
                    actual = compute(value)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
            self.assertFalse(session.residency_manager._residency)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_opaque_region_inside_checkpoint_preserves_session_and_gradients(self) -> None:
        """Feature: Opaque regions combined with activation checkpointing.

        Description: Recompute an opaque region from an autograd worker context.
        Expectation: Warmup retains its session and native replay preserves exact gradients.
        """
        active_sessions = []

        @skip_offload
        def opaque(value: torch.Tensor) -> torch.Tensor:
            """Record session visibility within the opaque recomputation."""
            active_sessions.append(OffloadSession.get_active())
            return value.square().cos()

        def compute(value: torch.Tensor) -> torch.Tensor:
            """Differentiate through opaque and ordinary saved activation users."""
            activation = value.sin()
            loss = (checkpoint(opaque, activation, use_reentrant=False) + activation.square()).sum()
            loss.backward()
            return loss

        reference = self.data.clone().requires_grad_()
        expected = compute(reference)
        active_sessions.clear()
        session = self._session()
        with session:
            compute(self.data.clone().requires_grad_())
        self.assertEqual(active_sessions, [session, session])
        self.assertIsNotNone(session._saved_replay)
        for _ in range(2):
            value = self.data.clone().requires_grad_()
            with session:
                actual = compute(value)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_repeated_saved_use_keeps_storage_until_last_consumer(self) -> None:
        """Feature: Repeated immutable saved-value access.

        Description: A backward function reads one saved activation twice.
        Expectation: Transfer once, retain the restored storage, and preserve exact gradients.
        """
        class ReadTwice(torch.autograd.Function):
            """Model checkpoint input savers that consult saved tensors twice."""

            @staticmethod
            def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
                """Save one immutable activation for repeated backward reads."""
                ctx.save_for_backward(tensor)
                return tensor.square()

            @staticmethod
            def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
                """Read the saved value twice before returning its gradient."""
                left, = ctx.saved_tensors
                right, = ctx.saved_tensors
                return grad * left + grad * right

        def compute(value: torch.Tensor) -> torch.Tensor:
            """Differentiate through the repeated saved-value consumer."""
            loss = ReadTwice.apply(value.sin()).sum()
            loss.backward()
            return loss

        reference = self.data.clone().requires_grad_()
        expected = compute(reference)
        session = self._session()
        with session:
            compute(self.data.clone().requires_grad_())
        self.assertIsNotNone(session._saved_replay)
        manager = session.residency_manager
        for _ in range(2):
            value = self.data.clone().requires_grad_()
            with (patch.object(manager, "copy_d2h", wraps=manager.copy_d2h) as d2h,
                  patch.object(manager, "copy_h2d", wraps=manager.copy_h2d) as h2d,
                  patch("hyper_parallel.auto_parallel.hyper_offload.api.session.ActivationDispatchMode",
                        side_effect=RuntimeError("native replay installed a global mode"))):
                with session:
                    actual = compute(value)
            self.assertEqual(d2h.call_count, 1)
            self.assertEqual(h2d.call_count, 1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
            self.assertFalse(manager._residency)


if __name__ == "__main__":
    unittest.main()
