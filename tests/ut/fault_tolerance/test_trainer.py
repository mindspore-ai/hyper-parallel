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
"""Trainer reentry and counter conventions, with actual CPU optimizer updates."""

import unittest
from contextlib import ExitStack, contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.fault_tolerance.state import ModelOptimizerState, TrainState
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.runtime.fault_tolerance import (
    _synchronize_checkpoint_saved_step, recovery_data_iterator, recovery_epoch_initialized,
    recovery_train_loop, recovery_train_step,
)
from hyper_parallel.trainer.runtime.model_integration import DisabledModelIntegrationSession
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


class _Loader:
    """A consumed cursor that survives iterator replacement."""

    def __init__(self) -> None:
        """Start with no consumed batches or live iterators."""
        self.position = 0
        self.iterations = 0
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Preserve the restored cursor when the requested epoch has not changed."""
        if epoch != self.epoch:
            self.epoch = epoch
            self.position = 0

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Create a fresh iterator from the most recently restored cursor."""
        self.iterations += 1
        while self.position < 6:
            self.position += 1
            yield torch.full((1, 2), self.position / 10)

    def state_dict(self) -> dict[str, int]:
        """Capture consumed batches independently of live generator frames."""
        return {"position": self.position, "epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Position the next iterator after the committed batch."""
        self.position = state["position"]
        self.epoch = state["epoch"]


class _MemorySession:
    """Replace control-plane scheduling while exercising real local state restore."""

    def __init__(self, inject: bool) -> None:
        """Choose whether run() simulates one interrupted forward/backward."""
        self.inject = inject
        self.failed = False
        self.reboot = False
        self.npu = None
        self.step = 0
        self.codec = self.train_state = self.snapshot = None
        self.closed = False
        self.events = []
        self._initial_state_pending = False

    def attach(self, model: Any, optimizer: Any, state: Any, step: int) -> None:
        """Bind the actual codecs that the Trainer adapter constructs."""
        self.codec = ModelOptimizerState(model, optimizer)
        self.train_state = state
        self.step = step

    @contextmanager
    def update(self, step: int) -> Iterator[None]:
        """Capture only after optimizer, scheduler and progress counters finish."""
        yield
        self.snapshot = (self.codec.rank_state(), self.train_state.capture())
        self.step = step

    def pause(self) -> None:
        """The CPU-only scheduler has no pending Controller commands."""

    def finish_restore(self) -> None:
        """Finalize committed RNG after native reentry initialization."""
        self.train_state.finish_restore()

    def publish_initial_state(self, step: int) -> None:
        """Record the post-callback progress only on the original startup."""
        if self._initial_state_pending:
            self.step = step
            self.snapshot = (self.codec.rank_state(), self.train_state.capture())
            self._initial_state_pending = False

    def make_iterator(self, factory: Callable[[], Iterator[Any]] = None) -> Iterator[Any]:
        """Expose the same iterator recreation boundary as the real session."""
        return self.train_state.make_iterator(factory=factory)

    def close(self) -> None:
        """Observe outer runtime cleanup without creating hardware resources."""
        self.closed = True
        self.events.append("close")

    def run(self, body: Any) -> Any:
        """Roll back one partial forward and reenter the same training body."""
        self._initial_state_pending = not self.reboot
        try:
            result = body(self)
        except RuntimeError as error:
            if str(error) != "injected forward failure":
                raise
            self.codec.restore_rank_state(self.snapshot[0]["buffers"], self.snapshot[0]["extra"])
            self.train_state.restore(self.snapshot[1])
            result = body(self)
        self.events.append("sdk_exit")
        return result


class _CpuBaseTrainer(BaseTrainer):
    """Use real native Trainer loops and callbacks with a mocked distributed boundary."""

    def destroy_distributed(self) -> None:
        """Record native teardown before the SDK exits."""
        self.recovery.events.append("destroy_distributed")


class TestTrainerReentry(unittest.TestCase):
    """Both Base and Text counter positions must restore the same transaction."""

    def test_original_train_begin_executes_on_each_reentry(self):
        """Feature: Original Trainer initialization after replica recovery.

        Description: Restore a replacement, enter its native train twice and include a checkpoint callback.
        Expectation: Each native train entry runs its own original initialization and callback dispatch.
        """
        events = []
        callback = MagicMock()
        checkpoint = MagicMock(spec=CheckpointerCallback)
        callback.on_train_begin.side_effect = lambda state: events.append(("callback", state.global_step))
        checkpoint.on_train_begin.side_effect = lambda state: events.append(("checkpoint", state.global_step))
        session = SimpleNamespace(reboot=True, npu=None, attach=MagicMock(), close=MagicMock(),
                                  publish_initial_state=MagicMock())
        base = SimpleNamespace(
            recovery=session, state=TrainerState(), train_dataloader=_Loader(), lr_scheduler=None,
            model=None, optimizer=None, _callbacks=[callback, checkpoint],
            abort_profilers=MagicMock(), destroy_distributed=MagicMock())
        trainer = SimpleNamespace(base=base, on_train_end=MagicMock())

        def _begin(active):
            events.append(("trainer", active.base.state.global_step))
            BaseTrainer.on_train_begin(active.base)

        trainer.on_train_begin = MethodType(_begin, trainer)
        result = object()

        def _native_train(active):
            active.on_train_begin()
            events.append(("train", active.base.state.global_step))
            return result

        def _run(body):
            self.assertEqual(events, [])
            session.reboot = False
            base.state.global_step = 8
            body(session)
            return body(session)

        session.run = _run
        with patch("hyper_parallel.trainer.runtime.fault_tolerance._synchronize_checkpoint_saved_step"):
            self.assertIs(recovery_train_loop(_native_train)(trainer), result)
        self.assertEqual(events, [("trainer", 8), ("callback", 8), ("checkpoint", 8), ("train", 8)] * 2)
        self.assertEqual(callback.on_train_begin.call_count, 2)
        self.assertEqual(checkpoint.on_train_begin.call_count, 2)

    def test_replacement_rng_is_restored_after_begin_epoch_and_iterator(self):
        """Feature: Final RNG restore after replacement lifecycle initialization.

        Description: Consume random values in original begin, epoch begin and native iterator construction.
        Expectation: The first resumed training draw matches the commit and both callbacks execute once.
        """
        callback = MagicMock()
        callback.on_train_begin.side_effect = lambda state: torch.rand(3)
        callback.on_epoch_begin.side_effect = lambda state: torch.rand(3)
        session = SimpleNamespace(reboot=True, npu=None, attach=MagicMock(), close=MagicMock())
        base = SimpleNamespace(
            recovery=session, state=TrainerState(), train_dataloader=_Loader(), lr_scheduler=None,
            model=None, optimizer=None, _callbacks=[callback],
            abort_profilers=MagicMock(), destroy_distributed=MagicMock())
        trainer = SimpleNamespace(base=base, on_train_end=MagicMock())

        def _begin(active):
            BaseTrainer.on_train_begin(active.base)

        trainer.on_train_begin = MethodType(_begin, trainer)

        def _factory():
            torch.rand(3)
            return iter(())

        def _native_train(active):
            active.on_train_begin()
            BaseTrainer.on_epoch_begin(active.base)
            recovery_data_iterator(active.base, _factory)
            recovery_epoch_initialized(active.base)
            return torch.rand(3)

        def _run(body):
            training = session.attach.call_args.args[2]
            session.train_state = training
            session.make_iterator = training.make_iterator
            session.finish_restore = training.finish_restore
            base.state.global_step = 8
            saved = training.capture()
            expected = torch.rand(3)
            torch.rand(11)
            training.restore(saved)
            session.reboot = False
            actual = body(session)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return actual

        session.run = _run
        with patch("hyper_parallel.trainer.runtime.fault_tolerance._synchronize_checkpoint_saved_step"):
            recovery_train_loop(_native_train)(trainer)
        callback.on_train_begin.assert_called_once_with(base.state)
        callback.on_epoch_begin.assert_called_once_with(base.state)

    def test_rng_restore_hook_only_runs_for_pending_rollback(self):
        """Feature: One-shot initialization completion after each rollback.

        Description: Enter normal epochs, restore twice and interleave additional epoch initialization calls.
        Expectation: Normal epochs never call the finalizer; each rollback invokes it once and clears readiness.
        """
        loader = _Loader()
        state = TrainState(loader, {})
        session = SimpleNamespace(train_state=state, finish_restore=MagicMock(wraps=state.finish_restore))
        base = SimpleNamespace(recovery=session)
        saved = state.capture()
        for _ in range(3):
            recovery_epoch_initialized(base)
        session.finish_restore.assert_not_called()
        for attempt in range(2):
            state.restore(saved)
            self.assertTrue(state.rng_restore_pending)
            state.make_iterator()
            recovery_epoch_initialized(base)
            self.assertFalse(state.rng_restore_pending)
            for _ in range(3):
                recovery_epoch_initialized(base)
            self.assertEqual(session.finish_restore.call_count, attempt + 1)

    def test_rng_restore_hook_retains_pending_state_on_failure(self):
        """Feature: Retryable one-shot initialization completion.

        Description: Fail device RNG application, retry the hook and enter another normal epoch.
        Expectation: Failure keeps readiness; successful retry clears it and later epochs do not call the finalizer.
        """
        device = SimpleNamespace(get_rng_state=torch.get_rng_state,
                                 set_rng_state=MagicMock(side_effect=RuntimeError("RNG restore failed")))
        state = TrainState(_Loader(), {}, device=device)
        session = SimpleNamespace(train_state=state, finish_restore=MagicMock(wraps=state.finish_restore))
        base = SimpleNamespace(recovery=session)
        state.restore(state.capture())
        state.make_iterator()
        with self.assertRaisesRegex(RuntimeError, "RNG restore failed"):
            recovery_epoch_initialized(base)
        self.assertTrue(state.rng_restore_pending)
        device.set_rng_state.side_effect = None
        recovery_epoch_initialized(base)
        self.assertFalse(state.rng_restore_pending)
        recovery_epoch_initialized(base)
        self.assertEqual(session.finish_restore.call_count, 2)

    def test_native_train_end_precedes_sdk_exit_failure(self):
        """Feature: Native train-end callbacks independent of SDK shutdown success.

        Description: Complete the native body and raise an error afterward in the SDK exit path.
        Expectation: The real train-end dispatcher has already run once when the exit error propagates.
        """
        events = []
        callback = MagicMock()
        callback.on_train_end.side_effect = lambda state: events.append("end")
        session = SimpleNamespace(reboot=False, npu=None, attach=MagicMock(), close=MagicMock())
        trainer = _CpuBaseTrainer.__new__(_CpuBaseTrainer)
        vars(trainer).update(
            recovery=session, state=TrainerState(), train_dataloader=_Loader(), lr_scheduler=None,
            model=None, optimizer=None, _callbacks=[callback], abort_profilers=MagicMock())

        def _native_train(active):
            active.on_train_end()

        def _run(body):
            body(session)
            events.append("sdk_exit_error")
            raise ValueError("SDK shutdown failed")

        session.run = _run
        with patch("hyper_parallel.trainer.runtime.fault_tolerance._synchronize_checkpoint_saved_step"):
            with self.assertRaisesRegex(ValueError, "SDK shutdown failed"):
                recovery_train_loop(_native_train)(trainer)
        self.assertEqual(events, ["end", "sdk_exit_error"])
        callback.on_train_end.assert_called_once_with(trainer.state)

    def test_train_loop_closes_session_when_recovery_preparation_fails(self):
        """Feature: Trainer cleanup before entering the recovery run.

        Description: Fail recovery preparation, with successful or failing session cleanup.
        Expectation: The session closes without profiler cleanup and errors follow normal finally chaining.
        """
        for cleanup_failure in (False, True):
            with self.subTest(cleanup_failure=cleanup_failure):
                session = SimpleNamespace(close=MagicMock())
                base = SimpleNamespace(recovery=session, abort_profilers=MagicMock())
                trainer = SimpleNamespace(base=base)
                if cleanup_failure:
                    session.close.side_effect = RuntimeError("cleanup failed")
                error = ValueError("preparation failed")
                original = MagicMock()
                run = recovery_train_loop(original)
                with patch("hyper_parallel.trainer.runtime.fault_tolerance._train_recoverable", side_effect=error):
                    with self.assertRaises(RuntimeError if cleanup_failure else ValueError) as caught:
                        run(trainer)
                if cleanup_failure:
                    self.assertIs(caught.exception.__context__, error)
                else:
                    self.assertIs(caught.exception, error)
                base.abort_profilers.assert_not_called()
                session.close.assert_called_once()
                original.assert_not_called()

    def test_train_loop_success_reports_session_cleanup_failure(self):
        """Feature: Trainer cleanup after successful training.

        Description: Finish training and fail the session close operation.
        Expectation: The close error propagates and success is not reported.
        """
        session = SimpleNamespace(close=MagicMock(side_effect=RuntimeError("cleanup failed")))
        base = SimpleNamespace(recovery=session, abort_profilers=MagicMock())
        trainer = SimpleNamespace(base=base)
        run = recovery_train_loop(MagicMock())
        with patch("hyper_parallel.trainer.runtime.fault_tolerance._train_recoverable", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                run(trainer)
        base.abort_profilers.assert_not_called()
        session.close.assert_called_once()

    def test_replacement_uses_coordinators_completed_save_step(self):
        """Feature: Synchronize the single checkpoint callback's completed save step.

        Description: Find a checkpoint callback after an unrelated callback and broadcast one saved-step value.
        Expectation: Rank zero's completed save step replaces the new worker's default marker.
        """
        callback = object.__new__(CheckpointerCallback)
        callback._last_saved_step = -1
        base = SimpleNamespace(_callbacks=[object(), callback])
        session = SimpleNamespace(local_rank=0, step=8)
        tensor = torch.tensor

        def _cpu_tensor(values, **kwargs):
            return tensor(values, dtype=kwargs["dtype"])

        def _broadcast(ledger, src):
            self.assertEqual(src, 0)
            self.assertEqual(ledger.numel(), 1)
            ledger.fill_(7)

        with patch("hyper_parallel.trainer.runtime.fault_tolerance.torch.tensor", side_effect=_cpu_tensor), \
                patch("hyper_parallel.trainer.runtime.fault_tolerance.torch.device", return_value="cpu"), \
                patch("hyper_parallel.trainer.runtime.fault_tolerance.dist.broadcast", side_effect=_broadcast):
            _synchronize_checkpoint_saved_step(base, session)
        self.assertEqual(callback._last_saved_step, 7)

    def test_missing_checkpoint_callback_skips_save_step_broadcast(self):
        """Feature: No checkpoint coordination when the callback is absent.

        Description: Enter checkpoint synchronization with only unrelated callbacks.
        Expectation: No saved-step Tensor is allocated and no collective is issued.
        """
        with patch("hyper_parallel.trainer.runtime.fault_tolerance.torch.tensor") as tensor, \
                patch("hyper_parallel.trainer.runtime.fault_tolerance.dist.broadcast") as broadcast:
            _synchronize_checkpoint_saved_step(SimpleNamespace(_callbacks=[object()]), SimpleNamespace(local_rank=0))
        tensor.assert_not_called()
        broadcast.assert_not_called()

    @staticmethod
    def _run(inject: bool, increment_before_forward: bool, muon: bool = False,
             trainer_type: type = TextTrainer, epochs: int = 1) -> Any:
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1)
        optimizer = (ChainedOptimizer(model, {"muon": Muon([model.weight], ns_steps=2),
                                             "adamw": AdamW([model.bias], lr=0.01)})
                     if muon else AdamW(model.parameters(), lr=0.01))
        loader = _Loader()
        session = _MemorySession(inject)
        session.finish_restore = MagicMock(wraps=session.finish_restore)
        schedulers = ([torch.optim.lr_scheduler.StepLR(leaf, 2, gamma=0.5)
                       for leaf in optimizer.chained_optimizers] if muon
                      else torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5))
        base = _CpuBaseTrainer.__new__(_CpuBaseTrainer)
        callback = MagicMock()
        callback.on_train_begin.side_effect = lambda state: session.events.append("begin")
        callback.on_epoch_begin.side_effect = lambda state: torch.rand(3)
        callback.on_train_end.side_effect = lambda state: session.events.append("end")
        vars(base).update(
            model=model, optimizer=optimizer, state=TrainerState(), recovery=session,
            lr_scheduler=schedulers, local_rank=0, _callbacks=[callback],
            model_integration=DisabledModelIntegrationSession(), train_dataloader=loader,
            train_steps=6 // epochs, train_iters=6, train_epochs=epochs,
            config=SimpleNamespace(dataloader=SimpleNamespace(use_background_prefetcher=False, drop_last=True)))
        trainer_type = BaseTrainer if increment_before_forward else trainer_type
        trainer = base if increment_before_forward else trainer_type.__new__(trainer_type)
        if trainer is not base:
            trainer.base = base

        def _step(active, iterator):
            del active
            data = next(iterator)
            if increment_before_forward:
                base.state.global_step += 1
            (model(data) * torch.rand(())).square().mean().backward()
            if inject and not session.failed and session.step == 3:
                session.failed = True
                raise RuntimeError("injected forward failure")
            BaseTrainer.step_optimizers_and_schedulers(base)
            if not increment_before_forward:
                base.state.global_step += 1
            base.on_step_end()

        native = MagicMock(wraps=trainer_type.train.__wrapped__)
        with ExitStack() as stack:
            step = MethodType(recovery_train_step(_step), trainer)
            stack.enter_context(patch.object(trainer, "train_step", step))
            stack.enter_context(patch("hyper_parallel.trainer.runtime.memory_profiler.memory_profiler.stop"))
            for module in ("base", "text_trainer", "vlm_trainer"):
                stack.enter_context(patch(f"hyper_parallel.trainer.{module}.synchronize"))
                stack.enter_context(patch(f"hyper_parallel.trainer.{module}.print_device_mem_info"))
            recovery_train_loop(native)(trainer)
        base.native_train_calls = native.call_count
        return base, session

    def test_native_profiler_cleanup_runs_during_recoverable_failure(self):
        """Feature: Native profiler cleanup during process recovery.

        Description: Interrupt each native Trainer loop and allow the SDK stand-in to restore and retry it.
        Expectation: Memory profiler abort executes before retry and training still reaches the final step.
        """
        for trainer_type in (BaseTrainer, TextTrainer, VLMTrainer):
            with self.subTest(trainer=trainer_type.__name__), \
                    patch("hyper_parallel.trainer.runtime.memory_profiler.memory_profiler.abort") as abort:
                base, session = self._run(True, trainer_type is BaseTrainer, trainer_type=trainer_type)
                abort.assert_called_once_with()
                self.assertTrue(session.failed)
                self.assertEqual(base.state.global_step, 6)

    def test_only_body_reenters_and_cursor_is_recreated(self):
        """Feature: Reentry into each Trainer's native loop.

        Description: Interrupt CPU training with pre-forward or post-optimizer counters.
        Expectation: Original train reenters, state matches baseline and lifecycle hooks run once.
        """
        for increment_before_forward in (True, False):
            with self.subTest(base_style=increment_before_forward):
                baseline, baseline_session = self._run(False, increment_before_forward)
                self.assertEqual(set(baseline_session.train_state.replica_providers), {"progress", "scheduler_0"})
                recovered, session = self._run(True, increment_before_forward)
                self.assertTrue(session.failed)
                self.assertEqual(recovered.state.global_step, 6)
                self.assertEqual(recovered.train_dataloader.position, 6)
                self.assertEqual(recovered.train_dataloader.iterations, 2)
                callbacks = vars(recovered)["_callbacks"]
                self.assertEqual(callbacks[0].on_train_begin.call_count, 2)
                callbacks[0].on_train_end.assert_called_once()
                self.assertEqual(session.events, ["begin", "begin", "end", "destroy_distributed", "sdk_exit", "close"])
                baseline_session.finish_restore.assert_not_called()
                session.finish_restore.assert_called_once()
                self.assertEqual(baseline.native_train_calls, 1)
                self.assertEqual(recovered.native_train_calls, 2)
                self.assertEqual(recovered.lr_scheduler.state_dict(), baseline.lr_scheduler.state_dict())
                for actual, expected in zip(recovered.model.parameters(), baseline.model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_muon_adamw_trainer_reentry_matches_uninterrupted_updates(self):
        baseline, _ = self._run(False, False, muon=True)
        recovered, session = self._run(True, False, muon=True)
        self.assertTrue(session.failed)
        self.assertEqual(recovered.state.global_step, 6)
        self.assertEqual(recovered.train_dataloader.position, 6)
        for actual, expected in zip(recovered.lr_scheduler, baseline.lr_scheduler):
            self.assertEqual(actual.state_dict(), expected.state_dict())
        for actual, expected in zip(recovered.model.parameters(), baseline.model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_vlm_uses_its_native_train_loop(self):
        """Feature: Shared recovery of the native VLM Trainer loop.

        Description: Reenter the actual VLM train method after an interrupted CPU step.
        Expectation: VLM callbacks, counters, loader and optimizer match uninterrupted execution.
        """
        baseline, _ = self._run(False, False, muon=True, trainer_type=VLMTrainer)
        recovered, session = self._run(True, False, muon=True, trainer_type=VLMTrainer)
        self.assertEqual(recovered.native_train_calls, 2)
        self.assertEqual(recovered.state.global_step, 6)
        self.assertEqual(recovered.train_dataloader.position, 6)
        self.assertEqual(session.events, ["begin", "begin", "end", "destroy_distributed", "sdk_exit", "close"])
        for actual, expected in zip(recovered.model.parameters(), baseline.model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_loops_resume_across_epoch_boundaries(self):
        """Feature: Native Trainer epoch transitions after recovery.

        Description: Interrupt the first step after a completed epoch in all three native loops.
        Expectation: Each original loop resumes the correct data and completes two epochs.
        """
        for trainer_type in (BaseTrainer, TextTrainer, VLMTrainer):
            with self.subTest(trainer=trainer_type.__name__):
                before = trainer_type is BaseTrainer
                baseline, baseline_session = self._run(False, before, trainer_type=trainer_type, epochs=2)
                recovered, session = self._run(True, before, trainer_type=trainer_type, epochs=2)
                self.assertEqual(recovered.state.global_step, 6)
                self.assertEqual(recovered.state.epoch, 2)
                self.assertEqual(recovered.train_dataloader.state_dict(), baseline.train_dataloader.state_dict())
                self.assertEqual(session.events, ["begin", "begin", "end", "destroy_distributed", "sdk_exit", "close"])
                baseline_session.finish_restore.assert_not_called()
                session.finish_restore.assert_called_once()
                for actual, expected in zip(recovered.model.parameters(), baseline.model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
