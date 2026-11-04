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
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.high_availability.recovery_state import ModelOptimizerState
from hyper_parallel.data.batching.build_dataloader import TokenBatchLoader
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.runtime.process_recovery import (
    _train_recoverable,
    _build_train_state,
    recovery_constructor,
    recovery_epoch_initialized,
    recovery_train_loop,
    recovery_train_step,
    validate_recovery_config,
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
        self._initial_progress_pending = False

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


    def initialize_training_progress(self, step: int) -> None:
        """Record the post-callback progress only on the original startup."""
        if self._initial_progress_pending:
            self.step = step
            self._initial_progress_pending = False


    def close(self) -> None:
        """Observe outer runtime cleanup without creating hardware resources."""
        self.closed = True
        self.events.append("close")

    def run(self, body: Any) -> Any:
        """Roll back one partial forward and reenter the same training body."""
        self._initial_progress_pending = not self.reboot
        try:
            result = body()
        except RuntimeError as error:
            if str(error) != "injected forward failure":
                raise
            self.codec.restore_rank_state(self.snapshot[0]["buffers"], self.snapshot[0]["extra"])
            self.train_state.restore(self.snapshot[1])
            result = body()
        self.events.append("sdk_exit")
        return result


class _CpuBaseTrainer(BaseTrainer):
    """Use real native Trainer loops and callbacks with a mocked distributed boundary."""

    def destroy_distributed(self) -> None:
        """Record native teardown before the SDK exits."""
        self.recovery.events.append("destroy_distributed")


class TestTrainerReentry(unittest.TestCase):
    """Both Base and Text counter positions must restore the same transaction."""

    def test_dynamic_source_generator_resumes_after_iterator_rebuild(self) -> None:
        """Feature: Dynamic batching retains its inner loader's committed RNG.

        Description: Reconstruct a real TokenBatchLoader after restoring its Trainer state.
        Expectation: Epoch finalization removes the extra inner iterator seed draw.
        """
        loader = TokenBatchLoader(
            [{"input_ids": torch.tensor([1, 2])}],
            collate_fn=list,
            batch_size=1,
            dp_world_size=1,
            max_seq_len=2,
            min_buffered_samples=1,
        )
        base = SimpleNamespace(train_dataloader=loader, lr_scheduler=None, state=TrainerState())
        state = _build_train_state(base, None)
        base.recovery = SimpleNamespace(train_state=state)
        iter(loader)
        saved = state.capture()
        expected = loader.source_dataloader.generator.get_state()
        self.assertNotIn("_hp_generator_state", loader.source_dataloader.state_dict())
        state.restore(saved)

        @recovery_epoch_initialized
        def _initialize(active):
            """Construct the actual inner source iterator through the native wrapper."""
            return iter(active.train_dataloader)

        _initialize(base)
        torch.testing.assert_close(loader.source_dataloader.generator.get_state(), expected, rtol=0, atol=0)

    def test_enabled_profiling_providers_use_existing_snapshot_transport(self) -> None:
        """Feature: Profiling progress is protected only for enabled features.

        Description: Bind and restore training state with profiling enabled or disabled.
        Expectation: Enabled providers participate in existing snapshots; disabled ones add no state.
        """
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                profiler = MagicMock(config=SimpleNamespace(enable=enabled))
                profiler.state_dict.return_value = {"training_step": 3}
                memory = MagicMock()
                memory.state_dict.return_value = {"current_step": 4}
                model = torch.nn.Linear(2, 1)
                base = SimpleNamespace(
                    model=model,
                    optimizer=AdamW(model.parameters()),
                    train_dataloader=_Loader(),
                    lr_scheduler=None,
                    state=TrainerState(),
                    profiler_callback=profiler,
                    config=SimpleNamespace(memory=SimpleNamespace(enable=enabled, mem_info=False)),
                )
                session = SimpleNamespace(npu=None, attach=MagicMock())

                def _run(body, context=(enabled, session)):
                    enabled, session = context
                    state = session.attach.call_args.args[2]
                    snapshot = state.capture()
                    keys = set(snapshot["replica_state"])
                    self.assertEqual(keys, {"progress", "profiler", "memory_profiler"} if enabled else {"progress"})
                    state.restore(snapshot)
                    return body()

                session.run = _run
                with patch("hyper_parallel.trainer.runtime.process_recovery.memory_profiler", memory):
                    self.assertEqual(_train_recoverable(base, base, session, lambda trainer: "done"), "done")
                self.assertEqual(profiler.load_state_dict.call_count, int(enabled))
                self.assertEqual(memory.load_state_dict.call_count, int(enabled))

    def test_constructor_leaves_checkpoint_cleanup_to_train_binding(self) -> None:
        """Feature: Bind the recovery session during Trainer construction.

        Description: Construct direct and wrapped Trainers with and without checkpointing.
        Expectation: Store the session without registering checkpoint cleanup.
        """
        for wrapped in (False, True):
            for checkpointing in (False, True):
                with self.subTest(wrapped=wrapped, checkpointing=checkpointing):
                    trainer = SimpleNamespace()
                    checkpoint = MagicMock(spec=CheckpointerCallback)
                    checkpoint.checkpointer = MagicMock()
                    config = SimpleNamespace(worker_recovery=SimpleNamespace(enable_worker_reboot=True))
                    with patch("hyper_parallel.trainer.runtime.process_recovery.validate_recovery_config"), \
                            patch("hyper_parallel.trainer.runtime.process_recovery.RecoverySession") as factory:
                        session = factory.return_value

                        def _initialize(
                                active,
                                active_config,
                                context=(checkpoint, checkpointing, config, session, wrapped),
                            ):
                            checkpoint, checkpointing, config, session, wrapped = context
                            self.assertIs(active_config, config)
                            session.register_cleanup.assert_not_called()
                            base = SimpleNamespace() if wrapped else active
                            if wrapped:
                                active.base = base
                            base.train_dataloader = SimpleNamespace(num_workers=0)
                            if checkpointing:
                                base.checkpoint_callback = checkpoint
                            base._callbacks = [object(), checkpoint] if checkpointing else [object()]

                        recovery_constructor(_initialize)(trainer, config)
                        self.assertIs((trainer.base if wrapped else trainer).recovery, session)
                        session.register_cleanup.assert_not_called()
                        session.close.assert_not_called()

    def test_cpu_offload_cp_and_attention_swap_keep_other_recovery_restrictions(self) -> None:
        """Feature: Recovery accepts audited memory features while retaining unrelated restrictions.

        Description: Accept CPU offload with CP and attention swap;
            accept full/selective recomputation separately while rejecting PP and compile.
        Expectation: Offload, CP, attention swap and recomputation pass; PP and compile remain rejected.
        """
        config = SimpleNamespace(
            accelerator=SimpleNamespace(cp_size=2, pp_size=1),
            training=SimpleNamespace(
                backend="hccl",
                init_device="meta",
                eval_iters=0,
                eval_steps=0,
                eval_epochs=0,
                low_precision=SimpleNamespace(enabled=False),
            ),
            compile=SimpleNamespace(enabled=False),
            fsdp_config=SimpleNamespace(enable_offload=True, reshard_after_backward=True),
            optimizer=SimpleNamespace(swap=SimpleNamespace(enabled=False)),
            activation_swap="attention",
            activation_checkpoint=SimpleNamespace(mode="off"),
            dataloader=SimpleNamespace(use_background_prefetcher=False),
            checkpoint=SimpleNamespace(is_async=False, is_peft=False),
            model_integration=SimpleNamespace(mode="off"),
            peft=None,
            magi=None,
            profiler=SimpleNamespace(enable=False),
            memory=SimpleNamespace(enable=False),
            stress_detect=SimpleNamespace(op_stress_detect=SimpleNamespace(detection_interval=0)),
        )
        for mode in ("off", "full", "selective"):
            config.activation_swap = "attention" if mode == "off" else "none"
            config.activation_checkpoint.mode = mode
            validate_recovery_config(config)
        for init_device in ("meta", "cpu", "cuda", "npu"):
            with self.subTest(init_device=init_device):
                config.training.init_device = init_device
                validate_recovery_config(config)
        config.checkpoint.is_async = True
        config.checkpoint.is_peft = True
        config.training.low_precision.enabled = True
        config.profiler.enable = True
        config.memory.enable = True
        config.stress_detect.op_stress_detect.detection_interval = 10
        config.fsdp_config.reshard_after_backward = False
        validate_recovery_config(config)
        config.peft = object()
        with self.assertRaisesRegex(ValueError, "PEFT"):
            validate_recovery_config(config)
        config.peft = None
        for owner, name, enabled in (
            (config.accelerator, "pp_size", 2),
            (config.compile, "enabled", True),
        ):
            with self.subTest(feature=name):
                previous = getattr(owner, name)
                setattr(owner, name, enabled)
                with self.assertRaisesRegex(ValueError, "PP|eager"):
                    validate_recovery_config(config)
                setattr(owner, name, previous)

    def test_original_train_begin_executes_on_each_reentry(self):
        """Feature: Original Trainer initialization after replica recovery.

        Description: Restore a replacement, enter its native train twice and include a checkpoint callback.
        Expectation: Each native train entry runs its own original initialization and callback dispatch.
        """
        events = []
        callback = MagicMock()
        checkpoint = MagicMock(spec=CheckpointerCallback)
        checkpoint.checkpointer = MagicMock()
        callback.on_train_begin.side_effect = lambda state: events.append(("callback", state.global_step))
        checkpoint.on_train_begin.side_effect = lambda state: events.append(("checkpoint", state.global_step))
        session = SimpleNamespace(reboot=True, npu=None, attach=MagicMock(), close=MagicMock(),
                                  initialize_training_progress=MagicMock(), register_cleanup=MagicMock())
        session.initialize_training_progress.side_effect = lambda step: events.append(("progress", step))
        base = SimpleNamespace(
            recovery=session, state=TrainerState(), train_dataloader=_Loader(), lr_scheduler=None,
            model=None, optimizer=None, _callbacks=[callback, checkpoint], checkpoint_callback=checkpoint,
            abort_profilers=MagicMock(), destroy_distributed=MagicMock())
        trainer = SimpleNamespace(base=base, on_train_end=MagicMock())

        def _attach(*_args):
            session.register_cleanup.assert_called_once_with(checkpoint.cleanup_state)

        session.attach.side_effect = _attach

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
            body()
            return body()

        session.run = _run
        self.assertIs(recovery_train_loop(_native_train)(trainer), result)
        self.assertEqual(events, [
            ("trainer", 8), ("callback", 8), ("checkpoint", 8), ("progress", 8), ("train", 8),
        ] * 2)
        self.assertEqual(callback.on_train_begin.call_count, 2)
        self.assertEqual(checkpoint.on_train_begin.call_count, 2)
        session.register_cleanup.assert_called_once_with(checkpoint.cleanup_state)

    def test_reentry_restores_rng_after_begin_epoch_and_iterator(self):
        """Feature: RNG restoration follows replacement lifecycle initialization.

        Description: Consume random values in original begin, epoch begin and native iterator construction.
        Expectation: The resumed draw matches the commit despite extra initialization draws;
        both callbacks still execute once.
        """
        callback = MagicMock()
        callback.on_train_begin.side_effect = lambda state: torch.rand(3)
        callback.on_epoch_begin.side_effect = lambda state: torch.rand(3)
        session = SimpleNamespace(
            reboot=True,
            npu=None,
            attach=MagicMock(),
            close=MagicMock(),
            initialize_training_progress=MagicMock(),
        )
        base = SimpleNamespace(
            recovery=session, state=TrainerState(), train_dataloader=_Loader(), lr_scheduler=None,
            model=None, optimizer=None, _callbacks=[callback], checkpoint_callback=None,
            abort_profilers=MagicMock(), destroy_distributed=MagicMock())
        trainer = SimpleNamespace(base=base, on_train_end=MagicMock())

        def _begin(active):
            BaseTrainer.on_train_begin(active.base)

        trainer.on_train_begin = MethodType(_begin, trainer)

        def _factory():
            torch.rand(3)
            return iter(())

        @recovery_epoch_initialized
        def _initialize_epoch(active):
            """Run native epoch initialization before restoring the committed RNG."""
            BaseTrainer.on_epoch_begin(active.base)
            return _factory()

        def _native_train(active):
            active.on_train_begin()
            _initialize_epoch(active)
            return torch.rand(3)

        def _run(body):
            training = session.attach.call_args.args[2]
            session.train_state = training
            base.state.global_step = 8
            saved = training.capture()
            expected = torch.rand(3)
            torch.rand(11)
            training.restore(saved)
            session.reboot = False
            actual = body()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return actual

        session.run = _run
        recovery_train_loop(_native_train)(trainer)
        callback.on_train_begin.assert_called_once_with(base.state)
        callback.on_epoch_begin.assert_called_once_with(base.state)


    def test_native_train_end_precedes_sdk_exit_failure(self):
        """Feature: Native train-end callbacks independent of SDK shutdown success.

        Description: Complete the native body and raise an error afterward in the SDK exit path.
        Expectation: The real train-end dispatcher has already run once when the exit error propagates.
        """
        events = []
        callback = MagicMock()
        callback.on_train_end.side_effect = lambda state: events.append("end")
        session = SimpleNamespace(reboot=False, npu=None, attach=MagicMock(), close=MagicMock())
        trainer = object.__new__(_CpuBaseTrainer)
        vars(trainer).update(
            recovery=session, state=TrainerState(), train_dataloader=_Loader(), lr_scheduler=None,
            model=None, optimizer=None, _callbacks=[callback], checkpoint_callback=None, abort_profilers=MagicMock())

        def _native_train(active):
            active.on_train_end()

        def _run(body):
            body()
            events.append("sdk_exit_error")
            raise ValueError("SDK shutdown failed")

        session.run = _run
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
                with patch("hyper_parallel.trainer.runtime.process_recovery._train_recoverable", side_effect=error):
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
        with patch("hyper_parallel.trainer.runtime.process_recovery._train_recoverable", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                run(trainer)
        base.abort_profilers.assert_not_called()
        session.close.assert_called_once()

    def test_checkpoint_cleanup_resets_marker_before_aborting_async_save(self) -> None:
        """Feature: Clear checkpoint bookkeeping and persistence through one cleanup entry.

        Description: Clean callback state with old save markers and repeat after a later save.
        Expectation: Async abort observes -1, including when its failure must propagate.
        """
        for saved_step in (-1, 7, 8):
            with self.subTest(saved_step=saved_step):
                callback = object.__new__(CheckpointerCallback)
                callback._last_saved_step = saved_step
                callback.checkpointer = MagicMock()

                def _abort(context=(callback,)):
                    callback, = context
                    self.assertEqual(callback._last_saved_step, -1)

                callback.checkpointer.abort_async_save.side_effect = _abort
                callback.cleanup_state()
                self.assertEqual(callback._last_saved_step, -1)
                callback.checkpointer.abort_async_save.assert_called_once_with()
                callback._last_saved_step = 9
                callback.checkpointer.abort_async_save.side_effect = RuntimeError("abort failed")
                with self.assertRaisesRegex(RuntimeError, "abort failed"):
                    callback.cleanup_state()
                self.assertEqual(callback._last_saved_step, -1)

    def test_train_entry_does_not_change_checkpoint_markers(self) -> None:
        """Feature: Keep checkpoint marker handling outside native train entry.

        Description: Reenter original and replacement workers with different previous save markers.
        Expectation: Entries preserve the callback's marker and never perform checkpoint collectives.
        """
        for replacement in (False, True):
            for saved_step in (-1, 7, 8):
                with self.subTest(replacement=replacement, saved_step=saved_step):
                    callback = object.__new__(CheckpointerCallback)
                    callback._last_saved_step = saved_step
                    model = torch.nn.Linear(2, 1)
                    base = SimpleNamespace(
                        model=model,
                        optimizer=AdamW(model.parameters()),
                        train_dataloader=_Loader(),
                        lr_scheduler=None,
                        state=SimpleNamespace(global_step=8),
                        checkpoint_callback=callback,
                    )
                    session = SimpleNamespace(reboot=replacement, npu=None, attach=MagicMock(),
                                              register_cleanup=MagicMock())
                    observed = []

                    def _train(trainer, context=(base, callback, observed)):
                        base, callback, observed = context
                        self.assertIs(trainer, base)
                        observed.append(callback._last_saved_step)
                        callback._last_saved_step = 8
                        return "done"

                    def _run(body, context=(session,)):
                        session, = context
                        session.reboot = False
                        body()
                        return body()

                    session.run = _run
                    with patch("torch.distributed.all_reduce") as reduction:
                        self.assertEqual(_train_recoverable(base, base, session, _train), "done")
                    self.assertEqual(observed, [saved_step, 8])
                    reduction.assert_not_called()

    def test_missing_checkpoint_callback_does_not_require_marker_reset(self) -> None:
        """Feature: Optional checkpoint callback during train entry.

        Description: Enter and reenter training without a checkpoint_callback attribute.
        Expectation: Both native entries execute without checkpoint communication or new attributes.
        """
        model = torch.nn.Linear(2, 1)
        base = SimpleNamespace(
            model=model,
            optimizer=AdamW(model.parameters()),
            train_dataloader=_Loader(),
            lr_scheduler=None,
            state=SimpleNamespace(global_step=0),
        )
        session = SimpleNamespace(reboot=True, npu=None, attach=MagicMock())
        train = MagicMock(return_value="done")

        def _run(body):
            """Execute the native training path with local state capture and simulated recovery."""
            session.reboot = False
            body()
            return body()

        session.run = _run
        with patch("torch.distributed.all_reduce") as reduction:
            self.assertEqual(_train_recoverable(base, base, session, train), "done")
        self.assertEqual(train.call_count, 2)
        self.assertFalse(hasattr(base, "checkpoint_callback"))
        reduction.assert_not_called()

    @staticmethod
    def _run(inject: bool, increment_before_forward: bool, muon: bool = False,
             trainer_type: type = TextTrainer, epochs: int = 1) -> Any:
        """Compare the native training loop with an optional recovery retry."""
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1)
        optimizer = (ChainedOptimizer(model, {"muon": Muon([model.weight], ns_steps=2),
                                             "adamw": AdamW([model.bias], lr=0.01)})
                     if muon else AdamW(model.parameters(), lr=0.01))
        loader = _Loader()
        session = _MemorySession(inject)
        schedulers = ([torch.optim.lr_scheduler.StepLR(leaf, 2, gamma=0.5)
                       for leaf in optimizer.chained_optimizers] if muon
                      else torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5))
        base = object.__new__(_CpuBaseTrainer)
        callback = MagicMock()
        callback.on_train_begin.side_effect = lambda state: session.events.append("begin")
        callback.on_epoch_begin.side_effect = lambda state: None
        callback.on_train_end.side_effect = lambda state: session.events.append("end")
        vars(base).update(
            model=model, optimizer=optimizer, state=TrainerState(), recovery=session,
            lr_scheduler=schedulers, local_rank=0, _callbacks=[callback], checkpoint_callback=None,
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
                baseline, _ = self._run(False, before, trainer_type=trainer_type, epochs=2)
                recovered, session = self._run(True, before, trainer_type=trainer_type, epochs=2)
                self.assertEqual(recovered.state.global_step, 6)
                self.assertEqual(recovered.state.epoch, 2)
                self.assertEqual(recovered.train_dataloader.state_dict(), baseline.train_dataloader.state_dict())
                self.assertEqual(session.events, ["begin", "begin", "end", "destroy_distributed", "sdk_exit", "close"])
                for actual, expected in zip(recovered.model.parameters(), baseline.model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
