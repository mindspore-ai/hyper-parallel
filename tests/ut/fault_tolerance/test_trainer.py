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
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.fault_tolerance.state import ModelOptimizerState
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.runtime.fault_tolerance import (
    _TrainerLogState, _synchronize_checkpoint_ledger, _train_recoverable,
)
from hyper_parallel.trainer.runtime.model_integration import DisabledModelIntegrationSession
from hyper_parallel.trainer.state import TrainerState


class _Loader:
    """A consumed cursor that survives iterator replacement."""

    def __init__(self) -> None:
        """Start with no consumed batches or live iterators."""
        self.position = 0
        self.iterations = 0

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Create a fresh iterator from the most recently restored cursor."""
        self.iterations += 1
        while self.position < 6:
            self.position += 1
            yield torch.full((1, 2), self.position / 10)

    def state_dict(self) -> dict[str, int]:
        """Capture consumed batches independently of live generator frames."""
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Position the next iterator after the committed batch."""
        self.position = state["position"]


class _MemorySession:
    """Replace control-plane scheduling while exercising real local state restore."""

    def __init__(self, inject: bool) -> None:
        """Choose whether run() simulates one interrupted forward/backward."""
        self.inject = inject
        self.failed = False
        self.reboot = False
        self.npu = None
        self.step = 0
        self.codec = self.training = self.snapshot = None

    def attach(self, model: Any, optimizer: Any, state: Any, step: int) -> None:
        """Bind the actual codecs that the Trainer adapter constructs."""
        self.codec = ModelOptimizerState(model, optimizer)
        self.training = state
        self.step = step

    @contextmanager
    def update(self, step: int) -> Iterator[None]:
        """Capture only after optimizer, scheduler and progress counters finish."""
        yield
        self.snapshot = (self.codec.rank_state(), self.training.capture())
        self.step = step

    def pause(self) -> None:
        """The CPU-only scheduler has no pending Controller commands."""

    def make_iterator(self) -> Iterator[Any]:
        """Expose the same iterator recreation boundary as the real session."""
        return self.training.make_iterator(preserve_rng=self.failed)

    def run(self, body: Any) -> Any:
        """Roll back one partial forward and reenter the same training body."""
        try:
            return body(self)
        except RuntimeError as error:
            if str(error) != "injected forward failure":
                raise
            self.codec.restore_rank_state(self.snapshot[0]["buffers"], self.snapshot[0]["extra"])
            self.training.restore(self.snapshot[1])
            return body(self)


class TestTrainerReentry(unittest.TestCase):
    """Both Base and Text counter positions must restore the same transaction."""

    def test_replacement_uses_coordinators_completed_save_step(self):
        callback = object.__new__(CheckpointerCallback)
        callback._last_saved_step = -1
        base = SimpleNamespace(_callbacks=[callback])
        session = SimpleNamespace(local_rank=0, step=8)
        tensor = torch.tensor

        def _cpu_tensor(values, **kwargs):
            return tensor(values, dtype=kwargs["dtype"])

        def _broadcast(ledger, src):
            self.assertEqual(src, 0)
            ledger.fill_(7)

        with patch("hyper_parallel.trainer.runtime.fault_tolerance.torch.tensor", side_effect=_cpu_tensor), \
                patch("hyper_parallel.trainer.runtime.fault_tolerance.dist.broadcast", side_effect=_broadcast):
            _synchronize_checkpoint_ledger(base, session)
        self.assertEqual(callback._last_saved_step, 7)

    def test_repaired_metrics_identify_the_source_without_mutating_the_original(self):
        metrics = {"step": 8, "loss": 0.25}
        base = SimpleNamespace(recovery=SimpleNamespace(rank=5), _recovery_step_metrics=metrics)
        provider = _TrainerLogState(base)
        snapshot = provider.state_dict()
        self.assertEqual(snapshot, {"step": 8, "loss": 0.25, "source_rank": 5})
        self.assertNotIn("source_rank", metrics)
        provider.load_state_dict(snapshot)
        self.assertEqual(base._restored_step_metrics["source_rank"], 5)

    @staticmethod
    def _run(inject: bool, increment_before_forward: bool, muon: bool = False) -> Any:
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
        trainer = SimpleNamespace(
            model=model, optimizer=optimizer, state=TrainerState(), recovery=session,
            lr_scheduler=schedulers,
            model_integration=DisabledModelIntegrationSession(), train_dataloader=loader,
            train_steps=6, train_iters=6, train_epochs=1,
            on_train_begin=MagicMock(), on_epoch_begin=MagicMock(), on_epoch_end=MagicMock(),
            on_train_end=MagicMock(), destroy_distributed=MagicMock())

        def _step(iterator):
            data = next(iterator)
            if increment_before_forward:
                trainer.state.global_step += 1
            model(data).square().mean().backward()
            if inject and not session.failed and loader.position == 4:
                session.failed = True
                raise RuntimeError("injected forward failure")
            BaseTrainer.step_optimizers_and_schedulers(trainer)

        trainer.train_step = _step
        with patch("hyper_parallel.trainer.runtime.fault_tolerance.memory_profiler.stop"):
            _train_recoverable(trainer, trainer, session)
        return trainer, session

    def test_only_body_reenters_and_cursor_is_recreated(self):
        for increment_before_forward in (True, False):
            with self.subTest(base_style=increment_before_forward):
                baseline, baseline_session = self._run(False, increment_before_forward)
                self.assertEqual(baseline_session.training.replica_providers,
                                 frozenset({"progress", "logs", "scheduler_0"}))
                recovered, session = self._run(True, increment_before_forward)
                self.assertTrue(session.failed)
                self.assertEqual(recovered.state.global_step, 6)
                self.assertEqual(recovered.train_dataloader.position, 6)
                self.assertEqual(recovered.train_dataloader.iterations, 2)
                recovered.on_train_begin.assert_called_once()
                recovered.on_train_end.assert_called_once()
                recovered.destroy_distributed.assert_called_once()
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
