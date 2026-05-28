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
"""Unit tests for ``hyper_parallel.trainer.callbacks.base``.

Most callbacks are I/O wrappers (TensorBoard, W&B, DCP save/load). The
load-bearing logic worth UT here is what determines whether a callback
**fires** or **stays silent** at all:

1. ``Callback`` base class hooks are no-op (subclasses opt in).
2. ``LoggingCallback`` only emits at multiples of ``log_steps`` and
   computes ``tokens_per_sec`` / ``tflops`` / ``mfu`` correctly.
3. ``GradientHealthCallback`` raises only when ``check_nan_inf`` is on
   AND ``grad_norm`` is non-finite (silent otherwise).
4. ``GCCallback`` fires ``gc.collect`` only at multiples of ``gc_steps``.
5. ``ProgressCallback`` degrades gracefully when ``tqdm`` is missing.

Everything else (DCP / safetensors / W&B / TB writers) needs real
hardware or external deps and is exercised via integration tests.
"""
# pylint: disable=protected-access
import os
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.trainer.base import TrainerState
from hyper_parallel.trainer.callbacks import base as cb_mod
from hyper_parallel.trainer.callbacks.base import (
    Callback,
    GCCallback,
    GradientHealthCallback,
    LoggingCallback,
    ProgressCallback,
)


def _make_trainer(*, log_cfg=None, train_cfg=None, data_cfg=None, debug_cfg=None,
                  lr_scheduler=None):
    """Build a SimpleNamespace ``trainer`` shaped like ``BaseTrainer`` from the callbacks' POV."""
    args = types.SimpleNamespace(
        logging=log_cfg,
        train=train_cfg or types.SimpleNamespace(global_batch_size=4),
        data=data_cfg or types.SimpleNamespace(max_seq_len=8),
        debug=debug_cfg,
    )
    return types.SimpleNamespace(
        args=args,
        lr_scheduler=lr_scheduler,
        _last_global_tokens=None,
        dispatch_log_event=None,
    )


class TestCallbackBase(unittest.TestCase):
    """The ``Callback`` base class must be a no-op for every hook."""

    def test_all_hooks_are_no_op(self):
        """Each lifecycle hook can be called with a state object and returns ``None``."""
        cb = Callback(trainer=MagicMock())
        state = TrainerState()
        # Every hook returns None and never raises.
        self.assertIsNone(cb.on_init_end(state))
        self.assertIsNone(cb.on_train_begin(state))
        self.assertIsNone(cb.on_train_end(state))
        self.assertIsNone(cb.on_epoch_begin(state))
        self.assertIsNone(cb.on_epoch_end(state))
        self.assertIsNone(cb.on_step_begin(state))
        self.assertIsNone(cb.on_step_end(state, loss=1.0, grad_norm=0.5))
        self.assertIsNone(cb.on_substep_end(state))
        self.assertIsNone(cb.on_pre_optimizer_step(state, grad_norm=0.5))
        self.assertIsNone(cb.on_log(state, metrics={}))
        self.assertIsNone(cb.on_save(state, checkpoint_dir="/tmp/x"))
        self.assertIsNone(cb.on_load(state, checkpoint_dir="/tmp/x"))
        self.assertIsNone(cb.on_evaluate(state, metrics={}))


class TestLoggingCallback(unittest.TestCase):
    """``LoggingCallback`` gates emission on ``log_steps`` and computes throughput."""

    def _build(self, *, log_steps=2, report_throughput=False,
               model_flops_per_token=None, peak_tflops=None,
               global_batch_size=4, max_seq_len=8, lr=1e-4):
        """Build a ``LoggingCallback`` against a synthetic trainer with the given config."""
        log_cfg = types.SimpleNamespace(
            log_steps=log_steps,
            report_global_loss=False,
            report_throughput=report_throughput,
            model_flops_per_token=model_flops_per_token,
            peak_tflops=peak_tflops,
        )
        scheduler = MagicMock()
        scheduler.get_last_lr.return_value = [lr]
        trainer = _make_trainer(
            log_cfg=log_cfg,
            train_cfg=types.SimpleNamespace(global_batch_size=global_batch_size),
            data_cfg=types.SimpleNamespace(max_seq_len=max_seq_len),
            lr_scheduler=scheduler,
        )
        return LoggingCallback(trainer), trainer

    def test_emits_only_on_log_steps_multiples(self):
        """``on_step_end`` produces a metric record only when ``global_step % log_steps == 0``."""
        cb, _ = self._build(log_steps=3)
        state = TrainerState()
        cb.on_step_begin(state)
        for step in (1, 2, 4, 5):  # not multiples of 3
            state.global_step = step
            cb.on_step_end(state, loss=1.0, grad_norm=0.5)
        self.assertEqual(state.log_history, [], (
            f"No metrics record should be emitted on non-multiples, got {state.log_history}"
        ))

        state.global_step = 3
        cb.on_step_end(state, loss=1.0, grad_norm=0.5)
        self.assertEqual(len(state.log_history), 1, (
            f"Expected exactly one record at step=3, got {len(state.log_history)}"
        ))
        record = state.log_history[0]
        self.assertEqual(record["step"], 3)
        self.assertEqual(record["loss"], 1.0)
        self.assertEqual(record["grad_norm"], 0.5)
        self.assertEqual(record["lr"], 1e-4)

    def test_throughput_off_omits_optional_metrics(self):
        """``report_throughput=False`` → ``tokens_per_sec`` stays ``None`` in the record."""
        cb, _ = self._build(log_steps=1, report_throughput=False)
        state = TrainerState()
        cb.on_step_begin(state)
        state.global_step = 1
        cb.on_step_end(state, loss=2.0, grad_norm=0.1)
        self.assertEqual(len(state.log_history), 1)
        self.assertIsNone(state.log_history[0]["tokens_per_sec"], (
            f"Throughput off should leave tokens_per_sec=None, got {state.log_history[0]}"
        ))

    def test_throughput_uses_last_global_tokens_when_available(self):
        """``tokens_per_sec`` is computed from ``trainer._last_global_tokens`` / elapsed."""
        cb, trainer = self._build(
            log_steps=1, report_throughput=True,
            global_batch_size=2, max_seq_len=4,
        )
        trainer._last_global_tokens = 1000
        state = TrainerState()
        state.global_step = 1

        # Pin elapsed = 2.0s so the expected tokens_per_sec = 500.
        with patch.object(cb_mod, "time") as mock_time:
            cb._step_start_time = 100.0
            mock_time.time.return_value = 102.0
            cb.on_step_end(state, loss=1.0, grad_norm=0.5)

        tps = state.log_history[-1]["tokens_per_sec"]
        self.assertAlmostEqual(tps, 1000 / 2.0, places=3, msg=(
            f"tokens_per_sec must use _last_global_tokens / elapsed, got {tps}"
        ))


class TestGradientHealthCallback(unittest.TestCase):
    """Only fire when explicitly enabled AND grad_norm is non-finite."""

    def _build(self, *, check_nan_inf=True):
        debug_cfg = types.SimpleNamespace(check_nan_inf=check_nan_inf)
        return GradientHealthCallback(_make_trainer(debug_cfg=debug_cfg))

    def test_disabled_by_default_does_not_raise(self):
        """``check_nan_inf=False`` → guard is silent even on NaN."""
        cb = self._build(check_nan_inf=False)
        cb.on_pre_optimizer_step(TrainerState(), grad_norm=float("nan"))  # no raise

    def test_finite_grad_norm_is_silent(self):
        """Enabled + finite grad_norm → no error."""
        cb = self._build()
        cb.on_pre_optimizer_step(TrainerState(), grad_norm=1.5)  # no raise

    def test_nan_grad_raises_on_rank_zero(self):
        """Enabled + NaN grad_norm → rank-0 raises ``RuntimeError`` with diagnostic."""
        cb = self._build()
        with patch.object(cb_mod, "platform") as mock_platform:
            mock_platform.get_rank.return_value = 0
            with self.assertRaises(RuntimeError) as ctx:
                cb.on_pre_optimizer_step(TrainerState(), grad_norm=float("nan"))
        self.assertIn("Non-finite grad_norm", str(ctx.exception))

    def test_inf_grad_does_not_raise_on_non_rank_zero(self):
        """Non-rank-0 must NOT raise on non-finite grad — only rank-0 raises; NCCL tears down the rest."""
        cb = self._build()
        with patch.object(cb_mod, "platform") as mock_platform:
            mock_platform.get_rank.return_value = 3
            cb.on_pre_optimizer_step(TrainerState(), grad_norm=float("inf"))  # no raise


class TestGCCallback(unittest.TestCase):
    """``GCCallback`` only fires at multiples of ``gc_steps``."""

    def test_disabled_when_gc_steps_zero(self):
        """``gc_steps=0`` → never invokes ``gc.collect``."""
        with patch.object(cb_mod, "gc") as mock_gc:
            cb = GCCallback(_make_trainer(debug_cfg=types.SimpleNamespace(gc_steps=0)))
            for step in range(1, 10):
                state = TrainerState()
                state.global_step = step
                cb.on_step_end(state)
            mock_gc.collect.assert_not_called()
            mock_gc.disable.assert_not_called()

    def test_enabled_collects_at_multiples_only(self):
        """``gc_steps=4`` → invokes ``gc.collect`` only at step % 4 == 0."""
        with patch.object(cb_mod, "gc") as mock_gc:
            cb = GCCallback(_make_trainer(debug_cfg=types.SimpleNamespace(gc_steps=4)))
            # Construction disables the auto generational collector once.
            mock_gc.disable.assert_called_once()

            collect_at = []
            for step in range(1, 13):
                state = TrainerState()
                state.global_step = step
                cb.on_step_end(state)
                if mock_gc.collect.call_count > len(collect_at):
                    collect_at.append(step)
        self.assertEqual(collect_at, [4, 8, 12], (
            f"gc.collect must fire at step %4==0 only, got {collect_at}"
        ))


class TestProgressCallback(unittest.TestCase):
    """``ProgressCallback`` degrades gracefully when ``tqdm`` is unavailable."""

    def test_missing_tqdm_keeps_pbar_none(self):
        """``ImportError`` from tqdm → ``_pbar`` stays ``None`` and updates are no-ops."""
        cb = ProgressCallback(_make_trainer())
        with patch.object(cb_mod, "platform") as mock_platform, \
             patch.dict("sys.modules", {"tqdm": None}):
            mock_platform.get_rank.return_value = 0
            cb.on_train_begin(TrainerState())
        self.assertIsNone(cb._pbar, (
            f"pbar must stay None when tqdm is missing, got {cb._pbar!r}"
        ))
        # Subsequent on_step_end / on_train_end must be silent no-ops.
        cb.on_step_end(TrainerState(), loss=1.0, grad_norm=0.5)
        cb.on_train_end(TrainerState())

    def test_non_rank_zero_skips_pbar_creation(self):
        """Non-rank-0 must not allocate a tqdm bar (avoids per-rank noise)."""
        cb = ProgressCallback(_make_trainer())
        with patch.object(cb_mod, "platform") as mock_platform:
            mock_platform.get_rank.return_value = 5
            cb.on_train_begin(TrainerState())
        self.assertIsNone(cb._pbar, (
            f"Non-rank-0 must skip tqdm allocation, got {cb._pbar!r}"
        ))


if __name__ == "__main__":
    unittest.main()
