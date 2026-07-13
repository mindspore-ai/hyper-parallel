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
"""Unit tests for trainer callback base types and built-ins."""
# pylint: disable=protected-access,wrong-import-position
import os
import types
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.trainer.base import TrainerState
from hyper_parallel.trainer.callbacks import base as cb_mod
from hyper_parallel.trainer.callbacks.base import (
    BaseCallback,
    GCCallback,
    GradientHealthCallback,
    LoggingCallback,
    ProgressCallback,
    TrainerCallbackContext,
    TrainerControl,
)


def _context(state=None, *, rank=0, local_rank=None, world_size=1, logs=None):
    """Build a callback context for unit tests."""
    local_rank = rank if local_rank is None else local_rank
    return TrainerCallbackContext(
        state=state or TrainerState(),
        control=TrainerControl(),
        global_rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_world_rank0=rank == 0,
        is_local_rank0=local_rank == 0,
        logs=dict(logs or {}),
    )


class TestBaseCallback(unittest.TestCase):
    """The BaseCallback class is a no-op for every hook."""

    def test_all_hooks_are_no_op(self):
        """Every public hook accepts context plus payload and returns None."""
        callback = BaseCallback()
        context = _context()
        hook_names = [
            name
            for name in dir(BaseCallback)
            if name.startswith("on_") and callable(getattr(BaseCallback, name))
        ]
        for hook_name in hook_names:
            with self.subTest(hook_name=hook_name):
                self.assertIsNone(getattr(callback, hook_name)(context, payload=1))


class TestTrainerControl(unittest.TestCase):
    """TrainerControl merge and reset semantics."""

    def test_merge_uses_or_and_metadata_update(self):
        """Boolean signals use OR; later metadata overwrites same keys."""
        control = TrainerControl(should_log=True, metadata={"source": "a", "x": 1})
        control.merge(
            TrainerControl(
                should_save=True,
                should_log=False,
                metadata={"source": "b", "y": 2},
            )
        )
        self.assertTrue(control.should_log)
        self.assertTrue(control.should_save)
        self.assertEqual(control.metadata, {"source": "b", "x": 1, "y": 2})

    def test_step_reset_keeps_training_stop(self):
        """Step reset clears step flags but preserves training stop."""
        control = TrainerControl(
            should_training_stop=True,
            should_skip_step=True,
            should_log=True,
            metadata={"record": 1},
        )
        control.reset_step_flags()
        self.assertTrue(control.should_training_stop)
        self.assertFalse(control.should_skip_step)
        self.assertFalse(control.should_log)
        self.assertEqual(control.metadata, {})


class TestTrainerCallbackContext(unittest.TestCase):
    """TrainerCallbackContext construction semantics."""

    def test_from_trainer_reads_rank_fields_without_platform_lookup(self):
        """Rank metadata comes from BaseTrainer attributes, not platform fallback."""
        trainer = types.SimpleNamespace(
            state=TrainerState(),
            global_rank=3,
            local_rank=1,
            world_size=8,
            is_world_rank0=False,
            is_local_rank0=False,
            _callback_logs={"loss": 1.0},
        )
        with (
            patch.object(cb_mod.platform, "get_rank", side_effect=AssertionError("platform.get_rank called")),
            patch.object(
                cb_mod.platform,
                "get_world_size",
                side_effect=AssertionError("platform.get_world_size called"),
            ),
        ):
            context = TrainerCallbackContext.from_trainer(trainer, TrainerControl())

        self.assertEqual(context.rank, 3)
        self.assertEqual(context.global_rank, 3)
        self.assertEqual(context.local_rank, 1)
        self.assertEqual(context.world_size, 8)
        self.assertFalse(context.is_world_rank0)
        self.assertFalse(context.is_local_rank0)
        self.assertFalse(hasattr(context, "is_rank0"))
        self.assertEqual(context.logs, {"loss": 1.0})


class TestLoggingCallback(unittest.TestCase):
    """LoggingCallback emits records only on configured log steps."""

    def test_emits_only_on_log_steps_multiples(self):
        """on_step_end returns should_log only on multiples of log_steps."""
        callback = LoggingCallback(log_steps=3, report_throughput=False)
        state = TrainerState()
        callback.on_step_begin(_context(state))
        for step in (1, 2, 4, 5):
            state.global_step = step
            result = callback.on_step_end(_context(state), loss=1.0, grad_norm=0.5)
            self.assertIsNone(result)
        self.assertEqual(state.log_history, [])

        state.global_step = 3
        result = callback.on_step_end(_context(state), loss=1.0, grad_norm=0.5, lr=1e-4)
        self.assertIsNotNone(result)
        self.assertTrue(result.should_log)
        self.assertEqual(len(state.log_history), 1)
        record = state.log_history[0]
        self.assertEqual(record["step"], 3)
        self.assertEqual(record["loss"], 1.0)
        self.assertEqual(record["grad_norm"], 0.5)
        self.assertEqual(record["lr"], 1e-4)

    def test_throughput_uses_payload_tokens(self):
        """tokens_per_sec is computed from payload tokens and elapsed time."""
        callback = LoggingCallback(log_steps=1, report_throughput=True)
        state = TrainerState()
        state.global_step = 1
        with patch.object(cb_mod, "time") as mock_time:
            callback._step_start_time = 100.0
            mock_time.time.return_value = 102.0
            callback.on_step_end(_context(state), loss=1.0, grad_norm=0.5, tokens=1000)
        self.assertAlmostEqual(state.log_history[-1]["tokens_per_sec"], 500.0)


class TestGradientHealthCallback(unittest.TestCase):
    """GradientHealthCallback raises only when enabled and non-finite."""

    def test_disabled_does_not_raise(self):
        """Disabled callback is silent even for NaN."""
        callback = GradientHealthCallback(enabled=False)
        callback.on_before_optimizer_step(_context(), grad_norm=float("nan"))

    def test_nan_grad_raises_on_rank_zero(self):
        """Enabled NaN grad_norm raises on rank 0."""
        callback = GradientHealthCallback(enabled=True)
        with self.assertRaises(RuntimeError) as ctx:
            callback.on_before_optimizer_step(_context(rank=0), grad_norm=float("nan"))
        self.assertIn("Non-finite grad_norm", str(ctx.exception))

    def test_inf_grad_does_not_raise_on_non_rank_zero(self):
        """Non-rank-zero logs but does not raise."""
        callback = GradientHealthCallback(enabled=True)
        callback.on_before_optimizer_step(_context(rank=3), grad_norm=float("inf"))


class TestGCCallback(unittest.TestCase):
    """GCCallback fires at multiples of gc_steps."""

    def test_disabled_when_gc_steps_zero(self):
        """gc_steps=0 never invokes gc.collect."""
        with patch.object(cb_mod, "gc") as mock_gc:
            callback = GCCallback(gc_steps=0)
            for step in range(1, 10):
                state = TrainerState()
                state.global_step = step
                callback.on_step_end(_context(state))
            mock_gc.collect.assert_not_called()
            mock_gc.disable.assert_not_called()

    def test_enabled_collects_at_multiples_only(self):
        """gc_steps=4 invokes gc.collect only on multiples."""
        with patch.object(cb_mod, "gc") as mock_gc:
            callback = GCCallback(gc_steps=4)
            mock_gc.disable.assert_called_once()
            collect_at = []
            for step in range(1, 13):
                state = TrainerState()
                state.global_step = step
                callback.on_step_end(_context(state))
                if mock_gc.collect.call_count > len(collect_at):
                    collect_at.append(step)
        self.assertEqual(collect_at, [4, 8, 12])


class TestProgressCallback(unittest.TestCase):
    """ProgressCallback degrades gracefully when tqdm is unavailable."""

    def test_missing_tqdm_keeps_pbar_none(self):
        """ImportError from tqdm leaves _pbar unset."""
        callback = ProgressCallback()
        with patch.dict("sys.modules", {"tqdm": None}):
            callback.on_train_begin(_context())
        self.assertIsNone(callback._pbar)
        callback.on_step_end(_context(), loss=1.0, grad_norm=0.5)
        callback.on_train_end(_context())

    def test_non_rank_zero_skips_pbar_creation(self):
        """Non-rank-zero must not allocate tqdm."""
        callback = ProgressCallback()
        callback.on_train_begin(_context(rank=3))
        self.assertIsNone(callback._pbar)


class TestBuildDefaultCallbacks(unittest.TestCase):
    """Default callback factory reads nested config."""

    def test_logging_config_read_from_train_logging(self):
        """Factory applies nested logging config values."""
        logging_cfg = types.SimpleNamespace(
            log_steps=7,
            report_global_loss=True,
            report_throughput=False,
            model_flops_per_token=None,
            peak_tflops=None,
        )
        args = types.SimpleNamespace(train=types.SimpleNamespace(logging=logging_cfg))
        callbacks = cb_mod.build_default_callbacks(args)
        logging_callback = callbacks[0]
        self.assertEqual(logging_callback.log_steps, 7)
        self.assertTrue(logging_callback.report_global_loss)
        self.assertFalse(logging_callback.report_throughput)


if __name__ == "__main__":
    unittest.main()
