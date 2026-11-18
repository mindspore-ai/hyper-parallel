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
"""Unit tests for cadence-controlled router auxiliary-loss logging."""

import io
import json
import logging
import subprocess
import tempfile
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.components.modules.moe_utils import MoEMonitorCallback
from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.config.monitor import MonitorConfig
from hyper_parallel.trainer.runtime.monitor.session import MonitorSession


class TestLoggingCallback(unittest.TestCase):
    """Verify router metrics use the normal rank, cadence and terminal output path."""

    @staticmethod
    def _trainer(monitor=None, logging_steps: int = 2, rank: int = 0) -> SimpleNamespace:
        """Build the logging callback's minimal Trainer surface."""
        return SimpleNamespace(
            mesh=None, global_rank=rank, moe_monitor=monitor,
            config=SimpleNamespace(training=SimpleNamespace(logging_steps=logging_steps)),
            step_env_metrics={"training/total_loss": 2.5},
        )

    def test_global_aux_metric_is_not_overwritten_by_local_mean(self) -> None:
        """Preserve the shared AUX value in terminal output, including zero."""
        for global_value in (4.0, 0.0):
            with self.subTest(global_value=global_value):
                trainer = self._trainer()
                trainer.step_env_metrics["training/router_aux_loss"] = global_value
                callback = LoggingCallback(trainer)
                state = TrainerState(global_step=2)
                with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
                    callback.on_step_end(state, loss=2.5, loss_dict={"router_aux_loss": 2.0}, grad_norm=1.0)
                self.assertIn(f"training/router_aux_loss={global_value:g}", log.call_args.args[1])

    def test_meterless_fallback_preserves_supplied_step_value(self) -> None:
        """Log the supplied local step value without inferring normalization."""
        trainer = self._trainer()
        trainer.num_micro_batches = 2
        callback = LoggingCallback(trainer)
        state = TrainerState(global_step=2)
        with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
            callback.on_step_end(state, loss=2.5, loss_dict={"router_aux_loss": 4.0}, grad_norm=1.0)
        message = log.call_args.args[1]
        self.assertIn("training/router_aux_loss=4", message.split(), f"expected supplied AUX=4, got={message}")

    def test_callback_import_does_not_load_model_runtime(self) -> None:
        """
        Feature: CPU callback imports.
        Description: Import logging in a fresh interpreter with model dependencies unavailable.
        Expectation: Logging callbacks load without importing the Transformers model runtime.
        """
        script = dedent("""
            import importlib.abc
            import sys

            class RejectModelRuntime(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname == "transformers.modeling_utils":
                        raise ImportError(f"Unexpected model dependency: {fullname}")
                    return None

            sys.meta_path.insert(0, RejectModelRuntime())
            import hyper_parallel.trainer.callbacks as callbacks
            from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback

            assert callbacks.LoggingCallback is LoggingCallback, (
                f"Expected {LoggingCallback!r}, got {callbacks.LoggingCallback!r}")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=60,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Expected import exit code 0, got {result.returncode}: {result.stderr}",
        )

    def test_raw_aux_loss_is_logged_with_bias_updates_disabled(self) -> None:
        """
        Feature: Auxiliary-loss logging.
        Description: The real monitor supplies mean raw router losses independently of bias updates.
        Expectation: Rank zero prints one aux metric without modifying shared metrics or the main loss.
        """
        layers = [SimpleNamespace(last_aux_loss=torch.tensor(value)) for value in (1.0, 1.5)]
        monitor = MoEMonitorCallback(SimpleNamespace(modules=lambda: iter(layers)))
        trainer = self._trainer(monitor)
        callback = LoggingCallback(trainer)
        state = TrainerState(global_step=2)
        with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
            callback.on_step_end(state, loss=2.5, loss_dict={}, grad_norm=1.0)
            callback.on_step_end(state, loss=2.5, loss_dict={}, grad_norm=1.0)
        log.assert_called_once()
        message = log.call_args.args[1]
        self.assertIn("training/router_aux_loss=1.25", message)
        self.assertIn("training/total_loss=2.5", message)
        self.assertEqual(trainer.step_env_metrics, {"training/total_loss": 2.5})

    def test_skipped_logs_do_not_read_aux_device_scalars(self) -> None:
        """
        Feature: Auxiliary-loss logging.
        Description: Disabled logs, nonzero ranks and intermediate steps must not read auxiliary scalars.
        Expectation: The aux property is never evaluated and no message is emitted.
        """
        for cadence, rank, step in ((0, 0, 2), (2, 1, 2), (2, 0, 1)):
            with self.subTest(cadence=cadence, rank=rank, step=step):
                monitor = Mock()
                trainer = self._trainer(monitor, logging_steps=cadence, rank=rank)
                callback = LoggingCallback(trainer)
                with (
                    patch.object(type(monitor), "last_mean_aux_loss", new_callable=PropertyMock, create=True) as read,
                    patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log,
                ):
                    callback.on_step_end(TrainerState(global_step=step), loss=2.5, loss_dict={}, grad_norm=1.0)
                read.assert_not_called()
                log.assert_not_called()

    def test_missing_aux_is_omitted_and_zero_aux_is_logged(self) -> None:
        """
        Feature: Auxiliary-loss logging.
        Description: Missing or disabled aux is omitted while a computed zero remains visible.
        Expectation: Ordinary loss logging continues and zero aux is not mistaken for a missing metric.
        """
        for monitor in (None, SimpleNamespace(last_mean_aux_loss=None), SimpleNamespace(last_mean_aux_loss=0.0)):
            with self.subTest(monitor=monitor):
                callback = LoggingCallback(self._trainer(monitor))
                with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
                    callback.on_step_end(TrainerState(global_step=2), loss=2.5, loss_dict={}, grad_norm=1.0)
                message = log.call_args.args[1]
                if monitor is not None and monitor.last_mean_aux_loss is not None:
                    self.assertIn("training/router_aux_loss=0", message)
                else:
                    self.assertNotIn("training/router_aux_loss", message)

    def test_aux_log_uses_single_monitor_writer(self) -> None:
        """
        Feature: Structured output.
        Description: Router auxiliary metrics share the canonical monitor writer.
        Expectation: One writer call contains the aux field, with no duplicate console call.
        """
        trainer = self._trainer(SimpleNamespace(last_mean_aux_loss=1.25))
        trainer.monitor = SimpleNamespace(enabled=True, log_step=Mock(), config=MonitorConfig(sinks=["jsonl"]))
        callback = LoggingCallback(trainer)
        state = TrainerState(global_step=2)
        with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
            callback.on_step_end(state, loss=2.5, loss_dict={}, grad_norm=1.0)
            callback.on_step_end(state, loss=2.5, loss_dict={}, grad_norm=1.0)
        trainer.monitor.log_step.assert_called_once()
        self.assertEqual(trainer.monitor.log_step.call_args.args[1]["training/router_aux_loss"], 1.25)
        log.assert_not_called()

    def test_progress_and_metrics_share_one_console_line(self) -> None:
        """Feature: Combined progress and logging.

        Description: Emit step 3/20 twice through real logging and the monitor writer.
        Expectation: One line and one structured event; every metric occurs only once.
        """
        for active, console in ((False, True), (True, True), (True, False)):
            with self.subTest(active=active, console=console), tempfile.TemporaryDirectory() as directory:
                trainer = self._trainer(logging_steps=1)
                trainer.train_iters = 20
                trainer.config.monitor = MonitorConfig(
                    enabled=active, output_dir=directory, sinks=["console", "jsonl"] if console else ["jsonl"],
                )
                trainer.monitor = MonitorSession(trainer)
                trainer.monitor.start_output()
                self.addCleanup(trainer.monitor.close)
                callback = LoggingCallback(trainer)
                state = TrainerState(global_step=3)
                stream = io.StringIO()
                handler = logging.StreamHandler(stream)
                # Catch both callback and writer output so a second emitter cannot pass unnoticed.
                with self.assertLogs("hyper_parallel.trainer", level="INFO") if console else self.assertNoLogs(
                    "hyper_parallel.trainer", level="INFO"
                ) as captured:
                    if console:
                        captured_handler = logging.getLogger("hyper_parallel.trainer")
                        captured_handler.addHandler(handler)
                    try:
                        callback.on_step_end(state, loss=2.5, loss_dict={}, grad_norm=1.0)
                        callback.on_step_end(state, loss=2.5, loss_dict={}, grad_norm=1.0)
                    finally:
                        if console:
                            captured_handler.removeHandler(handler)
                if console:
                    self.assertEqual(len(captured.records), 1)
                    line = stream.getvalue()
                    self.assertEqual(line.count("\n"), 1)
                    self.assertNotIn("\r", line)
                    self.assertIn("Training:  15%|", line)
                    self.assertIn("3/20 | step=3 epoch=0", line)
                    self.assertEqual(line.count("training/total_loss=2.5"), 1)
                    self.assertNotIn("loss=", line.split(" | step=", 1)[0])
                trainer.monitor.close()
                if active:
                    records = [json.loads(line) for line in (
                        Path(directory) / trainer.monitor.run_id / "rank_0.jsonl"
                    ).read_text().splitlines()]
                    steps = [record for record in records if record["event"] == "train_step"]
                    self.assertEqual(len(steps), 1)
                    self.assertEqual(steps[0]["step"], 3)
                    self.assertEqual(steps[0]["metrics"], {"training/total_loss": 2.5})
                    self.assertNotIn("Training:", json.dumps(steps))

    def test_missing_tqdm_preserves_scalar_logging(self) -> None:
        """Feature: Optional tqdm. Description: Missing formatter. Expectation: Scalars still print."""
        trainer = self._trainer()
        trainer.train_iters = 20
        callback = LoggingCallback(trainer)
        with (
            patch("hyper_parallel.trainer.callbacks.logging_callback.tqdm", None),
            patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log,
        ):
            callback.on_step_end(TrainerState(global_step=2), loss=2.5, loss_dict={}, grad_norm=1.0)
        log.assert_called_once()
        self.assertTrue(log.call_args.args[1].startswith("step=2 epoch=0 "))
