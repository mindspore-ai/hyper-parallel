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
"""Unit tests for the Trainer tqdm callback."""

import sys
import unittest
from io import StringIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from hyper_parallel.trainer.callbacks import TqdmCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


class _FakeTqdm:
    """Format deterministic progress snapshots without terminal output."""

    @staticmethod
    def format_meter(**kwargs: Any) -> str:
        """Return a compact representation of the supplied progress state."""
        return f"{kwargs['prefix']}: {kwargs['n']}/{kwargs['total']} [{kwargs['postfix']}]"


class TestTqdmCallback(unittest.TestCase):
    """Verify interactive and redirected progress output behavior."""

    @staticmethod
    def _build_trainer() -> SimpleNamespace:
        """Build the callback's minimal Trainer dependency surface."""
        return SimpleNamespace(
            global_rank=0,
            mesh=None,
            train_iters=5,
            step_train_metrics={"training/total_loss": 2.5},
            step_env_metrics={"performance/tokens_per_second": 128.1254},
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_redirected_stream_writes_one_snapshot_per_step(self) -> None:
        """Emit one newline and no carriage returns when stderr is not a TTY.

        Feature: Redirected Trainer progress output.
        Description: Complete one step with a structured log message on a non-TTY stream.
        Expectation: Exactly one finalized newline-terminated progress snapshot is written.
        """
        stream = StringIO()
        callback = TqdmCallback(self._build_trainer())
        state = TrainerState(global_step=0, epoch=0)

        with patch.object(sys, "stderr", stream), patch.dict(
            sys.modules,
            {"tqdm": SimpleNamespace(tqdm=_FakeTqdm)},
        ):
            callback.on_train_begin(state)
            self.assertEqual(
                stream.getvalue(),
                "",
                f"progress bar initialization must not emit output, got={stream.getvalue()!r}",
            )

            message_handled = callback.write("step=1 epoch=0 training/total_loss=2.5")
            self.assertTrue(message_handled, f"active progress bar must handle messages, got={message_handled}")
            state.global_step = 1
            callback.on_step_end(state)
            callback.on_train_end(state)

        output = stream.getvalue()
        newline_count = output.count("\n")
        self.assertEqual(newline_count, 1, f"expected one progress snapshot, got={newline_count}, output={output!r}")
        self.assertNotIn("\r", output, f"redirected progress must not contain carriage returns, got={output!r}")
        self.assertIn("Training: 1/5", output, f"progress position missing from output, got={output!r}")
        self.assertIn("loss=2.5", output, f"progress metrics missing from output, got={output!r}")
        self.assertIn("step=1 epoch=0", output, f"structured log message missing from output, got={output!r}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_optional_tflops_metric_reaches_the_postfix(self) -> None:
        """Show the per-device TFLOP/s metric beside the existing throughput fields.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Publish a step whose environment metrics carry ``performance/tflops_per_device``.
        Expectation: The progress postfix renders the value with three decimals.
        """
        stream = StringIO()
        trainer = self._build_trainer()
        trainer.step_env_metrics = {
            "performance/tokens_per_second": 65536.0,
            "performance/step_time": 0.5,
            "performance/tflops_per_device": 23.5584,
        }
        callback = TqdmCallback(trainer)
        state = TrainerState(global_step=0, epoch=0)

        with patch.object(sys, "stderr", stream), patch.dict(
            sys.modules,
            {"tqdm": SimpleNamespace(tqdm=_FakeTqdm)},
        ):
            callback.on_train_begin(state)
            state.global_step = 1
            callback.on_step_end(state)
            callback.on_train_end(state)

        output = stream.getvalue()
        self.assertIn("TFLOP/s=23.558", output, f"tflops metric missing from postfix, got={output!r}")
        self.assertIn("tokens/s=65536.000", output, f"throughput metric missing from postfix, got={output!r}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_absent_tflops_metric_is_not_rendered(self) -> None:
        """Leave the postfix unchanged when the optional metric is disabled.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Publish a step whose environment metrics carry no FLOPs entry.
        Expectation: The progress postfix shows no TFLOP/s label.
        """
        stream = StringIO()
        callback = TqdmCallback(self._build_trainer())
        state = TrainerState(global_step=0, epoch=0)

        with patch.object(sys, "stderr", stream), patch.dict(
            sys.modules,
            {"tqdm": SimpleNamespace(tqdm=_FakeTqdm)},
        ):
            callback.on_train_begin(state)
            state.global_step = 1
            callback.on_step_end(state)
            callback.on_train_end(state)

        output = stream.getvalue()
        self.assertNotIn("TFLOP/s", output, f"disabled metric must not render, got={output!r}")
        self.assertIn("tokens/s=128.125", output, f"existing metric missing from postfix, got={output!r}")


if __name__ == "__main__":
    unittest.main()
