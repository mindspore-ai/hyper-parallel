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
"""Validate observational metrics independently of backward loss and log cadence."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


class TestLoggingMetrics(unittest.TestCase):
    """Provider collection must remain collective-safe on non-printing ranks."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_all_ranks_collect_once_even_without_output(self):
        """Feature: Scalar logging.

        Description: Skip printing by cadence, rank, or configuration.
        Expectation: Every provider is still consumed exactly once for each step.
        """
        for rank, cadence in [(0, 2), (1, 1), (0, 0)]:
            model = SimpleNamespace(get_logging_metrics=Mock(return_value={"training/lm_loss": 2.0}))
            optimizer = SimpleNamespace(get_logging_metrics=Mock(return_value={"optimizer/max": 4.0}))
            trainer = SimpleNamespace(mesh=None, global_rank=rank, model=model, optimizer=optimizer,
                                      config=SimpleNamespace(training=SimpleNamespace(logging_steps=cadence)),
                                      step_env_metrics={"training/loss": 3.0}, step_train_metrics={})
            callback = LoggingCallback(trainer)
            callback._write = Mock()
            backward_losses = {"loss": 3.0}
            callback.on_step_end(TrainerState(global_step=1), loss_dict=backward_losses)
            callback.on_step_end(TrainerState(global_step=1), loss_dict=backward_losses)
            model.get_logging_metrics.assert_called_once()
            optimizer.get_logging_metrics.assert_called_once()
            callback._write.assert_not_called()
            self.assertEqual(backward_losses, {"loss": 3.0})
            self.assertEqual(trainer.step_train_metrics["training/lm_loss"], 2.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rank_zero_prints_full_precision_metrics(self):
        """Feature: Terminal metrics.

        Description: Run the normal rank-zero callback with an optional provider.
        Expectation: The output includes new fields and nine significant digits.
        """
        trainer = SimpleNamespace(mesh=None, global_rank=0,
                                  model=SimpleNamespace(get_logging_metrics=lambda: {"training/mtp_loss": 0.123456789}),
                                  config=SimpleNamespace(training=SimpleNamespace(logging_steps=1)),
                                  step_env_metrics={"training/loss": 2.0}, step_train_metrics={})
        callback = LoggingCallback(trainer)
        callback._write = Mock()
        callback.on_step_end(TrainerState(global_step=1))
        self.assertIn("training/mtp_loss=0.123456789", callback._write.call_args.args[0])
