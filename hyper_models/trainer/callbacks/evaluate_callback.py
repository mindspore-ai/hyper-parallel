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
"""Scheduled placeholder for in-training evaluation."""

import logging
from typing import Any, Literal

from .base import Callback, TrainerState


logger = logging.getLogger(__name__)


class EvaluateCallback(Callback):
    """Expose step and epoch evaluation hooks without running validation yet."""

    def __init__(self, trainer: Any) -> None:
        """Read evaluation intervals from the Trainer configuration.

        Args:
            trainer: Trainer that owns the callback lifecycle.
        """
        super().__init__(trainer)
        training_config = trainer.config.training
        self.eval_steps = training_config.eval_steps
        self.eval_epochs = training_config.eval_epochs
        self._last_evaluated_step: int | None = None

    def _evaluate(self, state: TrainerState, trigger: Literal["step", "epoch"]) -> None:
        """Report that the scheduled validation loop is still a placeholder."""
        if getattr(self.trainer, "global_rank", 0) == 0:
            logger.warning(
                "Evaluation triggered at step %s by %s, but the validation loop is not implemented yet.",
                state.global_step,
                trigger,
            )

    def _trigger_evaluation(self, state: TrainerState, trigger: Literal["step", "epoch"]) -> None:
        """Invoke the extension point at most once for each global step."""
        if self._last_evaluated_step == state.global_step:
            return
        self._evaluate(state, trigger)
        self._last_evaluated_step = state.global_step

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Trigger the placeholder at the configured optimizer-step cadence."""
        del kwargs
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
            self._trigger_evaluation(state, trigger="step")

    def on_epoch_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Trigger the placeholder at the configured completed-epoch cadence."""
        del kwargs
        if self.eval_epochs > 0 and (state.epoch + 1) % self.eval_epochs == 0:
            self._trigger_evaluation(state, trigger="epoch")
