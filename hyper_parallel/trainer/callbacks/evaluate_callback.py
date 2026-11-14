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
"""Scheduled validation with isolated training state."""

from __future__ import annotations

import logging
from typing import Any, Literal

from hyper_parallel.trainer.runtime.evaluation import validate_evaluation_config

from .base import Callback, TrainerState


logger = logging.getLogger(__name__)


class EvaluateCallback(Callback):
    """Run bounded validation at configured step or epoch boundaries."""

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
        self.last_metrics: dict[str, float] | None = None

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Reject an incomplete evaluation recipe before the training loop starts.

        Args:
            state: Current training progress at the callback boundary.
            **kwargs: Additional callback context.
        """
        del state, kwargs
        if self.eval_steps > 0 or self.eval_epochs > 0:
            validate_evaluation_config(self.trainer)

    def _evaluate(self, state: TrainerState, trigger: Literal["step", "epoch"]) -> None:
        """Run validation and publish global metrics without advancing training state."""
        self.last_metrics = self.trainer.evaluate()
        if getattr(self.trainer, "global_rank", 0) == 0:
            logger.info("Evaluation at step %s (%s): %s", state.global_step, trigger, self.last_metrics)

    def _trigger_evaluation(self, state: TrainerState, trigger: Literal["step", "epoch"]) -> None:
        """Invoke the extension point at most once for each global step."""
        if self._last_evaluated_step == state.global_step:
            return
        self._evaluate(state, trigger)
        self._last_evaluated_step = state.global_step

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Run validation at the configured optimizer-step cadence.

        Args:
            state: Current training progress at the callback boundary.
            **kwargs: Additional callback context.
        """
        if not kwargs.get("optimizer_step_applied", True):
            return
        cadence_step = state.optimizer_step if "optimizer_step_applied" in kwargs else state.global_step
        if self.eval_steps > 0 and cadence_step % self.eval_steps == 0:
            self._trigger_evaluation(state, trigger="step")

    def on_epoch_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Run validation at the configured completed-epoch cadence.

        Args:
            state: Current training progress at the callback boundary.
            **kwargs: Additional callback context.
        """
        del kwargs
        if self.eval_epochs > 0 and (state.epoch + 1) % self.eval_epochs == 0:
            self._trigger_evaluation(state, trigger="epoch")
