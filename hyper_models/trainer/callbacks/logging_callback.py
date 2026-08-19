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
"""Per-step structured terminal logging callback."""

import logging
from typing import Any

from .base import Callback, TrainerState


logger = logging.getLogger(__name__)


class LoggingCallback(Callback):
    """Log shared step metrics on global rank zero without breaking tqdm."""

    def __init__(self, trainer: Any) -> None:
        """Initialize the configured logging cadence.

        Args:
            trainer: Trainer that owns the callback lifecycle.
        """
        super().__init__(trainer)
        self.logging_steps = trainer.config.training.logging_steps
        self._last_logged_step: int | None = None

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a scalar metric compactly for one terminal line."""
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _format_message(cls, state: TrainerState, metrics: dict[str, Any]) -> str:
        """Build a deterministic, complete metric line for one optimizer step."""
        fields = [f"step={state.global_step}", f"epoch={state.epoch}"]
        fields.extend(
            f"{name}={cls._format_value(value)}"
            for name, value in sorted(metrics.items())
        )
        return " ".join(fields)

    def _write(self, message: str) -> None:
        """Write through the active tqdm bar or fall back to standard logging."""
        tqdm_callback = getattr(self.trainer, "tqdm_callback", None)
        tqdm_write = getattr(tqdm_callback, "write", None)
        if callable(tqdm_write) and tqdm_write(message):
            return
        logger.info("%s", message)

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Log all shared environment metrics at the configured cadence."""
        del kwargs
        if (
            self.logging_steps <= 0
            or getattr(self.trainer, "global_rank", 0) != 0
            or state.global_step % self.logging_steps != 0
            or self._last_logged_step == state.global_step
        ):
            return

        metrics = getattr(self.trainer, "step_env_metrics", {})
        if not metrics:
            return
        self._write(self._format_message(state, metrics))
        self._last_logged_step = state.global_step
