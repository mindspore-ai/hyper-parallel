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

from __future__ import annotations

from typing import Any

from hyper_parallel.trainer.runtime.logging import create_logger

from .base import Callback, TrainerState


logger = create_logger(__name__)


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
        self._last_collected_step: int | None = None

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a scalar metric compactly for one terminal line."""
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        try:
            return f"{float(value):.9g}"
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

    def _collect_provider_metrics(self, state: TrainerState) -> None:
        """Collect optional model/optimizer scalar metrics on every rank once per step.

        Providers own aggregation and distributed semantics, may use collectives,
        and must return detached, namespaced scalars. Metrics are observation only
        and never enter Trainer's loss dictionary or backward computation.
        """
        if self._last_collected_step == state.global_step:
            return
        optimizers = getattr(self.trainer, "optimizer", None)
        optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
        metrics = dict(getattr(self.trainer, "step_env_metrics", {}))
        additions = {}
        for provider in [getattr(self.trainer, "model", None), *optimizers]:
            collect = getattr(provider, "get_logging_metrics", None)
            if not callable(collect):
                continue
            supplied = collect()
            duplicates = (metrics.keys() | additions.keys()) & supplied.keys()
            if duplicates:
                raise ValueError(f"Logging metric names collide: {sorted(duplicates)}")
            additions.update(supplied)
        metrics.update(additions)
        self.trainer.step_env_metrics = metrics
        self.trainer.step_train_metrics = {**getattr(self.trainer, "step_train_metrics", {}), **additions}
        self._last_collected_step = state.global_step

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Log all shared environment metrics at the configured cadence.

        Args:
            state: Current optimizer step and epoch.
            **kwargs: Other callback observations, excluded from loss computation.
        """
        del kwargs
        self._collect_provider_metrics(state)
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
