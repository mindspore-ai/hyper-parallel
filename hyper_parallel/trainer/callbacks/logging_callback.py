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

import logging
from typing import Any

# Progress formatting is optional; scalar logging works without tqdm.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .base import Callback, TrainerState


logger = logging.getLogger(__name__)


class LoggingCallback(Callback):
    """Print one progress-and-metrics line on global rank zero; persist structured metrics separately."""

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

    def _format_message(self, state: TrainerState, metrics: dict[str, Any]) -> str:
        """Build a deterministic, complete metric line for one optimizer step."""
        fields = [f"step={state.global_step}", f"epoch={state.epoch}"]
        fields.extend(
            f"{name}={self._format_value(value)}"
            for name, value in sorted(metrics.items())
        )
        message = " ".join(fields)
        total = getattr(self.trainer, "train_iters", None)
        if tqdm is not None and total is not None and total > 0:
            progress = tqdm.format_meter(
                n=state.global_step, total=total, elapsed=0, prefix="Training",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt}",
                ascii=True,
            )
            message = f"{progress} | {message}"
        return message

    def on_step_end(
        self,
        state: TrainerState,
        loss: float,
        loss_dict: dict[str, float],
        grad_norm: float,
        **kwargs: Any,
    ) -> None:
        """Log all shared environment metrics at the configured cadence."""
        del loss, grad_norm, kwargs
        if self.logging_steps <= 0:
            return
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        if state.global_step % self.logging_steps != 0:
            return
        if self._last_logged_step == state.global_step:
            return

        metrics = dict(getattr(self.trainer, "step_env_metrics", {}))
        if "training/router_aux_loss" not in metrics:
            # Without the meter, callers supply their own step aggregation.
            # The logger does not normalize local values across micro-batches or ranks.
            aux_loss = loss_dict.get("router_aux_loss") if loss_dict is not None else None
            if aux_loss is None:
                moe_monitor = getattr(self.trainer, "moe_monitor", None)
                if moe_monitor is not None:
                    aux_loss = moe_monitor.last_mean_aux_loss
            if aux_loss is not None:
                metrics["training/router_aux_loss"] = float(aux_loss)
        if not metrics:
            return
        monitor = getattr(self.trainer, "monitor", None)
        if monitor is None or not monitor.enabled or "console" in monitor.config.sinks:
            logger.info("%s", self._format_message(state, metrics))
        if monitor is not None and monitor.enabled:
            monitor.log_step(state, metrics)
        self._last_logged_step = state.global_step
