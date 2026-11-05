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
        self._router_aux_sum: Any = None
        self._router_aux_count = 0
        self._micro_step_outputs_seen = False

    def on_step_begin(
            self, state: TrainerState, micro_batches: list[dict[str, Any]] | None = None, **kwargs: Any,
    ) -> None:
        """Reset per-step metric state, including steps that do not emit a log."""
        del state, micro_batches, kwargs
        self._router_aux_sum = None
        self._router_aux_count = 0
        self._micro_step_outputs_seen = False

    def on_micro_step_end(self, state: TrainerState, loss_dict: dict[str, Any], **kwargs: Any) -> None:
        """Accumulate detached router outputs without synchronizing device scalars."""
        del state, kwargs
        if self.logging_steps <= 0 or getattr(self.trainer, "global_rank", 0) != 0:
            return
        self._micro_step_outputs_seen = True
        aux_loss = loss_dict.get("router_aux_loss") if loss_dict is not None else None
        if aux_loss is not None:
            value = aux_loss.detach()
            if self._router_aux_sum is None:
                # Graph executors may reuse output storage on the next microbatch.
                self._router_aux_sum = value.clone()
            else:
                self._router_aux_sum += value
            self._router_aux_count += 1

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
        moe_monitor = getattr(self.trainer, "moe_monitor", None)
        if self._micro_step_outputs_seen:
            aux_loss = (self._router_aux_sum / self._router_aux_count
                        if self._router_aux_count else None)
        else:
            # Keep direct step-end callers compatible when no micro-step outputs were observed.
            aux_loss = loss_dict.get("router_aux_loss") if loss_dict is not None else None
            if aux_loss is None and moe_monitor is not None:
                aux_loss = moe_monitor.last_mean_aux_loss
        if aux_loss is not None:
            metrics["training/router_aux_loss"] = float(aux_loss)
        if not metrics:
            return
        self._write(self._format_message(state, metrics))
        self._last_logged_step = state.global_step
