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
"""Terminal progress callback for Trainer."""

import logging
from typing import Any

from .base import Callback, TrainerState


logger = logging.getLogger(__name__)


class TqdmCallback(Callback):
    """Display global training progress and shared metrics on global rank zero."""

    _PRIMARY_METRICS = (
        ("training/total_loss", "loss"),
        ("training/grad_norm", "grad_norm"),
        ("training/lr", "lr"),
    )
    _ENV_METRICS = (
        ("performance/tokens_per_second", "tokens/s"),
        ("performance/step_time", "step_time"),
    )

    def __init__(self, trainer: Any) -> None:
        """Initialize progress state without producing terminal output.

        Args:
            trainer: Trainer that owns the callback lifecycle.
        """
        super().__init__(trainer)
        self._progress_bar = None
        self._last_updated_step = 0
        self._missing_dependency_warned = False

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Convert a scalar or scalar tensor-like value to a float."""
        if value is None:
            return None
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        return float(value)

    @classmethod
    def _format_metric(cls, name: str, value: Any) -> str:
        """Format a shared metric using a unit-appropriate representation."""
        scalar = cls._to_float(value)
        if scalar is None:
            return "n/a"
        if name == "training/lr":
            return f"{scalar:.3e}"
        if name == "performance/tokens_per_second":
            return f"{scalar:.3f}"
        if name == "performance/step_time":
            return f"{scalar:.3f}s"
        return f"{scalar:.6g}"

    def _create_progress_bar(self, total: int, initial: int) -> Any:
        """Create tqdm lazily because it is an optional dependency."""
        try:
            from tqdm import tqdm  # pylint: disable=C0415
        except ImportError:
            if not self._missing_dependency_warned:
                logger.warning("TqdmCallback: 'tqdm' is not installed; progress bar disabled")
                self._missing_dependency_warned = True
            return None
        return tqdm(
            total=total,
            initial=initial,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

    def _postfix(self) -> dict[str, str]:
        """Build the progress postfix exclusively from shared metric dictionaries."""
        train_metrics = getattr(self.trainer, "step_train_metrics", {})
        env_metrics = getattr(self.trainer, "step_env_metrics", {})
        postfix = {
            label: self._format_metric(name, train_metrics[name])
            for name, label in self._PRIMARY_METRICS
            if name in train_metrics
        }

        primary_names = {name for name, _ in self._PRIMARY_METRICS}
        for name, value in sorted(train_metrics.items()):
            if name in primary_names:
                continue
            label = name.split("/", 1)[-1]
            postfix[label] = self._format_metric(name, value)

        for name, label in self._ENV_METRICS:
            if name in env_metrics:
                postfix[label] = self._format_metric(name, env_metrics[name])
        return postfix

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Create a single full-training progress bar on global rank zero."""
        del kwargs
        self._last_updated_step = state.global_step
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        self._progress_bar = self._create_progress_bar(
            total=self.trainer.train_steps,
            initial=state.global_step,
        )

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Publish shared metrics and advance once per completed global step."""
        del kwargs
        if self._progress_bar is None or state.global_step <= self._last_updated_step:
            return

        postfix = self._postfix()
        if postfix:
            self._progress_bar.set_postfix(postfix)
        self._progress_bar.update(state.global_step - self._last_updated_step)
        self._last_updated_step = state.global_step

    def write(self, message: str) -> bool:
        """Write above the active progress bar and redraw it.

        Args:
            message: Complete terminal line to display.

        Returns:
            Whether an active progress bar handled the message.
        """
        if self._progress_bar is None:
            return False
        self._progress_bar.write(message, file=getattr(self._progress_bar, "fp", None))
        return True

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Close the global progress bar idempotently."""
        del state, kwargs
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
