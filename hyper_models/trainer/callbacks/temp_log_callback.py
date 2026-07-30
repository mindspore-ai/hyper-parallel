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
"""Temporary tqdm progress callback for the Trainer demo."""

import logging
from typing import Any, Dict

from .base import Callback, TrainerState

logger = logging.getLogger(__name__)


class TempLogCallback(Callback):
    """Display per-step training progress and metrics on global rank zero."""

    def __init__(self, trainer: Any) -> None:
        """Initialize the callback without creating terminal output yet."""
        super().__init__(trainer)
        self._progress_bar = None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Convert a scalar value or tensor-like object to a Python float."""
        if value is None:
            return None
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        return float(value)

    @classmethod
    def _format_metric(cls, value: Any) -> str:
        """Format an optional scalar metric for one-line logging."""
        scalar = cls._to_float(value)
        return "n/a" if scalar is None else f"{scalar:.6g}"

    def _current_lr(self) -> float | None:
        """Read the first optimizer parameter group's current learning rate."""
        optimizer = self.trainer.optimizer
        optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        if not optimizers or not optimizers[0].param_groups:
            return None
        return float(optimizers[0].param_groups[0]["lr"])

    @staticmethod
    def _create_progress_bar(total: int, initial: int) -> Any:
        """Create tqdm lazily because it is an optional dependency."""
        try:
            from tqdm import tqdm  # pylint: disable=C0415
        except ImportError:
            logger.warning("TempLogCallback: 'tqdm' is not installed; progress bar disabled")
            return None
        return tqdm(
            total=total,
            initial=initial,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Create the progress bar when training starts."""
        del kwargs
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        self._progress_bar = self._create_progress_bar(
            total=self.trainer.train_steps,
            initial=state.global_step,
        )

    def on_step_end(
        self,
        state: TrainerState,
        loss: float,
        loss_dict: Dict[str, float],
        grad_norm: float,
        **kwargs: Any,
    ) -> None:
        """Update loss, gradient norm, learning rate, and named losses."""
        del kwargs, state
        if self._progress_bar is None:
            return

        postfix = {
            "loss": self._format_metric(loss),
            "grad_norm": self._format_metric(grad_norm),
            "lr": self._format_metric(self._current_lr()),
        }
        postfix.update({
            name: self._format_metric(value)
            for name, value in sorted((loss_dict or {}).items())
        })
        self._progress_bar.set_postfix(postfix)
        self._progress_bar.update(1)

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Close the progress bar after training."""
        del state, kwargs
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
