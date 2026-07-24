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
"""Callback system — StepState, TrainingCallback, CallbackManager, built-in callbacks.

Following design doc 03_training_loop.md §4.2.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from hyper_models.components.distributed.infrastructure import _is_rank_0

if TYPE_CHECKING:
    from hyper_models.trainer.config import TrainerConfig

logger = logging.getLogger(__name__)


# ── 4.2.1 StepState ──

@dataclass(frozen=True)
class StepState:
    """Per-step state snapshot passed to callbacks.

    Frozen dataclass — receivers are read-only.
    All timing flags are computed by StepScheduler and passed through.
    """
    step: int
    epoch: int
    is_final_step: bool

    # Timing flags (from StepScheduler)
    is_ckpt_step: bool
    is_val_step: bool
    is_log_step: bool
    is_gc_step: bool
    sigterm_received: bool

    # Training metrics
    loss: float
    grad_norm: Optional[float]
    lr: float
    tps: float
    mfu: float
    num_tokens: int


# ── 4.2.2 TrainingCallback ──

class TrainingCallback:
    """Training lifecycle callback.

    Only 3 callback points — never exposes training internals.
    """

    def on_step_end(self, state: StepState) -> None:
        """Called after each optimizer step."""
        pass

    def on_train_begin(self) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self) -> None:
        """Called at the end of training."""
        pass


# ── 4.2.3 CallbackManager ──

class CallbackManager:
    """Manages all registered callbacks, invoking in registration order."""

    def __init__(self):
        self._callbacks: list[TrainingCallback] = []

    def register(self, callback: TrainingCallback) -> None:
        self._callbacks.append(callback)

    def on_step_end(self, state: StepState) -> None:
        for cb in self._callbacks:
            cb.on_step_end(state)

    def on_train_begin(self) -> None:
        for cb in self._callbacks:
            cb.on_train_begin()

    def on_train_end(self) -> None:
        for cb in self._callbacks:
            cb.on_train_end()


# ── 4.2.4 Built-in Callbacks ──

class CheckpointCallback(TrainingCallback):
    """Save checkpoint on is_ckpt_step (periodic saves).

    Skip final step — handled by the training loop's explicit final save.
    """

    def __init__(self, recipe: Any):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_ckpt_step or state.is_final_step:
            return
        self.recipe.save_checkpoint(
            self.recipe.cfg.checkpoint.checkpoint_dir,
            state.epoch, state.step, state.loss,
            val_losses=getattr(self.recipe, "_last_val_losses", None),
        )
        self.recipe.step_scheduler.mark_epoch_ckpt_saved()


class EvaluateCallback(TrainingCallback):
    """Run validation on is_val_step."""

    def __init__(self, recipe: Any):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_val_step or not getattr(self.recipe, "val_dataloaders", None):
            return
        val_losses = {}
        for name, dl in self.recipe.val_dataloaders.items():
            val_losses[name] = self.recipe._run_validation_epoch(dl)
        self.recipe._last_val_losses = val_losses
        self.recipe.log_val_metrics(val_losses)


class LoggingCallback(TrainingCallback):
    """Log training metrics on is_log_step."""

    def __init__(self, recipe: Any):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_log_step:
            return
        logger.info(
            "step=%d loss=%.4f lr=%.2e grad_norm=%.4f tps=%.0f mfu=%.2f%%",
            state.step, state.loss, state.lr,
            state.grad_norm or 0.0, state.tps, state.mfu * 100,
        )


class TqdmCallback(TrainingCallback):
    """Update tqdm progress bar on each step (rank 0 only)."""

    def __init__(self, recipe: Any, total: Optional[int] = None):
        self.recipe = recipe
        self.total = total
        self.pbar = None

    def on_train_begin(self) -> None:
        if not _is_rank_0():
            return
        from tqdm import tqdm
        initial = getattr(self.recipe, "step_scheduler", None)
        initial_step = initial.step if initial is not None else 0
        self.pbar = tqdm(
            total=self.total, initial=initial_step,
            desc="Training", unit="step", dynamic_ncols=True,
        )

    def on_step_end(self, state: StepState) -> None:
        if self.pbar is None:
            return
        self.pbar.set_postfix(loss=f"{state.loss:.4f}", lr=f"{state.lr:.2e}")
        self.pbar.update(1)

    def on_train_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class WandbCallback(TrainingCallback):
    """Log metrics to WandB on is_log_step."""

    def __init__(self, recipe: Any, project: str = ""):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_log_step:
            return
        import wandb
        wandb.log({
            "loss": state.loss, "lr": state.lr,
            "grad_norm": state.grad_norm,
            "tps": state.tps, "mfu": state.mfu,
            "step": state.step,
        })


class GCCallback(TrainingCallback):
    """Trigger garbage collection on is_gc_step."""

    def __init__(self, recipe: Any):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.is_gc_step:
            return
        self.recipe._maybe_collect_garbage()


class SIGTERMHandler(TrainingCallback):
    """Graceful exit on SIGTERM.

    Does NOT save checkpoint here — final save is handled by the training loop.
    """

    def __init__(self, recipe: Any):
        self.recipe = recipe

    def on_step_end(self, state: StepState) -> None:
        if not state.sigterm_received:
            return
        logger.warning("SIGTERM received at step %d, exiting gracefully", state.step)
        self.recipe.step_scheduler.cleanup()
        self.recipe.step_scheduler.max_steps = state.step


# ── 4.2.5 Build factory ──

def build_callback_manager(
    recipe: Any,
    cfg: "TrainerConfig",
    pbar_total: Optional[int] = None,
) -> CallbackManager:
    """Build default CallbackManager with all built-in callbacks."""
    manager = CallbackManager()
    manager.register(CheckpointCallback(recipe))
    manager.register(EvaluateCallback(recipe))
    manager.register(LoggingCallback(recipe))
    manager.register(TqdmCallback(recipe, total=pbar_total))
    if getattr(cfg, "wandb", None) and getattr(cfg.wandb, "enabled", False):
        manager.register(WandbCallback(recipe, project=cfg.wandb.project))
    step_scheduler_cfg = getattr(cfg, "step_scheduler", None)
    if step_scheduler_cfg and getattr(step_scheduler_cfg, "gc_every_steps", None):
        manager.register(GCCallback(recipe))
    manager.register(SIGTERMHandler(recipe))
    return manager