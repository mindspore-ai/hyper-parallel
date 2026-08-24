# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CheckpointerCallback --- save/restore policy on top of a Checkpointer."""

import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from hyper_models.components.checkpoint import build_checkpointer
from hyper_models.components.checkpoint.dcp_checkpointer import (
    STEP_PREFIX,
    initialize_optimizer_state,
)
from hyper_models.components.utils import helper
from hyper_models.components.utils.device import (
    get_device_rng_state,
    set_device_rng_state,
)

from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer


logger = helper.create_logger(__name__)


def _as_list(value: Any) -> List[Any]:
    """Normalize an optional single-or-list component into a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _unwrap_single(values: List[Any]) -> Any:
    """Collapse a one-element list so single-component runs keep a flat state."""
    if len(values) == 1:
        return values[0]
    return values


class CheckpointerCallback(Callback):
    """Decide when to checkpoint and what goes in it.

    This callback owns *policy* --- the save cadence, duplicate suppression, and
    the mapping between trainer objects and the persisted payload. Writing that
    payload to disk and reading it back belongs to the
    :class:`~hyper_models.components.checkpoint.CheckpointerBase` it delegates
    to, so the storage format can change without touching this file.

    The payload is the model and optimizer state dicts plus an ``extra_state``
    bundle holding what those cannot represent: ``global_step`` / ``epoch``, the
    LR scheduler, the dataloader position, and the CPU / device / Python RNG
    states.

    Saving and restoring are independent: ``save_ckpt`` gates the write path,
    ``restore_from`` the read path. Turning saving off while pointing at a
    checkpoint is therefore a supported combination --- start from these weights
    and write nothing further.

    Restore runs in :meth:`on_train_begin`, i.e. after the model, optimizer,
    scheduler and dataloader exist but before the first training step.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        """Read the checkpoint configuration and build the checkpointer."""
        super().__init__(trainer)
        ckpt_cfg = trainer.config.checkpoint
        # ``save_ckpt`` gates the write path only --- this callback is also
        # registered for restore-only runs. Folding it into the cadence fields
        # makes "saving is off" structural (there is simply no cadence) instead
        # of a second enable check inside every save hook.
        self._save_ckpt = ckpt_cfg.save_ckpt
        self._checkpoint_dir = ckpt_cfg.checkpoint_dir
        self._save_steps = ckpt_cfg.save_steps if self._save_ckpt else 0
        self._save_epochs = ckpt_cfg.save_epochs if self._save_ckpt else 0
        self._is_async = ckpt_cfg.is_async
        self._is_peft = ckpt_cfg.is_peft
        self._save_optimizer = ckpt_cfg.save_optimizer
        self._save_train_state = ckpt_cfg.save_train_state
        self._save_extra_state_per_rank = ckpt_cfg.save_extra_state_per_rank

        self._restore_from = ckpt_cfg.restore_from
        self._restore_optimizer = ckpt_cfg.restore_optimizer
        self._restore_train_state = ckpt_cfg.restore_train_state

        self._last_saved_step: int = -1
        self.checkpointer = build_checkpointer(
            extra_state_per_rank=self._save_extra_state_per_rank,
        )

    # ------------------------------------------------------------------
    # Hook dispatchers
    # ------------------------------------------------------------------

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Log the checkpoint configuration and restore any requested state."""
        logger.info(
            "Checkpoint configuration: "
            "checkpoint_dir=%s, save_ckpt=%s, save_steps=%s, save_epochs=%s, "
            "is_async=%s, is_peft=%s, "
            "save_extra_state_per_rank=%s, restore_from=%s",
            self._checkpoint_dir,
            self._save_ckpt,
            self._save_steps,
            self._save_epochs,
            self._is_async,
            self._is_peft,
            self._save_extra_state_per_rank,
            self._restore_from,
        )
        self._load_checkpoint()

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Save on the configured step cadence."""
        if self._save_steps > 0 and state.global_step % self._save_steps == 0:
            if state.global_step == self._last_saved_step:
                return
            self._save_checkpoint(state)

    def on_epoch_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Save on the configured epoch cadence."""
        if self._save_epochs > 0 and (state.epoch + 1) % self._save_epochs == 0:
            if state.global_step != self._last_saved_step:
                self._save_checkpoint(state)
            else:
                logger.info(
                    "Skipping duplicate checkpoint save at epoch_end "
                    "(global_step %s already saved at step_end).",
                    state.global_step,
                )

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Persist the final step, then drain any in-flight async save.

        Always saved when saving is on and the step is not already on disk:
        losing the last stretch of training to a cadence that happened not to
        land on the final step is never what anyone wants.
        """
        if (
            self._save_ckpt
            and state.global_step > 0
            and state.global_step != self._last_saved_step
        ):
            # The process is about to exit, so the last checkpoint is written
            # synchronously regardless of ``is_async``.
            self._save_checkpoint(state, force_sync=True)
        self.wait_for_pending_save()

    def wait_for_pending_save(self) -> None:
        """Block until the checkpointer's in-flight async save is persisted."""
        self.checkpointer.maybe_wait_for_async_save()

    # ------------------------------------------------------------------
    # Payload assembly
    # ------------------------------------------------------------------

    def _model_state_dict(self) -> Dict[str, Any]:
        """Return the model state to persist, trainable-only under PEFT."""
        model = self.trainer.model
        state_dict = model.state_dict()
        if not self._is_peft:
            return state_dict

        trainable = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        return {name: value for name, value in state_dict.items() if name in trainable}

    def _collect_extra_state(self, state: TrainerState) -> Dict[str, Any]:
        """Build the extra_state bundle (progress / scheduler / dataloader / RNG)."""
        # Prefer the iterator snapshot: with background prefetching the loader has
        # already advanced past the batch the training step actually consumed.
        dataloader_state: Dict[str, Any] = {}
        data_iterator = getattr(self.trainer, "data_iterator", None)
        if data_iterator is not None and hasattr(data_iterator, "state_dict"):
            dataloader_state = data_iterator.state_dict()
        elif self.trainer.train_dataloader is not None and hasattr(
            self.trainer.train_dataloader, "state_dict"
        ):
            dataloader_state = self.trainer.train_dataloader.state_dict()

        schedulers = _as_list(self.trainer.lr_scheduler)
        lr_scheduler_sd = _unwrap_single([sch.state_dict() for sch in schedulers])

        return {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "lr_scheduler": lr_scheduler_sd,
            "train_dataloader": dataloader_state,
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
                "torch_device": get_device_rng_state(),
                "python": random.getstate(),
            },
        }

    def _steps_per_epoch(self) -> int:
        """Return the optimizer steps one epoch contains.

        ``trainer.train_steps`` is the *run total* (``steps_per_epoch *
        num_train_epochs``), so mapping a restored ``global_step`` back onto an
        ``(epoch, step)`` position needs the per-epoch count, which is the
        dataloader's length. An unsized (streaming) loader has no epoch boundary
        of its own, so the run total stands in for it.
        """
        try:
            steps_per_epoch = len(self.trainer.train_dataloader)
        except TypeError:
            steps_per_epoch = 0
        return max(steps_per_epoch or int(self.trainer.train_steps or 0), 1)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_checkpoint(self, state: TrainerState, force_sync: bool = False) -> None:
        """Assemble the payload for this step and hand it to the checkpointer."""
        save_dir = os.path.join(self._checkpoint_dir, f"{STEP_PREFIX}{state.global_step}")
        save_async = self._is_async and not force_sync

        logger.info(
            "Saving checkpoint: global_step=%s, epoch=%s, dir=%s, is_async=%s, "
            "extra_state_per_rank=%s, optimizer=%s, train_state=%s",
            state.global_step,
            state.epoch,
            save_dir,
            save_async,
            self._save_extra_state_per_rank,
            self._save_optimizer,
            self._save_train_state,
        )

        checkpoint_state: Dict[str, Any] = {"model": self._model_state_dict()}
        if self._save_optimizer:
            checkpoint_state["optimizer"] = _unwrap_single(
                [optimizer.state_dict() for optimizer in _as_list(self.trainer.optimizer)]
            )
        if self._save_train_state:
            checkpoint_state["extra_state"] = self._collect_extra_state(state)

        self.checkpointer.save(
            save_dir,
            checkpoint_state,
            global_step=state.global_step,
            save_async=save_async,
        )

        # Bookkeeping reflects the dispatched step immediately, including in async
        # mode: otherwise on_epoch_end would queue the same step again while the
        # first save is still in flight.
        self._last_saved_step = state.global_step

    # ------------------------------------------------------------------
    # Load / restore
    # ------------------------------------------------------------------

    def _resolve_restore_path(self) -> Optional[str]:
        """Resolve ``restore_from`` (including ``LATEST``) to a directory."""
        if self._restore_from is None:
            logger.info("No checkpoint to restore (restore_from is None).")
            return None

        restore_path = self._restore_from
        if restore_path.upper() == "LATEST":
            logger.info(
                "restore_from='LATEST', searching for the latest checkpoint in %s",
                self._checkpoint_dir,
            )
            resolved = self.checkpointer.find_latest_checkpoint(self._checkpoint_dir)
            if resolved is None:
                logger.warning(
                    "restore_from='LATEST' but no checkpoint found in %s; "
                    "starting from scratch.",
                    self._checkpoint_dir,
                )
                return None
            logger.info("Resolved LATEST checkpoint: %s", resolved)
            return resolved

        if not os.path.isdir(restore_path):
            raise FileNotFoundError(f"Checkpoint directory not found: {restore_path}")
        return restore_path

    def _load_checkpoint(self) -> None:
        """Restore a checkpoint into the trainer's live objects."""
        restore_path = self._resolve_restore_path()
        if restore_path is None:
            return

        logger.info("Loading checkpoint from %s", restore_path)

        optimizers = _as_list(self.trainer.optimizer) if self._restore_optimizer else []
        for optimizer in optimizers:
            if not initialize_optimizer_state(optimizer):
                logger.warning(
                    "Could not materialize optimizer state before loading; "
                    "optimizer moments may not be restored from %s.",
                    restore_path,
                )

        checkpoint_state: Dict[str, Any] = {"model": self._model_state_dict()}
        if optimizers:
            checkpoint_state["optimizer"] = _unwrap_single(
                [optimizer.state_dict() for optimizer in optimizers]
            )

        # The skeleton gives an embedded extra_state bundle keys to be read into;
        # the checkpointer decides whether it is actually needed for this layout.
        extra_state_skeleton = (
            self._collect_extra_state(self.trainer.state)
            if self._restore_train_state
            else None
        )

        self.checkpointer.load(
            restore_path,
            checkpoint_state,
            strict_model=not self._is_peft,
            extra_state_skeleton=extra_state_skeleton,
        )

        self.trainer.model.load_state_dict(
            checkpoint_state["model"], strict=not self._is_peft
        )
        # ``checkpoint_state["optimizer"]`` was built from ``optimizers`` above
        # (line 320) and DCP only fills that skeleton's existing tensor leaves
        # in place --- it never adds or removes list entries. So this list is
        # always exactly as long as ``optimizers``; a checkpoint that actually
        # carries fewer optimizer entries is reported by
        # ``_ModelStrictLoadPlanner`` (dcp_checkpointer.py) during
        # ``self.checkpointer.load()`` above, not by a length comparison here.
        optimizer_sds = _as_list(checkpoint_state.get("optimizer"))
        for optimizer, optimizer_sd in zip(optimizers, optimizer_sds):
            optimizer.load_state_dict(optimizer_sd)

        if self._restore_train_state:
            self._apply_extra_state(checkpoint_state["extra_state"])
        else:
            logger.info(
                "restore_train_state=False: loaded weights only from %s "
                "(step, scheduler, dataloader and RNG start fresh).",
                restore_path,
            )

        helper.empty_cache()
        logger.info(
            "Checkpoint loaded successfully: path=%s, global_step=%s, "
            "start_epoch=%s, start_step=%s",
            restore_path,
            self.trainer.state.global_step,
            self.trainer.start_epoch,
            self.trainer.start_step,
        )

    def _apply_extra_state(self, extra: Dict[str, Any]) -> None:
        """Restore progress, scheduler, dataloader position and RNG state."""
        trainer = self.trainer
        trainer.state.global_step = extra["global_step"]
        trainer.state.epoch = extra.get("epoch", 0)

        steps_per_epoch = self._steps_per_epoch()
        trainer.start_epoch = trainer.state.global_step // steps_per_epoch
        trainer.start_step = trainer.state.global_step % steps_per_epoch

        # The restored step is already on disk. Without this, resuming a run that
        # had nothing left to do would have ``on_train_end`` rewrite the very
        # checkpoint it just loaded.
        self._last_saved_step = trainer.state.global_step

        lr_scheduler_sd = extra.get("lr_scheduler")
        schedulers = _as_list(trainer.lr_scheduler)
        if lr_scheduler_sd and schedulers:
            scheduler_sds = _as_list(lr_scheduler_sd)
            if len(scheduler_sds) != len(schedulers):
                logger.warning(
                    "Checkpoint carries %s LR scheduler state dict(s) but this "
                    "run has %s scheduler(s); only the first %s pair(s) are "
                    "restored and any extra scheduler(s) keep their freshly "
                    "initialized state.",
                    len(scheduler_sds),
                    len(schedulers),
                    min(len(scheduler_sds), len(schedulers)),
                )
            for scheduler, scheduler_sd in zip(schedulers, scheduler_sds):
                scheduler.load_state_dict(scheduler_sd)

        dataloader_sd = extra.get("train_dataloader")
        if dataloader_sd and hasattr(trainer.train_dataloader, "load_state_dict"):
            trainer.train_dataloader.load_state_dict(dataloader_sd)
        elif dataloader_sd:
            logger.warning(
                "Checkpoint carries a dataloader position but %s is not stateful; "
                "the resumed epoch replays samples from its start.",
                type(trainer.train_dataloader).__name__,
            )

        rng_state = extra.get("rng_state") or {}
        torch_cpu_rng = rng_state.get("torch_cpu")
        if torch_cpu_rng is not None:
            torch.set_rng_state(torch_cpu_rng)
        set_device_rng_state(rng_state.get("torch_device"))
        python_rng = rng_state.get("python")
        if python_rng is not None:
            random.setstate(python_rng)


__all__ = ["CheckpointerCallback"]
