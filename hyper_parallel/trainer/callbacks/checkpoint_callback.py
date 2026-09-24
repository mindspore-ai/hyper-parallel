# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
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

__all__ = ["CheckpointerCallback"]

import math
import os
import random
import shutil
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.checkpoint import build_checkpointer
from hyper_parallel.components.checkpoint.dcp_checkpointer import (
    STEP_PREFIX,
    initialize_optimizer_state,
)
from hyper_parallel.components.optim.mixed_precision_optimizer import (
    MixedPrecisionOptimizer,
)
from hyper_parallel.models._transformers import CheckpointManager
from hyper_parallel.models._transformers.model_builder import apply_model_init_dtype
from hyper_parallel.trainer.runtime.device import (
    get_device_rng_state,
    set_device_rng_state,
)
from hyper_parallel.trainer.runtime.distributed import all_reduce
from hyper_parallel.trainer.runtime.logging import create_logger
from hyper_parallel.trainer.runtime.memory import empty_cache
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer


logger = create_logger(__name__)
_HF_CHECKPOINT_DIR = "hf_ckpt"
_HF_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


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
    :class:`~hyper_parallel.components.checkpoint.CheckpointerBase` it delegates
    to, so the storage format can change without touching this file.

    The payload is the model and optimizer state dicts plus an ``extra_state``
    bundle holding what those cannot represent: ``global_step`` / ``epoch``, the
    LR scheduler, the dataloader position, and the CPU / device / Python RNG
    states.

    DCP saving, Hugging Face export, and restoring are independently controlled
    by ``save_ckpt``, ``save_hf_weights``, and ``restore_from``. This permits a
    restore-only run as well as exporting Hugging Face weights from an existing
    DCP checkpoint without writing another DCP checkpoint.

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
        save_time_interval_minutes = float(ckpt_cfg.save_time_interval_minutes)
        if not math.isfinite(save_time_interval_minutes):
            raise ValueError("checkpoint.save_time_interval_minutes must be finite")
        self._save_time_interval_seconds = (
            max(save_time_interval_minutes, 0.0) * 60.0
            if self._save_ckpt
            else 0.0
        )
        self._next_time_save: Optional[float] = None
        self._save_total_limit = ckpt_cfg.save_total_limit
        self._is_async = ckpt_cfg.is_async
        self._is_peft = ckpt_cfg.is_peft
        self._save_optimizer = ckpt_cfg.save_optimizer
        self._save_train_state = ckpt_cfg.save_train_state
        self._save_hf_weights = ckpt_cfg.save_hf_weights
        self._save_extra_state_per_rank = ckpt_cfg.save_extra_state_per_rank

        if self._save_hf_weights and self._is_peft:
            raise ValueError(
                "checkpoint.save_hf_weights does not yet support PEFT checkpoints"
            )

        self._restore_from = ckpt_cfg.restore_from
        self._restore_optimizer = ckpt_cfg.restore_optimizer
        self._restore_train_state = ckpt_cfg.restore_train_state

        self._last_saved_step: int = -1
        self._last_hf_saved_step: int = -1
        self.checkpointer = build_checkpointer(
            extra_state_per_rank=self._save_extra_state_per_rank,
        )
        self._hf_checkpoint_manager = (
            CheckpointManager(trainer.model) if self._save_hf_weights else None
        )

    # ------------------------------------------------------------------
    # Hook dispatchers
    # ------------------------------------------------------------------

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Log the checkpoint configuration and restore any requested state."""
        logger.info(
            "Checkpoint configuration: "
            "checkpoint_dir=%s, save_ckpt=%s, save_hf_weights=%s, "
            "save_steps=%s, save_epochs=%s, save_time_interval_minutes=%s, "
            "save_total_limit=%s, "
            "is_async=%s, is_peft=%s, "
            "save_extra_state_per_rank=%s, restore_from=%s",
            self._checkpoint_dir,
            self._save_ckpt,
            self._save_hf_weights,
            self._save_steps,
            self._save_epochs,
            self._save_time_interval_seconds / 60.0,
            self._save_total_limit,
            self._is_async,
            self._is_peft,
            self._save_extra_state_per_rank,
            self._restore_from,
        )
        self._load_checkpoint()
        if self._save_time_interval_seconds > 0:
            self._next_time_save = time.monotonic() + self._save_time_interval_seconds

    def on_step_end(  # pylint: disable=arguments-differ
        self, state: TrainerState, **kwargs: Any
    ) -> None:
        """Save on either the configured step or elapsed-time cadence."""
        step_due = self._save_steps > 0 and state.global_step % self._save_steps == 0
        time_due = self._time_checkpoint_due()
        if not step_due and not time_due:
            return

        if state.global_step != self._last_saved_step:
            self._save_checkpoint(state)
        if time_due:
            self._advance_time_deadline()

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
        if (
            self._save_hf_weights
            and state.global_step > 0
            and state.global_step != self._last_hf_saved_step
        ):
            save_dir = os.path.join(
                self._checkpoint_dir, f"{STEP_PREFIX}{state.global_step}"
            )
            self._save_hf_checkpoint(save_dir, state.global_step)
            self._rotate_checkpoints()

    def wait_for_pending_save(self) -> None:
        """Block until the checkpointer's in-flight async save is persisted."""
        self.checkpointer.maybe_wait_for_async_save()
        self._rotate_checkpoints()

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

    def _time_checkpoint_due(self) -> bool:
        """Return one rank-consistent decision for the elapsed-time cadence."""
        if self._next_time_save is None:
            return False

        local_due = int(time.monotonic() >= self._next_time_save)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return bool(all_reduce(local_due, op="max"))
        return bool(local_due)

    def _advance_time_deadline(self) -> None:
        """Move the time deadline forward without accumulating save latency."""
        if self._next_time_save is None:
            return
        now = time.monotonic()
        elapsed = max(now - self._next_time_save, 0.0)
        elapsed_intervals = int(elapsed // self._save_time_interval_seconds)
        self._next_time_save += (elapsed_intervals + 1) * self._save_time_interval_seconds

    def _save_checkpoint(self, state: TrainerState, force_sync: bool = False) -> None:
        """Assemble the payload for this step and hand it to the checkpointer."""
        if (
                self._save_total_limit is not None
                and self._save_total_limit > 0
                and self._is_async
                and self._last_saved_step >= 0
        ):
            # Complete and rotate the preceding async checkpoint before
            # starting a new one. This temporarily allows limit + 1
            # directories instead of deleting the last known-good checkpoint
            # while its replacement is still being written.
            self.wait_for_pending_save()

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
        if self._save_hf_weights:
            # The full-weight gather is memory-intensive, so never overlap it
            # with an asynchronous DCP persistence job.
            self.checkpointer.maybe_wait_for_async_save()
            self._save_hf_checkpoint(save_dir, state.global_step)
            self._rotate_checkpoints()
        elif not save_async:
            self._rotate_checkpoints()

    def _save_hf_checkpoint(self, save_dir: str, global_step: int) -> None:
        """Collect and export one Transformers-compatible model checkpoint."""
        if self._hf_checkpoint_manager is None:
            raise RuntimeError("Hugging Face checkpoint manager is not initialized")

        hf_dir = os.path.join(save_dir, _HF_CHECKPOINT_DIR)
        logger.info(
            "Saving Hugging Face weights: global_step=%s, dir=%s",
            global_step,
            hf_dir,
        )

        write_assets = self._hf_checkpoint_manager.save_pretrained(hf_dir)
        if write_assets:
            self._save_hf_assets(hf_dir)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self._last_hf_saved_step = global_step
        if write_assets:
            logger.info(
                "Hugging Face checkpoint saved successfully: global_step=%s, dir=%s",
                global_step,
                hf_dir,
            )

    @staticmethod
    def _checkpoint_step(name: str) -> Optional[int]:
        """Parse an exact ``global_step_<integer>`` directory name."""
        if not name.startswith(STEP_PREFIX):
            return None
        suffix = name[len(STEP_PREFIX):]
        if not suffix or not suffix.isascii() or not suffix.isdigit():
            return None
        if len(suffix) > 1 and suffix.startswith("0"):
            return None
        return int(suffix)

    @staticmethod
    def _is_complete_checkpoint(path: str) -> bool:
        """Return whether a managed step directory contains a complete payload."""
        try:
            with os.scandir(path) as entries:
                if any(
                        entry.is_file(follow_symlinks=False)
                        and (
                            entry.name == ".metadata"
                            or (
                                entry.name[:-len(".metadata")].isascii()
                                and entry.name[:-len(".metadata")].isdigit()
                                and entry.name.endswith(".metadata")
                            )
                        )
                        for entry in entries
                ):
                    return True
        except OSError:
            return False

        hf_dir = os.path.join(path, _HF_CHECKPOINT_DIR)
        if os.path.islink(hf_dir) or not os.path.isdir(hf_dir):
            return False
        return any(os.path.isfile(os.path.join(hf_dir, name)) for name in _HF_WEIGHT_FILES)

    def _list_managed_checkpoints(self) -> List[tuple[int, str]]:
        """List completed, non-symlink checkpoint directories from oldest to newest."""
        checkpoint_root = os.path.realpath(self._checkpoint_dir)
        try:
            entries = os.scandir(checkpoint_root)
        except OSError:
            return []

        checkpoints = []
        with entries:
            for entry in entries:
                step = self._checkpoint_step(entry.name)
                if step is None or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    continue
                resolved_path = os.path.realpath(entry.path)
                if os.path.dirname(resolved_path) != checkpoint_root:
                    logger.warning("Skipping checkpoint outside configured root: %s", entry.path)
                    continue
                if self._is_complete_checkpoint(resolved_path):
                    checkpoints.append((step, resolved_path))

        checkpoints.sort(key=lambda item: item[0])
        return checkpoints

    def _rotate_checkpoints(self) -> None:
        """Delete oldest completed checkpoints while preserving active saves."""
        limit = self._save_total_limit
        if limit is None or limit <= 0:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        checkpoints = self._list_managed_checkpoints()
        delete_count = len(checkpoints) - limit
        if delete_count <= 0:
            return

        protected_steps = {
            step
            for step in (self._last_saved_step, self._last_hf_saved_step)
            if step >= 0
        }
        deletable = [item for item in checkpoints if item[0] not in protected_steps]
        if len(deletable) < delete_count:
            logger.warning(
                "Checkpoint limit %s cannot be met without deleting an active checkpoint; "
                "keeping %s completed checkpoints.",
                limit,
                len(checkpoints) - len(deletable),
            )
            delete_count = len(deletable)

        for step, checkpoint_path in deletable[:delete_count]:
            # Revalidate immediately before deletion so a replaced path or
            # symlink can never redirect rotation outside checkpoint_dir.
            checkpoint_root = os.path.realpath(self._checkpoint_dir)
            if (
                    os.path.islink(checkpoint_path)
                    or not os.path.isdir(checkpoint_path)
                    or os.path.dirname(os.path.realpath(checkpoint_path)) != checkpoint_root
                    or self._checkpoint_step(os.path.basename(checkpoint_path)) != step
            ):
                logger.warning("Skipping unsafe checkpoint deletion target: %s", checkpoint_path)
                continue
            try:
                shutil.rmtree(checkpoint_path)
            except OSError as exc:
                logger.warning("Failed to delete old checkpoint %s: %s", checkpoint_path, exc)
                continue
            logger.info(
                "Deleted old checkpoint due to save_total_limit=%s: %s",
                limit,
                checkpoint_path,
            )

    def _save_hf_assets(self, hf_dir: str) -> None:
        """Write the tokenizer or processor assets next to exported weights."""
        processor = getattr(self.trainer, "processor", None)
        if processor is not None:
            processor.save_pretrained(hf_dir)
            return

        chat_template = getattr(self.trainer, "chat_template", None)
        if chat_template is not None:
            chat_template.save_pretrained(hf_dir)
            return

        tokenizer = getattr(self.trainer, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(hf_dir)

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
        apply_model_init_dtype(
            self.trainer.model,
            self.trainer.config.model_init_dtype,
        )
        # ``checkpoint_state["optimizer"]`` was built from ``optimizers`` above
        # and DCP only fills that skeleton's existing tensor leaves in place ---
        # it never adds or removes list entries. The checkpoint planner reports
        # missing persisted optimizer entries during ``checkpointer.load()``.
        optimizer_sds = _as_list(checkpoint_state.get("optimizer"))
        for optimizer, optimizer_sd in zip(optimizers, optimizer_sds):
            optimizer.load_state_dict(optimizer_sd)
        if not optimizers:
            for optimizer in _as_list(self.trainer.optimizer):
                if isinstance(optimizer, MixedPrecisionOptimizer):
                    optimizer.reload_model_params()

        if self._restore_train_state:
            self._apply_extra_state(checkpoint_state["extra_state"])
        else:
            logger.info(
                "restore_train_state=False: loaded weights only from %s "
                "(step, scheduler, dataloader and RNG start fresh).",
                restore_path,
            )

        empty_cache()
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
