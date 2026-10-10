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

import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.checkpoint import build_checkpointer
from hyper_parallel.components.checkpoint.dcp_checkpointer import (
    STEP_PREFIX,
    initialize_optimizer_state,
)
from hyper_parallel.components.optim.mixed_precision_optimizer import (
    MixedPrecisionOptimizer,
)
from hyper_parallel.core.distributed_checkpoint.utils import flatten_state_dict
from hyper_parallel.models._transformers.model_builder import (
    validate_model_init_dtype,
)
from hyper_parallel.trainer.runtime.logging import create_logger
from hyper_parallel.trainer.runtime.memory import empty_cache
from hyper_parallel.trainer.runtime.device import (
    get_device_rng_state,
    set_device_rng_state,
)
from hyper_parallel.models.external_state import CheckpointRuntime, get_model_external_state
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer


logger = create_logger(__name__)


def _as_list(value: Any) -> List[Any]:
    """Normalize an optional single-or-list component into a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _state_leaf_keys(value: Any) -> frozenset[str]:
    """Use DCP's own traversal to list persisted dense optimizer leaves."""
    flat, _ = flatten_state_dict({"optimizer": value})
    return frozenset(key.removeprefix("optimizer.") for key in flat)


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
        self._restore_dataloader_state = ckpt_cfg.restore_dataloader_state

        external = get_model_external_state(getattr(trainer, "model", None))
        if external is not None:
            if self._is_async or self._is_peft:
                raise ValueError("Host external state requires synchronous non-PEFT checkpointing")
            if self._save_ckpt and (not self._save_optimizer or not self._save_train_state):
                raise ValueError("Host checkpoint requires optimizer and train-state saving")
            if self._restore_optimizer != self._restore_train_state:
                raise ValueError("Host restore requires optimizer and train state together")
            if trainer.mesh.pp_size > 1 and not self._save_extra_state_per_rank:
                raise ValueError("Host PP checkpoint requires per-rank extra state")

        self._last_saved_step: int = -1
        self.checkpointer = build_checkpointer(
            extra_state_per_rank=self._save_extra_state_per_rank,
        )

    # ------------------------------------------------------------------
    # Hook dispatchers
    # ------------------------------------------------------------------

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Log the checkpoint configuration and restore any requested state.

        Args:
            state: Trainer or external model state.
        """
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

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Surface a failed async save, then save on the configured step cadence.

        Args:
            state: Trainer or external model state.
        """
        # Asked every step on purpose. An async save that failed can only be
        # reported on a thread that cannot raise into this loop, so waiting for the
        # next save to notice would throw away every step in between.
        self.checkpointer.raise_for_failed_async_save()
        if self._save_steps > 0 and state.global_step % self._save_steps == 0:
            if state.global_step == self._last_saved_step:
                return
            self._save_checkpoint(state)

    def on_epoch_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Save on the configured epoch cadence.

        Args:
            state: Trainer or external model state.
        """
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

        Args:
            state: Trainer or external model state.
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
        external = get_model_external_state(model)
        if external is not None:
            missing = set(external.parameters) - set(state_dict)
            if missing:
                raise RuntimeError(f"External model FQNs missing from state_dict: {sorted(missing)}")
            state_dict = {key: value for key, value in state_dict.items()
                          if key not in external.parameters}
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

    def _resume_position(self) -> Tuple[int, int]:
        """Return the ``(epoch, step)`` position the training loop resumes from.

        Read off the same two values the loops use rather than re-derived from
        ``global_step``: they re-enter ``state.epoch`` and skip the optimizer steps
        its predecessors consumed, and ``state.epoch`` is restored from the
        checkpoint. Dividing ``global_step`` instead would be a second answer that
        can disagree with the loop, and it would have to divide by
        ``train_steps`` --- the optimizer steps one epoch holds --- not by the
        dataloader's length, which counts micro-batches.

        Returns:
            The epoch training resumes in, and its offset within that epoch.
        """
        state = self.trainer.state
        return state.epoch, state.global_step - state.epoch * self.trainer.train_steps

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

        external = get_model_external_state(self.trainer.model)
        if external is not None:
            external.before_checkpoint_save(CheckpointRuntime(
                step_dir=save_dir, global_step=state.global_step,
                mesh_context=self.trainer.mesh,
                save_optimizer=self._save_optimizer,
                save_train_state=self._save_train_state,
                restore_optimizer=self._restore_optimizer,
                restore_train_state=self._restore_train_state,
                persisted_optimizer_keys=_state_leaf_keys(checkpoint_state["optimizer"]),
            ))

        model_integration = getattr(self.trainer, "model_integration", None)
        if model_integration is not None:
            model_integration.capture_checkpoint_payload(
                "before_save",
                save_dir,
                checkpoint_state,
            )

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

    def _build_restore_skeleton(self, restore_path: str, external: Any) -> tuple[Dict[str, Any], List[Any]]:
        """Initialize lazy dense moments and create the DCP load skeleton."""
        optimizers = _as_list(self.trainer.optimizer) if self._restore_optimizer else []
        for optimizer in optimizers:
            target = external.optimizer_for_dcp() if external is not None else optimizer
            if not initialize_optimizer_state(target):
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
        return checkpoint_state, optimizers

    def _load_optimizer_payload(self, checkpoint_state: Dict[str, Any], optimizers: List[Any],
                                persisted_keys: Optional[frozenset[str]]) -> None:
        """Restore dense optimizer leaves or reload dense main parameters."""
        optimizer_sds = _as_list(checkpoint_state.get("optimizer"))
        for optimizer, optimizer_sd in zip(optimizers, optimizer_sds):
            optimizer.load_state_dict(optimizer_sd)
            drop_primed = getattr(optimizer, "drop_unpersisted_dense_state", None)
            if drop_primed is not None and persisted_keys is not None:
                drop_primed(persisted_keys)
        if not optimizers:
            for optimizer in _as_list(self.trainer.optimizer):
                if isinstance(optimizer, MixedPrecisionOptimizer):
                    optimizer.reload_model_params()

    def _apply_restored_payload(self, checkpoint_state: Dict[str, Any], optimizers: List[Any],
                                external: Any, runtime: Any, restore_path: str,
                                persisted_keys: Optional[frozenset[str]]) -> None:
        """Install the loaded dense and external model, optimizer, and train state."""
        if external is not None:
            full_model_keys = set(self.trainer.model.state_dict())
            dense_model_keys = set(checkpoint_state["model"])
            if full_model_keys - dense_model_keys != set(external.parameters):
                raise RuntimeError("Dense checkpoint model keys differ from exact external exclusions")
        load_result = self.trainer.model.load_state_dict(
            checkpoint_state["model"], strict=not self._is_peft and external is None
        )
        if external is not None and load_result is not None and (
                set(load_result.missing_keys) != set(external.parameters)
                or load_result.unexpected_keys):
            raise RuntimeError("Dense checkpoint model keys differ from exact external exclusions")
        validate_model_init_dtype(
            self.trainer.model,
            self.trainer.config.model_init_dtype,
        )
        self._load_optimizer_payload(checkpoint_state, optimizers, persisted_keys)

        if self._restore_train_state:
            self._apply_extra_state(checkpoint_state["extra_state"])
        else:
            logger.info(
                "restore_train_state=False: loaded weights only from %s "
                "(step, scheduler, dataloader and RNG start fresh).",
                restore_path,
            )

        if external is not None:
            external.after_checkpoint_load(runtime)
            if not self._restore_optimizer:
                dense = external.optimizer_for_dcp()
                if isinstance(dense, MixedPrecisionOptimizer):
                    dense.reload_model_params()
                external.after_weights_only_load()

    def _load_checkpoint(self) -> None:
        """Restore a checkpoint into the trainer's live objects."""
        restore_path = self._resolve_restore_path()
        if restore_path is None:
            return

        logger.info("Loading checkpoint from %s", restore_path)
        external = get_model_external_state(self.trainer.model)
        runtime = None
        requirements = None
        if external is not None:
            runtime = CheckpointRuntime(
                step_dir=restore_path, global_step=None,
                mesh_context=self.trainer.mesh,
                save_optimizer=self._save_optimizer,
                save_train_state=self._save_train_state,
                restore_optimizer=self._restore_optimizer,
                restore_train_state=self._restore_train_state,
            )
            requirements = external.before_checkpoint_load(runtime)

        checkpoint_state, optimizers = self._build_restore_skeleton(restore_path, external)

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
        if requirements is not None and self._restore_optimizer:
            missing_optimizer_keys = (requirements.persisted_optimizer_keys
                                      - _state_leaf_keys(checkpoint_state["optimizer"]))
            if missing_optimizer_keys:
                raise RuntimeError(f"Dense optimizer skeleton lost persisted keys: {sorted(missing_optimizer_keys)}")
        model_integration = getattr(self.trainer, "model_integration", None)
        if model_integration is not None:
            model_integration.capture_checkpoint_payload(
                "after_load",
                restore_path,
                checkpoint_state,
            )

        self._apply_restored_payload(
            checkpoint_state, optimizers, external, runtime, restore_path,
            requirements.persisted_optimizer_keys if requirements is not None else None,
        )

        empty_cache()
        # Computed here because this log is the only consumer: the training loops
        # derive their own starting position from ``state`` instead, so carrying it
        # on the trainer would be state nobody reads.
        start_epoch, start_step = self._resume_position()
        logger.info(
            "Checkpoint loaded successfully: path=%s, global_step=%s, "
            "start_epoch=%s, start_step=%s",
            restore_path,
            self.trainer.state.global_step,
            start_epoch,
            start_step,
        )

    def _apply_extra_state(self, extra: Dict[str, Any]) -> None:
        """Restore progress, scheduler, dataloader position and RNG state."""
        trainer = self.trainer
        trainer.state.global_step = extra["global_step"]
        persisted_epoch = extra.get("epoch", 0)
        trainer.state.epoch = persisted_epoch

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
        if dataloader_sd and not self._restore_dataloader_state:
            logger.info(
                "restore_dataloader_state=False: retaining the configured "
                "dataloader start instead of restoring its checkpoint cursor."
            )
        elif dataloader_sd and hasattr(trainer.train_dataloader, "load_state_dict"):
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
