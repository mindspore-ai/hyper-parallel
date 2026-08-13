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
# ============================================================================
"""Distributed-checkpoint (DCP) backend for :class:`CheckpointerBase`."""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.distributed_checkpoint import (
    StandardLoadPlanner,
    async_save as dcp_async_save,
    load as dcp_load,
    save as dcp_save,
)
from hyper_models.components.checkpoint.checkpointer import CheckpointerBase
from hyper_models.components.utils import helper


logger = helper.create_logger(__name__)

# Layout of a checkpoint root:
#
#   checkpoints/
#   ├── latest_checkpoint_iteration.txt       # step number of the newest checkpoint
#   └── global_step_42/
#       ├── .metadata                         # DCP global metadata, written last
#       ├── __0_0.safetensors, ...            # DCP shard payloads
#       └── extra_state/
#           └── extra_state_rank_{R}.pt       # per-rank training state
#
# ``.metadata`` doubles as the completeness signal: ``FileSystemWriter``'s
# ``finalize_checkpoint`` only writes it after every rank's shard writes have
# been collected, so a directory carrying it holds a fully persisted checkpoint
# and one missing it was interrupted mid-save.
STEP_PREFIX = "global_step_"
LATEST_CHECKPOINT_FILE = "latest_checkpoint_iteration.txt"
_EXTRA_STATE_DIR = "extra_state"
_EXTRA_STATE_FILE = "extra_state_rank_{rank}.pt"
_DCP_METADATA_FILE = ".metadata"

# ``extra_state`` must carry this key; the checkpointer stamps a sentinel over
# it in the skeleton handed to an embedded-bundle load, so a load that read
# nothing back is distinguishable from a genuine restore. A real checkpoint is
# never written at a step this low.
_STEP_KEY = "global_step"
_UNRESTORED_STEP = -1


def _get_rank() -> int:
    """Return this process' global rank, or 0 outside a process group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _barrier() -> None:
    """Synchronize all ranks, tolerating single-process (non-distributed) runs."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class _ModelStrictLoadPlanner(StandardLoadPlanner):
    """Allow partial optimizer state while still requiring a complete model.

    A parameter that never received a gradient --- a frozen weight, an MoE expert
    no batch routed to, or simply a checkpoint written before the first
    ``step()`` --- has no optimizer state to save. Meanwhile the load side primes
    the optimizer (see :func:`initialize_optimizer_state`) so DCP has entries to
    fill, which means it legitimately asks for keys the checkpoint does not
    carry. Under the planner's default strict mode that is a hard error, so
    ``allow_partial_load`` lets those entries keep the zero-initialised values
    the optimizer would have created on its next step anyway.

    ``allow_partial_load`` is global to the planner and cannot be scoped to the
    optimizer subtree, so model completeness is re-checked here: a truncated
    model checkpoint must still fail loudly instead of resuming half-loaded.
    """

    def __init__(self, strict_model: bool = True) -> None:
        """Load partially, optionally demanding every requested model key."""
        super().__init__(allow_partial_load=True)
        self.strict_model = strict_model

    def build_local_plan(self):
        """Validate model coverage, then defer to the standard planner."""
        if self.state_dict is None or self.metadata is None:
            return super().build_local_plan()

        available = self.metadata.state_dict_metadata
        missing = [fqn for fqn in self.state_dict if fqn not in available]

        missing_model = sorted(fqn for fqn in missing if fqn.startswith("model."))
        if self.strict_model and missing_model:
            preview = ", ".join(missing_model[:10])
            suffix = " ..." if len(missing_model) > 10 else ""
            raise RuntimeError(
                f"Checkpoint is missing {len(missing_model)} model key(s) required "
                f"for a full-model resume: {preview}{suffix}"
            )

        missing_optimizer = sorted(fqn for fqn in missing if fqn.startswith("optimizer."))
        if missing_optimizer:
            preview = ", ".join(missing_optimizer[:3])
            logger.warning(
                "%s optimizer entries are absent from the checkpoint and keep their "
                "zero-initialised values (e.g. %s). Expected for parameters that "
                "never received a gradient, or for a checkpoint written with "
                "save_optimizer=false.",
                len(missing_optimizer),
                preview,
            )

        return super().build_local_plan()


def initialize_optimizer_state(optimizer: torch.optim.Optimizer) -> bool:
    """Materialize optimizer state so a DCP load has entries to fill.

    ``Optimizer.state`` is empty until the first ``step()``, and DCP loads
    strictly into the keys the caller's state_dict already contains. Restoring a
    freshly built optimizer therefore reads *nothing* and silently resumes with
    zeroed moments --- the model continues from the checkpoint while Adam starts
    over. Priming with zero gradients creates the ``exp_avg`` / ``exp_avg_sq``
    entries the checkpoint will overwrite.

    The priming step runs with ``lr`` and ``weight_decay`` forced to zero, so the
    parameters themselves are left untouched.

    Returns:
        Whether the optimizer now carries per-parameter state.
    """
    # ``ChainedOptimizer`` (this repo's composition wrapper for split param
    # groups, e.g. Muon/AdamW or decay/no-decay) has no ``.state`` of its own
    # --- only each sub-optimizer it dispatches to does. ``param_groups`` /
    # ``step`` / ``zero_grad`` are already delegated below via its own
    # properties, so duck-typing this one flat-optimizer assumption is enough.
    sub_optimizers = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    if all(opt.state for opt in sub_optimizers):
        return True

    params = [
        param
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ]
    if not params:
        return False
    if any(param.grad is not None for param in params):
        logger.warning(
            "Optimizer parameters already carry gradients; skipping state priming "
            "to avoid corrupting them. Optimizer state may not be restored."
        )
        return False

    saved_hyperparams = [
        (group.get("lr"), group.get("weight_decay")) for group in optimizer.param_groups
    ]
    try:
        for param in params:
            param.grad = torch.zeros_like(param)
        for group in optimizer.param_groups:
            if "lr" in group:
                group["lr"] = 0.0
            if "weight_decay" in group:
                group["weight_decay"] = 0.0
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            optimizer.step()
    finally:
        for group, (lr, weight_decay) in zip(optimizer.param_groups, saved_hyperparams):
            if lr is not None:
                group["lr"] = lr
            if weight_decay is not None:
                group["weight_decay"] = weight_decay
        optimizer.zero_grad(set_to_none=True)

    return all(opt.state for opt in sub_optimizers)


class DistributedCheckpointer(CheckpointerBase):
    """Persist a training payload through ``hyper_parallel``'s DCP.

    ``extra_state`` can live in either of two layouts, chosen by
    ``extra_state_per_rank`` at save time:

    * ``True`` --- one ``torch.save`` file per rank under ``extra_state/``.
      Always correct.
    * ``False`` --- embedded in the DCP payload. Fewer files, but DCP
      deduplicates entries sharing an FQN across ranks (see
      ``distributed_checkpoint/util.py::remove_redundant_plans``), so every rank
      restores whichever rank's copy won. Only valid when the extra state is
      rank-invariant.

    :meth:`load` auto-detects which layout a checkpoint on disk used, so the
    flag may differ from the run that produced it.
    """

    def __init__(self, extra_state_per_rank: bool = True) -> None:
        """Select the ``extra_state`` layout used by :meth:`save`."""
        self.extra_state_per_rank = extra_state_per_rank
        self._async_save_response: Optional[Any] = None
        self._async_save_dir: Optional[str] = None
        self._async_save_step: int = -1

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extra_state_path(checkpoint_dir: str, rank: int) -> str:
        """Return this rank's extra_state file inside ``checkpoint_dir``."""
        return os.path.join(
            checkpoint_dir, _EXTRA_STATE_DIR, _EXTRA_STATE_FILE.format(rank=rank)
        )

    @staticmethod
    def _is_complete(checkpoint_path: str) -> bool:
        """Report whether ``checkpoint_path`` holds a fully persisted checkpoint.

        Presence of the DCP metadata file is the signal: it is written only once
        every rank's shards are on disk. ``use_collectives=False`` names it
        ``{rank}.metadata`` instead of a single shared ``.metadata``, so both
        spellings count.
        """
        if not os.path.isdir(checkpoint_path):
            return False
        if os.path.isfile(os.path.join(checkpoint_path, _DCP_METADATA_FILE)):
            return True
        return any(
            name.endswith(_DCP_METADATA_FILE) for name in os.listdir(checkpoint_path)
        )

    @staticmethod
    def _latest_pointer_path(checkpoint_dir: str) -> str:
        """Return the path of the newest-checkpoint pointer file."""
        return os.path.join(checkpoint_dir, LATEST_CHECKPOINT_FILE)

    def _write_latest_pointer(self, checkpoint_dir: str, global_step: int) -> None:
        """Record which checkpoint is newest, so resume need not scan the tree.

        Only the step number is stored: the directory name is
        ``STEP_PREFIX + step``, and that prefix is a module constant rather than
        anything configurable, so writing it down would just be a second copy of
        a value the reader already has.
        """
        with open(self._latest_pointer_path(checkpoint_dir), "w", encoding="utf-8") as handle:
            handle.write(f"{global_step}\n")

    def _read_latest_pointer(self, checkpoint_dir: str) -> Optional[str]:
        """Resolve the pointer file to a usable checkpoint directory.

        Returns ``None`` when the file is absent, unreadable, or names a
        checkpoint that is missing or incomplete --- every one of which is a
        reason to fall back to scanning rather than to fail.
        """
        pointer_path = self._latest_pointer_path(checkpoint_dir)
        if not os.path.isfile(pointer_path):
            return None

        try:
            with open(pointer_path, encoding="utf-8") as handle:
                global_step = int(handle.read().strip())
        except (OSError, ValueError):
            logger.warning(
                "Ignoring unreadable %s; falling back to scanning %s.",
                pointer_path,
                checkpoint_dir,
            )
            return None

        candidate = os.path.join(checkpoint_dir, f"{STEP_PREFIX}{global_step}")
        if not self._is_complete(candidate):
            logger.warning(
                "%s points at %s, which is missing or incomplete; falling back to "
                "scanning %s.",
                pointer_path,
                candidate,
                checkpoint_dir,
            )
            return None
        return candidate

    def _list_checkpoints(self, checkpoint_dir: str) -> List[tuple]:
        """List ``(step, path)`` for every complete checkpoint, oldest first."""
        if not os.path.isdir(checkpoint_dir):
            return []

        found = []
        for entry in os.listdir(checkpoint_dir):
            entry_path = os.path.join(checkpoint_dir, entry)
            if not os.path.isdir(entry_path) or not entry.startswith(STEP_PREFIX):
                continue
            try:
                step = int(entry[len(STEP_PREFIX):])
            except ValueError:
                continue
            if self._is_complete(entry_path):
                found.append((step, entry_path))

        found.sort(key=lambda item: item[0])
        return found

    def find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """Find the newest complete checkpoint under ``checkpoint_dir``.

        The pointer file written after each save answers this in one read. It is
        only a cache, though --- checkpoints produced before it existed, or
        copied in by hand, are still found by scanning the directory. A
        checkpoint interrupted mid-save has no DCP metadata and is skipped by
        both paths, so resume never lands on a half-written one.
        """
        from_pointer = self._read_latest_pointer(checkpoint_dir)
        if from_pointer is not None:
            return from_pointer

        candidates = self._list_checkpoints(checkpoint_dir)
        if not candidates:
            return None
        return candidates[-1][1]

    # ------------------------------------------------------------------
    # Async coordination
    # ------------------------------------------------------------------

    def maybe_wait_for_async_save(self) -> None:
        """Block until an in-flight async save finishes, then mark it complete.

        Safe to call when nothing is pending. This is the single entry point for
        async-save coordination --- it runs before every new save, before a
        restore, and at train end, so two saves never overlap and the process
        never exits with a half-persisted checkpoint.
        """
        if self._async_save_response is None:
            return

        rank = _get_rank()
        try:
            logger.info("[rank %s] waiting for async DCP save to finish...", rank)
            self._async_save_response.persist_completion.result()
            self._finalize_checkpoint(self._async_save_dir, self._async_save_step)
            logger.info("[rank %s] async DCP save has been finished...", rank)
        except Exception:
            logger.exception("[rank %s] async DCP save failed", rank)
            raise
        finally:
            self._async_save_response = None
            self._async_save_dir = None
            self._async_save_step = -1

    def _finalize_checkpoint(self, save_dir: Optional[str], step: int) -> None:
        """Publish the finished checkpoint as the newest one.

        The leading barrier holds rank 0 until every rank has finished writing,
        so the pointer is never published over a checkpoint that is still being
        assembled; the trailing one keeps the ranks in step afterwards.
        """
        if save_dir is None:
            return

        _barrier()
        if _get_rank() == 0:
            self._write_latest_pointer(os.path.dirname(save_dir), step)
        _barrier()

        logger.info(
            "Distributed checkpoint saved successfully: global_step=%s, dir=%s",
            step,
            save_dir,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        state: Dict[str, Any],
        *,
        global_step: int,
        save_async: bool = False,
    ) -> None:
        """Write ``state`` into ``path``; see :meth:`CheckpointerBase.save`."""
        # Never let two saves overlap: they would compete for host memory and,
        # in async mode, race on the checkpoint directory.
        self.maybe_wait_for_async_save()

        payload = dict(state)
        extra_state = payload.pop("extra_state", None)

        if extra_state is not None and self.extra_state_per_rank:
            os.makedirs(os.path.join(path, _EXTRA_STATE_DIR), exist_ok=True)
            _barrier()
            # Written before the shards so that every persisted model/optimizer
            # payload is already backed by the progress state to resume from it.
            torch.save(extra_state, self._extra_state_path(path, _get_rank()))
        else:
            os.makedirs(path, exist_ok=True)
            _barrier()
            if extra_state is not None:
                payload["extra_state"] = extra_state

        # Free residual training-step allocations before DCP allocates its
        # collective buffers for the gather pass.
        helper.empty_cache()

        try:
            if save_async:
                self._async_save_response = dcp_async_save(payload, checkpoint_id=path)
                self._async_save_dir = path
                self._async_save_step = global_step
                logger.info("Async DCP save dispatched: %s", path)
            else:
                dcp_save(payload, checkpoint_id=path)
                logger.info("Sync DCP save completed: %s", path)
        except Exception:
            logger.exception("Failed to save checkpoint at %s", path)
            raise

        helper.empty_cache()

        if not save_async:
            self._finalize_checkpoint(path, global_step)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        path: str,
        state: Dict[str, Any],
        *,
        strict_model: bool = True,
        extra_state_skeleton: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fill ``state`` from ``path``; see :meth:`CheckpointerBase.load`."""
        self.maybe_wait_for_async_save()

        # Auto-detect the layout the checkpoint was written with, so the
        # configured ``extra_state_per_rank`` may differ from the producing run.
        extra_state_file = self._extra_state_path(path, _get_rank())
        from_file = os.path.isfile(extra_state_file)

        if extra_state_skeleton is not None and not from_file:
            skeleton = dict(extra_state_skeleton)
            skeleton[_STEP_KEY] = _UNRESTORED_STEP
            state["extra_state"] = skeleton

        try:
            dcp_load(
                state,
                checkpoint_id=path,
                planner=_ModelStrictLoadPlanner(strict_model=strict_model),
            )
        except Exception:
            logger.exception("Failed to load checkpoint from %s", path)
            raise

        if extra_state_skeleton is not None:
            state["extra_state"] = self._read_extra_state(
                path, state, extra_state_file, from_file
            )

        # Hold every rank here until all of them have finished reading, so no
        # rank starts training against a checkpoint others are still loading.
        _barrier()
        return state

    def _read_extra_state(
        self,
        path: str,
        state: Dict[str, Any],
        extra_state_file: str,
        from_file: bool,
    ) -> Dict[str, Any]:
        """Return the extra_state bundle from wherever the checkpoint kept it."""
        if from_file:
            logger.info("Reading extra_state from %s", extra_state_file)
            return torch.load(extra_state_file, weights_only=False)

        extra = state.get("extra_state") or {}
        if extra.get(_STEP_KEY, _UNRESTORED_STEP) == _UNRESTORED_STEP:
            raise FileNotFoundError(
                f"Checkpoint {path} carries no training state for rank "
                f"{_get_rank()}: neither a per-rank file ({extra_state_file}) nor "
                "an extra_state entry inside the DCP. Resume needs the progress "
                "state saved beside the model shards; set "
                "checkpoint.restore_train_state=false to load the weights only."
            )
        logger.info("Read extra_state embedded in the DCP state dict")
        return extra


__all__ = [
    "STEP_PREFIX",
    "DistributedCheckpointer",
    "initialize_optimizer_state",
]
