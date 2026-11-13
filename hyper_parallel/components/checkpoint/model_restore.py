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
"""Model-only checkpoint restore for standalone tasks.

The training restore path lives in ``CheckpointerCallback`` and runs from
``on_train_begin``. Standalone eval/predict do not build callbacks at all, so
they need an entry point that resolves a checkpoint, loads the dense model
state, and installs any model-external sidecar -- without an optimizer, an lr
scheduler, or a DataLoader cursor.

The strategy stays in the callback (what a training run restores); this module
owns only the standalone ordering. Duplicating the path-resolution and state
mapping here instead of reusing the callback is deliberate: the callback's
skeleton is built around a live optimizer, and giving it a dummy one would make
"model-only" a claim rather than a property.
"""

from __future__ import annotations

__all__ = ["RestoreOutcome", "restore_model_weights"]

from dataclasses import dataclass
from typing import Any, Dict, Optional

from hyper_parallel.components.checkpoint import build_checkpointer
from hyper_parallel.models._transformers.model_builder import validate_model_init_dtype
from hyper_parallel.models.external_state import CheckpointRuntime, get_model_external_state
from hyper_parallel.trainer.runtime.logging import create_logger

logger = create_logger(__name__)


@dataclass
class RestoreOutcome:
    """What a standalone restore resolved and applied.

    Args:
        path: Resolved checkpoint directory, or ``None`` when nothing was
            restored.
        step: Checkpointed optimizer step recorded in the resolved directory
            name, or ``None`` for a weights-only load.
        external_applied: Whether a model-external sidecar participated.
    """

    path: Optional[str] = None
    step: Optional[int] = None
    external_applied: bool = False


def _resolve_path(config: Any, checkpointer: Any) -> Optional[str]:
    """Resolve ``checkpoint.restore_from`` (including ``LATEST``) to a directory.

    Unlike the training callback, an unresolvable request fails loudly: a
    standalone task that silently falls back to freshly initialized weights
    would report metrics for a model nobody trained.
    """
    restore_from = config.checkpoint.restore_from
    if restore_from is None:
        return None
    if restore_from.upper() == "LATEST":
        checkpoint_dir = config.checkpoint.checkpoint_dir
        resolved = checkpointer.find_latest_checkpoint(checkpoint_dir)
        if resolved is None:
            raise FileNotFoundError(
                f"checkpoint.restore_from='LATEST' but no checkpoint was found under "
                f"{checkpoint_dir!r}"
            )
        logger.info("Resolved LATEST checkpoint: %s", resolved)
        return resolved
    import os  # pylint: disable=import-outside-toplevel

    if not os.path.isdir(restore_from):
        raise FileNotFoundError(f"Checkpoint directory not found: {restore_from}")
    return restore_from


def _step_from_path(path: str) -> Optional[int]:
    """Recover the optimizer step encoded in a checkpoint directory name.

    Saved step directories are named with :data:`STEP_PREFIX` followed by the
    step number. Any other layout yields ``None`` rather than a guess.
    """
    from hyper_parallel.components.checkpoint.dcp_checkpointer import (  # pylint: disable=import-outside-toplevel
        STEP_PREFIX,
    )

    name = path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if not name.startswith(STEP_PREFIX):
        return None
    digits = name[len(STEP_PREFIX):]
    return int(digits) if digits.isdigit() else None


def restore_model_weights(
        model: Any,
        config: Any,
        mesh_context: Any,
        *,
        strict: bool = True,
) -> RestoreOutcome:
    """Restore model weights into an already-materialized, parallelized model.

    Args:
        model: Model produced by the shared build entry point; its parameters
            must already be laid out by every parallel strategy it will run
            under.
        config: Resolved trainer configuration providing the ``checkpoint``
            section.
        mesh_context: Resolved mesh, used to build the external-state runtime.
        strict: Require every requested model key to exist in the checkpoint.
            Disable only for adapter-only (PEFT) checkpoints.

    Returns:
        The restore outcome; ``path`` is ``None`` when no restore was requested.

    Raises:
        FileNotFoundError: ``restore_from`` names a checkpoint that does not
            exist or cannot be resolved.
        RuntimeError: The loaded state does not match the model's keys, or a
            model-external sidecar cannot be restored without an optimizer.
    """
    checkpointer = build_checkpointer(
        extra_state_per_rank=bool(config.checkpoint.save_extra_state_per_rank),
    )
    path = _resolve_path(config, checkpointer)
    if path is None:
        logger.info("No checkpoint to restore (checkpoint.restore_from is None).")
        return RestoreOutcome()

    logger.info("Restoring model weights from %s", path)
    external = get_model_external_state(model)
    runtime = None
    load_plan = None
    if external is not None:
        runtime = CheckpointRuntime(
            step_dir=path,
            global_step=None,
            mesh_context=mesh_context,
            include_optimizer=False,
            include_train_state=False,
        )
        load_plan = external.before_checkpoint_load(runtime)

    # The skeleton is the full model state dict minus external parameters: the
    # loaders fill only keys the caller declares, so the skeleton must be
    # complete before the call.
    state_dict = model.state_dict()
    if external is not None:
        missing = set(external.parameters) - set(state_dict)
        if missing:
            raise RuntimeError(f"External model FQNs missing from state_dict: {sorted(missing)}")
        dense_state = {key: value for key, value in state_dict.items() if key not in external.parameters}
    else:
        dense_state = dict(state_dict)

    payload: Dict[str, Any] = {"model": dense_state}
    loaded = checkpointer.load(path, payload, strict_model=strict)
    load_result = model.load_state_dict(loaded["model"], strict=strict)
    if external is not None and load_result is not None:
        if set(load_result.missing_keys) != set(external.parameters) or load_result.unexpected_keys:
            raise RuntimeError(
                "Dense checkpoint model keys differ from exact external exclusions: "
                f"missing={sorted(load_result.missing_keys)[:8]}, "
                f"unexpected={sorted(load_result.unexpected_keys)[:8]}"
            )

    validate_model_init_dtype(model, config.model_init_dtype)

    if external is not None:
        # A sidecar that needs optimizer state to rehydrate cannot be honoured
        # by a model-only task; failing here is better than evaluating half a
        # model.
        try:
            external.after_checkpoint_load(runtime, load_plan)
            external.after_weights_only_load()
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                "model-external state cannot be restored without an optimizer; "
                "this checkpoint needs a full training-context restore"
            ) from exc

    return RestoreOutcome(
        path=path,
        step=_step_from_path(path),
        external_applied=external is not None,
    )
