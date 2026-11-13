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
"""Run-mode validation and assembly planning for standalone tasks.

Two checks share this module:

* :func:`validate_run_config` is a pure configuration check that runs before
  distributed initialization. It only rejects combinations that are already
  known to be unsupported or self-contradictory.
* :func:`build_steps_for` returns the explicit assembly plan a concrete
  Trainer consumes, so skipping training-only components is a declared
  decision rather than an incidental side effect of a shorter ``__init__``.

Capability checks that need the resolved mesh or model (pipeline width, TP/CP
statistics domains, checkpoint layout) live in
:func:`validate_run_capabilities`, which the Trainer calls after those objects
exist.
"""

from __future__ import annotations

__all__ = [
    "BuildStep",
    "build_steps_for",
    "validate_run_capabilities",
    "validate_run_config",
]

from typing import Any

from hyper_parallel.trainer.config.evaluation import EvaluationConfig


class BuildStep:
    """Names of the Trainer assembly steps a run mode may execute."""

    SETUP = "setup"
    MODEL = "model"
    LOSS = "loss"
    MODEL_ASSETS = "model_assets"
    DATA_TRANSFORM = "data_transform"
    DATASET = "dataset"
    EVAL_DATA = "eval_data"
    COLLATE = "collate"
    DATALOADER = "dataloader"
    EVAL_BATCH_PLAN = "eval_batch_plan"
    GET_BATCH = "get_batch"
    MODEL_RESTORE = "model_restore"
    TRAIN_ITERS = "train_iters"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"
    TRAINING_CONTEXT = "training_context"
    GRAPH_COMPILER = "graph_compiler"
    CALLBACKS = "callbacks"


#: Assembly steps every run mode shares.
_COMMON_STEPS = (
    BuildStep.SETUP,
    BuildStep.MODEL,
    BuildStep.MODEL_ASSETS,
)

#: Training-only steps: absent from both standalone plans.
_TRAINING_ONLY_STEPS = (
    BuildStep.OPTIMIZER,
    BuildStep.LR_SCHEDULER,
    BuildStep.TRAINING_CONTEXT,
    BuildStep.GRAPH_COMPILER,
    BuildStep.CALLBACKS,
)

_STEPS_BY_MODE: dict[str, tuple[str, ...]] = {
    "train": (
        *_COMMON_STEPS,
        BuildStep.LOSS,
        BuildStep.DATA_TRANSFORM,
        BuildStep.DATASET,
        BuildStep.COLLATE,
        BuildStep.DATALOADER,
        BuildStep.GET_BATCH,
        BuildStep.TRAIN_ITERS,
        *_TRAINING_ONLY_STEPS,
    ),
    "eval": (
        *_COMMON_STEPS,
        BuildStep.LOSS,
        BuildStep.DATA_TRANSFORM,
        BuildStep.DATASET,
        BuildStep.COLLATE,
        BuildStep.DATALOADER,
        BuildStep.EVAL_BATCH_PLAN,
        BuildStep.GET_BATCH,
        BuildStep.MODEL_RESTORE,
    ),    "predict": (
        *_COMMON_STEPS,
        BuildStep.MODEL_RESTORE,
    ),
}


def build_steps_for(run_mode: str, evaluation_mode: str | None = None) -> tuple[str, ...]:
    """Return the ordered assembly steps a run mode executes.

    Args:
        run_mode: ``train``, ``eval``, or ``predict``.
        evaluation_mode: ``loss`` or ``generation``; reserved for the
            generation-based evaluator, which adds no assembly step in P0.

    Returns:
        The step names in execution order.

    Raises:
        ValueError: ``run_mode`` is not a supported mode.
    """
    del evaluation_mode  # P0 has a single evaluation assembly plan.
    try:
        return _STEPS_BY_MODE[run_mode]
    except KeyError as exc:
        raise ValueError(
            f"run_mode must be one of: train, eval, predict; got {run_mode!r}"
        ) from exc


def _reject(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when a validated condition does not hold."""
    if condition:
        raise ValueError(message)


def validate_run_config(config: Any) -> None:
    """Validate run-mode configuration before distributed initialization.

    This is a pure function over the resolved ``TrainerConfig``: it must not
    initialize a process group, build a model, or touch the filesystem.

    Args:
        config: Resolved trainer configuration.

    Raises:
        ValueError: The configuration is invalid for its run mode.
    """
    run_mode = config.run_mode
    build_steps_for(run_mode)
    evaluation: EvaluationConfig = config.evaluation

    if run_mode == "train":
        _reject(
            config.optimizer is None,
            "run_mode='train' requires an 'optimizer' configuration section",
        )
        _reject(
            evaluation.mode == "generation",
            "evaluation.mode='generation' is not supported in a training run",
        )
        return

    # ---- standalone eval -------------------------------------------------
    if run_mode == "eval":
        if evaluation.mode == "generation":
            raise NotImplementedError(
                "evaluation.mode='generation' is reserved for the generation-based "
                "evaluator and is not implemented yet; configure mode='loss'"
            )
        if not evaluation.metrics:
            raise ValueError(
                "run_mode='eval' requires at least one entry in evaluation.metrics"
            )
        if config.dataset is None and evaluation.dataset is None:
            raise ValueError(
                "run_mode='eval' requires a 'dataset' section (or evaluation.dataset)"
            )
        if config.dataloader is None and evaluation.dataloader is None:
            raise ValueError(
                "run_mode='eval' requires a 'dataloader' section (or evaluation.dataloader)"
            )
        _reject(
            config.compile.enabled,
            "run_mode='eval' does not support compile.enabled; the evaluator runs "
            "in eager mode under torch.no_grad()",
        )
        _reject(
            getattr(config.activation_checkpoint, "mode", "off") not in (None, "off"),
            "run_mode='eval' does not enable activation checkpointing; there is no "
            "backward pass to recompute for",
        )
        _reject(
            config.activation_swap != "none",
            "run_mode='eval' does not enable activation swap",
        )
        if evaluation.max_samples is None and config.training.eval_iters <= 0:
            raise ValueError(
                "run_mode='eval' needs a finite data range: set "
                "evaluation.max_samples, or training.eval_iters > 0"
            )
        return

    # ---- standalone predict ----------------------------------------------
    if run_mode == "predict":
        raise NotImplementedError(
            "run_mode='predict' is not implemented yet; the offline Predictor "
            "lands in P1"
        )


def validate_run_capabilities(config: Any, mesh_context: Any) -> None:
    """Reject topologies and weights the current run mode cannot execute.

    Called after distributed setup and model construction, when the mesh and
    the resolved checkpoint are known.

    Args:
        config: Resolved trainer configuration.
        mesh_context: Resolved mesh context.

    Raises:
        NotImplementedError: The topology is not supported for this run mode.
    """
    if config.run_mode != "eval":
        return

    unsupported = {
        "pp_size": getattr(mesh_context, "pp_size", 1),
        "tp_size": getattr(mesh_context, "tp_size", 1),
        "cp_size": getattr(mesh_context, "cp_size", 1),
        "ep_size": getattr(mesh_context, "ep_size", 1),
    }
    enabled = {name: size for name, size in unsupported.items() if size > 1}
    if enabled:
        raise NotImplementedError(
            "run_mode='eval' currently supports dense data parallelism only; "
            f"got {enabled}. Loss statistics domains for TP/CP/EP/PP require "
            "their own numerical acceptance, and PP additionally needs a "
            "forward-only schedule (P3)"
        )
