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

Configuration checks and initialization planning share this module:

* :func:`validate_run_config` is a pure configuration check that runs before
  distributed initialization. It only rejects combinations that are already
  known to be unsupported or self-contradictory.
* :func:`get_build_steps` returns the component initialization steps enabled
  for a run mode. The Trainer uses these names to select which components to
  build. Selecting the names does not execute those initialization steps.

Capability checks that need the resolved mesh or model (pipeline width,
parameter sharding, sequence parallelism) live in
:func:`validate_run_capabilities`, which the Trainer calls after those objects
exist.
"""

from __future__ import annotations

__all__ = [
    "BuildStep",
    "get_build_steps",
    "validate_run_capabilities",
    "validate_run_config",
]

from typing import Any


class BuildStep:
    """Names of component initialization steps enabled by each run mode."""

    SETUP = "setup"
    MODEL = "model"
    LOSS = "loss"
    MODEL_ASSETS = "model_assets"
    DATA_TRANSFORM = "data_transform"
    DATASET = "dataset"
    COLLATE = "collate"
    DATALOADER = "dataloader"
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

#: Training-only steps: absent from the prediction plan.
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
    "predict": (
        *_COMMON_STEPS,
        BuildStep.MODEL_RESTORE,
    ),
}


def get_build_steps(run_mode: str) -> tuple[str, ...]:
    """Return the component initialization steps enabled for a run mode.

    This function selects step names without building any components. The
    Trainer owns their execution order; they do not describe generation rounds.

    Args:
        run_mode: ``train`` or ``predict``.

    Returns:
        The enabled component initialization step names.

    Raises:
        ValueError: ``run_mode`` is not a supported mode.
    """
    try:
        return _STEPS_BY_MODE[run_mode]
    except KeyError as exc:
        raise ValueError(
            f"run_mode must be one of: train, predict; got {run_mode!r}"
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
    get_build_steps(run_mode)

    if run_mode == "train":
        _reject(
            config.optimizer is None,
            "run_mode='train' requires an 'optimizer' configuration section",
        )
        return

    # ---- standalone predict ----------------------------------------------
    if run_mode == "predict":
        prediction = config.prediction
        _reject(not prediction.input, "run_mode='predict' requires prediction.input")
        _reject(not prediction.output_file, "run_mode='predict' requires prediction.output_file")
        _reject(config.compile.enabled, "run_mode='predict' does not support compile.enabled")
        _reject(
            getattr(config.activation_checkpoint, "mode", "off") not in (None, "off"),
            "run_mode='predict' does not enable activation checkpointing",
        )
        _reject(
            getattr(getattr(config, "hyper_offload", None), "enable", False),
            "run_mode='predict' does not enable training hyper_offload",
        )
        availability = getattr(config, "high_availability", None)
        recovery = getattr(availability, "worker_recovery", None)
        _reject(
            getattr(recovery, "enable", False),
            "run_mode='predict' does not enable training worker recovery",
        )
        validate_run_capabilities(config, getattr(config, "accelerator", None))


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
    if config.run_mode != "predict":
        return

    unsupported = {
        "pp_size": getattr(mesh_context, "pp_size", 1),
        "tp_size": getattr(mesh_context, "tp_size", 1),
        "cp_size": getattr(mesh_context, "cp_size", 1),
        "ep_size": getattr(mesh_context, "ep_size", 1),
    }
    fsdp = getattr(config, "fsdp_config", None)
    unsupported["dp_shard_size"] = getattr(mesh_context, "dp_shard_size", None) or getattr(fsdp, "dp_shard_size", 1)
    unsupported["edp_shard_size"] = (
        getattr(mesh_context, "edp_shard_size", None) or getattr(fsdp, "edp_shard_size", 1)
    )
    if getattr(mesh_context, "sequence_parallel", False):
        raise NotImplementedError(f"run_mode={config.run_mode!r} does not support sequence parallelism")
    enabled = {name: size for name, size in unsupported.items() if size is not None and size > 1}
    if enabled:
        raise NotImplementedError(
            f"run_mode={config.run_mode!r} currently supports independent data parallelism; "
            f"got {enabled}. Sharded or model-parallel generation requires "
            "a compatible decoding schedule and separate numerical acceptance"
        )
