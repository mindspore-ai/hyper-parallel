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
"""Finalization of deferred pretrained loading for HyperParallel models.

Reading the weights is
:class:`~hyper_parallel.components.checkpoint.huggingface_checkpointer.HuggingFaceCheckpointer`'s
job. What is here runs after it, on the :class:`LoadReport` it produced: tensors the checkpoint
covered are marked initialized and checked for having survived distributed wrapping unchanged, tied
aliases are resolved, and everything the checkpoint did not cover is handed to the model's own
``initialize_weights()``.

:func:`load_pretrained_weights` chains the two for a caller that owns an already-finalized model,
and :class:`CheckpointManager` exports such a model back to Transformers format.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed import is_available, is_initialized

from hyper_parallel import DTensor
from hyper_parallel.components.checkpoint.huggingface_checkpointer import (
    HuggingFaceCheckpointer,
)
from hyper_parallel.components.checkpoint.load_groups import (
    LoadReport,
    join_fqn,
    local_target_tensor,
)
from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion

# PEP 604 unions (``str | Path``, ``int | None``) are valid on this interpreter, but the checker
# evaluates them and reports a false ``unsupported-binary-operation`` at every site.
# pylint: disable=unsupported-binary-operation

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Export one finalized model as a Transformers checkpoint.

    Loading moved to
    :class:`~hyper_parallel.components.checkpoint.huggingface_checkpointer.HuggingFaceCheckpointer`.
    What stays here is the way back out, which reads the conversions loading recorded on the model
    to restore the released tensor names and layouts.
    """

    def __init__(self, model: nn.Module) -> None:
        """Bind the manager to one finalized model."""
        self.model = model

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        max_shard_size: int | str = "5GB",
        save_original_format: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Gather model weights and save a Transformers-compatible checkpoint.

        All distributed ranks must call this method. Collectives produce each
        full tensor on every rank, but only rank 0 retains CPU weights and
        writes files.

        Returns:
            True on the writing rank and False on all other ranks.
        """
        save_method = getattr(self.model, "save_pretrained", None)
        if not callable(save_method):
            raise TypeError("CheckpointManager.save_pretrained requires a Transformers model")
        is_main_process = self._is_main_process()
        state_dict = self._gather_full_state_dict(keep_state_dict=is_main_process)
        if not is_main_process:
            return False
        used_base = getattr(
            self.model, "_hp_used_base_weight_conversions", None
        )
        used_replacements = getattr(
            self.model, "_hp_used_replacement_weight_conversions", None
        )
        if save_original_format and used_replacements:
            original_mapping = getattr(self.model, "_weight_conversions", None)
            try:
                self.model._weight_conversions = used_replacements  # pylint: disable=protected-access
                state_dict = revert_weight_conversion(self.model, state_dict)
                if used_base:
                    self.model._weight_conversions = used_base  # pylint: disable=protected-access
                    state_dict = revert_weight_conversion(self.model, state_dict)
            finally:
                self.model._weight_conversions = original_mapping  # pylint: disable=protected-access
            save_original_format = False
        save_method(
            save_directory,
            state_dict=state_dict,
            is_main_process=True,
            max_shard_size=max_shard_size,
            save_original_format=save_original_format,
            **kwargs,
        )
        return True

    def _gather_full_state_dict(self, *, keep_state_dict: bool) -> dict[str, Any]:
        """Gather sharded model tensors into a full CPU state dictionary."""

        state_dict = self.model.state_dict(keep_vars=True)
        gathered = {}
        for name, value in state_dict.items():
            if isinstance(value, DTensor):
                value = value.full_tensor()
            elif isinstance(value, torch.Tensor):
                layout = getattr(value, "_sharding_spec", None)
                if layout is not None:
                    value = DTensor.from_local_with_layout(value.detach(), layout).full_tensor()
                else:
                    value = value.detach()
            if keep_state_dict:
                gathered[name] = value.cpu() if isinstance(value, torch.Tensor) else value
        return gathered

    @staticmethod
    def _is_main_process() -> bool:
        return not (is_available() and is_initialized()) or torch.distributed.get_rank() == 0


# ── Checkpoint finalization (moved from infrastructure.py, 05 §15.2.1) ──


@dataclass(frozen=True)
class _FinalizeTarget:
    """Registered model tensor considered during pretrained finalization."""

    fqn: str
    module: nn.Module
    tensor_name: str
    tensor: torch.Tensor
    is_parameter: bool
    is_non_persistent: bool


@dataclass(frozen=True)
class _TargetSnapshot:
    """Identity and layout invariants for a loaded model tensor."""

    tensor_id: int
    local_tensor_id: int
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    layout_id: int | None


def _build_finalize_targets(model: nn.Module) -> dict[str, _FinalizeTarget]:
    """Build a registry including persistent and non-persistent model state."""
    targets = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        for tensor_name, parameter in module._parameters.items():  # pylint: disable=W0212
            if parameter is None:
                continue
            fqn = join_fqn(module_name, tensor_name)
            targets[fqn] = _FinalizeTarget(fqn, module, tensor_name, parameter, True, False)

        non_persistent = module._non_persistent_buffers_set  # pylint: disable=W0212
        for tensor_name, buffer in module._buffers.items():  # pylint: disable=W0212
            if buffer is None:
                continue
            fqn = join_fqn(module_name, tensor_name)
            targets[fqn] = _FinalizeTarget(
                fqn,
                module,
                tensor_name,
                buffer,
                False,
                tensor_name in non_persistent,
            )
    return targets


def _validate_materialized(targets: list[_FinalizeTarget]) -> None:
    """Reject model state that remains on meta after storage materialization."""
    meta_keys = [target.fqn for target in targets if local_target_tensor(target.tensor).is_meta]
    if meta_keys:
        preview = ", ".join(meta_keys[:10])
        raise ValueError(
            f"Model finalization found {len(meta_keys)} tensors on meta device; first keys: {preview}"
        )


def _snapshot_target(target: _FinalizeTarget) -> _TargetSnapshot:
    """Capture identity and shape invariants without copying tensor data."""
    local_tensor = local_target_tensor(target.tensor)
    layout = getattr(target.tensor, "layout", None)
    if layout is None:
        layout = getattr(target.tensor, "_sharding_spec", None)
    return _TargetSnapshot(
        tensor_id=id(target.tensor),
        local_tensor_id=id(local_tensor),
        global_shape=tuple(target.tensor.shape),
        local_shape=tuple(local_tensor.shape),
        layout_id=id(layout) if layout is not None else None,
    )


def _mark_loaded_targets_initialized(
    targets: dict[str, _FinalizeTarget],
    loaded_keys: set[str],
) -> dict[str, _TargetSnapshot]:
    """Mark loaded tensors initialized and retain their structural invariants."""
    snapshots = {}
    unknown_keys = sorted(loaded_keys - targets.keys())
    if unknown_keys:
        preview = ", ".join(unknown_keys[:10])
        raise ValueError(
            f"Load report contains {len(unknown_keys)} model keys not registered after wrapping; first keys: {preview}"
        )
    loaded_targets = [targets[key] for key in loaded_keys]
    _validate_materialized(loaded_targets)
    for target in loaded_targets:
        target.tensor._is_hf_initialized = True  # pylint: disable=W0212
        target.module._is_hf_initialized = True  # pylint: disable=W0212
        snapshots[target.fqn] = _snapshot_target(target)
    return snapshots


def _shares_local_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two tensors represent the same local parameter storage."""
    if first is second:
        return True
    first_local = local_target_tensor(first)
    second_local = local_target_tensor(second)
    return (
        first_local.device == second_local.device
        and first_local.untyped_storage().data_ptr() == second_local.untyped_storage().data_ptr()
    )


def _resolve_tied_aliases(
    model: nn.Module,
    targets: dict[str, _FinalizeTarget],
    loaded_keys: set[str],
) -> set[str]:
    """Mark and validate loaded aliases of tied model parameters."""
    tied_mapping = getattr(model, "all_tied_weights_keys", {}) or {}
    initialized_aliases = set()
    for target_name, source_name in tied_mapping.items():
        target = targets.get(target_name)
        source = targets.get(source_name)
        if target is None or source is None:
            continue
        if target_name not in loaded_keys and source_name not in loaded_keys:
            continue
        if tuple(target.tensor.shape) != tuple(source.tensor.shape):
            raise ValueError(
                f"Tied parameters must have matching shapes: {target_name}={tuple(target.tensor.shape)} vs "
                f"{source_name}={tuple(source.tensor.shape)}"
            )
        if not _shares_local_storage(target.tensor, source.tensor):
            raise ValueError(
                f"Tied parameters no longer share local storage after distributed wrapping: "
                f"{target_name} and {source_name}"
            )
        target.tensor._is_hf_initialized = True  # pylint: disable=W0212
        source.tensor._is_hf_initialized = True  # pylint: disable=W0212
        target.module._is_hf_initialized = True  # pylint: disable=W0212
        source.module._is_hf_initialized = True  # pylint: disable=W0212
        initialized_aliases.update((target_name, source_name))
    return initialized_aliases


def _matches_any_pattern(key: str, patterns: set[str]) -> bool:
    """Return whether a model state key matches any configured regex pattern."""
    return any(re.search(pattern, key) is not None for pattern in patterns)


def _adjust_loading_keys(
    model: nn.Module,
    missing_keys: set[str],
    unexpected_keys: set[str],
) -> tuple[set[str], set[str], int, int]:
    """Apply Transformers-compatible ignore patterns to loading results."""
    missing_patterns = set(getattr(model, "_keys_to_ignore_on_load_missing", None) or set())
    unexpected_patterns = set(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or set())
    adjusted_missing = {
        key for key in missing_keys if not _matches_any_pattern(key, missing_patterns)
    }
    adjusted_unexpected = {
        key for key in unexpected_keys if not _matches_any_pattern(key, unexpected_patterns)
    }
    return (
        adjusted_missing,
        adjusted_unexpected,
        len(missing_keys) - len(adjusted_missing),
        len(unexpected_keys) - len(adjusted_unexpected),
    )


def _prepare_initialization_targets(
    model: nn.Module,
    targets: list[_FinalizeTarget],
) -> None:
    """Reopen only modules that own missing or non-persistent state."""
    for module in model.modules():
        module._is_hf_initialized = True  # pylint: disable=W0212
    owner_modules = {}
    for target in targets:
        target.tensor._is_hf_initialized = False  # pylint: disable=W0212
        owner_modules[id(target.module)] = target.module
    for module in owner_modules.values():
        module._is_hf_initialized = False  # pylint: disable=W0212


def _initialize_model_state_after_loading(model: nn.Module) -> None:
    """Run the guarded Transformers initialization entry point."""
    initialize_weights = getattr(model, "initialize_weights", None)
    if not callable(initialize_weights):
        raise ValueError(
            "Deferred pretrained loading requires a callable model.initialize_weights()"
        )
    initialize_weights()


def _validate_loaded_target_snapshots(
    targets: dict[str, _FinalizeTarget],
    snapshots: dict[str, _TargetSnapshot],
) -> None:
    """Ensure finalization did not replace or reshape loaded distributed state."""
    for key, expected in snapshots.items():
        target = targets[key]
        actual = _snapshot_target(target)
        if actual != expected:
            raise ValueError(
                f"Model finalization replaced or reshaped loaded tensor {key}: "
                f"expected={expected}, actual={actual}"
            )
        if not getattr(target.tensor, "_is_hf_initialized", False):
            raise ValueError(f"Loaded tensor lost its initialized marker during finalization: {key}")


def _validate_initialization_targets(targets: list[_FinalizeTarget]) -> None:
    """Ensure requested model state was initialized by the model contract."""
    uninitialized = [
        target.fqn
        for target in targets
        if not getattr(target.tensor, "_is_hf_initialized", False)
    ]
    if uninitialized:
        preview = ", ".join(uninitialized[:10])
        raise ValueError(
            f"Model initialize_weights() left {len(uninitialized)} tensors uninitialized; first keys: {preview}"
        )


def _validate_missing_initialization_targets(initialization_targets, missing_keys):
    """Reject missing distributed parameters requiring DTensor-aware initialization."""
    missing_sharded = [
        target.fqn
        for target in initialization_targets
        if target.fqn in missing_keys and isinstance(target.tensor, DTensor)
    ]
    if missing_sharded:
        preview = ", ".join(sorted(missing_sharded)[:10])
        raise ValueError(
            f"Missing distributed parameters require DTensor-aware initialization; first keys: {preview}"
        )


def _finalize_model_loading(
    model: nn.Module,
    load_report: LoadReport,
    *,
    strict: bool,
) -> LoadReport:
    """Finalize deferred pretrained loading without replacing distributed parameters."""
    targets = _build_finalize_targets(model)
    loaded_keys = set(load_report.loaded_keys)
    missing_keys = set(load_report.missing_keys)
    unexpected_keys = set(load_report.unexpected_keys)
    loaded_snapshots = _mark_loaded_targets_initialized(targets, loaded_keys)

    tied_aliases = _resolve_tied_aliases(model, targets, loaded_keys)
    missing_keys.difference_update(tied_aliases)
    missing_keys, unexpected_keys, ignored_missing, ignored_unexpected = _adjust_loading_keys(
        model,
        missing_keys,
        unexpected_keys,
    )
    if strict and missing_keys:
        preview = ", ".join(sorted(missing_keys)[:10])
        raise RuntimeError(
            f"Checkpoint did not load {len(missing_keys)} owned model tensors after finalization; "
            f"first keys: {preview}"
        )

    initialization_targets = [
        target
        for target in targets.values()
        if not target.is_non_persistent and target.fqn in missing_keys
    ]
    _validate_missing_initialization_targets(initialization_targets, missing_keys)
    _validate_materialized(initialization_targets)
    _prepare_initialization_targets(model, initialization_targets)
    if initialization_targets:
        _initialize_model_state_after_loading(model)
        for target in initialization_targets:
            target.tensor._is_hf_initialized = True  # pylint: disable=W0212
    _validate_loaded_target_snapshots(targets, loaded_snapshots)
    _validate_initialization_targets(initialization_targets)
    _validate_materialized(list(targets.values()))
    _resolve_tied_aliases(model, targets, loaded_keys)

    finalized_report = LoadReport(
        loaded_keys=tuple(sorted(loaded_keys)),
        missing_keys=tuple(sorted(missing_keys)),
        unexpected_keys=tuple(sorted(unexpected_keys)),
    )
    logger.info(
        "Finalized pretrained model state: loaded=%d initialized=%d ignored_missing=%d "
        "ignored_unexpected=%d unresolved_missing=%d unexpected=%d",
        len(finalized_report.loaded_keys),
        len(initialization_targets),
        ignored_missing,
        ignored_unexpected,
        len(finalized_report.missing_keys),
        len(finalized_report.unexpected_keys),
    )
    return finalized_report


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: str | Path,
    *,
    strict: bool = True,
) -> LoadReport:
    """Load Hugging Face pretrained weights into an already-finalized model.

    The model build path (``apply_model_infrastructure``) stages the checkpoint and
    finalizes it internally; this wrapper is for callers that own a finalized model
    -- for example a demo builder that constructs with ``from_config`` for a random
    init and then wants the real weights instead.  Staging alone is not enough:
    :func:`_finalize_model_loading` performs the deferred copy and marks the loaded
    tensors initialized, so both steps belong here. The loader is picked the same
    way as on the build path, through ``HYPER_PARALLEL_HF_LOADER``.

    Args:
        model: Finalized (sharded and materialized) model to load into.
        pretrained_path: Local Hugging Face checkpoint directory or file.
        strict: Raise when an owned model tensor is left unloaded; a partial load
            is a silently half-random model, so this defaults to ``True``.

    Returns:
        The load report, after finalization.

    Raises:
        RuntimeError: If ``strict`` is set and owned tensors are missing.
    """
    state = {"model": model}
    HuggingFaceCheckpointer().load(str(pretrained_path), state, strict_model=False)
    return _finalize_model_loading(model, state["load_report"], strict=strict)
