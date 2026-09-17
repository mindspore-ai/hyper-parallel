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
"""Checkpoint management for finalized HyperParallel models."""

import logging
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from torch.distributed import is_available, is_initialized
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    dot_natural_key,
    get_model_conversion_mapping,
    revert_weight_conversion,
)

from hyper_parallel import DTensor
from hyper_parallel.models._transformers.checkpoint_conversion import (
    CheckpointIndex,
    LoadGroup,
    LoadReport,
    SourceModelView,
    alias_names_by_target,
    base_weights_mapping,
    build_load_groups,
    build_load_targets,
    build_replacement_routes,
    convert_group,
    copy_into_target,
    join_fqn,
    local_target_tensor,
    make_tensor_loader,
    resolve_checkpoint_index,
    validate_load_result,
)

logger = logging.getLogger(__name__)

# Picks the pretrained loader when CheckpointManager.load_checkpoint is not told which one to use.
HF_LOADER_ENV = "HYPER_PARALLEL_HF_LOADER"
_HF_LOADERS = ("legacy", "dcp")


def resolve_hf_loader(loader: str | None = None) -> str:
    """
    Name the pretrained loader to use.

    Args:
        loader (str | None): ``"legacy"`` reads whole checkpoint tensors on every rank and shards them in
            memory. ``"dcp"`` plans the same conversions as distributed checkpoint reads, so that each
            rank reads only the regions of the checkpoint its shards need. Default None, which reads
            ``HYPER_PARALLEL_HF_LOADER`` and falls back to ``"legacy"``.

    Returns:
        str: ``"legacy"`` or ``"dcp"``.

    Raises:
        ValueError: If the loader named is neither.
    """
    choice = (loader or os.environ.get(HF_LOADER_ENV) or "legacy").strip().lower()
    if choice not in _HF_LOADERS:
        raise ValueError(
            f"Unknown pretrained loader {choice!r}; expected one of {', '.join(_HF_LOADERS)} "
            f"(set through the loader argument or {HF_LOADER_ENV})"
        )
    return choice


class DCPBackend(Protocol):
    """Contract implemented by the distributed-checkpoint subsystem."""

    def load(
        self,
        state_dict: dict[str, Any],
        *,
        checkpoint_id: str | Path,
        **kwargs: Any,
    ) -> Any:
        """Load a DCP checkpoint into the supplied sharded state dict."""

    def save(
        self,
        state_dict: dict[str, Any],
        *,
        checkpoint_id: str | Path,
        **kwargs: Any,
    ) -> Any:
        """Save the supplied sharded state dict as DCP."""


class CheckpointManager:
    """Manage pretrained and resumable checkpoints for one finalized model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        dcp_backend: DCPBackend | None = None,
    ) -> None:
        """Bind the manager to one finalized model and an optional DCP backend."""
        self.model = model
        self.dcp_backend = dcp_backend

    def load_checkpoint(
        self,
        pretrained_path: str,
        *,
        strict: bool = True,
        weights_mapping: list[WeightRenaming | WeightConverter] | None = None,
        loader: str | None = None,
    ) -> LoadReport:
        """
        Load complete Hugging Face weights into the finalized model.

        Args:
            pretrained_path (str): A safetensors file, a checkpoint directory or a Hub repository id.
            strict (bool): Raise if a model tensor is left unloaded. Default True.
            weights_mapping (list[WeightRenaming | WeightConverter] | None): Rules renaming and converting
                checkpoint tensors. Default None, for the model's Transformers conversion mapping.
            loader (str | None): Which loader reads the weights; see :func:`resolve_hf_loader`.
                Default None.

        Returns:
            LoadReport: What was loaded, what is missing and what the checkpoint holds beyond the model.
        """
        if not pretrained_path:
            raise ValueError("pretrained_path must be provided when load_base_model=True")
        if resolve_hf_loader(loader) == "dcp":
            # Imported here: the planner module imports this one.
            from hyper_parallel.models._transformers.hf_load_planner import (  # pylint: disable=C0415
                load_hf_checkpoint,
            )

            return load_hf_checkpoint(
                self.model, pretrained_path, weights_mapping=weights_mapping, strict=strict
            )
        if weights_mapping is None:
            weights_mapping = get_model_conversion_mapping(
                self.model,
                key_mapping=None,
                hf_quantizer=None,
            )

        checkpoint_index = resolve_checkpoint_index(pretrained_path)
        targets = build_load_targets(self.model)
        replacement_mapping = getattr(
            self.model,
            "_hp_replacement_weight_conversions",
            None,
        )
        source_shapes = getattr(
            self.model,
            "_hp_checkpoint_source_shapes",
            None,
        )
        if replacement_mapping and source_shapes:
            return self._load_with_replacement_conversions(
                checkpoint_index,
                targets,
                weights_mapping,
                replacement_mapping,
                source_shapes,
                pretrained_path,
                strict,
            )
        groups, unexpected_keys, weight_mapping = build_load_groups(
            self.model,
            checkpoint_index.keys(),
            targets,
            weights_mapping=weights_mapping,
            make_loader=partial(make_tensor_loader, checkpoint_index),
        )
        aliases_by_target = alias_names_by_target(targets)
        loaded_keys = set()
        loaded_target_ids = set()

        for group in groups:
            converted = self._convert_group(group)
            for target_name, tensor in converted.items():
                target = targets.get(target_name)
                if target is None:
                    unexpected_keys += (target_name,)
                    continue
                tensor = tensor[0] if isinstance(tensor, list) else tensor
                target_id = id(target)
                if target_id not in loaded_target_ids:
                    copy_into_target(target_name, tensor, target)
                    loaded_target_ids.add(target_id)
                loaded_keys.update(aliases_by_target[target_id])

        missing_keys = tuple(sorted(set(targets) - loaded_keys, key=dot_natural_key))
        unexpected_keys = tuple(sorted(set(unexpected_keys), key=dot_natural_key))
        self._validate_load_result(missing_keys, unexpected_keys, strict)
        used_conversions = [transform for transform in weight_mapping if transform.was_used()]
        self.model._weight_conversions = used_conversions  # pylint: disable=W0212
        report = LoadReport(
            loaded_keys=tuple(sorted(loaded_keys, key=dot_natural_key)),
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
        logger.info(
            "Loaded %d model tensors from %s",
            len(report.loaded_keys),
            pretrained_path,
        )
        return report

    def _load_with_replacement_conversions(
        self,
        checkpoint_index: CheckpointIndex,
        targets: dict[str, torch.Tensor],
        weights_mapping: list[WeightRenaming | WeightConverter],
        replacement_mapping: list[WeightRenaming | WeightConverter],
        source_shapes: dict[str, tuple[int, ...]],
        pretrained_path: str,
        strict: bool,
    ) -> LoadReport:
        """Normalize original weights before applying replacement conversions."""
        base_mapping = base_weights_mapping(weights_mapping, replacement_mapping)
        source_model = SourceModelView(self.model, source_shapes)
        base_groups, unexpected_keys, _ = build_load_groups(
            source_model,
            checkpoint_index.keys(),
            source_model.targets,
            weights_mapping=base_mapping,
            make_loader=partial(make_tensor_loader, checkpoint_index),
        )
        routes = build_replacement_routes(
            self.model,
            tuple(source_shapes),
            targets,
            replacement_mapping,
        )
        aliases_by_target = alias_names_by_target(targets)
        loaded_keys = set()
        loaded_target_ids = set()
        used_replacements = []

        def copy_converted(converted: dict[str, torch.Tensor]) -> None:
            """Copy converted tensors into their finalized model targets."""
            nonlocal unexpected_keys
            for target_name, tensor in converted.items():
                target = targets.get(target_name)
                if target is None:
                    unexpected_keys += (target_name,)
                    continue
                tensor = tensor[0] if isinstance(tensor, list) else tensor
                target_id = id(target)
                if target_id not in loaded_target_ids:
                    copy_into_target(target_name, tensor, target)
                    loaded_target_ids.add(target_id)
                loaded_keys.update(aliases_by_target[target_id])

        for base_group in base_groups:
            normalized = self._convert_group(base_group, model=source_model)
            for source_name, tensor in normalized.items():
                tensor = tensor[0] if isinstance(tensor, list) else tensor
                route = routes.get(source_name)
                if route is None:
                    copy_converted({source_name: tensor})
                    continue
                state, target_name, source_pattern = route
                state.group.transform.add_tensor(
                    target_name,
                    source_name,
                    source_pattern,
                    lambda value=tensor: value,
                )
                state.received[source_pattern] += 1
                if not state.completed and state.received == state.expected:
                    copy_converted(self._convert_group(state.group))
                    state.completed = True
                    used_replacements.append(state.group.transform)

        missing_keys = tuple(sorted(set(targets) - loaded_keys, key=dot_natural_key))
        unexpected_keys = tuple(sorted(set(unexpected_keys), key=dot_natural_key))
        self._validate_load_result(missing_keys, unexpected_keys, strict)
        used_base = [transform for transform in base_mapping if transform.was_used()]
        self.model._hp_used_base_weight_conversions = used_base  # pylint: disable=protected-access
        self.model._hp_used_replacement_weight_conversions = (  # pylint: disable=protected-access
            used_replacements
        )
        self.model._weight_conversions = used_base + used_replacements  # pylint: disable=protected-access
        report = LoadReport(
            loaded_keys=tuple(sorted(loaded_keys, key=dot_natural_key)),
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
        logger.info("Loaded %d model tensors from %s", len(report.loaded_keys), pretrained_path)
        return report

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

    def load_dcp(
        self,
        checkpoint_id: str | Path,
        *,
        strict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Delegate DCP loading, then apply the restored sharded model state."""
        backend = self._require_dcp_backend()
        model_state = self.model.state_dict()
        state_dict = {"model": model_state}
        result = backend.load(
            state_dict,
            checkpoint_id=checkpoint_id,
            **kwargs,
        )
        self.model.load_state_dict(state_dict["model"], strict=strict)
        return result

    def save_dcp(self, checkpoint_id: str | Path, **kwargs: Any) -> Any:
        """Delegate sharded model-state saving to the configured DCP backend."""
        backend = self._require_dcp_backend()
        return backend.save(
            {"model": self.model.state_dict()},
            checkpoint_id=checkpoint_id,
            **kwargs,
        )

    def _convert_group(
        self,
        group: LoadGroup,
        *,
        model: nn.Module | SourceModelView | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert all checkpoint tensors belonging to one load group."""
        return convert_group(group, self.model if model is None else model)

    @staticmethod
    def _validate_load_result(
        missing_keys: tuple[str, ...],
        unexpected_keys: tuple[str, ...],
        strict: bool,
    ) -> None:
        """Validate missing keys and report ignored checkpoint tensors."""
        validate_load_result(missing_keys, unexpected_keys, strict)

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

    def _require_dcp_backend(self) -> DCPBackend:
        if self.dcp_backend is None:
            raise NotImplementedError(
                "DCP backend is not configured; inject the distributed-checkpoint "
                "implementation through CheckpointManager(dcp_backend=...)"
            )
        return self.dcp_backend

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


def _prepare_initialization_targets(targets: list[_FinalizeTarget]) -> None:
    """Clear stale initialization flags only for state that must be rebuilt."""
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

    _validate_materialized(initialization_targets)
    _prepare_initialization_targets(initialization_targets)
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
    pretrained_path: str,
    *,
    strict: bool = True,
) -> LoadReport:
    """Backward-compatible functional wrapper around CheckpointManager."""
    return CheckpointManager(model).load_checkpoint(pretrained_path, strict=strict)
