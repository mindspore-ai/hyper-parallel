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
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.distributed import is_available, is_initialized
from transformers import PreTrainedModel
from hyper_parallel.core.distributed_checkpoint import (
    FileSystemReader,
    TensorStorageMetadata,
    load as load_dcp_state,
)
from hyper_parallel.core.distributed_checkpoint.utils import str_to_dtype
from hyper_parallel.core.fully_shard.api import HSDPModule

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
        dcp_checkpoint: str | Path | None = None,
        dcp_expert_layout: Optional[str] = None,
        max_shard_size: int | str = "5GB",
        save_original_format: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Gather model weights and save a Transformers-compatible checkpoint.

        When ``dcp_checkpoint`` is provided, rank 0 reconstructs model weights
        from the completed distributed checkpoint while the other ranks return
        immediately. This avoids running full-model collectives against the
        live training parameters. Without a DCP source, all distributed ranks
        must call this method so the live sharded weights can be gathered.

        Args:
            save_directory: Destination Hugging Face checkpoint directory.
            dcp_checkpoint: Optional completed DCP directory containing the
                model shards to export.
            dcp_expert_layout: For an unreplaced HF model, normalize DCP expert
                weights using ``auto``, ``hf``, or ``grouped_gemm`` layout.
                None preserves the live model's registered conversions.
            max_shard_size: Maximum size of each Hugging Face weight shard.
            save_original_format: Whether to reverse registered weight
                conversions before saving. Qwen3.5 dense/MoE text models always
                use their native text namespace, not the source VLM namespace.
            **kwargs: Additional arguments forwarded to ``save_pretrained``.

        Returns:
            True on the writing rank and False on all other ranks.
        """
        save_method = getattr(self.model, "save_pretrained", None)
        if dcp_expert_layout is not None and dcp_checkpoint is None:
            raise ValueError("dcp_expert_layout requires a DCP checkpoint")
        if not callable(save_method):
            raise TypeError("CheckpointManager.save_pretrained requires a Transformers model")
        is_main_process = self._is_main_process()
        if dcp_checkpoint is not None:
            if not is_main_process:
                return False
            state_dict = self._load_model_state_dict_from_dcp(dcp_checkpoint)
            if dcp_expert_layout is not None:
                state_dict = self._normalize_dcp_export_layout(state_dict, dcp_expert_layout)
        else:
            state_dict = self._gather_full_state_dict(keep_state_dict=is_main_process)
        if not is_main_process:
            return False
        export_model = self._hf_export_model()
        canonical_text = getattr(getattr(export_model, "config", None), "model_type", None) in (
            "qwen3_5_text", "qwen3_5_moe_text",
        )
        used_base = getattr(
            self.model, "_hp_used_base_weight_conversions", None
        )
        used_replacements = getattr(
            self.model, "_hp_used_replacement_weight_conversions", None
        )
        if (save_original_format or canonical_text) and used_replacements:
            original_mapping = getattr(export_model, "_weight_conversions", None)
            try:
                export_model._weight_conversions = used_replacements  # pylint: disable=protected-access
                state_dict = revert_weight_conversion(export_model, state_dict)
                if used_base and not canonical_text:
                    export_model._weight_conversions = used_base  # pylint: disable=protected-access
                    state_dict = revert_weight_conversion(export_model, state_dict)
            finally:
                export_model._weight_conversions = original_mapping  # pylint: disable=protected-access
            save_original_format = False
        if canonical_text:
            # Loading a text model from VLM weights records a PrefixChange.
            # Reversing it would export VLM keys with a text-only config.
            save_original_format = False
        export_model.save_pretrained(
            save_directory,
            state_dict=state_dict,
            is_main_process=True,
            max_shard_size=max_shard_size,
            save_original_format=save_original_format,
            **kwargs,
        )
        return True

    def _hf_export_model(self) -> nn.Module:
        """Create a metadata-only HF view without modifying the live training model."""
        if not isinstance(self.model, PreTrainedModel):
            return self.model
        export_model = copy(self.model)
        model_class = type(self.model)
        if HSDPModule in model_class.__bases__:
            model_class = next(base for base in model_class.__bases__ if base is not HSDPModule)
        export_model.__class__ = model_class
        # HF overwrites architectures/dtype during saving. The shallow module
        # copy shares parameters without allocating weights, but not metadata.
        export_model.config = deepcopy(self.model.config)
        if getattr(self.model, "generation_config", None) is not None:
            export_model.generation_config = deepcopy(self.model.generation_config)
        return export_model

    def _normalize_dcp_export_layout(
        self,
        state_dict: dict[str, torch.Tensor],
        layout: str,
    ) -> dict[str, torch.Tensor]:
        """Restore batched expert layout against an unreplaced HF model."""
        if layout not in ("auto", "hf", "grouped_gemm"):
            raise ValueError(f"Unsupported DCP expert layout: {layout}")
        if getattr(self.model, "_hp_used_replacement_weight_conversions", None):
            raise ValueError("DCP layout normalization requires an unreplaced Hugging Face model")
        expected = self.model.state_dict()
        result = dict(state_dict)
        expert_groups: dict[str, list[str]] = defaultdict(list)
        for name in state_dict:
            if re.search(r"(?:^|\.)experts\.(?:gate_up_proj|up_proj|down_proj)$", name):
                expert_groups[name.rsplit(".", 1)[0]].append(name)
        for prefix, names in expert_groups.items():
            if any(name not in expected or state_dict[name].ndim != 3 for name in names):
                raise ValueError(f"Unsupported DCP expert parameters under {prefix}; expected batched HF weights")
            hf_matches = all(state_dict[name].shape == expected[name].shape for name in names)
            transposed_matches = all(
                state_dict[name].transpose(-2, -1).shape == expected[name].shape for name in names
            )
            if layout == "auto" and hf_matches and transposed_matches:
                raise ValueError(
                    f"Ambiguous expert layout under {prefix}; explicitly select hf or grouped_gemm"
                )
            transpose = layout == "grouped_gemm" or (layout == "auto" and not hf_matches)
            if transpose and not transposed_matches:
                raise ValueError(
                    f"DCP expert shapes under {prefix} do not match the requested HF layout: "
                    + ", ".join(
                        f"{name}: DCP {tuple(state_dict[name].shape)}, HF {tuple(expected[name].shape)}"
                        for name in names
                    )
                )
            if transpose:
                for name in names:
                    result[name] = state_dict[name].transpose(-2, -1).contiguous()
                logger.info("Restored HF expert layout for %s", prefix)
        for name, tensor in result.items():
            target = expected.get(name)
            if target is None:
                raise ValueError(f"DCP parameter {name!r} is absent from the HF model; check --model-dir")
            if tensor.shape != target.shape:
                raise ValueError(
                    f"DCP parameter {name!r} has shape {tuple(tensor.shape)}, "
                    f"but HF expects {tuple(target.shape)}; check model config and expert layout"
                )
        return result

    @staticmethod
    def _model_key_from_dcp(
        fqn: str,
        planner_data: dict[str, tuple[Any, ...]] | None,
    ) -> str | None:
        """Return the model-local key for one flattened DCP entry."""
        path = planner_data.get(fqn) if planner_data is not None else None
        if path is not None:
            if len(path) < 2 or path[0] != "model":
                return None
            return ".".join(str(part) for part in path[1:])
        if fqn.startswith("model."):
            return fqn[len("model."):]
        return None

    def _load_model_state_dict_from_dcp(
        self,
        checkpoint_id: str | Path,
    ) -> dict[str, torch.Tensor]:
        """Reconstruct only model tensors from a completed DCP on rank 0."""
        reader = FileSystemReader(checkpoint_id)
        metadata = reader.load_metadata()
        planner_data = metadata.planner_data
        if planner_data is not None and not isinstance(planner_data, dict):
            raise TypeError("DCP planner_data must be a mapping for Hugging Face export")

        dcp_state: dict[str, torch.Tensor] = {}
        model_keys: dict[str, str] = {}
        seen_model_keys: set[str] = set()
        for fqn, storage_metadata in metadata.state_dict_metadata.items():
            model_key = self._model_key_from_dcp(fqn, planner_data)
            if model_key is None:
                continue
            if not isinstance(storage_metadata, TensorStorageMetadata):
                raise TypeError(
                    f"DCP model entry {fqn!r} is not a tensor and cannot be exported"
                )
            if model_key in seen_model_keys:
                raise ValueError(f"Duplicate model key {model_key!r} in DCP checkpoint")
            dcp_state[fqn] = torch.empty(
                storage_metadata.size,
                dtype=str_to_dtype(storage_metadata.properties.dtype),
                device="cpu",
            )
            model_keys[fqn] = model_key
            seen_model_keys.add(model_key)

        if not dcp_state:
            raise ValueError(f"No model tensors found in DCP checkpoint {checkpoint_id}")

        load_dcp_state(
            dcp_state,
            checkpoint_id=checkpoint_id,
            no_dist=True,
            broadcast_replicated_tensors=False,
        )
        return {
            model_keys[fqn]: tensor.detach()
            for fqn, tensor in dcp_state.items()
        }

    @torch.no_grad()
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
