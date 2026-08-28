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

import json
import logging
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from torch.distributed import is_available, is_initialized
from hyper_parallel.auto_models.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    dot_natural_key,
    get_model_conversion_mapping,
    rename_source_key,
    revert_weight_conversion,
)

from hyper_parallel import DTensor, Partial, distribute_tensor

logger = logging.getLogger(__name__)

_SAFE_WEIGHTS_NAME = "model.safetensors"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_SNAPSHOT_PATTERNS = ("*.safetensors", "*.safetensors.index.json")


@dataclass(frozen=True)
class LoadReport:
    """Summary of one pretrained-weight load."""

    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


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


@dataclass(frozen=True)
class _CheckpointIndex:
    """Map checkpoint tensor names to their safetensors shard files."""

    files_by_key: dict[str, Path]

    def keys(self) -> tuple[str, ...]:
        """Return checkpoint keys in deterministic natural order."""
        return tuple(sorted(self.files_by_key, key=dot_natural_key))

    def load_tensor(self, key: str) -> torch.Tensor:
        """Materialize one checkpoint tensor on CPU."""
        file_path = self.files_by_key.get(key)
        if file_path is None:
            raise ValueError(f"Checkpoint key is not indexed: {key}")
        with safe_open(str(file_path), framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(key)


@dataclass
class _LoadGroup:
    first_target_name: str
    transform: WeightRenaming | WeightConverter


@dataclass(frozen=True)
class _TensorShape:
    shape: torch.Size


class _SourceModelView:
    """Expose pre-replacement parameter shapes to Transformers conversion ops."""

    def __init__(self, model: nn.Module, shapes: dict[str, tuple[int, ...]]) -> None:
        """Build a lightweight model view from captured tensor shapes."""
        self.config = getattr(model, "config", None)
        self.base_model_prefix = getattr(model, "base_model_prefix", None)
        self._targets = {
            name: _TensorShape(torch.Size(shape)) for name, shape in shapes.items()
        }

    def get_parameter(self, name: str) -> _TensorShape:
        """Return source parameter metadata used by shape-aware converters."""
        try:
            return self._targets[name]
        except KeyError as exc:
            raise AttributeError(f"source model has no parameter {name!r}") from exc


@dataclass
class _ReplacementLoadGroup:
    group: _LoadGroup
    expected: Counter[str]
    received: Counter[str]
    completed: bool = False


def _index_single_file(file_path: Path) -> _CheckpointIndex:
    with safe_open(str(file_path), framework="pt", device="cpu") as checkpoint:
        files_by_key = {key: file_path for key in checkpoint.keys()}
    return _CheckpointIndex(files_by_key)


def _index_sharded_checkpoint(directory: Path, index_path: Path) -> _CheckpointIndex:
    """Index tensors described by a sharded safetensors index file."""

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read safetensors index {index_path}: {exc}") from exc
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Safetensors index has no non-empty weight_map: {index_path}")

    files_by_key = {}
    for key, relative_path in weight_map.items():
        file_path = directory / relative_path
        if not file_path.is_file():
            raise ValueError(f"Safetensors shard for {key} does not exist: {file_path}")
        files_by_key[key] = file_path
    return _CheckpointIndex(files_by_key)


def _resolve_checkpoint_index(pretrained_path: str) -> _CheckpointIndex:
    """Resolve a local or Hub checkpoint into a tensor-to-file index."""

    path = Path(pretrained_path).expanduser()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"MVP only supports safetensors checkpoints, got: {path}")
        return _index_single_file(path)

    if path.is_dir():
        checkpoint_directory = path
    else:
        checkpoint_directory = Path(
            snapshot_download(
                repo_id=pretrained_path,
                allow_patterns=list(_SNAPSHOT_PATTERNS),
            )
        )

    index_path = checkpoint_directory / _SAFE_WEIGHTS_INDEX_NAME
    if index_path.is_file():
        return _index_sharded_checkpoint(checkpoint_directory, index_path)

    single_file = checkpoint_directory / _SAFE_WEIGHTS_NAME
    if single_file.is_file():
        return _index_single_file(single_file)
    raise ValueError(
        "MVP requires model.safetensors or model.safetensors.index.json under "
        f"{checkpoint_directory}"
    )


def _join_fqn(module_name: str, tensor_name: str) -> str:
    return f"{module_name}.{tensor_name}" if module_name else tensor_name


def _build_load_targets(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect persistent parameters and buffers owned by the model."""

    targets = {}
    # Direct module registries preserve tied aliases and let us exclude
    # non-persistent buffers without materializing an FSDP state_dict.
    for module_name, module in model.named_modules(remove_duplicate=False):
        for tensor_name, parameter in module._parameters.items():  # pylint: disable=W0212
            if parameter is not None:
                targets[_join_fqn(module_name, tensor_name)] = parameter
        non_persistent = module._non_persistent_buffers_set  # pylint: disable=W0212
        for tensor_name, buffer in module._buffers.items():  # pylint: disable=W0212
            if buffer is not None and tensor_name not in non_persistent:
                targets[_join_fqn(module_name, tensor_name)] = buffer
    return targets


def _make_tensor_loader(index: _CheckpointIndex, source_key: str) -> Callable[[], torch.Tensor]:
    return lambda: index.load_tensor(source_key)


def _build_load_groups(
    model: nn.Module,
    checkpoint_index: _CheckpointIndex,
    targets: dict[str, torch.Tensor],
    *,
    weights_mapping: list[WeightRenaming | WeightConverter],
) -> tuple[
    tuple[_LoadGroup, ...],
    tuple[str, ...],
    list[WeightRenaming | WeightConverter],
]:
    """Build checkpoint conversion groups and report unmatched transforms."""

    weight_mapping = weights_mapping
    unsupported = [
        transform
        for transform in weight_mapping
        if not isinstance(transform, (WeightRenaming, WeightConverter))
    ]
    if unsupported:
        names = ", ".join(type(transform).__name__ for transform in unsupported)
        raise ValueError(f"Unsupported Transformers weight transforms in MVP: {names}")

    renamings = [transform for transform in weight_mapping if isinstance(transform, WeightRenaming)]
    converters = [transform for transform in weight_mapping if isinstance(transform, WeightConverter)]
    converters_by_pattern = defaultdict(list)
    for converter in converters:
        for pattern in converter.source_patterns:
            converters_by_pattern[pattern].append(converter)
    groups: OrderedDict[str, _LoadGroup] = OrderedDict()
    unexpected_keys = []
    base_model_prefix = getattr(model, "base_model_prefix", None)

    for source_key in checkpoint_index.keys():
        target_name, source_pattern = rename_source_key(
            source_key,
            renamings,
            converters,
            base_model_prefix=base_model_prefix,
            meta_state_dict=targets,
        )
        if target_name not in targets and source_key in targets:
            target_name, source_pattern = rename_source_key(
                source_key,
                [],
                [],
                base_model_prefix=base_model_prefix,
                meta_state_dict=targets,
            )
        if target_name not in targets:
            unexpected_keys.append(source_key)
            continue

        if source_pattern is None:
            source_pattern = source_key
            transform = WeightRenaming(source_patterns=source_key, target_patterns=target_name)
        else:
            candidates = converters_by_pattern.get(source_pattern, [])
            scoped_candidates = [
                converter
                for converter in candidates
                if converter.scope_prefix is not None
                and (
                    target_name == converter.scope_prefix
                    or target_name.startswith(f"{converter.scope_prefix}.")
                )
            ]
            if len(scoped_candidates) == 1:
                converter = scoped_candidates[0]
            else:
                unscoped_candidates = [
                    converter for converter in candidates if converter.scope_prefix is None
                ]
                converter = unscoped_candidates[0] if len(unscoped_candidates) == 1 else None
            if converter is None:
                raise ValueError(
                    "No unique WeightConverter found for matched source pattern "
                    f"{source_pattern!r} and target {target_name!r}"
                )
            transform = deepcopy(converter)

        group = groups.setdefault(
            target_name,
            _LoadGroup(first_target_name=target_name, transform=transform),
        )
        group.transform.add_tensor(
            target_name,
            source_key,
            source_pattern,
            _make_tensor_loader(checkpoint_index, source_key),
        )

    return tuple(groups.values()), tuple(unexpected_keys), weight_mapping


def _build_replacement_routes(
    model: nn.Module,
    source_names: tuple[str, ...],
    targets: dict[str, torch.Tensor],
    transforms: list[WeightRenaming | WeightConverter],
) -> dict[str, tuple[_ReplacementLoadGroup, str, str]]:
    """Route normalized Transformers parameters into replacement converters."""
    routing_transforms = deepcopy(transforms)
    renamings = [
        transform
        for transform in routing_transforms
        if isinstance(transform, WeightRenaming)
    ]
    converters = [
        transform
        for transform in routing_transforms
        if isinstance(transform, WeightConverter)
    ]
    converters_by_pattern = defaultdict(list)
    for converter in converters:
        for pattern in converter.source_patterns:
            converters_by_pattern[pattern].append(converter)

    groups: OrderedDict[str, _ReplacementLoadGroup] = OrderedDict()
    routes = {}
    base_model_prefix = getattr(model, "base_model_prefix", None)
    for source_name in source_names:
        target_name, source_pattern = rename_source_key(
            source_name,
            renamings,
            converters,
            base_model_prefix=base_model_prefix,
            meta_state_dict=targets,
        )
        if source_pattern is None and target_name == source_name:
            continue
        if target_name not in targets:
            continue

        if source_pattern is None:
            collected_pattern = source_name
            transform: WeightRenaming | WeightConverter = WeightRenaming(
                source_patterns=source_name,
                target_patterns=target_name,
            )
        else:
            collected_pattern = source_pattern
            candidates = converters_by_pattern.get(source_pattern, [])
            scoped_candidates = [
                converter
                for converter in candidates
                if converter.scope_prefix is not None
                and (
                    target_name == converter.scope_prefix
                    or target_name.startswith(f"{converter.scope_prefix}.")
                )
            ]
            if len(scoped_candidates) != 1:
                raise ValueError(
                    "No unique replacement WeightConverter found for source "
                    f"{source_name!r} and target {target_name!r}"
                )
            transform = deepcopy(scoped_candidates[0])

        state = groups.get(target_name)
        if state is None:
            state = _ReplacementLoadGroup(
                group=_LoadGroup(target_name, transform),
                expected=Counter(),
                received=Counter(),
            )
            groups[target_name] = state
        state.expected[collected_pattern] += 1
        routes[source_name] = (state, target_name, collected_pattern)
    return routes


def _local_target_tensor(target: torch.Tensor) -> torch.Tensor:
    return target.to_local() if isinstance(target, DTensor) else target


def _target_layout(target: torch.Tensor) -> Any:
    if isinstance(target, DTensor) and target.layout is not None:
        return target.layout
    return getattr(target, "_sharding_spec", None)


def _shard_for_target(target_name: str, full_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shard a full checkpoint tensor according to its target layout."""

    layout = _target_layout(target)
    if layout is None:
        return full_tensor
    if any(isinstance(placement, Partial) for placement in layout.placements):
        raise ValueError(f"Partial placement is not supported for pretrained loading: {target_name}")

    local_dtensor = distribute_tensor(
        full_tensor,
        layout.mesh,
        layout.alias_placements,
        src_data_rank=None,
    )
    return local_dtensor.to_local()


def _copy_into_target(target_name: str, full_tensor: torch.Tensor, target: torch.Tensor) -> None:
    """Copy a checkpoint tensor into its materialized local target."""

    local_tensor = _shard_for_target(target_name, full_tensor, target)
    destination = _local_target_tensor(target)
    if destination.is_meta:
        raise ValueError(f"Target must be materialized before loading: {target_name}")
    if tuple(local_tensor.shape) != tuple(destination.shape):
        raise ValueError(
            f"Local shape mismatch for {target_name}: checkpoint shard "
            f"{tuple(local_tensor.shape)} vs target {tuple(destination.shape)}"
        )
    local_tensor = local_tensor.to(device=destination.device, dtype=destination.dtype)
    with torch.no_grad():
        destination.copy_(local_tensor)
    target._is_hf_initialized = True  # pylint: disable=W0212


def _alias_names_by_target(targets: dict[str, torch.Tensor]) -> dict[int, set[str]]:
    aliases = defaultdict(set)
    for target_name, target in targets.items():
        aliases[id(target)].add(target_name)
    return aliases


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
    ) -> LoadReport:
        """Load complete Hugging Face weights into the finalized model."""
        if not pretrained_path:
            raise ValueError("pretrained_path must be provided when load_base_model=True")
        if weights_mapping is None:
            weights_mapping = get_model_conversion_mapping(
                self.model,
                key_mapping=None,
                hf_quantizer=None,
            )

        checkpoint_index = _resolve_checkpoint_index(pretrained_path)
        targets = _build_load_targets(self.model)
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
        groups, unexpected_keys, weight_mapping = _build_load_groups(
            self.model,
            checkpoint_index,
            targets,
            weights_mapping=weights_mapping,
        )
        aliases_by_target = _alias_names_by_target(targets)
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
                    _copy_into_target(target_name, tensor, target)
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
        checkpoint_index: _CheckpointIndex,
        targets: dict[str, torch.Tensor],
        weights_mapping: list[WeightRenaming | WeightConverter],
        replacement_mapping: list[WeightRenaming | WeightConverter],
        source_shapes: dict[str, tuple[int, ...]],
        pretrained_path: str,
        strict: bool,
    ) -> LoadReport:
        """Normalize original weights before applying replacement conversions."""
        replacement_ids = {id(transform) for transform in replacement_mapping}
        base_mapping = [
            transform
            for transform in weights_mapping
            if id(transform) not in replacement_ids
        ]
        source_model = _SourceModelView(self.model, source_shapes)
        base_groups, unexpected_keys, _ = _build_load_groups(
            source_model,
            checkpoint_index,
            source_model._targets,  # pylint: disable=protected-access
            weights_mapping=base_mapping,
        )
        routes = _build_replacement_routes(
            self.model,
            tuple(source_shapes),
            targets,
            replacement_mapping,
        )
        aliases_by_target = _alias_names_by_target(targets)
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
                    _copy_into_target(target_name, tensor, target)
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
        group: _LoadGroup,
        *,
        model: nn.Module | _SourceModelView | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert all checkpoint tensors belonging to one load group."""

        try:
            return group.transform.convert(
                group.first_target_name,
                model=self.model if model is None else model,
                config=getattr(self.model if model is None else model, "config", None),
                hf_quantizer=None,
                loading_info=None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to convert checkpoint tensors for {group.first_target_name}: {exc}"
            ) from exc

    @staticmethod
    def _validate_load_result(
        missing_keys: tuple[str, ...],
        unexpected_keys: tuple[str, ...],
        strict: bool,
    ) -> None:
        """Validate missing keys and report ignored checkpoint tensors."""

        if strict and missing_keys:
            preview = ", ".join(missing_keys[:10])
            raise RuntimeError(
                f"Checkpoint did not load {len(missing_keys)} owned model tensors; "
                f"first keys: {preview}"
            )
        if unexpected_keys:
            logger.warning(
                "Ignored %d checkpoint tensors not owned by this model/rank; first keys: %s",
                len(unexpected_keys),
                ", ".join(unexpected_keys[:10]),
            )

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


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: str,
    *,
    strict: bool = True,
) -> LoadReport:
    """Backward-compatible functional wrapper around CheckpointManager."""
    return CheckpointManager(model).load_checkpoint(pretrained_path, strict=strict)
