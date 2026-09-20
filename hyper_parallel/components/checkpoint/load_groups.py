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
"""Hugging Face checkpoint names and layouts, mapped onto the tensors of a finalized model.

Both pretrained loaders are built on what is here. :class:`HuggingFaceCheckpointer` reads whole checkpoint
tensors and converts them in memory, while :class:`HFLoadPlanner` plans the same conversions as
distributed checkpoint reads. Keys are renamed, converters are chosen and replacement conversions are
routed the same way in both, so the two load the same values under the same names.
"""
import json
import logging
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn

from hyper_parallel import DTensor, Partial, distribute_tensor
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    dot_natural_key,
    rename_source_key,
)

logger = logging.getLogger(__name__)

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_SNAPSHOT_PATTERNS = ("*.safetensors", "*.safetensors.index.json")

# Picks the converter a checkpoint key matched by ``source pattern`` belongs to, for the model tensor
# ``target name``, out of every converter with that source pattern. None when no single one fits.
ConverterSelector = Callable[[str, str, Sequence[WeightConverter]], Optional[WeightConverter]]


@dataclass(frozen=True)
class LoadReport:
    """Summary of one pretrained-weight load."""

    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointIndex:
    """Map checkpoint tensor names to their safetensors shard files."""

    files_by_key: dict[str, Path]

    def keys(self) -> tuple[str, ...]:
        """Return checkpoint keys in deterministic natural order."""
        return sorted_checkpoint_keys(self.files_by_key)

    def load_tensor(self, key: str) -> torch.Tensor:
        """Materialize one checkpoint tensor on CPU."""
        file_path = self.files_by_key.get(key)
        if file_path is None:
            raise ValueError(f"Checkpoint key is not indexed: {key}")
        with safe_open(str(file_path), framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(key)


@dataclass
class LoadGroup:
    """Checkpoint tensors converted together, and the transform converting them."""

    first_target_name: str
    transform: WeightRenaming | WeightConverter


@dataclass(frozen=True)
class _TensorShape:
    """Shape of a model tensor, as shape-aware conversion operations ask for it."""

    shape: torch.Size


class SourceModelView:
    """Expose pre-replacement parameter shapes to Transformers conversion ops."""

    def __init__(self, model: nn.Module, shapes: dict[str, tuple[int, ...]]) -> None:
        """Build a lightweight model view from captured tensor shapes."""
        self.config = getattr(model, "config", None)
        self.base_model_prefix = getattr(model, "base_model_prefix", None)
        self._targets = {
            name: _TensorShape(torch.Size(shape)) for name, shape in shapes.items()
        }

    @property
    def targets(self) -> dict[str, _TensorShape]:
        """Shape of every pre-replacement tensor, by name."""
        return self._targets

    def get_parameter(self, name: str) -> _TensorShape:
        """Return source parameter metadata used by shape-aware converters."""
        try:
            return self._targets[name]
        except KeyError as exc:
            raise AttributeError(f"source model has no parameter {name!r}") from exc


@dataclass
class ReplacementLoadGroup:
    """A replacement conversion waiting for the normalized tensors it is routed."""

    group: LoadGroup
    expected: Counter[str]
    received: Counter[str]
    completed: bool = False


def sorted_checkpoint_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """Checkpoint keys in natural order, experts 2 before 10, which is the order loads collect them in."""
    return tuple(sorted(keys, key=dot_natural_key))


def _index_single_file(file_path: Path) -> CheckpointIndex:
    """Index every tensor of one safetensors file."""
    with safe_open(str(file_path), framework="pt", device="cpu") as checkpoint:
        files_by_key = {key: file_path for key in checkpoint.keys()}
    return CheckpointIndex(files_by_key)


def _index_sharded_checkpoint(directory: Path, index_path: Path) -> CheckpointIndex:
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
    return CheckpointIndex(files_by_key)


def resolve_checkpoint_location(pretrained_path: str) -> Path:
    """
    Resolve a pretrained path into a local safetensors file or checkpoint directory.

    Args:
        pretrained_path (str): A ``.safetensors`` file, a checkpoint directory, or a Hub repository id,
            which is downloaded first.

    Returns:
        Path: The file or the directory.

    Raises:
        ValueError: If the path is a file of another kind.
    """
    path = Path(pretrained_path).expanduser()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"MVP only supports safetensors checkpoints, got: {path}")
        return path
    if path.is_dir():
        return path
    return Path(
        snapshot_download(
            repo_id=pretrained_path,
            allow_patterns=list(_SNAPSHOT_PATTERNS),
        )
    )


def resolve_checkpoint_index(pretrained_path: str) -> CheckpointIndex:
    """Resolve a local or Hub checkpoint into a tensor-to-file index."""
    location = resolve_checkpoint_location(pretrained_path)
    if location.is_file():
        return _index_single_file(location)

    index_path = location / SAFE_WEIGHTS_INDEX_NAME
    if index_path.is_file():
        return _index_sharded_checkpoint(location, index_path)

    single_file = location / SAFE_WEIGHTS_NAME
    if single_file.is_file():
        return _index_single_file(single_file)
    raise ValueError(
        "MVP requires model.safetensors or model.safetensors.index.json under "
        f"{location}"
    )


def join_fqn(module_name: str, tensor_name: str) -> str:
    """Name of a tensor registered on the module named ``module_name``."""
    return f"{module_name}.{tensor_name}" if module_name else tensor_name


def build_load_targets(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect persistent parameters and buffers owned by the model."""
    targets = {}
    # Direct module registries preserve tied aliases and let us exclude
    # non-persistent buffers without materializing an FSDP state_dict.
    for module_name, module in model.named_modules(remove_duplicate=False):
        for tensor_name, parameter in module._parameters.items():  # pylint: disable=W0212
            if parameter is not None:
                targets[join_fqn(module_name, tensor_name)] = parameter
        non_persistent = module._non_persistent_buffers_set  # pylint: disable=W0212
        for tensor_name, buffer in module._buffers.items():  # pylint: disable=W0212
            if buffer is not None and tensor_name not in non_persistent:
                targets[join_fqn(module_name, tensor_name)] = buffer
    return targets


def make_tensor_loader(index: CheckpointIndex, source_key: str) -> Callable[[], torch.Tensor]:
    """A loader reading one checkpoint tensor when a conversion asks for it."""
    return lambda: index.load_tensor(source_key)


def _scoped_candidates(candidates: Sequence[WeightConverter], target_name: str) -> list[WeightConverter]:
    """The candidates scoped to a module that holds ``target_name``."""
    return [
        converter
        for converter in candidates
        if converter.scope_prefix is not None
        and (
            target_name == converter.scope_prefix
            or target_name.startswith(f"{converter.scope_prefix}.")
        )
    ]


def select_unique_converter(
    source_pattern: str,
    target_name: str,
    candidates: Sequence[WeightConverter],
) -> Optional[WeightConverter]:
    """
    Pick the converter a matched checkpoint key belongs to.

    A converter scoped to a module holding the target wins when it is the only one. Otherwise the only
    unscoped converter does.

    Args:
        source_pattern (str): The source pattern the key matched.
        target_name (str): The model tensor the key was renamed to.
        candidates (Sequence[WeightConverter]): Every converter with that source pattern.

    Returns:
        Optional[WeightConverter]: The converter, or None when neither rule picks exactly one.
    """
    del source_pattern
    scoped_candidates = _scoped_candidates(candidates, target_name)
    if len(scoped_candidates) == 1:
        return scoped_candidates[0]
    unscoped_candidates = [converter for converter in candidates if converter.scope_prefix is None]
    return unscoped_candidates[0] if len(unscoped_candidates) == 1 else None


def _split_transforms(
    transforms: Sequence[Any],
) -> tuple[list[WeightRenaming], list[WeightConverter], dict[str, list[WeightConverter]]]:
    """Renamings, converters, and the converters by each of their source patterns."""
    renamings = [transform for transform in transforms if isinstance(transform, WeightRenaming)]
    converters = [transform for transform in transforms if isinstance(transform, WeightConverter)]
    converters_by_pattern = defaultdict(list)
    for converter in converters:
        for pattern in converter.source_patterns:
            converters_by_pattern[pattern].append(converter)
    return renamings, converters, converters_by_pattern


def build_load_groups(
    model: Any,
    source_keys: Iterable[str],
    targets: dict[str, Any],
    *,
    weights_mapping: list[WeightRenaming | WeightConverter],
    make_loader: Callable[[str], Any],
    select_converter: ConverterSelector = select_unique_converter,
) -> tuple[
    tuple[LoadGroup, ...],
    tuple[str, ...],
    list[WeightRenaming | WeightConverter],
]:
    """
    Build checkpoint conversion groups and report unmatched checkpoint keys.

    Args:
        model (Any): The model, or the view of it the keys are renamed against.
        source_keys (Iterable[str]): Checkpoint keys, in the order their tensors are collected.
        targets (dict[str, Any]): Model tensors by name.
        weights_mapping (list[WeightRenaming | WeightConverter]): Rules renaming and converting keys.
        make_loader (Callable[[str], Any]): Called with a key, returns what the group collects for it.
        select_converter (ConverterSelector): Picks the converter of a matched key.
            Default :func:`select_unique_converter`.

    Returns:
        tuple: The groups in order of their first key, the keys renamed to no target, and the mapping.
    """
    unsupported = [
        transform
        for transform in weights_mapping
        if not isinstance(transform, (WeightRenaming, WeightConverter))
    ]
    if unsupported:
        names = ", ".join(type(transform).__name__ for transform in unsupported)
        raise ValueError(f"Unsupported Transformers weight transforms in MVP: {names}")

    renamings, converters, converters_by_pattern = _split_transforms(weights_mapping)
    groups: OrderedDict[str, LoadGroup] = OrderedDict()
    unexpected_keys = []
    base_model_prefix = getattr(model, "base_model_prefix", None)

    for source_key in source_keys:
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
            converter = select_converter(
                source_pattern, target_name, converters_by_pattern.get(source_pattern, [])
            )
            if converter is None:
                raise ValueError(
                    "No unique WeightConverter found for matched source pattern "
                    f"{source_pattern!r} and target {target_name!r}"
                )
            transform = deepcopy(converter)

        group = groups.setdefault(
            target_name,
            LoadGroup(first_target_name=target_name, transform=transform),
        )
        group.transform.add_tensor(target_name, source_key, source_pattern, make_loader(source_key))

    return tuple(groups.values()), tuple(unexpected_keys), weights_mapping


def build_replacement_routes(
    model: nn.Module,
    source_names: tuple[str, ...],
    targets: dict[str, torch.Tensor],
    transforms: list[WeightRenaming | WeightConverter],
) -> dict[str, tuple[ReplacementLoadGroup, str, str]]:
    """Route normalized Transformers parameters into replacement converters."""
    renamings, converters, converters_by_pattern = _split_transforms(deepcopy(transforms))
    groups: OrderedDict[str, ReplacementLoadGroup] = OrderedDict()
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
            scoped_candidates = _scoped_candidates(converters_by_pattern.get(source_pattern, []), target_name)
            if len(scoped_candidates) != 1:
                raise ValueError(
                    "No unique replacement WeightConverter found for source "
                    f"{source_name!r} and target {target_name!r}"
                )
            transform = deepcopy(scoped_candidates[0])

        state = groups.get(target_name)
        if state is None:
            state = ReplacementLoadGroup(
                group=LoadGroup(target_name, transform),
                expected=Counter(),
                received=Counter(),
            )
            groups[target_name] = state
        state.expected[collected_pattern] += 1
        routes[source_name] = (state, target_name, collected_pattern)
    return routes


def base_weights_mapping(
    weights_mapping: list[WeightRenaming | WeightConverter],
    replacement_mapping: list[WeightRenaming | WeightConverter],
) -> list[WeightRenaming | WeightConverter]:
    """The rules of ``weights_mapping`` that are not replacement conversions."""
    replacement_ids = {id(transform) for transform in replacement_mapping}
    return [transform for transform in weights_mapping if id(transform) not in replacement_ids]


def first_tensor(value: Any) -> Any:
    """The tensor a conversion returned under one name, which a renaming returns in a list."""
    return value[0] if isinstance(value, list) else value


def convert_group(group: LoadGroup, model: Any) -> dict[str, Any]:
    """
    Convert all checkpoint tensors belonging to one load group.

    Args:
        group (LoadGroup): The group, its tensors already collected.
        model (Any): The model, or the view of it, handed to the conversion operations.

    Returns:
        dict[str, Any]: Converted tensors by model tensor name.

    Raises:
        RuntimeError: If the conversion fails, naming the group.
    """
    try:
        return group.transform.convert(
            group.first_target_name,
            model=model,
            config=getattr(model, "config", None),
            hf_quantizer=None,
            loading_info=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to convert checkpoint tensors for {group.first_target_name}: {exc}"
        ) from exc


def local_target_tensor(target: torch.Tensor) -> torch.Tensor:
    """The part of a model tensor held on this rank."""
    return target.to_local() if isinstance(target, DTensor) else target


def target_layout(target: torch.Tensor) -> Any:
    """The layout a model tensor is sharded with, or None for a whole tensor."""
    if isinstance(target, DTensor) and target.layout is not None:
        return target.layout
    return getattr(target, "_sharding_spec", None)


def reject_partial_layout(target_name: str, layout: Any) -> None:
    """Refuse a layout with a Partial placement, which pretrained loads do not support."""
    if any(isinstance(placement, Partial) for placement in layout.placements):
        raise ValueError(f"Partial placement is not supported for pretrained loading: {target_name}")


def shard_for_target(target_name: str, full_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shard a full checkpoint tensor according to its target layout."""
    layout = target_layout(target)
    if layout is None:
        return full_tensor
    reject_partial_layout(target_name, layout)

    local_dtensor = distribute_tensor(
        full_tensor,
        layout.mesh,
        layout.alias_placements,
        src_data_rank=None,
    )
    return local_dtensor.to_local()


def copy_into_target(target_name: str, full_tensor: torch.Tensor, target: torch.Tensor) -> None:
    """Copy a checkpoint tensor into its materialized local target."""
    local_tensor = shard_for_target(target_name, full_tensor, target)
    destination = local_target_tensor(target)
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


def alias_names_by_target(targets: dict[str, torch.Tensor]) -> dict[int, set[str]]:
    """Every name each model tensor is registered under, by the identity of the tensor."""
    aliases = defaultdict(set)
    for target_name, target in targets.items():
        aliases[id(target)].add(target_name)
    return aliases


def validate_load_result(
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
