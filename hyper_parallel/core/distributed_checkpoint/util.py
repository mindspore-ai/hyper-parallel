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
"""Common utility functions."""
import dataclasses
from collections import defaultdict
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Union

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    CHUNK_INFO,
    ChunkInfo
)
from hyper_parallel.core.distributed_checkpoint.planner import SavePlan, WriteItem
from hyper_parallel.core.distributed_checkpoint.ragged_utils import compute_ragged_boxes
from hyper_parallel.core.distributed_checkpoint.reshard import infer_slice_area_by_rank
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.platform import get_platform


platform = get_platform()
Tensor = platform.Tensor


def check_path(path: Union[Path, str]) -> None:
    """
    Check whether path is existing or not.

    Args:
        path (Union[Path, str]): path to check. Can only a file name in current directory, a pure directory, or a file
        name with directory. When path contains a directory, the function will check whether the directory exists, if
        not, the directory will be created.
    """
    path_obj = Path(path) if isinstance(path, str) else path

    if path_obj.exists():
        return

    if path_obj.suffix:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    else:
        path_obj.mkdir(parents=True, exist_ok=True)


def has_valid_filename(path: Path) -> bool:
    """
    Check whether path has valid filename. A filename should contain name and suffix, name and suffix must contain
    letters, and then can have numbers and underscores.

    Args:
        path (Path): path to check.

    Return:
        bool: whether path has a valid filename.
    """
    conditions = (
        path.name,
        path.suffix,
        len(path.suffix) > 1,
        path.stem,
        any(c.isalpha() for c in path.stem),
        any(c.isalpha() for c in path.suffix[1:])
    )
    return all(conditions)


def narrow_tensor_by_index(tensor: Any, offsets: tuple, lengths: tuple) -> Any:
    """
    Narrow the tensor by (offsets, lengths) per dimension.

    Used for resharding operations to extract a slice from a tensor.
    Compatible with both torch and mindspore (uses slice indexing).

    Args:
        tensor (Any): The tensor to narrow (tensor-like object supporting indexing).
        offsets (tuple): Tuple of offsets per dimension.
        lengths (tuple): Tuple of lengths per dimension.

    Returns:
        Any: The narrowed tensor slice (tensor-like object).
    """
    if not offsets or not lengths:
        return tensor
    slices = tuple(
        slice(int(off), int(off) + int(ln))
        for off, ln in zip(offsets, lengths)
    )
    return tensor[slices]


def chunk_to_area(chunk: ChunkStorageMetadata) -> tuple[tuple[int, int], ...]:
    """
    Convert ChunkStorageMetadata to (start, end) area per dimension.

    Args:
        chunk (ChunkStorageMetadata): ChunkStorageMetadata instance with offsets and sizes.

    Returns:
        tuple[tuple[int, int], ...]: Tuple of (start, end) tuples for each dimension.
    """
    return tuple(
        (chunk.offsets[i], chunk.offsets[i] + chunk.sizes[i])
        for i in range(len(chunk.offsets))
    )


def create_chunk_list_for_tensor(obj: Union[Tensor, DTensor]) -> list[ChunkStorageMetadata]:
    """
    Create list of local chunks for the given object (DTensor or plain tensor).

    Used to determine what this rank needs to load (resharding).

    Args:
        obj (Union[Tensor, DTensor]): hyper DTensor or platform Tensor.

    Returns:
        list[ChunkStorageMetadata]: List of ChunkStorageMetadata representing
            local chunks needed by this rank.
    """
    if isinstance(obj, DTensor):
        layout = obj.layout
        if layout is None:
            shape = obj.shape if hasattr(obj, "shape") else obj.to_local().shape
            return [ChunkStorageMetadata(offsets=(0,) * len(shape), sizes=tuple(shape))]
        if layout.ragged_shard is not None:
            return [
                ChunkStorageMetadata(offsets=box.offsets, sizes=box.sizes)
                for box in compute_ragged_boxes(obj)
            ]

        mesh_shape = getattr(layout, "mesh_shape", None) or getattr(layout, "_mesh", None)
        tensor_map = getattr(layout, "tensor_map", None) or getattr(layout, "_tensor_map", None)
        rank_list = getattr(layout, "rank_list", None) or getattr(layout, "_rank_list", None)

        if mesh_shape is None or tensor_map is None or rank_list is None:
            shape = obj.shape if hasattr(obj, "shape") else obj.to_local().shape
            return [ChunkStorageMetadata(offsets=(0,) * len(shape), sizes=tuple(shape))]

        current_rank = platform.get_rank()
        if current_rank not in rank_list:
            return []

        inner_rank_id = rank_list.index(current_rank)
        full_shape = obj.shape
        slice_area = infer_slice_area_by_rank(
            mesh_shape=mesh_shape,
            tensor_map=tensor_map,
            rank_id=inner_rank_id,
            full_shape=full_shape,
        )
        offsets = tuple(s for s, _ in slice_area)
        sizes = tuple(e - s for s, e in slice_area)
        return [ChunkStorageMetadata(offsets=offsets, sizes=sizes)]

    if isinstance(obj, Tensor):
        # handle Tensor with shard information
        if hasattr(obj, CHUNK_INFO):
            if not isinstance(getattr(obj, CHUNK_INFO), ChunkInfo):
                raise ValueError("The attr CHUNK_INFO should be a ChunkInfo instance")
            chunk = getattr(obj, CHUNK_INFO).chunk
            return [chunk]
        # platform.Tensor has exactly one chunk in metadata (full tensor)
        shape = tuple(obj.shape)
        return [ChunkStorageMetadata(offsets=(0,) * len(shape), sizes=shape)]

    raise ValueError(f"Not support type {type(obj)} for creating chunk list ")


def remove_redundant_plans(
    all_plans: list[SavePlan],
    save_to_minimum_rank: bool = False,
) -> list[SavePlan]:
    """
    Remove duplicate entries across SavePlans. For each duplicate, only one plan
    keeps the entry. The selection prefers the smallest planned storage size
    (or the minimum rank when save_to_minimum_rank is True).

    Args:
        all_plans (list[SavePlan]): List of save plans to deduplicate.
        save_to_minimum_rank (bool): If True, assign duplicates to the minimum rank; else to plan with minimal storage.
            Default False.
    """
    # Build mapping from item index to set of plan indices containing it
    duplicate_map: dict[MetadataIndex, set[int]] = defaultdict(set)
    # Registry to retrieve WriteItem by its index
    item_registry: dict[MetadataIndex, WriteItem] = {}
    # Track which items remain in each plan after deduplication
    remaining_items: list[set[MetadataIndex]] = [
        {entry.index for entry in plan.items} for plan in all_plans
    ]

    # Collect all items and their plan associations
    for idx, plan in enumerate(all_plans):
        for entry in plan.items:
            duplicate_map[entry.index].add(idx)
            item_registry[entry.index] = entry

    storage_sizes = [0] * len(all_plans)

    # Separate unique items (appear in only one plan) from duplicates
    # Process unique items first to prevent them from affecting load balancing
    single_plan_items: list[tuple[MetadataIndex, int]] = []
    multi_plan_items: list[tuple[MetadataIndex, set[int]]] = []

    for item_key, containing_plans in duplicate_map.items():
        if len(containing_plans) == 1:
            single_plan_items.append((item_key, next(iter(containing_plans))))
        else:
            multi_plan_items.append((item_key, containing_plans))

    # First pass: handle items that appear in only one plan
    for item_key, target_idx in single_plan_items:
        entry = item_registry[item_key]
        storage_sizes[target_idx] += entry.tensor_storage_size() or 1

    # Second pass: assign duplicate items to the plan with minimal storage size
    for item_key, containing_plans in multi_plan_items:
        if save_to_minimum_rank:
            target_plan = min(containing_plans)
        else:
            target_plan = min(
                containing_plans, key=lambda p_idx: storage_sizes[p_idx]
            )

        entry = item_registry[item_key]
        storage_sizes[target_plan] += entry.tensor_storage_size() or 1
        # Remove this item from all other plans
        for p_idx in containing_plans - {target_plan}:
            remaining_items[p_idx].discard(item_key)

    if len(all_plans) != len(remaining_items):
        raise AssertionError("len(all_plans) != len(remaining_items)")

    # Generate deduplicated plans with only remaining items
    return [
        dataclasses.replace(
            plan, items=[entry for entry in plan.items if entry.index in item_set]
        )
        for plan, item_set in zip(all_plans, remaining_items)
    ]


def traverse_state_dict(
    state_dict: Any,
    visitor: Any,
) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping will be traversed and ``visitor`` will be applied to the leaf elements.
    ``visitor`` will only be applied to elements in a list or a tuple, if the
    container contains tensors or mappings.
    """

    def _is_terminal(value: Any) -> bool:
        """Leaf-like container: no nested mappings/lists/tuples/tensors to recurse into."""
        values: Collection
        if isinstance(value, Mapping):
            return False
        if isinstance(value, (list, tuple)):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list, tuple)) and not _is_terminal(entry):
                return False
            if isinstance(entry, Tensor):
                return False
        return True

    def _traverse_obj(path: tuple[Any, ...], value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def flatten_state_dict(state_dict: Any) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
    """Flatten a nested state dict to dotted FQN keys; returns ``(flat_dict, fqn -> path)``."""
    fqn_names: dict[str, Any] = {}
    mappings: dict[str, tuple[Any, ...]] = {}

    def flat_copy(path: tuple[Any, ...], value: Any) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in fqn_names:
            raise ValueError(
                f"Duplicate flattened FQN {new_fqn!r} when converting nested state_dict; "
                "two different values map to the same dotted name."
            )
        fqn_names[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(state_dict, flat_copy)
    return fqn_names, mappings


def set_element(root_dict: Any, path: tuple[Any, ...], value: Any) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    if not path:
        raise ValueError("path must be non-empty")
    cur_container: Any = root_dict

    def extend_list(lst: list[Any], idx: int) -> None:
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        next_key = path[i]
        def_val: Any = {} if isinstance(next_key, str) else []

        if isinstance(cur_container, Mapping):
            cur_container = cur_container.setdefault(prev_key, def_val)
        else:
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    last_key = path[-1]
    if isinstance(last_key, int):
        extend_list(cur_container, last_key)

    cur_container[last_key] = value
