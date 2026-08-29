# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
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

import time
import dataclasses
from pathlib import Path
from functools import wraps
from typing import Any, Union, Optional
from collections import defaultdict
from collections.abc import Collection, Mapping

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    CHUNK_INFO,
    ChunkInfo,
    BroadcastInfo,
)
from hyper_parallel.core.distributed_checkpoint.planner import SavePlan, WriteItem
from hyper_parallel.core.distributed_checkpoint.ragged_utils import compute_ragged_boxes
from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.platform import get_platform
from hyper_parallel.tools.logging import get_logger

platform = get_platform()
Tensor = platform.Tensor
BROADCAST_INFO = "broadcast_info"

# The one DCP logger: other distributed_checkpoint modules import this instead of
# registering a component of their own.
logger = get_logger("DCP")


def dcp_timer_decorator(func):
    """
    Used to collect statistics on the time consumed in each phase of the DCP.

    The timings are per-rank, so the rank is part of the message; enable them
    with ``HP_LOG_CONFIG=DCP:INFO``.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            rank_id = platform.get_rank()
        except ValueError:
            # No process group yet (offline converters, single-process tools).
            rank_id = 0
        logger.info("[rank=%d] >>> func %s start exec", rank_id, func.__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("[rank=%d] >>> func %s cost %.4f seconds", rank_id, func.__name__, execution_time)
        return result

    return wrapper

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
        slice_area = infer_slice_area_by_layout(
            layout,
            inner_rank_id,
            full_shape,
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


def infer_same_shard_ranks_for_dtensor(dtensor: DTensor) -> tuple[int, ...]:
    """
    Group global ranks that hold the same local shard for a DTensor.

    Shard identity is derived from each rank's global offsets, computed from the
    DTensor device mesh and placements (via layout).

    Args:
        dtensor (DTensor): The DTensor to analyze.

    Returns:
        tuple[int, ...]: Sorted global-rank tuples; each tuple is one
            same-shard group (length 1 when the shard is unique to one rank).
    """
    current_rank = platform.get_rank()
    layout = dtensor.layout
    if layout is None:
        return (current_rank,)

    mesh_shape = layout.mesh_shape
    tensor_map = layout.tensor_map
    rank_list = layout.rank_list

    if mesh_shape is None or tensor_map is None or rank_list is None:
        return (current_rank,)

    if current_rank not in rank_list:
        return (current_rank,)

    n_mesh_dims = len(mesh_shape)
    inner_rank_id = rank_list.index(current_rank)

    def dev_id_list_from_rank(inner_id: int) -> list[int]:
        dev_id_list = [0] * n_mesh_dims
        temp = inner_id
        for i in range(n_mesh_dims - 1, -1, -1):
            dev_id_list[i] = temp % mesh_shape[i]
            temp //= mesh_shape[i]
        return dev_id_list

    def compute_shard_key(dev_id_list: list[int]) -> tuple:
        key_parts = []
        for mapping in tensor_map:
            if isinstance(mapping, int):
                mapping = (mapping,) if mapping != -1 else ()
            elif not isinstance(mapping, tuple):
                mapping = (mapping,)
            if not mapping:
                continue
            shard_id = 0
            coef = 1
            for dim in reversed(mapping):
                if dim == -1:
                    continue
                shard_id += dev_id_list[-dim - 1] * coef
                coef *= mesh_shape[-dim - 1]
            key_parts.append(shard_id)
        return tuple(key_parts)

    current_dev_id_list = dev_id_list_from_rank(inner_rank_id)
    current_shard_key = compute_shard_key(current_dev_id_list)

    same_shard_ranks = []
    for idx, global_rank in enumerate(rank_list):
        if idx == inner_rank_id:
            same_shard_ranks.append(global_rank)
            continue
        other_dev_id_list = dev_id_list_from_rank(idx)
        if compute_shard_key(other_dev_id_list) == current_shard_key:
            same_shard_ranks.append(global_rank)

    return tuple(same_shard_ranks)


@dcp_timer_decorator
def all_gather_object(
    local_object: Any,
    world_size: int,
    use_collectives: bool,
) -> list[Any]:
    """
    Gather objects from all ranks.

    Args:
        local_object (Any): Local object for current rank.
        world_size (int): Total number of ranks.
        use_collectives (bool): Whether to use collective communication.

    Returns:
        list[Any]: List of all objects from all ranks.
    """
    if use_collectives and world_size > 1:
        all_objects = [None] * world_size
        platform.all_gather_object(all_objects, local_object)
        return all_objects
    return [local_object]


def _broadcast_within_existing_groups(
    state_dict: dict[str, Any],
    groups: dict[tuple, Any]
) -> dict[tuple, list[Any]]:
    """
    Broadcast the tensors whose same-shard group was pre-built by the caller.

    Args:
        state_dict (dict[str, Any]): Flat or nested state dict containing the entries whose
            local tensors were populated on the minimum rank of each group.
        groups (dict[tuple, Any]): Communication groups keyed by their rank tuple.

    Returns:
        dict[tuple, list[Any]]: The entries left untouched because ``groups`` has no group for
        them, keyed by the rank tuple of the group they still need.

    Raises:
        ValueError: If the broadcast info attached to an entry has an unexpected type.
    """
    missing_groups_ranks = {}
    for obj in state_dict.values():
        broadcast_info = getattr(obj, BROADCAST_INFO, None)
        if broadcast_info is None:
            continue
        if not isinstance(broadcast_info, BroadcastInfo):
            raise ValueError(f"The broadcast info attached to tensor must be of type {BroadcastInfo}.")
        group_ranks, src_rank = tuple(broadcast_info.group_ranks), broadcast_info.src_rank
        if group_ranks in groups:
            platform.broadcast(
                obj.to_local().detach() if isinstance(obj, DTensor) else obj.detach(),
                src_rank,
                groups[group_ranks]
            )
            delattr(obj, BROADCAST_INFO)
        else:
            missing_groups_ranks.setdefault(group_ranks, []).append(obj)
    return missing_groups_ranks


def _destroy_groups(groups: dict[tuple, Any]) -> None:
    """
    Release the groups built for one broadcast round.

    Only the ranks belonging to a group release it: a rank outside it never got a real group
    back from :meth:`Platform.new_group`, only a non-member placeholder.

    Releasing a group is best effort. The tensors are already broadcast by the time this runs,
    so a group that refuses to go away is a leak worth a warning, not a reason to fail the load.

    Args:
        groups (dict[tuple, Any]): Groups to release, keyed by their rank tuple.
    """
    current_rank = platform.get_rank()
    # Destroying a group is collective over its members, so keep the same deterministic order
    # the groups were created in.
    for group_ranks in sorted(groups):
        if current_rank not in group_ranks:
            continue
        try:
            platform.destroy_process_group(groups[group_ranks])
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to destroy the broadcast group %s: %s", group_ranks, e)


def _create_groups_and_broadcast(missing_groups_ranks: dict[tuple, list[Any]]) -> None:
    """
    Create the communication groups the caller did not provide, then broadcast through them.

    Creating a group is a collective call, so the rank tuples still missing are all-gathered
    first and every rank creates the whole set, not only the groups it needs itself.

    The groups live for this call only. Holding on to them would leak one set of communicators
    per load, because dropping the Python handle does not release the underlying communicator,
    and a loop that resumes repeatedly would run the backend out of them.

    Args:
        missing_groups_ranks (dict[tuple, list[Any]]): The entries waiting for a group, keyed
            by the rank tuple of the group they need.
    """
    logger.warning("There are missing groups %s. Then all gather the missing groups on each rank and "
                   "create them one by one, which will increase some time consumption.",
                   missing_groups_ranks)
    # all gather all missing groups, create them and broadcast the tensors
    all_missing_groups_ranks = all_gather_object(tuple(missing_groups_ranks.keys()),
                                                 platform.get_world_size(),
                                                 use_collectives=True)
    final_missing_groups_ranks = set(g for sub in all_missing_groups_ranks for g in sub)

    # The groups are only needed for the broadcasts right below, so create them without
    # registering them in the global group cache. Iterate in a deterministic order:
    # creating a group is a collective call and every rank must issue them in the same order.
    new_groups = {}
    for group_ranks in sorted(final_missing_groups_ranks):
        new_groups[group_ranks] = platform.new_group(group_ranks)
    for group_ranks, tensors in missing_groups_ranks.items():
        for tensor in tensors:
            broadcast_info = getattr(tensor, BROADCAST_INFO, None)
            if broadcast_info is None:
                continue
            platform.broadcast(
                tensor.to_local().detach() if isinstance(tensor, DTensor) else tensor.detach(),
                broadcast_info.src_rank,
                new_groups[group_ranks]
            )
            delattr(tensor, BROADCAST_INFO)

    # Only reached once every broadcast went through: releasing a group whose collectives just
    # failed can block instead of raising, which would turn a clean error into a hang.
    _destroy_groups(new_groups)


@dcp_timer_decorator
def broadcast_loaded_tensors(
    state_dict: dict[str, Any],
    groups: Optional[dict[tuple, Any]] = None
) -> None:
    """
    Broadcast loaded tensor shard from the src rank in each same-shard group.

    Non have attribute BROADCAST_INFO entries in ``state_dict`` are ignored.

    Args:
        state_dict (dict[str, Any]): Flat or nested state dict containing DTensors
            whose local tensors were populated on the minimum rank of each group.
        groups (dict): The Communication groups for broadcast.
    """
    # A caller that enabled broadcasting without pre-building groups lands in the
    # missing-groups path below, which creates them on demand.
    missing_groups_ranks = _broadcast_within_existing_groups(state_dict, groups or {})

    # if no group missing, return
    if not missing_groups_ranks:
        return

    _create_groups_and_broadcast(missing_groups_ranks)
