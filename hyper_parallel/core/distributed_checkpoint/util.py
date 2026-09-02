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
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterable, Optional, Union
from collections import deque
from collections.abc import Collection, Mapping

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    CHUNK_INFO,
    ChunkInfo,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    BroadcastSource,
    ReadItem,
    SavePlan,
)
from hyper_parallel.core.distributed_checkpoint.ragged_utils import (
    compute_ragged_boxes,
    get_ragged_box_tensor,
)
from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.platform import get_platform
from hyper_parallel.tools.logging import get_logger

platform = get_platform()
Tensor = platform.Tensor

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


def plan_ownership_masks(
    all_plans: list[SavePlan],
    save_to_minimum_rank: bool = False,
) -> list[bytearray]:
    """
    Decide which plan writes each item, as one keep-mask per plan.

    An item present in several plans is redundant: only one plan should write it. The owner is
    the plan with the smallest planned storage so far, or the lowest plan index when
    ``save_to_minimum_rank`` is True. Ownership is resolved from the plan order and the item
    sizes alone, so every rank running this over the same gathered plans reaches the same answer.

    Duplicates are assigned largest first (longest-processing-time): placing the big shards while
    the plans are still evenly loaded leaves the small ones to even out the remainder. Assigning
    in arrival order instead lets a late big shard land on an already-heavy plan, and the
    checkpoint's wall time is set by whichever plan writes the most.

    Masks are returned instead of filtered plans because the caller walks ``plan.items`` anyway:
    skipping on a mask costs no hashing and allocates no intermediate copy of every plan.

    Args:
        all_plans (list[SavePlan]): Local plans gathered from all ranks, indexed by rank.
        save_to_minimum_rank (bool): If True, assign duplicates to the lowest plan index; else to
            the plan holding the least data so far. Default False.

    Returns:
        list[bytearray]: One mask per plan, parallel to that plan's ``items``: 1 marks an item the
            plan owns and must write, 0 marks a duplicate another plan took.
    """
    # index -> [write_item, plan_idx, position, plan_idx, position, ...]. One flat list per
    # distinct item, so the common unique item costs a single dict lookup rather than an entry
    # in a duplicate map plus one in a registry plus one in a per-plan set.
    occurrences_by_index: dict[MetadataIndex, list] = {}
    for plan_idx, plan in enumerate(all_plans):
        for position, entry in enumerate(plan.items):
            occurrences = occurrences_by_index.get(entry.index)
            if occurrences is None:
                occurrences_by_index[entry.index] = [entry, plan_idx, position]
            else:
                occurrences.append(plan_idx)
                occurrences.append(position)

    masks = [bytearray(len(plan.items)) for plan in all_plans]
    storage_sizes = [0] * len(all_plans)

    # Unique items are assigned first so that they are all accounted for in storage_sizes before
    # duplicates are balanced against those sizes. Sizes are computed here, once per item.
    duplicates: list[tuple[int, list]] = []
    for occurrences in occurrences_by_index.values():
        item_size = occurrences[0].tensor_storage_size() or 1
        # The layout is [write_item] + (plan_idx, position) * holder_count: drop the slot the
        # item itself takes, then every two remaining slots are one plan holding it.
        holder_count = (len(occurrences) - 1) // 2
        if holder_count > 1:
            duplicates.append((item_size, occurrences))
            continue
        masks[occurrences[1]][occurrences[2]] = 1
        storage_sizes[occurrences[1]] += item_size

    # Largest first, so the shards with room to unbalance the plans are placed while every plan is
    # still a candidate. Python's sort is stable, including with reverse=True, so equally sized
    # duplicates keep their gather order and every rank still agrees on the owner. Sorting is
    # pointless when every duplicate goes to its lowest plan index regardless.
    if not save_to_minimum_rank:
        duplicates.sort(key=lambda pair: pair[0], reverse=True)

    for item_size, occurrences in duplicates:
        # Occurrences were appended in ascending plan order, so slot 1 is the lowest plan index
        # and the storage-size search breaks ties towards it.
        if save_to_minimum_rank:
            owner_slot = 1
        else:
            owner_slot = _least_loaded_owner_slot(occurrences, storage_sizes)
        owner_idx = occurrences[owner_slot]
        masks[owner_idx][occurrences[owner_slot + 1]] = 1
        storage_sizes[owner_idx] += item_size

    return masks


def _least_loaded_owner_slot(occurrences: list, storage_sizes: list[int]) -> int:
    """
    Pick the occurrence slot whose plan currently holds the least data.

    Args:
        occurrences (list): ``[write_item, plan_idx, position, ...]`` for one duplicated item.
        storage_sizes (list[int]): Bytes already assigned to each plan.

    Returns:
        int: Index into ``occurrences`` of the winning ``plan_idx`` (its position follows it).
    """
    # Slot 1 is the first plan_idx and every further holder sits two slots on. Comparing with a
    # strict < keeps the first minimum, which is the lowest plan index.
    best_slot = 1
    best_size = storage_sizes[occurrences[1]]
    for slot in range(3, len(occurrences), 2):
        size = storage_sizes[occurrences[slot]]
        if size < best_size:
            best_slot, best_size = slot, size
    return best_slot


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


# How many broadcasts one rank keeps going at once. Starting the next without waiting on
# the last is what lets a read overlap the send before it, but each one in flight holds
# resources inside the communication library, so the count is capped rather than left to
# grow with the number of shards in the checkpoint.
_MAX_BROADCASTS_IN_FLIGHT = 8

# Shards smaller than this travel together rather than one at a time. A broadcast of
# 64 KiB costs about as much as one of 1 MiB -- some 145 microseconds either way over
# four Ascend ranks -- and only reaches full speed past a few megabytes, so below that a
# load spends its time starting broadcasts rather than moving data. This sits just above
# where the two meet, measured at around 4.6 MiB. Note that a batch is held until its
# broadcast lands, so a load can have this much times the in-flight limit set aside.
DEFAULT_BROADCAST_BATCH_BYTES = 6 * 1024 * 1024


def _existing_group(group_ranks: tuple) -> Any:
    """
    A communication group over these ranks that is already there, or None if none is.

    The cache is where a group that already exists is normally found: the mesh a model is
    sharded over puts its tp columns and dp groups there, and they stay because training
    still needs them. Only when it holds nothing is the world worth considering - a
    parameter every rank has a copy of needs the group of every rank, which is the one the
    job runs on. That one has been there since initialization and is not in the cache, but
    it is no less already there, and remaking it would raise a second communicator over
    every rank for the sake of one read.

    Neither belongs to the load, so neither is destroyed when the load is done.

    Args:
        group_ranks (tuple): The ranks the group would hold.

    Returns:
        Any: The group, or None if it has to be created.
    """
    existing = platform.get_created_group(group_ranks)
    if existing is not None:
        return existing
    if group_ranks == tuple(range(platform.get_world_size())):
        return platform.get_world_group()
    return None


def _build_broadcast_groups(
    items: Iterable[ReadItem],
    supplied: dict[tuple, Any],
) -> tuple[dict[tuple, Any], set]:
    """
    The communication group of every shard this load broadcasts, and which of them it owns.

    Every rank has to call this, including one with no shard to broadcast at all: it
    all-gathers what is needed and takes part in creating all of it, and both of those are
    collective. A rank that stayed out because it had nothing of its own to add would leave
    the others waiting on it.

    A group that already exists is reused rather than made a second time - the mesh a model
    is sharded over has usually built the tp column or the dp group a replicated parameter
    needs. But **whether to reuse has to be decided the same way on every rank**:
    ``new_group`` is collective over the world, so a rank that skipped it because its own
    cache held the group, while another went ahead, would hang everybody. The cache is not
    symmetric on its own - ``DeviceMesh.from_group`` records only the group its rank is in -
    so the decision is taken from the gathered reports rather than from the local cache: a
    group is made afresh if it is absent on *any* rank that needs it, and reused only when
    it is absent on none.

    Groups made here are made with :meth:`platform.new_group`, which creates exactly the
    ranks it is given and hands the group straight back. :meth:`platform.create_group` would
    do more than is wanted: it takes the rank list as a *template*, expands it into a whole
    partition of the world, and keeps every group of it in a process-wide cache. A load uses
    each of these groups once - and the expansion refuses rank lists a pipeline-parallel
    model produces, such as a parameter tied across two stages that are not neighbours.

    Args:
        items (Iterable[ReadItem]): The read items of this rank finalized plan.
        supplied (dict[tuple, Any]): Communication groups the caller pre-built, keyed by
            their rank tuple. Kept as they are and never counted as owned.

    Returns:
        tuple[dict[tuple, Any], set]: A group for every rank tuple this rank broadcasts
            through, and the rank tuples whose groups this load made and must destroy.
    """
    needed = sorted({item.source.group_ranks for item in items
                     if item.source is not None and item.source.group_ranks not in supplied})
    absent = tuple(ranks for ranks in needed if _existing_group(ranks) is None)
    if absent:
        logger.warning("There are missing groups %s. Then all gather the missing groups on each rank and "
                       "create them one by one, which will increase some time consumption.", absent)

    gathered = all_gather_object(
        (tuple(needed), absent), platform.get_world_size(), use_collectives=True
    )
    wanted = sorted({ranks for reported, _ in gathered for ranks in reported})
    absent_somewhere = {ranks for _, reported in gathered for ranks in reported}

    groups = dict(supplied)
    owned = set()
    for group_ranks in wanted:
        # Both inputs to this came from the gather, so every rank takes the same branch.
        fresh = platform.new_group(group_ranks) if group_ranks in absent_somewhere else None
        if group_ranks not in needed:
            continue
        if fresh is None:
            groups[group_ranks] = _existing_group(group_ranks)
            continue
        groups[group_ranks] = fresh
        owned.add(group_ranks)
    return groups, owned


def ensure_broadcast_groups(
    items: Iterable[ReadItem],
    groups: Optional[dict[tuple, Any]] = None,
) -> dict[tuple, Any]:
    """
    The communication group of every shard this load broadcasts, built where one is missing.

    See :func:`_build_broadcast_groups` for how they are made. Whatever this builds is left
    to the caller to destroy; :func:`broadcast_groups_for_load` does that on its way out.

    Args:
        items (Iterable[ReadItem]): The read items of this rank finalized plan.
        groups (Optional[dict[tuple, Any]]): Communication groups the caller pre-built, keyed
            by their rank tuple. Kept as they are; only the missing ones are created.

    Returns:
        dict[tuple, Any]: A group for every rank tuple this rank broadcasts through.
    """
    return _build_broadcast_groups(items, dict(groups or {}))[0]


def _destroy_broadcast_groups(groups: dict[tuple, Any], owned: set) -> None:
    """
    Release the groups this load made, once it is done broadcasting through them.

    The broadcasts have been waited on but are not necessarily finished: a collective that
    has been waited on is only ordered against the device stream, so the transfer can still
    be queued when the call returns. Tearing the communicator down at that point kills the
    transfer - the receiving ranks silently keep whatever their buffer held - so drain the
    stream first.

    Releasing is then best effort: the tensors have really arrived by the time the loop
    runs, so a group that refuses to go away is a leak worth a warning rather than a reason
    to fail a load that already has its data.

    Only what this load made is passed here, and only by a rank that belongs to it:
    ``owned`` is built from this rank own reads, so a group it never joined never reaches
    this.

    Args:
        groups (dict[tuple, Any]): Every group the load broadcast through, keyed by rank tuple.
        owned (set): Rank tuples whose groups this load made and has to release.
    """
    if not owned:
        return
    platform.synchronize()
    # Destroying a group is collective over its members, so keep the same deterministic
    # order they were created in.
    for group_ranks in sorted(owned):
        try:
            platform.destroy_process_group(groups[group_ranks])
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to destroy the broadcast group %s: %s", group_ranks, e)


@contextmanager
def broadcast_groups_for_load(
    items: Iterable[ReadItem],
    groups: Optional[dict[tuple, Any]] = None,
) -> Any:
    """
    The communication groups one load broadcasts through, destroyed once it is done.

    A group made here exists to carry this load's broadcasts and nothing else, so it is torn
    down on the way out rather than left behind: a communicator holds device memory for as
    long as it lives, and a job that loads a checkpoint has no use for these afterwards.

    Only what this load made is destroyed. A group the caller pre-built and passed in
    belongs to the caller, and one that was already in the process-wide cache belongs to
    whoever put it there - the device mesh, most often, which needs it for the rest of
    training.

    Destroying is done in rank-tuple order, which every rank works out the same way, and
    only by the ranks of the group - the ones that reach the end of the read together.

    Args:
        items (Iterable[ReadItem]): The read items of this rank finalized plan.
        groups (Optional[dict[tuple, Any]]): Communication groups the caller pre-built.

    Yields:
        dict[tuple, Any]: A group for every rank tuple this rank broadcasts through.
    """
    built, owned = _build_broadcast_groups(items, dict(groups or {}))
    try:
        yield built
    finally:
        _destroy_broadcast_groups(built, owned)


def _start_broadcast(
    in_flight: deque,
    buffer: Any,
    source: BroadcastSource,
    groups: dict[tuple, Any],
    after: Optional[Any] = None,
) -> None:
    """
    Start one broadcast without waiting for it, making room for it first.

    Args:
        in_flight (deque): Broadcasts already going, oldest first. This one is added, after
            waiting on the oldest should too many already be going.
        buffer (Any): Contiguous memory to send from the source and receive into elsewhere.
        source (BroadcastSource): Which ranks take part and which of them sends.
        groups (dict[tuple, Any]): Communication groups, as :func:`ensure_broadcast_groups`
            returns them.
        after (Optional[Any]): Called once this broadcast has landed, for a send that is not
            finished when the bytes arrive -- a batch still to be dealt out to the shards it
            was gathered from.
    """
    while len(in_flight) >= _MAX_BROADCASTS_IN_FLIGHT:
        _finish_broadcast(in_flight.popleft())
    handle = platform.broadcast_async(buffer, source.src_rank, groups[source.group_ranks])
    if handle is None:
        if after is not None:
            after()
        return
    in_flight.append((handle, after))


def _finish_broadcast(pending: tuple) -> None:
    """Wait on one broadcast and do whatever was left until it had landed."""
    handle, after = pending
    handle.wait()
    if after is not None:
        after()


def broadcast_shard(
    in_flight: deque,
    state_dict: dict[str, Any],
    item: ReadItem,
    groups: dict[tuple, Any],
) -> None:
    """
    Start sending one shard between the ranks that hold it, without waiting for it.

    A shard travels whole rather than region by region: it is one local buffer, which is
    contiguous where a single region of it generally is not, and every rank holding it has it
    in the same shape. Shards are otherwise unrelated -- two of one tensor are two buffers,
    two groups and two broadcasts.

    Callers have to reach the shards of a group in the same order on every rank of it, since
    ranks that enter a group collectives in different orders deadlock. The order the global
    plan puts them in is the one thing every rank agrees on without asking.

    Args:
        in_flight (deque): Broadcasts already going, oldest first.
        state_dict (dict[str, Any]): Flat state dict holding the shard to send.
        item (ReadItem): Any item of the shard, which names it and who reads it.
        groups (dict[tuple, Any]): Communication groups, as :func:`ensure_broadcast_groups`
            returns them.
    """
    buffer = _shard_buffer(state_dict[item.dest_index.fqn], item.dest_index)
    _start_broadcast(in_flight, buffer, item.source, groups)


def wait_broadcasts(in_flight: deque) -> None:
    """
    Wait on every broadcast still going, oldest first.

    The buffers being sent are the state dict tensors themselves, so a load that carried on
    with one still in flight would be reading into memory a collective is still writing.

    Args:
        in_flight (deque): Broadcasts started so far. Emptied.
    """
    while in_flight:
        _finish_broadcast(in_flight.popleft())


class BroadcastBatcher:
    """
    Shards small enough that sending them one at a time would cost more than moving them.

    A broadcast costs about the same whether it carries 64 KiB or 1 MiB -- around 145
    microseconds either way, measured over four Ascend ranks, against 33 GiB/s once the
    shards are large. Below that crossing point a load spends its time starting broadcasts
    rather than moving data, and a checkpoint holds a great many small tensors: norms,
    biases, scalars, step counters. So shards under ``batch_bytes`` are gathered into one
    buffer, sent together, and dealt out again on the far side, while larger ones are sent
    as they are, the fixed cost being small against what they carry.

    Shards can only travel together when they agree on the group, the sending rank and the
    dtype, so one batch is kept per combination. Every rank of a group meets that group
    shards in the same order and so fills and sends the same batches at the same points,
    which is what keeps its collectives in step -- the same thing that lets shards be sent
    one at a time without agreeing on anything first.
    """

    def __init__(self, batch_bytes: int, groups: dict[tuple, Any]) -> None:
        """
        Args:
            batch_bytes (int): Shards this size or larger are sent on their own, and a batch
                is sent as soon as another shard would take it past this. Zero sends every
                shard on its own, which is what a platform whose tensors cannot be gathered
                into one buffer gets.
            groups (dict[tuple, Any]): Communication groups, as
                :func:`ensure_broadcast_groups` returns them.
        """
        self._batch_bytes = batch_bytes
        self._groups = groups
        self._batches: dict[tuple, list] = {}
        self._pending_bytes: dict[tuple, int] = {}
        self.batched = 0
        self.sent = 0

    def add(self, in_flight: deque, state_dict: dict[str, Any], item: ReadItem) -> None:
        """
        Hand one shard over to be sent, on its own or with others.

        Args:
            in_flight (deque): Broadcasts already going, oldest first.
            state_dict (dict[str, Any]): Flat state dict holding the shard to send.
            item (ReadItem): Any item of the shard, which names it and who sends it.
        """
        buffer = _shard_buffer(state_dict[item.dest_index.fqn], item.dest_index)
        nbytes = platform.get_tensor_storage_size(buffer)
        if nbytes >= self._batch_bytes:
            self.sent += 1
            _start_broadcast(in_flight, buffer, item.source, self._groups)
            return

        key = (item.source.group_ranks, item.source.src_rank, buffer.dtype)
        if self._pending_bytes.get(key, 0) + nbytes > self._batch_bytes:
            self._send(in_flight, key)
        self._batches.setdefault(key, []).append(buffer)
        self._pending_bytes[key] = self._pending_bytes.get(key, 0) + nbytes

    def flush(self, in_flight: deque) -> None:
        """
        Send whatever is still waiting to go with something else.

        Batches go in the order they were first added to, which follows the shards and so is
        the same on every rank that holds them.

        Args:
            in_flight (deque): Broadcasts already going, oldest first.
        """
        for key in list(self._batches):
            self._send(in_flight, key)

    def _send(self, in_flight: deque, key: tuple) -> None:
        """Send one batch, and arrange for it to be dealt out once it has landed."""
        buffers = self._batches.pop(key, [])
        self._pending_bytes.pop(key, None)
        if not buffers:
            return

        group_ranks, src_rank = key[:2]
        source = BroadcastSource(group_ranks=group_ranks, src_rank=src_rank)
        self.sent += 1
        if len(buffers) == 1:
            # Nothing to gather it with, so gathering it would only cost a round trip.
            _start_broadcast(in_flight, buffers[0], source, self._groups)
            return

        self.batched += len(buffers)
        # Left uninitialized: the views below cover it exactly, and it is written
        # whole either by gathering the shards into it or by the broadcast landing.
        staging = platform.new_tensor(
            (sum(buffer.numel() for buffer in buffers),),
            buffers[0].dtype,
            getattr(buffers[0], "device", None),
        )
        views, offset = [], 0
        for buffer in buffers:
            views.append(staging[offset:offset + buffer.numel()].reshape(buffer.shape))
            offset += buffer.numel()

        if source.src_rank == platform.get_rank():
            # The sender already holds every shard, so it gathers them and needs nothing back.
            platform.copy_each(views, buffers)
            _start_broadcast(in_flight, staging, source, self._groups)
            return
        _start_broadcast(
            in_flight, staging, source, self._groups,
            after=lambda: platform.copy_each(buffers, views),
        )


def _shard_buffer(obj: Any, index: MetadataIndex) -> Any:
    """
    The local buffer of one shard, which is what a single broadcast carries.

    Mirrors what :meth:`StandardLoadPlanner.acquire_tensor` narrows into, so that the rank
    reading a shard sends the same storage the others are waiting to have written.

    Args:
        obj (Any): The state dict entry the shard belongs to.
        index (MetadataIndex): Names the shard, as a read item destination does.

    Returns:
        Any: A tensor view over the shard, writable in place by a collective.
    """
    if isinstance(obj, DTensor):
        if obj.layout is not None and obj.layout.ragged_shard is not None:
            return get_ragged_box_tensor(obj, index).detach()
        return obj.to_local().detach()
    return obj.detach()
