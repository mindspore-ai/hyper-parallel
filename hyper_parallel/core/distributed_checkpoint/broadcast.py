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
"""Sending a shard one rank read to the ranks that hold a copy of it.

A checkpoint holds one copy of a replicated parameter, so on load exactly one rank of each
replica set reads it and broadcasts it to the others. This module owns that half of a load:
the communication groups it goes through, the sends themselves, and the batching that keeps
small shards from each costing a round trip.
"""
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterable, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from hyper_parallel.core.distributed_checkpoint.metadata import MetadataIndex
from hyper_parallel.core.distributed_checkpoint.planner import BroadcastSource, ReadItem
from hyper_parallel.core.distributed_checkpoint.ragged import get_ragged_box_tensor
from hyper_parallel.core.distributed_checkpoint.utils import (
    all_gather_object,
    get_tensor_storage_size,
    logger,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.high_availability.group_registry import GroupRegistry
from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS


def copy_each(dests: list, srcs: list) -> None:
    """
    Copy every pair in one fused operation rather than one at a time.

    Measured on an Ascend card, five hundred copies of 64 KiB take 11.8 ms one at a
    time and 1.4 ms fused: nearly all of what a small copy costs is starting it.

    Args:
        dests (list): Tensors written into.
        srcs (list): Tensors read from, pairing with ``dests`` by position.
    """
    if dests:
        torch._foreach_copy_(dests, srcs)  # pylint: disable=protected-access


def get_created_group(rank_list: Union[list[int], tuple[int, ...]]) -> Any:
    """
    The process group over these ranks that the process-wide cache already holds.

    Args:
        rank_list (Union[list[int], tuple[int, ...]]): The ranks the group would hold.

    Returns:
        Any: The cached group, or None when the cache holds none over these ranks.
    """
    return EXISTING_COMM_GROUPS.get(str(tuple(sorted(rank_list))))


def synchronize() -> None:
    """
    Block the host until the work queued on the current device stream has finished.

    Nothing can be queued on a device stream when the process has no device, and there are
    two ways to have none: a CPU-only install carries no ``torch.npu`` at all, and a gloo run
    on an accelerator box never initializes the one it has.
    """
    device = getattr(torch, "npu", None)
    if device is None or not device.is_initialized():
        return
    device.current_stream().synchronize()


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
    existing = get_created_group(group_ranks)
    if existing is not None:
        return existing
    if group_ranks == tuple(range(dist.get_world_size())):
        return _get_default_group()
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

    Groups made here are made with :func:`torch.distributed.new_group`, which creates exactly
    the ranks it is given and hands the group straight back. The project's ``create_group``
    would do more than is wanted: it takes the rank list as a *template*, expands it into a
    whole partition of the world, and keeps every group of it in a process-wide cache. A load
    uses each of these groups once - and the expansion refuses rank lists a pipeline-parallel
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
        (tuple(needed), absent), dist.get_world_size(), use_collectives=True
    )
    wanted = sorted({ranks for reported, _ in gathered for ranks in reported})
    absent_somewhere = {ranks for _, reported in gathered for ranks in reported}

    groups = dict(supplied)
    owned = set()
    create_group = dist.new_group
    registry = getattr(create_group, "__self__", None)
    if isinstance(registry, GroupRegistry):
        # These groups are destroyed when loading finishes, so they must not
        # enter the persistent recovery catalog or its sealed-group checks.
        create_group = registry.call_original_new_group
    for group_ranks in wanted:
        # Both inputs to this came from the gather, so every rank takes the same branch.
        fresh = create_group(ranks=list(group_ranks)) if group_ranks in absent_somewhere else None
        if group_ranks not in needed:
            continue
        if fresh is None:
            groups[group_ranks] = _existing_group(group_ranks)
            continue
        groups[group_ranks] = fresh
        owned.add(group_ranks)
    return groups, owned


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
    synchronize()
    # Destroying a group is collective over its members, so keep the same deterministic
    # order they were created in.
    for group_ranks in sorted(owned):
        try:
            dist.destroy_process_group(groups[group_ranks])
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
        groups (dict[tuple, Any]): Communication groups, as :func:`broadcast_groups_for_load`
            returns them.
        after (Optional[Any]): Called once this broadcast has landed, for a send that is not
            finished when the bytes arrive -- a batch still to be dealt out to the shards it
            was gathered from.
    """
    while len(in_flight) >= _MAX_BROADCASTS_IN_FLIGHT:
        _finish_broadcast(in_flight.popleft())
    handle = dist.broadcast(buffer, source.src_rank, groups[source.group_ranks], async_op=True)
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
        groups (dict[tuple, Any]): Communication groups, as :func:`broadcast_groups_for_load`
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
                shard on its own.
            groups (dict[tuple, Any]): Communication groups, as
                :func:`broadcast_groups_for_load` yields them.
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
        nbytes = get_tensor_storage_size(buffer)
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
        staging = torch.empty(
            size=(sum(buffer.numel() for buffer in buffers),),
            dtype=buffers[0].dtype,
            device=getattr(buffers[0], "device", None),
        )
        views, offset = [], 0
        for buffer in buffers:
            views.append(staging[offset:offset + buffer.numel()].reshape(buffer.shape))
            offset += buffer.numel()

        if source.src_rank == dist.get_rank():
            # The sender already holds every shard, so it gathers them and needs nothing back.
            copy_each(views, buffers)
            _start_broadcast(in_flight, staging, source, self._groups)
            return
        _start_broadcast(
            in_flight, staging, source, self._groups,
            after=lambda: copy_each(buffers, views),
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
