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
"""Native sparse transport with persistent credits and no virtual expert axis."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from hyper_parallel.core.multicore._loader import get_routed_experts_paths
from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor, RouteEdge
from hyper_parallel.core.multicore.modules.routed_experts.readiness import ExpertReadiness, upload_metadata
from hyper_parallel.core.multicore.modules.routed_experts.workspace import RingWorkspace


@lru_cache(maxsize=1)
def _load_native() -> ctypes.CDLL:
    """Load only the independent fat kernel and its Torch registration adapter."""
    kernel, adapter = get_routed_experts_paths()
    handle = ctypes.CDLL(str(kernel), mode=ctypes.RTLD_LOCAL)
    torch.ops.load_library(str(adapter))
    return handle


def _pack_blocks(edges: tuple[RouteEdge, ...], ranks: int, block_rows: int,
                 peer_column: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Expand dense host rows in stable peer/edge/block order without edge objects."""
    values = [(edge.source, edge.target, edge.expert, edge.source_row, edge.target_row, edge.rows) for edge in edges]
    if (any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63
            for row in values for value in row)
            or any(row[3] + row[5] >= 2**63 or row[4] + row[5] >= 2**63 for row in values)):
        raise ValueError("transport edge fields and exclusive row ends must be nonnegative int64 integers")
    host = np.asarray(values, dtype=np.int64).reshape(-1, 6)
    host = host[np.argsort(host[:, peer_column], kind="stable")]
    counts = (host[:, 5] - 1) // block_rows + 1
    totals = np.zeros(ranks, dtype=np.int64)
    np.add.at(totals, host[:, peer_column], counts)
    if np.any(totals >= 2**30):
        raise RuntimeError("transport sequences require a collective drained reset before reuse")
    ends = totals.cumsum()
    packed = np.repeat(host, counts, axis=0)
    starts = counts.cumsum() - counts
    displacement = (np.arange(len(packed), dtype=np.int64) - np.repeat(starts, counts)) * block_rows
    packed[:, 3] += displacement
    packed[:, 4] += displacement
    packed[:, 5] = np.minimum(block_rows, packed[:, 5] - displacement)
    return torch.from_numpy(packed), ends - totals, ends


def _directional_sequences(workspace: RingWorkspace, direction: int, ranks: int) -> tuple[list[int], list[int]]:
    """Validate mutable credit bases even when immutable block tables are cached."""
    send, receive = workspace.send_sequence[direction], workspace.receive_sequence[direction]
    for sequences in (send, receive):
        if (len(sequences) != ranks
                or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**30
                       for value in sequences)):
            raise RuntimeError("transport sequences require a collective drained reset before reuse")
    return send, receive


def compile_transport_metadata(route: RouteDescriptor, workspace: RingWorkspace,
                               reverse: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack actual blocks and persistent per-peer sequence bases on the host."""
    block_rows = workspace.spec.block_rows
    if isinstance(block_rows, bool) or not isinstance(block_rows, int) or not 1 <= block_rows <= 1024:
        raise ValueError("block_rows must be a positive integer")
    direction = int(reverse)
    send_sequence, receive_sequence = _directional_sequences(workspace, direction, route.ranks)
    key = route, "host_transport", block_rows
    cached = workspace.metadata.get(key)
    # Equal Python values can differ in type (False == 0); only the validated
    # immutable instance may bypass packing's strict field checks.
    if cached is None or cached[0] is not route:
        route.validate()
        if not 0 <= route.source_rows < 2**63 or not 0 <= route.received_rows < 2**63:
            raise ValueError("transport row extents must fit nonnegative int64")
        send, send_begin, send_end = _pack_blocks(route.outgoing, route.ranks, block_rows, 1)
        receive, receive_begin, receive_end = _pack_blocks(route.incoming, route.ranks, block_rows, 0)
        ranges = np.stack((send_begin, send_end, receive_begin, receive_end), axis=1)
        cached = route, send, receive, ranges
        workspace.metadata[key] = cached
    _, send, receive, ranges = cached
    if reverse:
        # Peer grouping is preserved when sender/receiver and their row columns swap.
        send, receive = (torch.from_numpy(receive.numpy()[:, [1, 0, 2, 4, 3, 5]].copy()),
                         torch.from_numpy(send.numpy()[:, [1, 0, 2, 4, 3, 5]].copy()))
        ranges = ranges[:, [2, 3, 0, 1]]
    else:
        send, receive = send.clone(), receive.clone()
    offsets = np.column_stack((ranges, np.asarray(send_sequence, dtype=np.int64),
                               np.asarray(receive_sequence, dtype=np.int64)))
    if (np.any(offsets[:, 4] + offsets[:, 1] - offsets[:, 0] >= 2**30)
            or np.any(offsets[:, 5] + offsets[:, 3] - offsets[:, 2] >= 2**30)):
        raise RuntimeError("transport sequences require a collective drained reset before reuse")
    return send, receive, torch.from_numpy(offsets)


def exchange_rows(source: torch.Tensor, route: RouteDescriptor, workspace: RingWorkspace, *,
                  mapping: torch.Tensor | None = None, reverse: bool = False,
                  output: torch.Tensor | None = None) -> torch.Tensor:
    """Exchange one sparse phase; mapping gathers source tokens without expansion.

    Args:
        source: Contiguous BF16 [rows, hidden] source, on the current NPU.
        route: Immutable route shared with the inverse phase.
        workspace: Root-wide positive-capacity window lease.
        mapping: Optional INT64 selection-to-source-row map, already on device.
        reverse: Return expert-major destination rows to source selection order.
        output: Optional preallocated output with the exact destination extent.

    Returns:
        Asynchronously populated expert-major rows, ordered on the current stream.
    """
    with workspace.submission(workspace.device):
        return _exchange_rows(source, route, workspace, mapping=mapping, reverse=reverse, output=output,
                              input_ready=None, publish=False)[0]


def _exchange_rows(source: torch.Tensor, route: RouteDescriptor, workspace: RingWorkspace, *,
                   mapping: torch.Tensor | None, reverse: bool, output: torch.Tensor | None,
                   input_ready: ExpertReadiness | None, publish: bool) -> tuple:
    """Submit one phase while holding the window's sequence and lifetime lease."""
    preparation = None
    error = None
    try:
        if input_ready is not None:
            input_ready.validate_polling()
        preparation = prepare_exchange(source, route, workspace, mapping=mapping, reverse=reverse,
                                       output=output, dependency=input_ready, publish=publish)
    except Exception as exc:  # Every rank must reach preflight, including allocation failures.
        error = exc
    workspace.check_preflight(error)
    _submit_payload(preparation, workspace, input_ready)
    return preparation.value, preparation.ready


def _check_tensor(value: torch.Tensor, workspace: RingWorkspace) -> None:
    """Check host-visible properties before the collective preparation decision."""
    if value.device != workspace.device or not value.is_contiguous():
        raise ValueError("transport tensors must be contiguous on the workspace device")
    storage = value.untyped_storage()
    if value.numel() and (storage.data_ptr() == 0
                          or storage.nbytes() < (value.storage_offset() + value.numel()) * value.element_size()):
        raise ValueError("transport tensor storage has been invalidated or is undersized")


def _check_transport_aliases(tensors: tuple[torch.Tensor, ...]) -> None:
    """Reject writable storage overlap after all tensor geometry checks."""
    for writable in (1, 3, 4, *range(5, len(tensors))):
        for index, value in enumerate(tensors):
            if (writable != index and value.numel() and tensors[writable].numel()
                    and tensors[writable].untyped_storage().data_ptr() == value.untyped_storage().data_ptr()):
                raise ValueError("transport writable storage must not overlap another argument")


def _cacheline_words(value: int) -> int:
    """Round peer-owned scalar tables to a 64-byte boundary."""
    return ((value + 7) // 8) * 8


def _check_transport_source(source: torch.Tensor, route: RouteDescriptor, workspace: RingWorkspace,
                            mapping: torch.Tensor | None, reverse: bool) -> None:
    """Check source rank and selection extent before allocating the destination."""
    if route.rank != workspace.rank or route.ranks != workspace.spec.ranks:
        raise ValueError("route rank does not match the transport root")
    if source.ndim != 2 or source.shape[1] != workspace.spec.hidden or source.dtype != torch.bfloat16:
        raise ValueError("transport source must match the BF16 window row width")
    source_rows = route.received_rows if reverse else route.source_rows
    if mapping is None and source.shape[0] != source_rows:
        raise ValueError("ungathered source must cover the exact source selection extent")


def _prepare_arguments(source: torch.Tensor, route: RouteDescriptor, workspace: RingWorkspace,
                       mapping: torch.Tensor | None, reverse: bool, output: torch.Tensor | None) -> tuple:
    """Allocate transport destinations and validate operands before native setup."""
    _check_transport_source(source, route, workspace, mapping, reverse)
    rows = route.source_rows if reverse else route.received_rows
    output = source.new_empty((rows, source.shape[1])) if output is None else output
    if output.shape != (rows, source.shape[1]) or output.dtype != source.dtype:
        raise ValueError("transport output does not match the destination extent")
    if mapping is not None and mapping.numel() != (route.received_rows if reverse else route.source_rows):
        raise ValueError("transport mapping must cover every source selection exactly once")
    indices = mapping if mapping is not None else torch.empty(0, dtype=torch.int64, device=source.device)
    if indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("transport mapping must be a one-dimensional int64 tensor")
    for value in (source, output, indices, workspace.data, workspace.signals, workspace.headers):
        _check_tensor(value, workspace)
    _check_transport_aliases((source, output, indices, workspace.data, workspace.signals, workspace.headers))
    return output, indices


def _prepare(source: torch.Tensor, route: RouteDescriptor, workspace: RingWorkspace,
             mapping: torch.Tensor | None, reverse: bool, output: torch.Tensor | None) -> tuple:
    """Allocate and validate every local argument before payload submission."""
    output, indices = _prepare_arguments(source, route, workspace, mapping, reverse, output)
    _load_native()
    current = torch.npu.current_stream(source.device)
    cache_key = route, reverse
    cached = workspace.metadata.get(cache_key)
    if cached is None or cached[4] is not route:
        send, receive, offsets = compile_transport_metadata(route, workspace, reverse)
        ranges = tuple(tuple(row[:4]) for row in offsets.tolist())
        send = upload_metadata(send, source.device)
        receive = upload_metadata(receive, source.device)
        initialized = torch.npu.Event()
        initialized.record(current)
        workspace.metadata[cache_key] = send, receive, ranges, initialized, route
    else:
        send, receive, ranges, initialized, _ = cached
        current.wait_event(initialized)
    direction = int(reverse)
    send_sequence, receive_sequence = _directional_sequences(workspace, direction, route.ranks)
    host_offsets = [list(row) + [send_sequence[peer], receive_sequence[peer]]
                    for peer, row in enumerate(ranges)]
    if any(max(row[4] + row[1] - row[0], row[5] + row[3] - row[2]) >= 2**30 for row in host_offsets):
        raise RuntimeError("transport sequences require a collective drained reset before reuse")
    offsets = torch.tensor(host_offsets, dtype=torch.int64)
    workers = 2 * route.ranks if route.ranks <= 24 else route.ranks
    progress = torch.zeros((workers, 8), dtype=torch.int64, device=source.device)
    return output, indices, send, receive, upload_metadata(offsets, source.device), host_offsets, progress, current


@dataclass(frozen=True)
class PendingExchange:
    """Hold transport output and fences until its actual consumers are queued."""

    value: torch.Tensor
    ready: ExpertReadiness | None
    complete: Any

    def wait(self) -> torch.Tensor:
        """Join the full payload only where all source selections are needed."""
        current = torch.npu.current_stream(self.value.device)
        current.wait_event(self.complete)
        self.value.record_stream(current)
        return self.value


@dataclass
class PreparedExchange:
    """Own one unsubmitted phase, including initialized output readiness.

    Instances belong to one workspace and one directional sequence snapshot.
    Preparation does not reserve or advance that sequence. The caller must
    converge all phase preparation failures before submitting any payload.
    """

    value: torch.Tensor
    ready: ExpertReadiness | None
    source: torch.Tensor
    route: RouteDescriptor
    workspace: RingWorkspace
    reverse: bool
    mapped: bool
    indices: torch.Tensor
    send: torch.Tensor
    receive: torch.Tensor
    offsets: torch.Tensor
    host_offsets: tuple[tuple[int, ...], ...]
    progress: torch.Tensor
    slots: torch.Tensor | None
    dependency: ExpertReadiness | None
    initialized: Any
    source_admitted: Any
    complete: Any
    generation: int
    dedup_state: torch.Tensor | None = None
    group_metadata: torch.Tensor | None = None
    reduce: bool = False
    publish_each_block: bool = False
    reduce_dense_ratio: float = 1.0
    forward_compact_max_unique_ratio: float = 1.0
    submitted: bool = False


def _task_readiness_metadata(route: RouteDescriptor, block_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile each tile's last intersecting block ordinal for every source peer."""
    experts = {expert: index for index, expert in enumerate(route.active_experts)}
    expected = np.zeros((route.ranks, len(experts)), dtype=np.int32)
    starts = (0, *route.boundaries[:-1])
    tile_counts = [(end - begin + 127) // 128 for begin, end in zip(starts, route.boundaries)]
    tile_offsets = np.cumsum((0, *tile_counts[:-1]))
    task_expected = np.zeros((route.ranks, sum(tile_counts)), dtype=np.int32)
    # Each peer's receiver consumes its blocks in stable edge order. A tile can
    # read after the last intersecting block for that peer has published its count.
    for peer in range(route.ranks):
        for edge in route.incoming:
            if edge.source != peer:
                continue
            slot = experts[edge.expert]
            first = starts[slot]
            for block in edge.blocks(block_rows):
                expected[peer, slot] += 1
                first_tile = tile_offsets[slot] + (block.target_row - first) // 128
                last_tile = tile_offsets[slot] + (block.target_row + block.rows - 1 - first) // 128
                task_expected[peer, first_tile:last_tile + 1] = expected[peer, slot]
    if task_expected.size and not np.all(task_expected.sum(axis=0) > 0):
        raise ValueError("incoming blocks do not cover every forward compute tile")
    return torch.from_numpy(expected), torch.from_numpy(task_expected)


def _create_readiness_metadata(route: RouteDescriptor, workspace: RingWorkspace,
                               publish: bool, publish_each_block: bool, current: Any) -> tuple:
    """Initialize one immutable readiness template in the original upload order."""
    experts = {expert: index for index, expert in enumerate(route.active_experts)}
    count = 1 + max((edge.expert for edge in (*route.incoming, *route.outgoing)), default=-1)
    slots = np.full((count,), -1, dtype=np.int64)
    for expert, index in experts.items():
        slots[expert] = index
    slots = upload_metadata(torch.from_numpy(slots), workspace.device)
    if publish_each_block:
        expected, task_expected = _task_readiness_metadata(route, workspace.spec.block_rows)
    else:
        counts = np.zeros((route.ranks, len(experts)), dtype=np.int32)
        for edge in route.incoming:
            counts[edge.source, experts[edge.expert]] += (
                edge.rows + workspace.spec.block_rows - 1) // workspace.spec.block_rows
        expected, task_expected = torch.from_numpy(counts), None
    target = upload_metadata(expected, workspace.device) if publish else None
    task_target = upload_metadata(task_expected, workspace.device) if publish and task_expected is not None else None
    initialized = torch.npu.Event()
    initialized.record(current)
    workspace.metadata[(route, "readiness")] = slots, expected, initialized, target, task_expected, task_target
    return slots, expected, initialized, target, task_expected, task_target


def _prepare_readiness(route: RouteDescriptor, workspace: RingWorkspace, reverse: bool,
                       dependency: ExpertReadiness | None, publish: bool, *,
                       publish_each_block: bool = False, current: Any = None) -> tuple:
    """Count actual received blocks per producer/expert, including self copies."""
    if publish and reverse:
        raise ValueError("source-major reverse output does not have compact expert readiness")
    if not publish and dependency is None and not reverse:
        return None, None
    current = torch.npu.current_stream(workspace.device) if current is None else current
    cached = workspace.metadata.get((route, "readiness"))
    if cached is None:
        slots, expected, initialized, target, task_expected, task_target = _create_readiness_metadata(
            route, workspace, publish, publish_each_block, current)
    else:
        slots, expected, initialized, target, task_expected, task_target = cached
        current.wait_event(initialized)
        if publish and (target is None or publish_each_block and task_target is None):
            if publish_each_block and task_expected is None:
                tile_counts, task_expected = _task_readiness_metadata(route, workspace.spec.block_rows)
                if not torch.equal(tile_counts, expected):
                    raise ValueError("cached expert and tile block counts disagree")
            if target is None:
                target = upload_metadata(expected, workspace.device)
            if publish_each_block:
                task_target = upload_metadata(task_expected, workspace.device)
            initialized = torch.npu.Event()
            initialized.record(current)
            workspace.metadata[(route, "readiness")] = (slots, expected, initialized, target,
                                                         task_expected, task_target)
    ready = (ExpertReadiness.create(expected, workspace.device, expected_device=target,
                                    task_expected=task_target if publish_each_block else None)
             if publish else None)
    return ready, slots


def prepare_exchange(source: torch.Tensor, route: RouteDescriptor, workspace: RingWorkspace, *,
                     mapping: torch.Tensor | None = None, reverse: bool = False,
                     output: torch.Tensor | None = None, dependency: ExpertReadiness | None = None,
                     publish: bool | None = None, dedup: bool = False,
                     group_metadata: torch.Tensor | None = None, reduce: bool = False,
                     publish_each_block: bool = False,
                     reduce_dense_ratio: float = 1.0,
                     forward_compact_max_unique_ratio: float = 1.0) -> PreparedExchange:
    """Allocate and initialize a token phase on the current stream without communication.

    Args:
        source: BF16 source storage; a later producer may populate it before submission.
        route: Immutable route for this workspace and direction.
        workspace: Window whose directional sequence is captured without advancing it.
        mapping: Optional device INT64 selection-to-source-row map.
        reverse: Return expert rows to the original source selections.
        output: Optional destination storage, including a buffer reused after an earlier T phase.
        dependency: Optional producer fence initialized before any dependent T work is queued.
        publish: Publish compact expert readiness; defaults to ``not reverse``.
        dedup: Count unique source tokens per destination peer on the device during forward dispatch.
        group_metadata: One 64-byte group metadata line per received expert row, retained for backward.
        reduce: Return one forward-recorded carrier per token and peer after summing duplicate expert rows.
        reduce_dense_ratio: Use dense reverse traffic for a peer when its unique fraction reaches this value.
        forward_compact_max_unique_ratio: Compact a forward block only below or at this unique fraction.

    Returns:
        A single-use preparation exposing ``value`` and initialized ``ready``.

    Raises:
        ValueError: The arguments violate the transport contract.
        RuntimeError: The workspace is unavailable or its sequence needs a drained reset.

    Note:
        No collective preflight is performed. The caller must converge preparation
        failures for the complete phase before any rank calls ``submit_prepared``.
        Reverse slot metadata is prepared even when its producer fence is bound later.
    """
    if workspace.closed or workspace.poisoned or workspace.closing:
        raise RuntimeError("routed-expert workspace is not usable")
    if dedup and (reverse or mapping is None or group_metadata is None):
        raise ValueError("device-counted dedup requires a forward mapping and target token output")
    if reduce and (not reverse or mapping is not None or group_metadata is None):
        raise ValueError("block return reduction requires reverse expert rows and forward group metadata")
    if dedup and reduce:
        raise ValueError("one token phase cannot count and reduce simultaneously")
    if not 0.0 <= reduce_dense_ratio <= 1.0:
        raise ValueError("dense return ratio must be between zero and one")
    if not 0.0 <= forward_compact_max_unique_ratio <= 1.0:
        raise ValueError("forward compact ratio must be between zero and one")
    if (dedup or reduce) and workspace.spec.hidden % 32:
        raise ValueError("device-counted transport requires 64-byte aligned BF16 rows")
    if not (dedup or reduce) and group_metadata is not None:
        raise ValueError("target token metadata requires counted dispatch or reduced return")
    if dedup and (source.shape[0] >= 2**31 or route.received_rows >= 2**31
                  or any(edge.target_row + edge.rows >= 2**31 for edge in route.outgoing)):
        raise ValueError("device-counted block references require token and target rows below 2**31")
    if group_metadata is not None:
        if (group_metadata.device != workspace.device or group_metadata.dtype != torch.int64
                or group_metadata.shape != (route.received_rows, 8) or not group_metadata.is_contiguous()):
            raise ValueError("target token metadata must provide one cache line per received expert row")
        _check_tensor(group_metadata, workspace)
        if group_metadata.numel() and group_metadata.data_ptr() % 64:
            raise ValueError("target token metadata must start on a 64-byte boundary")
    if publish is None:
        publish = not reverse
    if publish_each_block and (reverse or not publish):
        raise ValueError("per-block readiness requires a published forward exchange")
    values = _prepare(source, route, workspace, mapping, reverse, output)
    value, indices, send, receive, offsets, host_offsets, progress, current = values
    if group_metadata is not None:
        _check_transport_aliases((source, value, indices, workspace.data, workspace.signals,
                                  workspace.headers, group_metadata))
    ready, slots = _prepare_readiness(route, workspace, reverse, dependency, publish,
                                     publish_each_block=publish_each_block, current=current)
    _check_dependency(dependency, route, workspace)
    initialized = torch.npu.Event()
    initialized.record(current)
    dedup_state = None
    if dedup:
        # One sender owns each peer's token table; no atomic or host count is needed.
        dedup_state = torch.zeros((2, route.ranks, _cacheline_words(max(1, source.shape[0]))),
                                  dtype=torch.int64, device=workspace.device)
        initialized.record(current)
    return PreparedExchange(value, ready, source, route, workspace, reverse, mapping is not None,
                            indices, send, receive, offsets, tuple(tuple(row) for row in host_offsets),
                            progress, slots, dependency, initialized, torch.npu.Event(), torch.npu.Event(),
                            workspace.metadata.get(("transport_generation", int(reverse)), 0),
                            dedup_state, group_metadata, reduce, publish_each_block, reduce_dense_ratio,
                            forward_compact_max_unique_ratio)


def _check_dependency(dependency: ExpertReadiness | None, route: RouteDescriptor,
                      workspace: RingWorkspace) -> None:
    """Reject host-visible producer-fence errors before the phase preflight."""
    if dependency is None:
        return
    flags, expected = dependency.flags, dependency.expected
    for value in (flags, expected):
        _check_tensor(value, workspace)
    if (flags.dtype != torch.int32 or flags.ndim != 3 or not 1 <= flags.shape[0] <= 48
            or flags.shape[1:] != (len(route.active_experts), 16)
            or expected.dtype != torch.int32 or expected.shape != flags.shape[:2]):
        raise ValueError("transport dependency must cover the compact active experts")
    if flags.numel() and flags.data_ptr() % 64:
        raise ValueError("transport dependency cache lines must be 64-byte aligned")


def _check_preparation(preparation: PreparedExchange, workspace: RingWorkspace) -> tuple[int, tuple]:
    """Reject foreign, reused or stale preparations before dependency checks."""
    if preparation.workspace is not workspace:
        raise ValueError("prepared exchange belongs to another workspace")
    if preparation.submitted:
        raise RuntimeError("prepared exchange may only be submitted once")
    direction = int(preparation.reverse)
    generation_key = "transport_generation", direction
    if preparation.generation != workspace.metadata.get(generation_key, 0):
        raise RuntimeError("prepared exchange has a stale directional generation")
    for peer, row in enumerate(preparation.host_offsets):
        if (workspace.send_sequence[direction][peer] != row[4]
                or workspace.receive_sequence[direction][peer] != row[5]):
            raise RuntimeError("prepared exchange has stale directional sequences")
    return direction, generation_key


def _submit_payload(preparation: PreparedExchange, workspace: RingWorkspace,
                    dependency: ExpertReadiness | None, *, current: Any = None) -> None:
    """Submit under the workspace lease, rejecting stale or duplicate preparations."""
    direction, generation_key = _check_preparation(preparation, workspace)
    if dependency is not None and preparation.slots is None:
        raise ValueError("late readiness requires a prepared expert slot map")
    _check_dependency(dependency, preparation.route, workspace)
    if dependency is not None:
        dependency.validate_polling()
    current = torch.npu.current_stream(workspace.device) if current is None else current
    current.wait_event(preparation.initialized)
    if dependency is not None:
        current.wait_event(dependency.admitted)
        current.wait_event(dependency.poll_after.event)
    preparation.submitted = True
    workspace.last_progress = preparation.progress
    route, ready = preparation.route, preparation.ready
    torch.ops.hyper_parallel.routed_experts_transport(
        preparation.source, preparation.value, preparation.indices, preparation.send, preparation.receive,
        preparation.offsets, workspace.data[direction], workspace.signals[direction], preparation.progress,
        route.rank, route.ranks, workspace.spec.lanes, workspace.spec.block_rows, preparation.mapped,
        60_000_000_000, workspace.headers[direction], None if dependency is None else dependency.flags,
        None if dependency is None else dependency.expected, None if ready is None else ready.flags,
        preparation.slots, preparation.dedup_state, preparation.group_metadata,
        preparation.reduce, preparation.publish_each_block, preparation.reduce_dense_ratio,
        preparation.forward_compact_max_unique_ratio,
    )
    for peer, row in enumerate(preparation.host_offsets):
        if peer != route.rank:
            workspace.send_sequence[direction][peer] += row[1] - row[0]
            workspace.receive_sequence[direction][peer] += row[3] - row[2]
    workspace.metadata[generation_key] = preparation.generation + 1
    preparation.complete.record(current)


def submit_prepared(preparation: PreparedExchange, workspace: RingWorkspace, *,
                    dependency: ExpertReadiness | None = None) -> PendingExchange:
    """Submit prepared arguments on T after the caller's complete-phase preflight.

    A nonempty dependency overrides the producer fence supplied at preparation.
    Its admission and producer-predecessor events are joined; native consumers
    poll actual producer progress. Submit a prepared linear producer before
    submitting its transport consumer. Without a fence, call from the source
    producer's stream after its writes have been queued. No payload/control tensor
    allocation or collective preflight is performed on T. Same-direction intervening submissions invalidate
    this preparation, including phases with only local copies or zero rows.
    """
    dependency = preparation.dependency if dependency is None else dependency
    if dependency is not None:
        dependency.validate_polling()
    current = torch.npu.current_stream(workspace.device)
    if dependency is None:
        preparation.source_admitted.record(current)
    with torch.npu.stream(workspace.transport_stream):
        with workspace.submission(workspace.device):
            if dependency is None:
                workspace.transport_stream.wait_event(preparation.source_admitted)
            _submit_payload(preparation, workspace, dependency, current=workspace.transport_stream)
    if preparation.ready is not None:
        current.wait_event(preparation.ready.admitted)
        preparation.value.record_stream(current)
    return PendingExchange(preparation.value, preparation.ready, preparation.complete)


def begin_exchange(source: torch.Tensor, route: RouteDescriptor, workspace: RingWorkspace, *,
                   mapping: torch.Tensor | None = None, reverse: bool = False,
                   dependency: ExpertReadiness | None = None) -> PendingExchange:
    """Start a token phase on T; compact consumers can wait for individual experts.

    A dependency admits T before the producing GEMM finishes, avoiding a stream
    cycle. Without one, the source's current stream is joined before T reads it.
    """
    preparation, error = None, None
    try:
        if dependency is not None:
            dependency.validate_polling()
        preparation = prepare_exchange(source, route, workspace, mapping=mapping, reverse=reverse,
                                       dependency=dependency)
    except Exception as exc:
        error = exc
    workspace.check_preflight(error)
    return submit_prepared(preparation, workspace)
