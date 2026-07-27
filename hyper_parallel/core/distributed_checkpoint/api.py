# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Hyper Parallel Checkpoint API"""
import multiprocessing as mp
import queue
import threading
import traceback
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Collection, Optional, Union

from hyper_parallel.core.distributed_checkpoint.async_staging import build_staged_state_dict
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner, StandardLoadPlanner
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader, FileSystemWriter
from hyper_parallel.core.distributed_checkpoint.metadata import Metadata
from hyper_parallel.core.distributed_checkpoint.planner import SavePlanner, LoadPlanner
from hyper_parallel.core.distributed_checkpoint.storage import StorageReader, StorageWriter
from hyper_parallel.core.distributed_checkpoint.versioning import migrate_metadata
from hyper_parallel.platform import get_platform

platform = get_platform()


class _AsyncPersistStatus(Enum):
    """Queue payload status from :func:`_async_persist_worker` to the parent join thread."""

    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class AsyncSaveResponse:
    """Result of :func:`async_save`.

    Host staging runs synchronously before :func:`async_save` returns; only checkpoint
    **persistence** is asynchronous. ``persist_completion`` completes when the child
    process finishes :func:`_save_impl` (plan, collectives, disk I/O) and supplies
    :class:`Metadata`.
    """

    persist_completion: Future[Metadata]


def _gather_from_all_ranks(
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


def _validate_incremental_params(
    incremental_from: Optional[Union[Path, str]],
    changed_fqns: Optional[Collection[str]],
    storage_writer: Optional[StorageWriter],
) -> None:
    """Validate incremental save parameters.

    Args:
        incremental_from: Baseline checkpoint directory for incremental save.
        changed_fqns: Set of FQNs that have changed relative to the baseline.
        storage_writer: Custom storage writer, if any.

    Raises:
        ValueError: If incremental parameters are partially provided, if a
            custom *storage_writer* is used with incremental save, or if
            *changed_fqns* contains invalid entries.
    """
    if (incremental_from is None) != (changed_fqns is None):
        raise ValueError(
            "incremental_from and changed_fqns must be provided together or both omitted."
        )
    if incremental_from is not None and storage_writer is not None:
        raise ValueError(
            "Incremental save is only supported with the default FileSystemWriter; "
            "passing a custom storage_writer is not allowed."
        )
    if changed_fqns is not None:
        for fqn in changed_fqns:
            if not isinstance(fqn, str) or not fqn:
                raise ValueError(
                    f"Each item in changed_fqns must be a non-empty string, got {fqn!r}."
                )


def _build_save_plan(
    planner: SavePlanner,
    storage_writer: StorageWriter,
    world_size: int,
    use_collectives: bool,
) -> tuple[Any, Metadata]:
    """Build and finalize the save plan, returning (final_plan, metadata).

    Uses the planner cache when available; otherwise builds from scratch.

    Args:
        planner: Configured save planner.
        storage_writer: Configured storage writer.
        world_size: Total number of ranks.
        use_collectives: Whether to use collective communication.

    Returns:
        A pair ``(final_plan, metadata)``.
    """
    cached_res = planner.get_cached() if hasattr(planner, 'get_cached') else None
    if cached_res:
        return cached_res.final_plan, cached_res.metadata

    local_plan = planner.build_local_plan()
    local_plan = storage_writer.optimize_local_plan(local_plan)

    all_local_plans = _gather_from_all_ranks(local_plan, world_size, use_collectives)
    global_plans, metadata = planner.build_global_plan(all_local_plans)
    global_plans = storage_writer.optimize_global_plan(global_plans)

    rank = platform.get_rank()
    if use_collectives and world_size > 1 and global_plans:
        central_plan = global_plans[rank]
    elif global_plans:
        central_plan = global_plans[0]
    else:
        central_plan = local_plan

    final_plan = planner.finalize_plan(central_plan)
    if hasattr(planner, 'cache_result'):
        planner.cache_result(final_plan, metadata)
    return final_plan, metadata


def _save_impl(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: Optional[Union[Path, str]] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    no_dist: bool = False,
    use_collectives: bool = True,
    incremental_from: Optional[Union[Path, str]] = None,
    changed_fqns: Optional[Collection[str]] = None,
) -> Metadata:
    """Synchronous distributed checkpoint save (shared by :func:`save` and :func:`async_save`).

    Args:
        state_dict: The state_dict to save.
        checkpoint_id: Checkpoint directory path.
        storage_writer: Custom storage writer. Default None.
        planner: Custom save planner. Default None.
        no_dist: Single-process mode. Default False.
        use_collectives: Use collective communication. Default True.
        incremental_from: Baseline checkpoint directory for incremental save.
            Must be provided together with *changed_fqns*. Default None.
        changed_fqns: Set of dot-separated FQNs that have changed relative to
            the baseline. Must be provided together with *incremental_from*.
            Default None.

    Returns:
        Metadata: Metadata object for the saved checkpoint.

    Raises:
        ValueError: If incremental parameters are partially provided, if a
            custom *storage_writer* is used with incremental save, or if
            *changed_fqns* contains invalid entries.
    """
    _validate_incremental_params(incremental_from, changed_fqns, storage_writer)

    checkpoint_id = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id
    use_collectives = False if no_dist else use_collectives

    if storage_writer is None:
        if checkpoint_id is None:
            raise ValueError("Either storage_writer or checkpoint_id must be provided")
        storage_writer = FileSystemWriter(
            checkpoint_id,
            incremental_from=incremental_from,
            changed_fqns=changed_fqns,
        )
    else:
        if checkpoint_id:
            storage_writer.initialize_writer(checkpoint_id)

    planner = StandardSavePlanner() if planner is None else planner

    rank = platform.get_rank()
    world_size = platform.get_world_size()
    is_coordinator = rank == 0
    is_incremental = incremental_from is not None

    planner.configure_planner(
        state_dict=state_dict,
        is_coordinator=is_coordinator,
        rank=rank,
        use_collectives=use_collectives,
        incremental=is_incremental,
    )
    storage_writer.configure_writer(
        is_coordinator=is_coordinator,
        rank=rank,
        use_collectives=use_collectives,
    )

    final_plan, metadata = _build_save_plan(planner, storage_writer, world_size, use_collectives)

    write_results = storage_writer.execute_write(final_plan, planner)
    all_write_results = _gather_from_all_ranks(write_results, world_size, use_collectives)
    storage_writer.finalize_checkpoint(metadata, all_write_results)

    return metadata


def _async_persist_worker(
        result_queue: mp.Queue,
        staged: dict[str, Any],
        checkpoint_id: Optional[Union[Path, str]],
        storage_writer: Optional[StorageWriter],
        planner: Optional[SavePlanner],
        no_dist: bool,
        use_collectives: bool,
        incremental_from: Optional[Union[Path, str]],
        changed_fqns: Optional[Collection[str]],
) -> None:
    """Child-process entry: run :func:`_save_impl` and report ``Metadata`` or an error string on ``result_queue``."""
    try:
        meta = _save_impl(
            staged,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
            incremental_from=incremental_from,
            changed_fqns=changed_fqns,
        )
        result_queue.put((_AsyncPersistStatus.SUCCESS, meta))
    except Exception:  # pylint: disable=broad-except
        result_queue.put((_AsyncPersistStatus.FAILURE, traceback.format_exc()))


def _async_persist_wait_process(
        proc: mp.Process,
        result_queue: mp.Queue,
        persist_future: Future[Metadata],
) -> None:
    """Join persist ``proc`` and complete ``persist_future`` (runs on a daemon thread)."""
    proc.join()
    if persist_future.done():
        return
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        persist_future.set_exception(
            RuntimeError(
                f"async_persist process exited with code {proc.exitcode} and no result on queue"
            )
        )
        return
    if status == _AsyncPersistStatus.SUCCESS:
        persist_future.set_result(payload)
    elif status == _AsyncPersistStatus.FAILURE:
        persist_future.set_exception(RuntimeError(payload))
    else:
        persist_future.set_exception(
            RuntimeError(f"async_persist queue returned unexpected status: {status!r}")
        )


def save(
        state_dict: dict[str, Any],
        *,
        checkpoint_id: Optional[Union[Path, str]] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        no_dist: bool = False,
        use_collectives: bool = True,
        incremental_from: Optional[Union[Path, str]] = None,
        changed_fqns: Optional[Collection[str]] = None,
) -> Metadata:
    """
    Save a distributed checkpoint in SPMD style.

    This function saves a state_dict containing DTensors, where each rank
    only saves their local shards.

    Args:
        state_dict (dict[str, Any]): The state_dict to save.
        checkpoint_id (Optional[Union[Path, str]]): The ID/path of this checkpoint instance (can be Path or str).
            Default None.
        storage_writer (Optional[StorageWriter]): Instance of StorageWriter. If None, FileSystemWriter
            will be created based on checkpoint_id. Default None.
        planner (Optional[SavePlanner]): Instance of SavePlanner. If None, StandardSavePlanner will be used.
            Default None.
        no_dist (bool): If True, save in single process mode. Default False.
        use_collectives (bool): If True, use collective communication for coordination.
            If False, each rank saves its own shard data and rank-local metadata (.metadata_rank{rank}),
            with no cross-rank interaction. Default True.
        incremental_from (Optional[Union[Path, str]]): Baseline checkpoint directory for incremental save.
            Must be provided together with *changed_fqns*. Default None.
        changed_fqns (Optional[Collection[str]]): Set of dot-separated FQNs that have changed relative
            to the baseline. Must be provided together with *incremental_from*. Default None.

    Returns:
        Metadata: Metadata object for the saved checkpoint.
    """
    metadata = _save_impl(
        state_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        no_dist=no_dist,
        use_collectives=use_collectives,
        incremental_from=incremental_from,
        changed_fqns=changed_fqns,
    )
    platform.barrier()
    return metadata


def async_save(
        state_dict: dict[str, Any],
        *,
        checkpoint_id: Optional[Union[Path, str]] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        no_dist: bool = False,
        use_collectives: bool = True,
        incremental_from: Optional[Union[Path, str]] = None,
        changed_fqns: Optional[Collection[str]] = None,
) -> AsyncSaveResponse:
    """
    Asynchronous version of :func:`save` using a **background child process** for persistence.

    **Staging** (tensor / DTensor → host copy) runs **synchronously in the caller
    process** via :func:`build_staged_state_dict`, so no process pool is used for
    staging and the training stack sees a normal Python call path. When this
    function returns successfully, host staging is done and the original
    ``state_dict`` may be mutated.

    **Persistence** (plan, collectives, disk I/O) runs in **one** background
    :class:`multiprocessing.Process` that executes :func:`_save_impl` on the staged
    dict. A small daemon **thread** only joins that process and fills
    ``persist_completion``; it does not perform tensor work.

    The staged dict and ``storage_writer`` / ``planner`` must be picklable for the
    persist child process (same constraints as before for the worker path).

    .. warning::
        Experimental API. Always wait on ``persist_completion`` for a fully persisted checkpoint.

    Args:
        state_dict (dict[str, Any]): The state_dict to save.
        checkpoint_id (Optional[Union[Path, str]]): Same as :func:`save`.
        storage_writer (Optional[StorageWriter]): Same as :func:`save`.
        planner (Optional[SavePlanner]): Same as :func:`save`.
        no_dist (bool): Same as :func:`save`.
        use_collectives (bool): Same as :func:`save`.
        incremental_from (Optional[Union[Path, str]]): Same as :func:`save`.
        changed_fqns (Optional[Collection[str]]): Same as :func:`save`.

    Returns:
        AsyncSaveResponse: Contains ``persist_completion`` only; staging is synchronous.
    """
    persist_completion: Future[Metadata] = Future()

    staged = build_staged_state_dict(state_dict)

    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_async_persist_worker,
        args=(
            result_queue,
            staged,
            checkpoint_id,
            storage_writer,
            planner,
            no_dist,
            use_collectives,
            incremental_from,
            changed_fqns,
        ),
        name="HPAsyncCheckpointPersist",
    )
    proc.start()
    join_thread = threading.Thread(
        target=_async_persist_wait_process,
        args=(proc, result_queue, persist_completion),
        daemon=True,
        name="HPAsyncCheckpointPersistJoin",
    )
    join_thread.start()
    return AsyncSaveResponse(persist_completion=persist_completion)


def _build_load_plan(
    planner: LoadPlanner,
    storage_reader: StorageReader,
    world_size: int,
    use_collectives: bool,
) -> Any:
    """Build and finalize the load plan.

    Args:
        planner: Configured load planner.
        storage_reader: Configured storage reader.
        world_size: Total number of ranks.
        use_collectives: Whether to use collective communication.

    Returns:
        The finalized load plan.
    """
    local_plan = planner.build_local_plan()
    local_plan = storage_reader.optimize_local_plan(local_plan)

    all_local_plans = _gather_from_all_ranks(local_plan, world_size, use_collectives)
    global_plans = planner.build_global_plan(all_local_plans)
    global_plans = storage_reader.optimize_global_plan(global_plans)

    rank = platform.get_rank()
    if use_collectives and world_size > 1 and global_plans:
        central_plan = global_plans[rank]
    elif global_plans:
        central_plan = global_plans[0]
    else:
        central_plan = local_plan

    return planner.finalize_plan(central_plan)


def load(
        state_dict: dict[str, Any],
        *,
        checkpoint_id: Optional[Union[Path, str]] = None,
        storage_reader: Optional[StorageReader] = None,
        planner: Optional[LoadPlanner] = None,
        no_dist: bool = False,
        use_collectives: bool = True,
) -> None:
    """
    Load a distributed checkpoint into state_dict in SPMD style.

    Each rank will try to read the least amount of data necessary
    to fulfill the requested state_dict. When loading DTensor instances,
    each rank only reads data for their local shards.

    Args:
        state_dict (dict[str, Any]): The state_dict to load the checkpoint into (modified in-place).
        checkpoint_id (Optional[Union[Path, str]]): The ID/path of this checkpoint instance (can be Path or str).
            Default None.
        storage_reader (Optional[StorageReader]): Instance of StorageReader. If None, FileSystemReader
            will be created based on checkpoint_id. Default None.
        planner (Optional[LoadPlanner]): Instance of LoadPlanner. If None, StandardLoadPlanner will be used.
            Default None.
        no_dist (bool): If True, load without cross-rank synchronization. Default False.
        use_collectives (bool): If False, load from rank-local metadata (.metadata_rank{rank}),
            for checkpoints saved with save(use_collectives=False). No cross-rank interaction. Default True.

    Returns:
        None. The state_dict is modified in-place.
    """
    checkpoint_id = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id
    use_collectives = False if no_dist else use_collectives

    if storage_reader is None:
        if checkpoint_id is None:
            raise ValueError("Either storage_reader or checkpoint_id must be provided")
        storage_reader = FileSystemReader(checkpoint_id)
    else:
        if checkpoint_id:
            storage_reader.initialize_reader(checkpoint_id)

    planner = StandardLoadPlanner() if planner is None else planner
    rank = platform.get_rank()
    world_size = platform.get_world_size()
    is_coordinator = rank == 0

    try:
        metadata = storage_reader.load_metadata()
    except FileNotFoundError:
        metadata = storage_reader.load_metadata(rank=rank)
        use_collectives = False

    metadata = migrate_metadata(metadata)

    planner.configure_planner(
        state_dict=state_dict, metadata=metadata,
        is_coordinator=is_coordinator, rank=rank,
    )
    storage_reader.configure_reader(
        metadata=metadata, is_coordinator=is_coordinator,
        rank=rank, use_collectives=use_collectives,
    )

    final_plan = _build_load_plan(planner, storage_reader, world_size, use_collectives)
    storage_reader.execute_read(final_plan, planner)
