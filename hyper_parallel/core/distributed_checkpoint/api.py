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
import pickle
import queue
import threading
import traceback
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Union

from hyper_parallel.core.distributed_checkpoint.async_staging import build_staged_state_dict
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner, StandardLoadPlanner
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader, FileSystemWriter
from hyper_parallel.core.distributed_checkpoint.metadata import Metadata
from hyper_parallel.core.distributed_checkpoint.planner import SavePlanner, LoadPlanner
from hyper_parallel.core.distributed_checkpoint.storage import StorageReader, StorageWriter
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
        serialization_error = None
        serialized_object = b""
        try:
            serialized_object = pickle.dumps(local_object)
        except Exception as error:  # pylint: disable=W0718
            serialization_error = error
        _raise_if_stage_failed(
            serialization_error,
            "collective payload serialization",
            world_size,
            use_collectives,
        )
        serialized_objects = [b""] * world_size
        platform.all_gather_object(serialized_objects, serialized_object)
        deserialization_error = None
        all_objects = []
        try:
            all_objects = [pickle.loads(value) for value in serialized_objects]
        except Exception as error:  # pylint: disable=W0718
            deserialization_error = error
        _raise_if_stage_failed(
            deserialization_error,
            "collective payload deserialization",
            world_size,
            use_collectives,
        )
        return all_objects
    return [local_object]


def _raise_if_stage_failed(
    local_error: Optional[Exception],
    operation: str,
    world_size: int,
    use_collectives: bool,
) -> None:
    """Exchange rank-local errors before entering the next checkpoint phase."""
    if use_collectives and world_size > 1:
        errors = [None] * world_size
        platform.all_gather_object(
            errors,
            None if local_error is None else str(local_error),
        )
        if any(error is not None for error in errors):
            raise RuntimeError(
                f"Distributed checkpoint {operation} failed on one or more ranks: {errors}"
            ) from local_error
    elif local_error is not None:
        raise local_error


def _save_impl(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: Optional[Union[Path, str]] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    no_dist: bool = False,
    use_collectives: bool = True,
    rank_override: Optional[int] = None,
    world_size_override: Optional[int] = None,
) -> Metadata:
    """Synchronous distributed checkpoint save (shared by :func:`save` and :func:`async_save`)."""
    # Convert checkpoint_id to Path if it's a string
    checkpoint_id = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

    # Determine if we're in distributed mode
    use_collectives = False if no_dist else use_collectives

    # Get rank and coordinator info
    rank = (
        rank_override
        if rank_override is not None
        else (0 if no_dist else platform.get_rank())
    )
    world_size = (
        world_size_override
        if world_size_override is not None
        else (1 if no_dist else platform.get_world_size())
    )
    is_coordinator = rank == 0

    setup_error = None
    cached_res = None
    local_plan = None
    try:
        if storage_writer is None:
            if checkpoint_id is None:
                raise ValueError("Either storage_writer or checkpoint_id must be provided")
            storage_writer = FileSystemWriter(checkpoint_id)
        elif checkpoint_id:
            storage_writer.initialize_writer(checkpoint_id)
        planner = StandardSavePlanner() if planner is None else planner
        planner.configure_planner(
            state_dict=state_dict,
            is_coordinator=is_coordinator,
            rank=rank,
            use_collectives=use_collectives,
        )
        storage_writer.configure_writer(
            is_coordinator=is_coordinator,
            rank=rank,
            use_collectives=use_collectives,
        )
        cached_res = planner.get_cached() if hasattr(planner, "get_cached") else None
        if cached_res is None:
            local_plan = storage_writer.optimize_local_plan(planner.build_local_plan())
    except Exception as error:  # pylint: disable=W0718
        setup_error = error
    _raise_if_stage_failed(setup_error, "save planning setup", world_size, use_collectives)

    plan_error = None
    try:
        cache_states = _gather_from_all_ranks(
            (
                None
                if cached_res is None
                else (
                    cached_res.metadata.state_dict_metadata,
                    cached_res.metadata.planner_data,
                    cached_res.metadata.version,
                )
            ),
            world_size,
            use_collectives,
        )
        cache_hits = [cache_state is not None for cache_state in cache_states]
        if any(cache_hits) and not all(cache_hits):
            raise RuntimeError("Distributed checkpoint save-plan caches differ across ranks")
        if all(cache_hits) and any(cache_state != cache_states[0] for cache_state in cache_states[1:]):
            raise RuntimeError("Distributed checkpoint cached metadata differs across ranks")
        if cached_res:
            final_plan, metadata = cached_res.final_plan, cached_res.metadata
        else:
            # Gather all local plans and build global plan.
            all_local_plans = _gather_from_all_ranks(local_plan, world_size, use_collectives)
            global_plans, metadata = planner.build_global_plan(all_local_plans)
            global_plans = storage_writer.optimize_global_plan(global_plans)

            # Select central plan for current rank.
            if use_collectives and world_size > 1 and global_plans:
                central_plan = global_plans[rank]
            elif global_plans:
                central_plan = global_plans[0]
            else:
                central_plan = local_plan
            final_plan = planner.finalize_plan(central_plan)
            if hasattr(planner, "cache_result"):
                planner.cache_result(final_plan, metadata)
    except Exception as error:  # pylint: disable=W0718
        plan_error = error
    _raise_if_stage_failed(plan_error, "save global planning", world_size, use_collectives)

    # Propagate rank-local write failures before entering result collection.
    write_error = None
    try:
        write_results = storage_writer.execute_write(final_plan, planner)
    except Exception as error:  # pylint: disable=W0718
        write_error = error
        write_results = []
    _raise_if_stage_failed(write_error, "write", world_size, use_collectives)

    # Finalize checkpoint
    all_write_results = _gather_from_all_ranks(write_results, world_size, use_collectives)
    finalize_error = None
    try:
        storage_writer.finalize_checkpoint(metadata, all_write_results)
    except Exception as error:  # pylint: disable=W0718
        finalize_error = error
    _raise_if_stage_failed(finalize_error, "finalization", world_size, use_collectives)

    return metadata


def _async_persist_worker(
        result_queue: mp.Queue,
        staged: dict[str, Any],
        checkpoint_id: Optional[Union[Path, str]],
        storage_writer: Optional[StorageWriter],
        planner: Optional[SavePlanner],
        no_dist: bool,
        use_collectives: bool,
        rank: int,
        world_size: int,
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
            rank_override=rank,
            world_size_override=world_size,
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
    while True:
        try:
            status, payload = result_queue.get(timeout=0.1)
            break
        except queue.Empty:
            if proc.is_alive():
                continue
            proc.join()
            try:
                status, payload = result_queue.get(timeout=0.1)
                break
            except queue.Empty:
                persist_future.set_exception(
                    RuntimeError(
                        f"async_persist process exited with code {proc.exitcode} and no result on queue"
                    )
                )
                return
    proc.join()
    if persist_future.done():
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

    Returns:
        Metadata: Metadata object for the saved checkpoint.
    """
    return _save_impl(
        state_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )


def async_save(
        state_dict: dict[str, Any],
        *,
        checkpoint_id: Optional[Union[Path, str]] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        no_dist: bool = False,
        use_collectives: bool = False,
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

    Returns:
        AsyncSaveResponse: Contains ``persist_completion`` only; staging is synchronous.
    """
    persist_completion: Future[Metadata] = Future()
    if use_collectives and not no_dist:
        raise ValueError(
            "async_save does not support collectives from its persistence child; "
            "use use_collectives=False"
        )

    staged = build_staged_state_dict(state_dict)
    rank = 0 if no_dist else platform.get_rank()
    world_size = 1 if no_dist else platform.get_world_size()

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
            rank,
            world_size,
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
    # Convert checkpoint_id to Path if it's a string
    checkpoint_id = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

    # Determine if we're in distributed mode
    use_collectives = False if no_dist else use_collectives

    # Get rank and coordinator info
    rank = 0 if no_dist else platform.get_rank()
    world_size = 1 if no_dist else platform.get_world_size()
    is_coordinator = rank == 0

    metadata_error = None
    try:
        if storage_reader is None:
            if checkpoint_id is None:
                raise ValueError("Either storage_reader or checkpoint_id must be provided")
            storage_reader = FileSystemReader(checkpoint_id)
        elif checkpoint_id:
            storage_reader.initialize_reader(checkpoint_id)
        planner = StandardLoadPlanner() if planner is None else planner
        if use_collectives:
            metadata = storage_reader.load_metadata()
        else:
            metadata = storage_reader.load_metadata(rank=rank)
    except Exception as error:  # pylint: disable=W0718
        metadata_error = error
    _raise_if_stage_failed(metadata_error, "load metadata", world_size, use_collectives)

    setup_error = None
    try:
        planner.configure_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
            rank=rank,
        )
        storage_reader.configure_reader(
            metadata=metadata,
            is_coordinator=is_coordinator,
            rank=rank,
            use_collectives=use_collectives,
        )
        local_plan = storage_reader.optimize_local_plan(planner.build_local_plan())
    except Exception as error:  # pylint: disable=W0718
        setup_error = error
    _raise_if_stage_failed(setup_error, "load planning setup", world_size, use_collectives)

    # Gather all local plans and build global plan
    all_local_plans = _gather_from_all_ranks(local_plan, world_size, use_collectives)
    plan_error = None
    try:
        global_plans = planner.build_global_plan(all_local_plans)
        global_plans = storage_reader.optimize_global_plan(global_plans)
        if use_collectives and world_size > 1 and global_plans:
            central_plan = global_plans[rank]
        elif global_plans:
            central_plan = global_plans[0]
        else:
            central_plan = local_plan
        final_plan = planner.finalize_plan(central_plan)
    except Exception as error:  # pylint: disable=W0718
        plan_error = error
    _raise_if_stage_failed(plan_error, "load global planning", world_size, use_collectives)

    read_error = None
    try:
        storage_reader.execute_read(final_plan, planner)
    except Exception as error:  # pylint: disable=W0718
        read_error = error
    _raise_if_stage_failed(read_error, "read", world_size, use_collectives)
