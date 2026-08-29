# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
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

import os
import threading
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional, Union

from hyper_parallel.core.distributed_checkpoint.async_persist import (
    AsyncSaveResponse,
    FileType,
    build_staged_state_dict,
    execute_async_persist_with_gloo,
    execute_async_persist,
    resolve_async_persist_result,
    gather_all_results_from_storage,
)
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import (
    FileSystemReader,
    FileSystemWriter,
)
from hyper_parallel.core.distributed_checkpoint.metadata import (
    Metadata,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    SavePlanner,
    LoadPlanner,
)
from hyper_parallel.core.distributed_checkpoint.standard_planner import (
    StandardSavePlanner,
    StandardLoadPlanner,
)
from hyper_parallel.core.distributed_checkpoint.storage import (
    StorageReader,
    StorageWriter,
)
from hyper_parallel.core.distributed_checkpoint.util import (
    all_gather_object,
    dcp_timer_decorator,
    logger,
    platform,
)

_async_save_count = 0


@dcp_timer_decorator
def _save_impl(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: Optional[Union[Path, str]] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    no_dist: bool = False,
    use_collectives: bool = True,
    use_storage_comm: bool = False,
) -> Metadata:
    """Distributed checkpoint save implement(shared by :func:`save` and :func:`async_save`)."""
    # Convert checkpoint_id to Path if it's a string
    checkpoint_id = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

    # Determine if we're in distributed mode
    use_collectives = False if no_dist else use_collectives
    use_storage_comm = False if not use_collectives else use_storage_comm

    # Set up storage writer
    if storage_writer is None:
        if checkpoint_id is None:
            raise ValueError("Either storage_writer or checkpoint_id must be provided")
        storage_writer = FileSystemWriter(checkpoint_id)
    else:
        if checkpoint_id:
            storage_writer.initialize_writer(checkpoint_id)
        else:
            checkpoint_id = getattr(storage_writer, "checkpoint_dir", None)

    if use_storage_comm and checkpoint_id is None:
        raise ValueError(
            "Coordinating through the storage needs a checkpoint directory to exchange the "
            "plans and write results in: pass checkpoint_id, or a storage_writer exposing "
            "checkpoint_dir."
        )

    # Set up planner
    planner = StandardSavePlanner() if planner is None else planner

    # Get rank and coordinator info
    rank = platform.get_rank()
    world_size = platform.get_world_size()
    is_coordinator = rank == 0

    # Configure planner
    planner.configure_planner(
        state_dict=state_dict,
        is_coordinator=is_coordinator,
        rank=rank,
        use_collectives=use_collectives
    )

    # Configure storage writer (use_collectives for rank-local metadata when False)
    storage_writer.configure_writer(
        is_coordinator=is_coordinator,
        rank=rank,
        use_collectives=use_collectives
    )

    @dcp_timer_decorator
    def generate_final_plan_and_metadata():
        # Build local plan
        local_plan = planner.build_local_plan()
        local_plan = storage_writer.optimize_local_plan(local_plan)

        #Gather all local plans and build global plan
        all_local_plans = gather_all_results_from_storage(checkpoint_id,
                                                          is_coordinator,
                                                          world_size,
                                                          local_plan,
                                                          FileType.LOCAL_PLAN,
                                                          rank) if use_storage_comm \
            else all_gather_object(local_plan, world_size, use_collectives)
        global_plans, global_metadata = planner.build_global_plan(all_local_plans)
        global_plans = storage_writer.optimize_global_plan(global_plans)
        # Select central plan for current rank
        if use_collectives and world_size > 1 and global_plans:
            central_plan = global_plans[rank]
        elif global_plans:
            central_plan = global_plans[0]
        else:
            central_plan = local_plan

        # Finalize and cache plan
        finalized_plan = planner.finalize_plan(central_plan)
        # Add final plan and metadata to the cache
        if hasattr(planner, 'cache_result'):
            planner.cache_result(finalized_plan, global_metadata)
        return finalized_plan, global_metadata

    cached_res = planner.get_cached() if hasattr(planner, 'get_cached') else None
    if cached_res:
        # Get final plan and metadata from cache
        logger.info("Hit final plan and metadata cache.")
        final_plan, metadata = cached_res.final_plan, cached_res.metadata
    else:
        # First time generating a plan and metadata.
        final_plan, metadata = generate_final_plan_and_metadata()

    # Write data
    write_results = storage_writer.execute_write(final_plan, planner)

    # Finalize checkpoint
    all_write_results = gather_all_results_from_storage(checkpoint_id,
                                                        is_coordinator,
                                                        world_size,
                                                        write_results,
                                                        FileType.STORAGE_DATA,
                                                        rank) if use_storage_comm \
        else all_gather_object(write_results, world_size, use_collectives)
    storage_writer.finalize_checkpoint(metadata, all_write_results)

    return metadata


@dcp_timer_decorator
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
    metadata = _save_impl(
        state_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )
    platform.barrier()
    return metadata


def _create_persist_process(
    result_queue: mp.Queue,
    staged: dict[str, Any],
    checkpoint_id: Optional[Union[Path, str]],
    storage_writer: Optional[StorageWriter],
    planner: Optional[SavePlanner],
    no_dist: bool,
    use_collectives: bool,
    use_gloo: bool,
) -> mp.Process:
    """
    Build the child process persisting the staged state_dict for :func:`async_save`.

    Depending on the caller, the child either runs the plan and write result collectives on a
    dedicated gloo group it reinitialises itself, or exchanges those results through the
    storage, or runs without any cross-rank interaction at all.

    Args:
        result_queue (mp.Queue): IPC queue the child returns its status and metadata on.
        staged (dict[str, Any]): Host resident staged state_dict to persist.
        checkpoint_id (Optional[Union[Path, str]]): Same as :func:`async_save`.
        storage_writer (Optional[StorageWriter]): Same as :func:`async_save`.
        planner (Optional[SavePlanner]): Same as :func:`async_save`.
        no_dist (bool): Same as :func:`async_save`.
        use_collectives (bool): Same as :func:`async_save`.
        use_gloo (bool): Same as :func:`async_save`.

    Returns:
        mp.Process: The persist process, not started yet.

    Raises:
        AssertionError: If gloo communication is requested without MASTER_ADDR or MASTER_PORT.
    """
    world_size = platform.get_world_size()
    need_comm = use_collectives and not no_dist and world_size > 1
    if not need_comm or not use_gloo:
        use_storage_comm = need_comm and not use_gloo
        if use_storage_comm:
            logger.info("use storage communication for dcp async save.")
        return mp.Process(
            target=execute_async_persist,
            args=(
                result_queue,
                staged,
                checkpoint_id,
                storage_writer,
                planner,
                no_dist,
                use_collectives,
                use_storage_comm,
            ),
            name="AsyncCheckpointPersist",
        )

    logger.info("use Gloo communication for dcp async save.")
    rank = platform.get_rank()
    master_addr = os.environ.get("MASTER_ADDR", None)
    master_port = os.environ.get("MASTER_PORT", None)
    if master_addr is None:
        raise AssertionError("Async DCP needs MASTER_ADDR to use prefix store")
    if master_port is None:
        raise AssertionError("Async DCP needs MASTER_PORT to use prefix store")
    master_port = int(master_port)
    global _async_save_count
    proc = mp.Process(
        target=execute_async_persist_with_gloo,
        args=(
            result_queue,
            staged,
            checkpoint_id,
            storage_writer,
            planner,
            rank,
            world_size,
            master_addr,
            master_port,
        ),
        name=f"AsyncCheckpointPersistGloo_{_async_save_count}",
    )
    _async_save_count += 1
    return proc


@dcp_timer_decorator
def async_save(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: Optional[Union[Path, str]] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    no_dist: bool = False,
    use_collectives: bool = True,
    use_gloo: bool = False,
    callback: Optional[Callable[[], None]] = None,
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
    dict. A small background **thread** only joins that process and fills
    ``persist_completion``; it does not perform tensor work.

    Neither the thread nor the child process is a daemon, so a caller that exits with a
    checkpoint still in flight waits for it to land rather than losing it. Making only the
    thread a daemon would not shorten that wait either, since ``multiprocessing`` joins the
    non-daemon child at interpreter exit anyway; it would just drop ``persist_completion``
    and ``callback`` on the floor.

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
        use_gloo(bool): Whether to use Gloo. Default False.
        callback(Callable): Async save callback function.

    Returns:
        AsyncSaveResponse: Contains ``persist_completion`` only; staging is synchronous.
    """
    # Copy the state_dict to cpu memory for async save
    staged = build_staged_state_dict(state_dict)

    persist_completion: Future[Metadata] = Future()
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    proc = _create_persist_process(
        result_queue,
        staged,
        checkpoint_id,
        storage_writer,
        planner,
        no_dist,
        use_collectives,
        use_gloo,
    )

    # After the async save is completed, the user callback will be executed.
    def async_callback():
        if callback is not None:
            callback()

    # Deliberately not a daemon: it has to outlive the training loop long enough to resolve
    # persist_completion and run callback, and killing it early would not let the process exit
    # any sooner because the persist child is joined at interpreter exit regardless.
    join_thread = threading.Thread(
        target=resolve_async_persist_result,
        args=(proc, result_queue, persist_completion, async_callback),
        name="AsyncCheckpointPersistJoin",
    )
    proc.start()
    join_thread.start()
    ret = AsyncSaveResponse(persist_completion=persist_completion)
    return ret


def load(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: Optional[Union[Path, str]] = None,
    storage_reader: Optional[StorageReader] = None,
    planner: Optional[LoadPlanner] = None,
    no_dist: bool = False,
    use_collectives: bool = False,
    broadcast_from_minimum_rank: bool = False,
    broadcast_groups: Optional[dict[tuple, Any]] = None,
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
        use_collectives (bool): If False, each rank loads `.metadata` or `rank.metadata`,
            there is no communication between ranks, and each rank loads its own data. Default False.
        broadcast_from_minimum_rank (bool): Whether broadcast from minimum rank.
        broadcast_groups (dict): The Communication groups for broadcast.

    Returns:
        None. The state_dict is modified in-place.
    """
    # Convert checkpoint_id to Path if it's a string
    checkpoint_id = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

    # Determine if we're in distributed mode
    use_collectives = False if no_dist else use_collectives

    # Check the DTensor broadcast args
    if broadcast_groups and not broadcast_from_minimum_rank:
        raise ValueError("If not use broadcast_from_minimum_rank, the broadcast_groups should be None.")

    # Set up storage reader
    if storage_reader is None:
        if checkpoint_id is None:
            raise ValueError("Either storage_reader or checkpoint_id must be provided")
        storage_reader = FileSystemReader(checkpoint_id)
    else:
        if checkpoint_id:
            storage_reader.initialize_reader(checkpoint_id)

    # Set up planner
    planner = StandardLoadPlanner() if planner is None else planner

    # Get rank and coordinator info
    rank = platform.get_rank()
    world_size = platform.get_world_size()
    is_coordinator = rank == 0

    # Load metadata
    try:
        metadata = storage_reader.load_metadata()
    except FileNotFoundError:
        # Fallback to rank-local metadata (e.g. checkpoint saved with use_collectives=False)
        metadata = storage_reader.load_metadata(rank=rank)
        use_collectives = False

    # Configure planner
    planner.configure_planner(
        state_dict=state_dict,
        metadata=metadata,
        is_coordinator=is_coordinator,
        rank=rank,
        broadcast_from_minimum_rank=broadcast_from_minimum_rank,
    )

    # Configure storage reader
    storage_reader.configure_reader(
        metadata=metadata,
        is_coordinator=is_coordinator,
        rank=rank,
        broadcast_from_minimum_rank=broadcast_from_minimum_rank,
    )

    # Build local plan
    local_plan = planner.build_local_plan()
    local_plan = storage_reader.optimize_local_plan(local_plan)

    # Gather all local plans and build global plan
    all_local_plans = all_gather_object(local_plan, world_size, use_collectives)
    global_plans = planner.build_global_plan(all_local_plans)
    global_plans = storage_reader.optimize_global_plan(global_plans)

    # Select central plan for current rank
    if use_collectives and world_size > 1 and global_plans:
        central_plan = global_plans[rank]
    elif global_plans:
        central_plan = global_plans[0]
    else:
        central_plan = local_plan

    # Finalize plan
    final_plan = planner.finalize_plan(central_plan)

    # Execute read
    storage_reader.execute_read(final_plan, planner, broadcast_groups)
