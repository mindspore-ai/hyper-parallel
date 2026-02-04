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
from pathlib import Path
from typing import Any, Optional, Union

from hyper_parallel.core.checkpoint.standard_planner import StandardSavePlanner, StandardLoadPlanner
from hyper_parallel.core.checkpoint.filesystem_storage import FileSystemReader, FileSystemWriter
from hyper_parallel.core.checkpoint.metadata import Metadata
from hyper_parallel.core.checkpoint.planner import SavePlanner, LoadPlanner
from hyper_parallel.core.checkpoint.storage import StorageReader, StorageWriter
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import Platform


def _gather_from_all_ranks(
    platform: Platform,
    local_object: Any,
    world_size: int,
    use_collectives: bool,
) -> list[Any]:
    """
    Gather objects from all ranks.

    Args:
        platform (Platform): Platform instance for communication.
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


def save(
        state_dict: dict[str, Any],
        *,
        checkpoint_id: Optional[Union[Path, str]] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        no_dist: bool = False,
        use_collectives: bool = True,
        remove_redundancy: bool = True,
        save_to_minimum_rank: bool = False,
) -> Metadata:
    """
    Save a distributed checkpoint in SPMD style.

    This function saves a state_dict containing DTensors, where each rank
    only saves their local shards.

    Args:
        state_dict (dict[str, Any]): The state_dict to save.
        checkpoint_id (Optional[Union[Path, str]]): The ID/path of this checkpoint instance (can be Path or str).
        storage_writer (Optional[StorageWriter]): Instance of StorageWriter. If None, FileSystemWriter
            will be created based on checkpoint_id.
        planner (Optional[SavePlanner]): Instance of SavePlanner. If None, StandardSavePlanner will be used.
        no_dist (bool): If True, save in single process mode.
        use_collectives (bool): If True, use collective communication for coordination.
        remove_redundancy (bool): If True, deduplicate tensors across ranks.
        save_to_minimum_rank (bool): If True, deduplicated items are saved on the minimum rank.

    Returns:
        Metadata: Metadata object for the saved checkpoint.
    """
    platform = get_platform()

    # Convert checkpoint_id to Path if it's a string
    checkpoint_id = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

    # Determine if we're in distributed mode
    use_collectives = False if no_dist else use_collectives

    # Set up storage writer
    if storage_writer is None:
        if checkpoint_id is None:
            raise ValueError("Either storage_writer or checkpoint_id must be provided")
        storage_writer = FileSystemWriter(checkpoint_id)
    else:
        if checkpoint_id:
            storage_writer.initialize_writer(checkpoint_id)

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
        remove_redundancy=remove_redundancy,
        save_to_minimum_rank=save_to_minimum_rank
    )

    # Configure storage writer
    storage_writer.configure_writer(
        is_coordinator=is_coordinator,
        rank=rank,
        use_collectives=use_collectives
    )

    # Build local plan
    local_plan = planner.build_local_plan()
    local_plan = storage_writer.optimize_local_plan(local_plan)

    # Gather all local plans and build global plan
    all_local_plans = _gather_from_all_ranks(platform, local_plan, world_size, use_collectives)
    global_plans, metadata = planner.build_global_plan(all_local_plans)
    global_plans = storage_writer.optimize_global_plan(global_plans)

    # Select central plan for current rank
    if use_collectives and world_size > 1 and global_plans:
        central_plan = global_plans[rank]
    elif global_plans:
        central_plan = global_plans[0]
    else:
        central_plan = local_plan

    # Finalize plan
    final_plan = planner.finalize_plan(central_plan)

    # Write data
    write_results = storage_writer.execute_write(final_plan, planner)

    # Finalize checkpoint
    all_write_results = _gather_from_all_ranks(platform, write_results, world_size, use_collectives)
    if is_coordinator or not use_collectives:
        storage_writer.finalize_checkpoint(metadata, all_write_results)

    return metadata


def load(
        state_dict: dict[str, Any],
        *,
        checkpoint_id: Optional[Union[Path, str]] = None,
        storage_reader: Optional[StorageReader] = None,
        planner: Optional[LoadPlanner] = None,
        no_dist: bool = False,
) -> None:
    """
    Load a distributed checkpoint into state_dict in SPMD style.

    Each rank will try to read the least amount of data necessary
    to fulfill the requested state_dict. When loading DTensor instances,
    each rank only reads data for their local shards.

    Args:
        state_dict (dict[str, Any]): The state_dict to load the checkpoint into (modified in-place).
        checkpoint_id (Optional[Union[Path, str]]): The ID/path of this checkpoint instance (can be Path or str).
        storage_reader (Optional[StorageReader]): Instance of StorageReader. If None, FileSystemReader
            will be created based on checkpoint_id.
        planner (Optional[LoadPlanner]): Instance of LoadPlanner. If None, StandardLoadPlanner will be used.
        no_dist (bool): If True, load without cross-rank synchronization.

    Returns:
        None. The state_dict is modified in-place.
    """
    platform = get_platform()

    # Convert checkpoint_id to Path if it's a string
    checkpoint_id = Path(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

    # Determine if we're in distributed mode
    use_collectives = not no_dist

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
    except Exception:
        # Fallback to rank-local metadata
        metadata = storage_reader.load_metadata(rank=rank)
        use_collectives = False

    # Configure planner
    planner.configure_planner(
        state_dict=state_dict,
        metadata=metadata,
        is_coordinator=is_coordinator,
        rank=rank
    )

    # Configure storage reader
    storage_reader.configure_reader(
        metadata=metadata,
        is_coordinator=is_coordinator,
        rank=rank,
        use_collectives=use_collectives
    )

    # Build local plan
    local_plan = planner.build_local_plan()
    local_plan = storage_reader.optimize_local_plan(local_plan)

    # Gather all local plans and build global plan
    all_local_plans = _gather_from_all_ranks(platform, local_plan, world_size, use_collectives)
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
    storage_reader.execute_read(final_plan, planner)
