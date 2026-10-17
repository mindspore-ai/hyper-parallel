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
"""PyTorch distributed process group API."""
from datetime import timedelta
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS


def create_subgroups_for_parent(
        parent_pg: ProcessGroup,
        sub_size: int,
) -> dict[int, Optional[ProcessGroup]]:
    """Create equal subgroups for one process group.

    The fake backend has no peer processes to participate in the metadata
    exchange used by real backends. ``use_local_synchronization`` preserves
    the same rank membership contract without making callers know how the
    process group was initialized.
    """
    parent_ranks = tuple(sorted(dist.get_process_group_ranks(parent_pg)))
    if sub_size < 1 or len(parent_ranks) % sub_size:
        raise ValueError(
            f"sub_size must evenly divide parent group size, got {sub_size} and {len(parent_ranks)}"
        )
    current_rank = dist.get_rank()
    subgroup_count = len(parent_ranks) // sub_size
    if dist.get_backend(parent_pg) == "fake":
        return {
            index: dist.new_group(
                ranks=list(parent_ranks[index * sub_size:(index + 1) * sub_size]),
                use_local_synchronization=True,
            ) if current_rank in parent_ranks[index * sub_size:(index + 1) * sub_size] else None
            for index in range(subgroup_count)
        }

    world_size = dist.get_world_size()
    backend = dist.get_backend(parent_pg)
    if backend in ("nccl", "hccl", "mccl"):
        accelerator = getattr(torch, "npu", None)
        device = (
            torch.device("npu", accelerator.current_device())
            if accelerator is not None and accelerator.is_available()
            else torch.device("cuda", torch.cuda.current_device())
        )
    else:
        device = torch.device("cpu")
    local_tensor = torch.tensor(parent_ranks, dtype=torch.long, device=device)
    gathered = torch.empty((world_size, len(parent_ranks)), dtype=torch.long, device=device)
    dist.all_gather_into_tensor(gathered.view(-1), local_tensor)
    parent_rank_sets = sorted(set(tuple(row) for row in gathered.cpu().tolist()))
    result: dict[int, Optional[ProcessGroup]] = {}
    for ranks in parent_rank_sets:
        for index in range(len(ranks) // sub_size):
            subgroup_ranks = ranks[index * sub_size:(index + 1) * sub_size]
            subgroup = dist.new_group(list(subgroup_ranks))
            if ranks == parent_ranks:
                result[index] = subgroup if current_rank in subgroup_ranks else None
    return result


def _group_key(ranks: list[int]) -> str:
    """Build a stable cache key from process-group ranks."""
    return str(tuple(sorted(ranks)))


def init_process_group(
        backend: Optional[str] = None,
        *,
        init_method: Optional[str] = None,
        timeout: Optional[timedelta] = None,
        world_size: int = -1,
        rank: int = -1,
        store: Any = None,
        pg_options: Any = None,
        device_id: Any = None
) -> None:
    """
    Init global process group, this is the start of distributed job.

    Args:
        backend: The backend used for distributed communication.
        init_method: The method to initialize the process group.
        timeout: Timeout for operations executed against the process group.
        world_size: Number of processes participating in the job
        rank: Rank of the current process
        store: Key/value store for exchanging connection information
        pg_options: Process group options for backend-specific configurations
        device_id: Specific device this process will work on
    """
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        timeout=timeout,
        world_size=world_size,
        rank=rank,
        store=store,
        pg_options=pg_options,
        device_id=device_id,
    )


def destroy_process_group(group: Optional[ProcessGroup] = None) -> None:
    """
    Destroy a given process group.

    Args:
        group: The process group to be destroyed. If None, destroys the default group.

    """
    if group is None:
        EXISTING_COMM_GROUPS.clear()
    else:
        keys_to_destroy = [key for key, cached_group in EXISTING_COMM_GROUPS.items() if cached_group == group]
        for key in keys_to_destroy:
            del EXISTING_COMM_GROUPS[key]
    dist.destroy_process_group(group)


def get_process_group_ranks(group: Optional[ProcessGroup] = None) -> list[int]:
    """
    Get rank list of the given process group.

    Args:
        group: The process group to get ranks from. If None, uses the default group.

    Returns:
        List of ranks in the specified process group.

    """
    resolved_group = group if group is not None else dist.group.WORLD
    return dist.get_process_group_ranks(resolved_group)


def get_backend(group: Optional[ProcessGroup] = None) -> str:
    """
    Get the backend of the given process group.
    Args:
        group: The process group to get backend from. If None, uses the default group.

    Returns:
        The backend name of the specified process group.

    """
    return dist.get_backend(group)


def split_group(parent_pg: Optional[ProcessGroup] = None,
                split_ranks: Optional[list] = None,
                timeout: Optional[timedelta] = None,
                pg_options: Optional[Any] = None,
                group_desc: Optional[str] = None,
                ) -> Optional[ProcessGroup]:
    """
    Create split group relative to the parent process group.
    """
    del parent_pg, timeout, pg_options, group_desc
    if not split_ranks:
        raise ValueError("split_ranks cannot be None or empty")

    current_rank = dist.get_rank()
    current_group = None
    for ranks in split_ranks:
        key = _group_key(ranks)
        group = EXISTING_COMM_GROUPS.get(key)
        if group is None:
            group = dist.new_group(ranks=ranks)
            EXISTING_COMM_GROUPS[key] = group
        if current_rank in ranks:
            current_group = group
    return current_group


def get_group_local_rank(group: Optional[ProcessGroup] = None) -> int:
    """get group local rank id"""
    return dist.get_rank(group)


def mark_created_groups(process_group: Union[ProcessGroup, list[ProcessGroup]]) -> None:
    """
    mark created groups

    Args:
        process_group (Union[Any, list[Any]]): A process group or a list of process groups.

    """
    groups = process_group if isinstance(process_group, list) else [process_group]
    for group in groups:
        ranks = dist.get_process_group_ranks(group)
        EXISTING_COMM_GROUPS[_group_key(ranks)] = group
