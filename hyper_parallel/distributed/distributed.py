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
"""Distributed process group API"""
from datetime import timedelta
from typing import Optional, Any

from hyper_parallel import get_platform

platform = get_platform()


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
    platform.init_process_group(backend=backend, init_method=init_method, timeout=timeout, world_size=world_size,
                                rank=rank, store=store, pg_options=pg_options, device_id=device_id)


def destroy_process_group(group=None) -> None:
    """
    Destroy a given process group.

    Args:
        group: The process group to be destroyed. If None, destroys the default group.

    Raises:
        NotImplementedError: This method must be implemented by subclasses
    """
    platform.destroy_process_group(group=group)


def get_process_group_ranks(group=None) -> list[int]:
    """
    Get rank list of the given process group.

    Args:
        group: The process group to get ranks from. If None, uses the default group.

    Returns:
        List of ranks in the specified process group.

    Raises:
        NotImplementedError: This method must be implemented by subclasses
    """
    return platform.get_process_group_ranks(group=group)


def get_backend(group=None):
    """
    Get the backend of the given process group.
    Args:
        group: The process group to get backend from. If None, uses the default group.

    Returns:
        The backend name of the specified process group.

    Raises:
        NotImplementedError: This method must be implemented by subclasses
    """
    return platform.get_backend(group=group)


def split_group(parent_pg: Any = None,
                split_ranks: Optional[list] = None,
                timeout: Optional[timedelta] = None,
                pg_options: Optional[Any] = None,
                group_desc: Optional[str] = None,
                ) -> Any:
    """
    Create split group relative to the parent process group.
    """
    return platform.split_group(parent_pg=parent_pg, split_ranks=split_ranks, timeout=timeout, pg_options=pg_options,
                                group_desc=group_desc)
