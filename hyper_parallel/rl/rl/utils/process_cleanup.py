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
"""RL service shutdown, process-group teardown and runtime cache cleanup."""

import logging
import subprocess
from typing import TYPE_CHECKING

import torch.distributed as dist

from hyper_parallel.collectives import cc
from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache
from hyper_parallel.core.dtensor.dtensor import _LAYOUT_CACHE
from hyper_parallel.core.dtensor.tensor_redistribution import _tensor_redistribution
from hyper_parallel.core.fully_shard import hsdp_param
from hyper_parallel.core.pipeline_parallel._p2p import _P2P_MULTI_STREAM_GROUPS
from hyper_parallel.distributed.context_parallel.collectives import _HYBRID_MESH_CACHE


if TYPE_CHECKING:
    from rl.utils.monitoring.tracker import TrainingTracker

logger = logging.getLogger(__name__)


def cleanup_processes(
    tracker: "TrainingTracker | None",
    rollout_manager: object,
    rollout_engine: object,
    runtime_started: bool,
) -> None:
    """Close the supplied services, then destroy their distributed runtime.

    Args:
        tracker: Optional training tracker to finish.
        rollout_manager: Optional rollout manager with a close method.
        rollout_engine: Optional rollout engine with a close method.
        runtime_started: Whether the caller initialized the distributed runtime.
    """
    if tracker is not None:
        try:
            tracker.finish()
        except Exception as exc:  # pylint: disable=W0718
            logger.warning("Tracking cleanup failed: %s", exc)
    close_manager = getattr(rollout_manager, "close", None)
    if callable(close_manager):
        try:
            close_manager()
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            logger.warning("Rollout manager cleanup failed: %s", exc)
    close_rollout = getattr(rollout_engine, "close", None)
    if callable(close_rollout):
        try:
            close_rollout()
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            logger.warning("Rollout cleanup failed: %s", exc)
    if not runtime_started:
        return
    try:
        destroy_process_group()
    except (RuntimeError, ValueError) as exc:
        logger.warning("Distributed cleanup failed: %s", exc)


def destroy_process_group() -> None:
    """Destroy the RL default group and discard caches that reference its groups.

    DeviceMesh owns its mesh and group registry cleanup. RL also discards the
    remaining process-local runtime caches so subsequent runs cannot reuse
    destroyed process groups.

    Raises:
        RuntimeError: If the backend fails to destroy the group. Caches are
            still cleared before the error propagates.
    """
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    finally:
        clear_device_mesh_cache()
        for cache in (
            _P2P_MULTI_STREAM_GROUPS,
            _LAYOUT_CACHE,
            _HYBRID_MESH_CACHE,
        ):
            cache.clear()
        # Older checkouts have separate registries; master consolidated them into DeviceMesh's shared cache.
        for owner, name in ((cc, "_EXISTING_COMM_GROUPS"), (hsdp_param, "_GROUP_INFO_CACHE")):
            cache = getattr(owner, name, None)
            if cache is not None:
                cache.clear()
        _tensor_redistribution._transform_cache.clear()  # pylint: disable=protected-access
        _tensor_redistribution.is_init = False
        _tensor_redistribution.rank_id = None
