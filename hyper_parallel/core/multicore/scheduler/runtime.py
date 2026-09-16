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
"""Size and serialize dense runtime images without unused task/queue tails."""

import ctypes
import struct
from functools import lru_cache

from hyper_parallel.core.multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN,
    GROUP_LIST_CACHE_LINE_BYTES,
    MIN_EVENT_CAPACITY,
    MIN_GROUP_LIST_CAPACITY,
    NUM_WORKERS_CUBE,
    DynamicDataC,
    EventDescC,
    RuntimeConfigC,
    TaskDescC,
)

RUNTIME_HEADER_BYTES = ctypes.sizeof(RuntimeConfigC)
TASK_DESC_SIZE = ctypes.sizeof(TaskDescC)
RUNTIME_FIXED_BYTES = (
    RUNTIME_HEADER_BYTES + 16 + ctypes.sizeof(DynamicDataC) + ATOMIC_ADD_VALUE_LEN * 4
)


def _check_capacity(value: int, name: str, minimum: int = 0) -> None:
    """Reject values that cannot be represented by aligned runtime arrays."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum or value % 16:
        raise ValueError(f"{name} must be an integer >= {minimum} aligned to 16, got {value!r}.")


def grouped_matmul_group_list_capacity(local_experts: int) -> int:
    """Return scratch int64 slots for isolated, cache-line-aligned worker lists.

    Args:
        local_experts: Entries per worker, or zero for a graph without groups.

    Returns:
        Total slots, including padding to align the first worker's list.
    """
    if not isinstance(local_experts, int) or isinstance(local_experts, bool) or not 0 <= local_experts < 1 << 32:
        raise ValueError(f"local_experts must fit uint32, got {local_experts!r}.")
    alignment = GROUP_LIST_CACHE_LINE_BYTES // 8
    stride = (max(1, local_experts) + alignment - 1) // alignment * alignment
    return max(MIN_GROUP_LIST_CAPACITY, NUM_WORKERS_CUBE * stride + alignment)


def runtime_config_serialized_size(
    task_capacity: int, event_capacity: int = MIN_EVENT_CAPACITY, local_experts: int = 0
) -> int:
    """Return dense wire bytes for the single graph-sized runtime layout.

    Args:
        task_capacity: Number of task and per-kind queue slots, aligned to 16.
        event_capacity: Number of event slots, aligned to 16 and at least 1024.
        local_experts: Number of group-list entries required per worker.

    Returns:
        Image size representable by the device's uint32 byte offsets.
    """
    _check_capacity(task_capacity, "task_capacity")
    _check_capacity(event_capacity, "event_capacity", MIN_EVENT_CAPACITY)
    scratch_bytes = grouped_matmul_group_list_capacity(local_experts) * 8
    size = RUNTIME_FIXED_BYTES + event_capacity * 20 + task_capacity * (TASK_DESC_SIZE + 12) + scratch_bytes
    if size >= 1 << 32:
        raise ValueError("runtime descriptor exceeds uint32 device byte offsets")
    return size


@lru_cache(maxsize=16)
def _runtime_config_type(task_capacity: int, event_capacity: int, group_list_capacity: int) -> type:
    """Create one contiguous Host image sized for the graph's arrays."""
    return type("SizedRuntimeConfigC", (RuntimeConfigC,), {
        "_fields_": [
            ("all_event_num_triggers", ctypes.c_int32 * event_capacity),
            ("all_tasks", TaskDescC * task_capacity),
            ("all_events", EventDescC * event_capacity),
            ("task_index_num", ctypes.c_int32 * 4),
            ("cube_task_indices", ctypes.c_int32 * task_capacity),
            ("vector_task_indices", ctypes.c_int32 * task_capacity),
            ("mix_task_indices", ctypes.c_int32 * task_capacity),
            ("dynamic_data", DynamicDataC),
            ("grouped_matmul_group_list", ctypes.c_int64 * group_list_capacity),
            ("atomic_add_values", ctypes.c_int32 * ATOMIC_ADD_VALUE_LEN),
        ],
    })


def allocate_runtime_config(
    task_capacity: int, event_capacity: int = MIN_EVENT_CAPACITY, local_experts: int = 0
) -> RuntimeConfigC:
    """Allocate enough host task, queue and event slots before filling a graph.

    Args:
        task_capacity: Required task/queue slots, including termination.
        event_capacity: Event slots, including atomic-write padding.
        local_experts: Group-list entries per worker, used to size scratch.

    Returns:
        A zero-initialized ctypes image with sufficient array bounds.
    """
    if not isinstance(task_capacity, int) or isinstance(task_capacity, bool) or task_capacity < 0:
        raise ValueError(f"task_capacity must be a nonnegative integer, got {task_capacity!r}.")
    capacity = (task_capacity + 15) // 16 * 16
    runtime_config_serialized_size(capacity, event_capacity, local_experts)
    cfg = _runtime_config_type(capacity, event_capacity, grouped_matmul_group_list_capacity(local_experts))()
    cfg.task_capacity = capacity
    cfg.event_capacity = event_capacity
    cfg.dynamic_data.dynamic_group_size = local_experts
    return cfg


def runtime_config_task_capacity(cfg: RuntimeConfigC) -> int:
    """Cover task count, queue lengths and every queued descriptor reference.

    Args:
        cfg: Populated host runtime, which may contain a terminal beyond task_num.

    Returns:
        Required wire capacity aligned to 16 task slots.
    """
    required = int(cfg.task_num)
    queues = (cfg.cube_task_indices, cfg.vector_task_indices, cfg.mix_task_indices)
    for indices, count in zip(queues, cfg.task_index_num[:3]):
        if count < 0 or count > len(indices):
            raise ValueError(f"invalid runtime task-index count {count}")
        required = max(required, count)
        for task_id in indices[:count]:
            if task_id < 0 or task_id >= len(cfg.all_tasks):
                raise ValueError(f"invalid runtime task id {task_id}")
            required = max(required, task_id + 1)
    capacity = (required + 15) // 16 * 16
    if capacity > min(len(cfg.all_tasks), *(len(indices) for indices in queues)):
        raise ValueError(f"runtime requires {capacity} aligned task/queue slots")
    return capacity


def serialize_runtime_config(cfg: RuntimeConfigC) -> bytes:
    """Serialize dense tasks and queues up to their actual required capacity.

    All graphs use a 64-byte header with task count, worker count, task capacity
    and event capacity, followed by ready, local completion and protocol version.
    Ready is zero for EP1. Descriptors and queues remain dense.

    Args:
        cfg: Completed and validated host configuration.

    Returns:
        Bytes for a matching native runtime reader.
    """
    capacity = runtime_config_task_capacity(cfg)
    event_capacity = len(cfg.all_event_num_triggers)
    if len(cfg.all_events) != event_capacity:
        raise ValueError("runtime event and trigger arrays must have equal capacity")
    local_experts = cfg.dynamic_data.dynamic_group_size
    group_list_capacity = grouped_matmul_group_list_capacity(local_experts)
    if len(cfg.grouped_matmul_group_list) < group_list_capacity:
        raise ValueError("runtime group-list scratch is too small for dynamic_group_size")
    expected = runtime_config_serialized_size(capacity, event_capacity, local_experts)
    layout = type(cfg)
    address = ctypes.addressof(cfg)
    prefix = struct.pack(
        "<10I24x",
        cfg.task_num,
        cfg.num_workers,
        capacity,
        event_capacity,
        cfg.ready_event,
        cfg.cycle_profiling_enabled,
        cfg.aic_profile_record_capacity,
        cfg.aiv_profile_record_capacity,
        cfg.completion_event,
        cfg.protocol_version,
    )
    sections = [
        (layout.all_event_num_triggers.offset, event_capacity * 4),
        (layout.all_tasks.offset, capacity * TASK_DESC_SIZE),
        (layout.all_events.offset, event_capacity * ctypes.sizeof(EventDescC)),
        (layout.task_index_num.offset, 16),
        (layout.cube_task_indices.offset, capacity * 4),
        (layout.vector_task_indices.offset, capacity * 4),
        (layout.mix_task_indices.offset, capacity * 4),
        (layout.dynamic_data.offset, ctypes.sizeof(DynamicDataC)),
        (layout.grouped_matmul_group_list.offset, group_list_capacity * 8),
        (layout.atomic_add_values.offset, ATOMIC_ADD_VALUE_LEN * 4),
    ]
    data = prefix + b"".join(ctypes.string_at(address + offset, size) for offset, size in sections)
    if len(data) != expected:
        raise RuntimeError(f"runtime serialization produced {len(data)} bytes, expected {expected}")
    return data
