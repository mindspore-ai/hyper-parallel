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
"""
Runtime configuration: topology parameters, C++ ABI structs, and constants.

Combines:
- TaskSplitValue / init_task_split_value  — TP/EP/seq topology + per-rank counters
- ctypes.Structure mirrors of runtime_head.hpp — passed as binary blobs to C++ kernels
"""
import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

# ── Constants (match runtime_head.hpp exactly) ────────────────────────────────
MAX_TENSOR_DIMS      = 4
MAX_INPUTS_PER_TASK  = 4
MAX_OUTPUTS_PER_TASK = 4

MIN_EVENT_CAPACITY  = 1024
NUM_WORKERS_VECTOR   = 48
NUM_WORKERS_CUBE     = 24
MAX_GROUP_LIST       = 512
MAX_EXPERT_NUM_PER_RANK = 16
ATOMIC_ADD_VALUE_LEN = 8
READY_CACHE_LINE_BYTES = 64
INVALID_PROFILE_DESC_ID  = 0xFFFFFFFF
INVALID_PROFILE_OWNER_ID = 0xFFFFFFFF
EVENT_INVALID_ID         = 0xFFFFFFFF


def mega_moe_event_capacity(num_experts: int, ep_size: int) -> int:
    """Reserve graph events and the complete atomic write at the last trigger.

    Args:
        num_experts: Global expert count.
        ep_size: Expert-parallel group size.

    Returns:
        Cache-line-aligned event slots, with a minimum of 1024.
    """
    if ep_size <= 0 or num_experts <= 0 or num_experts % ep_size:
        raise ValueError("num_experts must be positive and divisible by ep_size")
    # Ready follows termination; both triggers write a complete atomic vector.
    final_event = num_experts + 3 * (num_experts // ep_size) + (4 if ep_size > 1 else 2)
    required = final_event + ATOMIC_ADD_VALUE_LEN
    return max(MIN_EVENT_CAPACITY, (required + 15) // 16 * 16)


# ── Enums ─────────────────────────────────────────────────────────────────────
class TaskAiCoreType(IntEnum):
    """Worker core categories encoded in RuntimeConfig."""

    TASK_AICORE_INVALID = 0
    TASK_AICORE_CUBE    = 1
    TASK_AICORE_VECTOR  = 2
    TASK_AICORE_MIX     = 3


class TaskType(IntEnum):
    """Task operation kinds understood by the Device scheduler."""

    TASK_TERMINATE            = 0
    TASK_BEGIN_TASK_GRAPH     = 10
    TASK_ADD_CUSTOM           = 101
    TASK_SWI_GLU              = 102
    TASK_MATMUL               = 103
    TASK_GROUPED_MATMUL       = 104
    TASK_SHMEM_PUT_MEM_SIGNAL = 105
    TASK_SWI_GLU_GRAD         = 106


class EventType(IntEnum):
    """Dependency and trigger event operations used by scheduled tasks."""

    EVENT_EMPTY                  = 900
    EVENT_LAUNCH_TASKS           = 901
    EVENT_LAUNCH_MASSIVE_TASKS   = 902
    EVENT_LAUNCH_DEPENDENT_TASKS = 903
    EVENT_END_OF_TASK_GRAPH      = 910
    EVENT_TERMINATION            = 911
    EVENT_INVALID                = 999


class DynamicType(IntEnum):
    """Runtime dynamic-data operations applied before task execution."""

    DYNAMIC_EMPTY        = 0
    DYNAMIC_DSV3_MOE = 101


# ── ctypes Structures (mirror runtime_head.hpp) ───────────────────────────────

class TensorDescC(ctypes.Structure):
    """Serialized tensor address and shape descriptor."""

    _fields_ = [
        ("tensor_type",     ctypes.c_uint32),
        ("num_dims",        ctypes.c_uint32),
        ("dim",             ctypes.c_uint32 * MAX_TENSOR_DIMS),
        ("stride",          ctypes.c_uint32 * MAX_TENSOR_DIMS),
        ("data_type",       ctypes.c_uint32),
        ("input_position",  ctypes.c_uint32),
        ("base_ptr_offset", ctypes.c_uint32),
        ("transpose_flag",  ctypes.c_uint32),
        ("dynamic_shape",   ctypes.c_uint32),
        ("dynamic_dim",     ctypes.c_uint32),
    ]


class TaskDescC(ctypes.Structure):
    """Serialized task descriptor shared by Host and Device schedulers."""
    _fields_ = [
        ("task_type",            ctypes.c_uint32),
        ("task_aicore_type",     ctypes.c_uint32),
        ("num_inputs",           ctypes.c_uint32),
        ("num_outputs",          ctypes.c_uint32),
        ("trigger_event",        ctypes.c_uint32),
        ("dependent_event",      ctypes.c_uint32),
        ("inputs",               TensorDescC * MAX_INPUTS_PER_TASK),
        ("outputs",              TensorDescC * MAX_OUTPUTS_PER_TASK),
        ("tiling_data_position", ctypes.c_uint32),
        ("tiling_data_offset",   ctypes.c_uint32),
        ("task_index",           ctypes.c_uint32),
        ("task_split_num",       ctypes.c_uint32),
        ("task_split_value",     ctypes.c_uint32),
        ("extra_value_0",        ctypes.c_uint32),
        ("extra_value_1",        ctypes.c_uint32),
        ("extra_value_2",        ctypes.c_uint32),
        ("profile_desc_id",      ctypes.c_uint32),
        ("profile_owner_id",     ctypes.c_uint32),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize optional profiling metadata to explicit invalid sentinels."""
        has_profile_desc = len(args) >= len(self._fields_) - 1 or "profile_desc_id" in kwargs
        has_profile_owner = len(args) >= len(self._fields_) or "profile_owner_id" in kwargs
        super().__init__(*args, **kwargs)
        if not has_profile_desc:
            self.profile_desc_id = INVALID_PROFILE_DESC_ID
        if not has_profile_owner:
            self.profile_owner_id = INVALID_PROFILE_OWNER_ID


class EventDescC(ctypes.Structure):
    """Serialized event operation descriptor."""

    _fields_ = [
        ("event_type",    ctypes.c_uint32),
        ("num_triggers",  ctypes.c_uint32),
        ("first_task_id", ctypes.c_uint32),
        ("last_task_id",  ctypes.c_uint32),
    ]


class DynamicDataC(ctypes.Structure):
    """Serialized dynamic-data update descriptor."""

    _fields_ = [
        ("dynamic_type",           ctypes.c_uint32),
        ("dynamic_input_position", ctypes.c_uint32),
        ("dynamic_group_size",     ctypes.c_uint32),
        ("dynamic_max_seq_len",    ctypes.c_int32),
    ]


class RuntimeConfigC(ctypes.Structure):
    """Common 64-byte header; allocated subclasses append graph-sized arrays."""

    _fields_ = [
        ("task_num", ctypes.c_uint32),
        ("num_workers", ctypes.c_uint32),
        ("task_capacity", ctypes.c_uint32),
        ("event_capacity", ctypes.c_uint32),
        ("ready_event", ctypes.c_uint32),
        ("cycle_profiling_enabled", ctypes.c_uint32),
        ("aic_profile_record_capacity", ctypes.c_uint32),
        ("aiv_profile_record_capacity", ctypes.c_uint32),
        ("_padding", ctypes.c_uint32 * 8),
    ]


class TilingDataC(ctypes.Structure):
    """Tiling data for add_custom ops (not GMM)."""
    _fields_ = [
        ("smallCoreDataNum",     ctypes.c_uint64),
        ("bigCoreDataNum",       ctypes.c_uint64),
        ("ubPartDataNum",        ctypes.c_uint64),
        ("smallCoreTailDataNum", ctypes.c_uint64),
        ("bigCoreTailDataNum",   ctypes.c_uint64),
        ("smallCoreLoopNum",     ctypes.c_uint64),
        ("bigCoreLoopNum",       ctypes.c_uint64),
        ("tailBlockNum",         ctypes.c_uint64),
    ]


class SwiGluTilingDataC(ctypes.Structure):
    """Serialized SwiGLU tiling values for one worker."""

    _fields_ = [
        ("is32BAligned",         ctypes.c_uint32),
        ("isDoubleBuffer",       ctypes.c_uint32),
        ("rowLen",               ctypes.c_uint64),
        ("colLen",               ctypes.c_uint64),
        ("baseRowLen",           ctypes.c_uint32),
        ("baseColLen",           ctypes.c_uint32),
        ("activateLeft",         ctypes.c_uint32),
        ("biasIsEmpty",          ctypes.c_uint32),
        ("quantScaleIsEmpty",    ctypes.c_uint32),
        ("activateScaleIsEmpty", ctypes.c_uint32),
        ("swiColLen",            ctypes.c_uint64),
        ("perRowLen",            ctypes.c_uint64),
        ("modRowLen",            ctypes.c_uint64),
        ("usedCoreNum",          ctypes.c_uint32),
    ]


class ClippedSwiGluTilingDataC(ctypes.Structure):
    """Serialized CANN ClippedSwiglu/ClippedSwigluGrad tiling values."""

    _fields_ = [
        ("core_num_all", ctypes.c_int64),
        ("dim_batch_size", ctypes.c_int64),
        ("dim_2h", ctypes.c_int64),
        ("is_long_h", ctypes.c_int64),
        ("is_group", ctypes.c_int64),
        ("is_interleaved", ctypes.c_int64),
        ("alpha", ctypes.c_float),
        ("limit", ctypes.c_float),
        ("bias", ctypes.c_float),
        ("ub_max_pair", ctypes.c_int64),
        ("group_num", ctypes.c_int64),
    ]


# ── Topology + runtime counters ───────────────────────────────────────────────

@dataclass
class TaskSplitValue:
    """
    Hardware topology parameters + per-rank runtime counters.

    Topology fields are user-supplied; derived properties compute
    sequence/expert partition sizes.  Runtime counters are reset by
    init_task_split_value() before each rank's fill loop.
    """
    # ── User inputs ───────────────────────────────────────────────────────────
    tp:             int = 4
    ep:             int = 4
    seq_size:       int = 8192
    all_expert_num: int = 32
    top_k:          int = 8

    def __post_init__(self) -> None:
        """Reject topology values that would invalidate runtime arithmetic."""
        values = {
            "tp": self.tp,
            "ep": self.ep,
            "seq_size": self.seq_size,
            "all_expert_num": self.all_expert_num,
            "top_k": self.top_k,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.top_k > self.all_expert_num:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed all_expert_num ({self.all_expert_num})."
            )
        if self.all_expert_num % self.ep:
            raise ValueError(
                f"all_expert_num ({self.all_expert_num}) must be divisible by ep ({self.ep})."
            )
        if self.single_rank_expert_num > MAX_EXPERT_NUM_PER_RANK:
            raise ValueError(
                "single_rank_expert_num cannot exceed the device scratch capacity "
                f"({MAX_EXPERT_NUM_PER_RANK}), got {self.single_rank_expert_num}."
            )

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def single_rank_expert_num(self) -> int:
        """Number of experts assigned to a single rank."""
        return self.all_expert_num // self.ep

    @property
    def seq_all(self) -> int:
        """Total sequence length after accounting for EP and TP parallelism."""
        return (self.seq_size * self.ep * self.top_k) // self.tp

    @property
    def per_expert_seq(self) -> int:
        """Sequence length per expert (across all EP ranks)."""
        return self.seq_all // self.top_k

    @property
    def per_rank_seq(self) -> int:
        """Sequence length assigned to each rank."""
        return self.seq_all // self.ep

    @property
    def per_expert_seq_to_other(self) -> int:
        """Sequence length per expert when sending to other ranks."""
        return self.seq_all // (self.ep * self.top_k)

    @property
    def all_event_num(self) -> int:
        """Total number of events needed for the schedule."""
        e = self.single_rank_expert_num
        return 1 + self.all_expert_num + e + e + e

    # ── Runtime counters (reset by init_task_split_value per rank) ────────────
    rank_id:              int = 0
    pre_pre_event_num:    int = 0
    pre_event_num:        int = 0
    pre_task_num:         int = 0
    pre_cube_task_num:    int = 0
    pre_vector_task_num:  int = 0
    pre_mix_task_num:     int = 0


def init_task_split_value(tsv: TaskSplitValue) -> None:
    """Reset per-rank runtime counters to zero."""
    tsv.pre_pre_event_num   = 0
    tsv.pre_event_num       = 0
    tsv.pre_task_num        = 0
    tsv.pre_cube_task_num   = 0
    tsv.pre_vector_task_num = 0
    tsv.pre_mix_task_num    = 0


def _validate_num_cube_cores(num_cube_cores: int) -> None:
    """Validate the hardware core count used by GMM task descriptors."""
    if not isinstance(num_cube_cores, int) or isinstance(num_cube_cores, bool) or num_cube_cores <= 0:
        raise ValueError(f"num_cube_cores must be a positive integer, got {num_cube_cores!r}.")


def _validate_task_bounds(task: TaskDescC, descriptor_index: int) -> None:
    """Validate the common task index range."""
    if task.task_split_num == 0 or task.task_index >= task.task_split_num:
        raise ValueError(
            f"task {descriptor_index} has invalid task_index/task_split_num "
            f"({task.task_index}/{task.task_split_num})."
        )


def _validate_swiglu_task(task: TaskDescC, descriptor_index: int, group_size: int) -> None:
    """Validate SwiGLU expert partition arithmetic."""
    if task.task_split_num < group_size or task.task_split_num % group_size:
        raise ValueError(
            f"SwiGLU task {descriptor_index} cannot be partitioned over {group_size} experts."
        )


def _validate_gmm_task(task: TaskDescC, descriptor_index: int, expected_tasks: int) -> None:
    """Validate that every local expert owns one task per Cube core."""
    if task.task_split_num != expected_tasks:
        raise ValueError(
            f"GMM task {descriptor_index} has task_split_num {task.task_split_num}; "
            f"expected {expected_tasks}."
        )


def _validate_shmem_task(task: TaskDescC, descriptor_index: int, tsv: TaskSplitValue) -> None:
    """Validate SHMEM task partitioning and remote-rank selection."""
    if task.task_split_num < tsv.all_expert_num or task.task_split_num % tsv.all_expert_num:
        raise ValueError(
            f"SHMEM task {descriptor_index} cannot be partitioned over "
            f"{tsv.all_expert_num} experts."
        )
    tasks_per_expert = task.task_split_num // tsv.all_expert_num
    target_rank_offset = task.task_index // (tsv.single_rank_expert_num * tasks_per_expert)
    if target_rank_offset >= tsv.ep:
        raise ValueError(
            f"SHMEM task {descriptor_index} targets rank offset {target_rank_offset}; "
            f"ep is {tsv.ep}."
        )


def validate_runtime_config(
    cfg: RuntimeConfigC,
    tsv: TaskSplitValue,
    num_cube_cores: int,
) -> None:
    """Validate task arithmetic before serializing a runtime configuration.

    The device worker intentionally assumes that generated task descriptors are
    internally consistent. Rejecting an invalid schedule here avoids both
    out-of-bounds descriptor access and device-side early exits that could omit
    event-counter or SHMEM-signal updates.

    Args:
        cfg: Populated runtime configuration awaiting serialization.
        tsv: Validated topology and task split values.
        num_cube_cores: Number of Cube cores used to generate GMM tasks.

    Raises:
        ValueError: If the runtime configuration violates a device-side invariant.
    """
    _validate_num_cube_cores(num_cube_cores)
    if cfg.num_workers != 2 * num_cube_cores:
        raise ValueError(
            f"num_workers ({cfg.num_workers}) must equal twice num_cube_cores ({num_cube_cores})."
        )
    if cfg.task_num > len(cfg.all_tasks):
        raise ValueError(f"task_num ({cfg.task_num}) exceeds allocated task capacity ({len(cfg.all_tasks)}).")
    event_capacity = len(cfg.all_event_num_triggers)
    queues = (cfg.cube_task_indices, cfg.vector_task_indices, cfg.mix_task_indices)
    for indices, count in zip(queues, cfg.task_index_num[:3]):
        if count < 0 or count > len(indices):
            raise ValueError(f"invalid runtime task-index count {count}")
        for task_id in indices[:count]:
            if task_id < 0 or task_id >= len(cfg.all_tasks):
                raise ValueError(f"invalid runtime task id {task_id}")
            task = cfg.all_tasks[task_id]
            if (task.dependent_event != 0xFFFFFFFF and task.dependent_event >= event_capacity) or (
                task.trigger_event + ATOMIC_ADD_VALUE_LEN > event_capacity
            ):
                raise ValueError(f"task {task_id} references an event outside allocated capacity {event_capacity}")

    local_experts = tsv.single_rank_expert_num
    group_size = cfg.dynamic_data.dynamic_group_size
    if group_size != local_experts:
        raise ValueError(
            f"dynamic_group_size ({group_size}) must equal local expert count ({local_experts})."
        )

    for descriptor_index in range(cfg.task_num):
        task = cfg.all_tasks[descriptor_index]
        task_type = task.task_type
        if task_type not in (
            TaskType.TASK_SWI_GLU,
            TaskType.TASK_SWI_GLU_GRAD,
            TaskType.TASK_GROUPED_MATMUL,
            TaskType.TASK_SHMEM_PUT_MEM_SIGNAL,
        ):
            continue
        _validate_task_bounds(task, descriptor_index)

        if task_type in (TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
            _validate_swiglu_task(task, descriptor_index, group_size)
        elif task_type == TaskType.TASK_GROUPED_MATMUL:
            _validate_gmm_task(task, descriptor_index, num_cube_cores * local_experts)
        else:
            _validate_shmem_task(task, descriptor_index, tsv)


def configure_ready_handshake(cfg: RuntimeConfigC, tsv: TaskSplitValue) -> None:
    """Reserve a local ready event after the graph's ordinary dependencies.

    Args:
        cfg: Allocated graph configuration whose counters include ready padding.
        tsv: Topology that determines whether peer readiness is needed.
    """
    if tsv.ep == 1:
        return
    ready_event = tsv.all_event_num + 3
    if ready_event + ATOMIC_ADD_VALUE_LEN > cfg.event_capacity:
        raise ValueError("ready handshake event exceeds event_capacity.")
    cfg.ready_event = ready_event
    cfg.all_event_num_triggers[ready_event] = 1


def event_workspace_bytes(ep_size: int, num_experts: int) -> int:
    """Return graph counters plus persistent per-rank ready generations.

    Args:
        ep_size: Number of expert-parallel ranks sharing the symmetric arena.
        num_experts: Global expert count used to size ordinary event counters.

    Returns:
        Required byte count for one direction's event workspace.
    """
    counter_bytes = mega_moe_event_capacity(num_experts, ep_size) * ctypes.sizeof(ctypes.c_int32)
    if ep_size <= 1:
        return counter_bytes
    return counter_bytes + (ep_size + 1) * READY_CACHE_LINE_BYTES
