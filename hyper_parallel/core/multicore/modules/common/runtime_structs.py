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
C++ runtime header structs mapped to ctypes.Structure.
Matches runtime_head.hpp exactly.
"""
import ctypes
from enum import IntEnum

# ── Constants (match runtime_head.hpp exactly) ────────────────────────────────
MAX_TENSOR_DIMS      = 4
MAX_INPUTS_PER_TASK  = 4
MAX_OUTPUTS_PER_TASK = 4

MAX_TASK_NUM         = 256 * 100
MAX_EVENT_NUM        = 1024
NUM_WORKERS_VECTOR   = 48
NUM_WORKERS_CUBE     = 24
QUEUE_CAPACITY       = 100
TASK_TYPE_INDEX_NUM  = 256 * 100
MAX_GROUP_LIST       = 512
ATOMIC_ADD_VALUE_LEN = 8

# ── Enums ─────────────────────────────────────────────────────────────────────
class TaskAiCoreType(IntEnum):
    TASK_AICORE_INVALID = 0
    TASK_AICORE_CUBE    = 1
    TASK_AICORE_VECTOR  = 2
    TASK_AICORE_MIX     = 3


class TaskType(IntEnum):
    TASK_TERMINATE            = 0
    TASK_BEGIN_TASK_GRAPH     = 10
    TASK_ADD_CUSTOM           = 101
    TASK_SWI_GLU              = 102
    TASK_MATMUL               = 103
    TASK_GROUPED_MATMUL       = 104
    TASK_SHMEM_PUT_MEM_SIGNAL = 105
    TASK_SWI_GLU_GRAD         = 106


class EventType(IntEnum):
    EVENT_EMPTY                  = 900
    EVENT_LAUNCH_TASKS           = 901
    EVENT_LAUNCH_MASSIVE_TASKS   = 902
    EVENT_LAUNCH_DEPENDENT_TASKS = 903
    EVENT_END_OF_TASK_GRAPH      = 910
    EVENT_TERMINATION            = 911
    EVENT_INVALID                = 999


class DynamicType(IntEnum):
    DYNAMIC_EMPTY        = 0
    DYNAMIC_DSV3_MOE_FFN = 101


# ── ctypes Structures ─────────────────────────────────────────────────────────

class TensorDescC(ctypes.Structure):
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
        ("extra_value_3",        ctypes.c_uint32),
        ("extra_value_4",        ctypes.c_uint32),
    ]


class EventDescC(ctypes.Structure):
    _fields_ = [
        ("event_type",    ctypes.c_uint32),
        ("num_triggers",  ctypes.c_uint32),
        ("first_task_id", ctypes.c_uint32),
        ("last_task_id",  ctypes.c_uint32),
    ]


class DynamicDataC(ctypes.Structure):
    _fields_ = [
        ("dynamic_type",           ctypes.c_uint32),
        ("dynamic_input_position", ctypes.c_uint32),
        ("dynamic_group_size",     ctypes.c_uint32),
        ("dynamic_max_seq_len",    ctypes.c_int32),
    ]


class RuntimeConfigC(ctypes.Structure):
    _fields_ = [
        ("task_num",                  ctypes.c_uint32),
        ("num_workers",               ctypes.c_uint32),
        ("queue_capacity",            ctypes.c_uint32),
        ("config_extra_value",        ctypes.c_uint32),
        ("all_event_num_triggers",    ctypes.c_int32   * MAX_EVENT_NUM),
        ("all_tasks",                 TaskDescC         * MAX_TASK_NUM),
        ("all_events",                EventDescC        * MAX_EVENT_NUM),
        ("task_index_num",            ctypes.c_int32   * 4),
        ("cube_task_indices",         ctypes.c_int32   * TASK_TYPE_INDEX_NUM),
        ("vector_task_indices",       ctypes.c_int32   * TASK_TYPE_INDEX_NUM),
        ("mix_task_indices",          ctypes.c_int32   * TASK_TYPE_INDEX_NUM),
        ("dynamic_data",              DynamicDataC),
        ("grouped_matmul_group_list", ctypes.c_int64   * MAX_GROUP_LIST),
        ("atomic_add_values",         ctypes.c_int32   * ATOMIC_ADD_VALUE_LEN),
    ]


# TilingData (used by add_custom, not GMM)
class TilingDataC(ctypes.Structure):
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
