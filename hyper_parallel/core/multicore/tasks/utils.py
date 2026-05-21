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
Shared utilities for all FillConfig implementations.

Counter helpers (advance_tsv_*) mirror the C++ tsv state advancement.
add_terminate / add_dynamic_data are called from gen_runtime_data after
the main fill loop.
"""
from hyper_parallel.core.multicore.scheduler.config import (
    TensorDescC, TaskDescC, RuntimeConfigC,
    TaskType, DynamicType,
    MAX_TENSOR_DIMS,
)


# ── TensorDescC quick constructor ─────────────────────────────────────────────

def make_tensor_desc(
    tensor_type:     int = 0,
    data_type:       int = 2,
    input_position:  int = 0,
    base_ptr_offset: int = 0,
    transpose_flag:  int = 0,
    dynamic_shape:   int = 0,
    dynamic_dim:     int = 0,
    shape:           list = None,
) -> TensorDescC:
    """Build a TensorDescC with the given fields."""
    t = TensorDescC()
    t.tensor_type     = tensor_type
    t.data_type       = data_type
    t.input_position  = input_position
    t.base_ptr_offset = base_ptr_offset
    t.transpose_flag  = transpose_flag
    t.dynamic_shape   = dynamic_shape
    t.dynamic_dim     = dynamic_dim
    if shape:
        for k, v in enumerate(shape[:MAX_TENSOR_DIMS]):
            t.dim[k] = v
    return t


# ── tsv counter helpers ───────────────────────────────────────────────────────

def advance_tsv_vector(tsv, task_num: int, event_group_size: int) -> None:
    """Advance counters after a vector op with event grouping."""
    tsv.pre_task_num        += task_num
    tsv.pre_vector_task_num += task_num
    tsv.pre_pre_event_num    = tsv.pre_event_num
    tsv.pre_event_num       += event_group_size


def advance_tsv_cube(tsv, task_num: int, event_group_size: int) -> None:
    """Advance counters after a cube op with event grouping."""
    tsv.pre_task_num      += task_num
    tsv.pre_cube_task_num += task_num
    tsv.pre_pre_event_num  = tsv.pre_event_num
    tsv.pre_event_num     += event_group_size


def advance_tsv_cube_only(tsv, task_num: int) -> None:
    """Advance only task/cube counters (no event update). Used by bwd GMM1."""
    tsv.pre_task_num      += task_num
    tsv.pre_cube_task_num += task_num


def advance_tsv_vector_only(tsv, task_num: int) -> None:
    """Advance only task/vector counters (no event update). Used by bwd combine."""
    tsv.pre_task_num        += task_num
    tsv.pre_vector_task_num += task_num


# ── Post-fill utilities ───────────────────────────────────────────────────────

def add_terminate(cfg: RuntimeConfigC, tsv, trigger_count: int) -> None:
    """
    Append a terminate task to cfg.

    trigger_count:
      forward:  combine_op.task_num // tsv.ep * tsv.ep
      backward: w1_grad_op.task_num + w2_grad_op.task_num + combine_op.task_num // tsv.ep * tsv.ep
    """
    cfg.all_event_num_triggers[tsv.all_event_num] = trigger_count
    task = TaskDescC()
    task.task_type       = TaskType.TASK_TERMINATE
    task.dependent_event = tsv.pre_pre_event_num + 1
    task.trigger_event   = tsv.pre_event_num + 1
    cfg.all_event_num_triggers[task.trigger_event] = 1

    cfg.all_tasks[tsv.pre_task_num]                 = task
    cfg.vector_task_indices[tsv.pre_vector_task_num] = tsv.pre_task_num
    cfg.task_index_num[1] += 1


def add_dynamic_data(cfg: RuntimeConfigC, tsv, dynamic_input_position: int) -> None:
    """
    Write the dynamic data record into cfg.

    dynamic_input_position: forward = 6, backward = 19.
    """
    cfg.dynamic_data.dynamic_type           = DynamicType.DYNAMIC_DSV3_MOE
    cfg.dynamic_data.dynamic_input_position = dynamic_input_position
    cfg.dynamic_data.dynamic_group_size     = tsv.single_rank_expert_num
    cfg.dynamic_data.dynamic_max_seq_len    = -1
