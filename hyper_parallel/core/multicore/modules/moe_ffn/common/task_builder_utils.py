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
Common utilities shared by forward and backward task builders.
Functions here are byte-for-byte identical between fwd and bwd C++ code.
"""

from hyper_parallel.core.multicore.modules.moe_ffn.common.runtime_structs import (
    TensorDescC, RuntimeConfigC,
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
    tsv.pre_task_num          += task_num
    tsv.pre_vector_task_num   += task_num
    tsv.pre_pre_event_num      = tsv.pre_event_num
    tsv.pre_event_num         += event_group_size


def advance_tsv_cube(tsv, task_num: int, event_group_size: int) -> None:
    """Advance counters after a cube op with event grouping."""
    tsv.pre_task_num        += task_num
    tsv.pre_cube_task_num   += task_num
    tsv.pre_pre_event_num    = tsv.pre_event_num
    tsv.pre_event_num       += event_group_size


def advance_tsv_cube_only(tsv, task_num: int) -> None:
    """Advance only task/cube counters (no event update). Used by bwd GMM1."""
    tsv.pre_task_num      += task_num
    tsv.pre_cube_task_num += task_num


def advance_tsv_vector_only(tsv, task_num: int) -> None:
    """Advance only task/vector counters (no event update). Used by bwd A2."""
    tsv.pre_task_num        += task_num
    tsv.pre_vector_task_num += task_num


# ── revise_task_queue (identical in fwd and bwd) ──────────────────────────────

def revise_task_queue(cfg: RuntimeConfigC, tsv,
                      dispatch_task_num: int, swiglu_task_num: int) -> None:
    """
    Reorder vector_task_indices for dispatch and combine segments based on rank_id.
    Directly translates C++ revise_task_queue.

    dispatch_task_num : task_num of the dispatch (A1) operator
    swiglu_task_num   : task_num of the swiglu / swiglu_grad operator
    """
    # snapshot current queue
    temp = list(cfg.vector_task_indices)

    single_rank_expert_num = tsv.single_rank_expert_num
    single_expert_task_num = dispatch_task_num // tsv.all_expert_num
    ep                     = tsv.ep
    rank_id                = tsv.rank_id
    single_rank_task_num   = dispatch_task_num // tsv.ep

    ep_rank = [(i + rank_id) % ep for i in range(ep)]

    # ── dispatch segment ─────────────────────────────────────────────────────
    start = 0
    index = 0
    for j in range(single_rank_expert_num):
        j_v = j * single_expert_task_num
        for k in range(single_expert_task_num):
            k_v = j_v + k
            for i in ep_rank:
                i_v = k_v + i * single_rank_task_num
                cfg.vector_task_indices[start + index] = temp[start + i_v]
                index += 1

    # ── combine segment ──────────────────────────────────────────────────────
    start = dispatch_task_num + swiglu_task_num
    index = 0
    for j in range(single_rank_expert_num):
        j_v = j * single_expert_task_num
        for k in range(single_expert_task_num):
            k_v = j_v + k
            for i in ep_rank:
                i_v = k_v + i * single_rank_task_num
                cfg.vector_task_indices[start + index] = temp[start + i_v]
                index += 1
