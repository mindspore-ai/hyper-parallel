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
Task-queue reordering passes applied after the main fill loop.

revise_task_queue          — RATR: rank-aware tile reorder for dispatch/combine.
revise_gmm_task_queue_bwd  — Backward GMM1/GMM4 expert interleave in cube_task_indices.
"""
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC


def revise_task_queue(cfg: RuntimeConfigC, tsv,
                      dispatch_task_num: int, swiglu_task_num: int) -> None:
    """
    Reorder vector_task_indices for dispatch and combine based on rank_id (RATR).

    dispatch_task_num: task_num of the dispatch (A1) operator.
    swiglu_task_num:   task_num of the swiglu / swiglu_grad operator.
    """
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


def revise_gmm_task_queue_bwd(cfg: RuntimeConfigC, tsv,
                               act_grad_task_num: int,
                               num_cube_cores: int = 24) -> None:
    """
    Backward-only: interleave w1_grad and act_grad experts in cube_task_indices.

    Result pattern: [w1_grad exp0, act_grad exp0, w1_grad exp1, act_grad exp1, ...]
    act_grad start offset = 0; w1_grad start offset = act_grad_task_num.
    """
    temp          = list(cfg.cube_task_indices)
    expert_single = tsv.single_rank_expert_num
    changes_num   = 2   # two streams: w1_grad (index=1) and act_grad (index=0)

    for i in range(expert_single * changes_num):
        index = 1 - (i % changes_num)   # alternates: 1, 0, 1, 0, ...
        m     = i // changes_num         # expert block: 0, 0, 1, 1, ...
        for j in range(num_cube_cores):
            dst = i * num_cube_cores + j
            src = index * act_grad_task_num + m * num_cube_cores + j
            cfg.cube_task_indices[dst] = temp[src]
