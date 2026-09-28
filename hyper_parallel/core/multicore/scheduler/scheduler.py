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
from __future__ import annotations

from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC, TaskSplitValue


def revise_task_queue(cfg: RuntimeConfigC, tsv: TaskSplitValue,
                      dispatch_task_num: int, swiglu_task_num: int, combine_task_num: int | None = None) -> None:
    """
    Reorder vector_task_indices for dispatch and combine based on rank_id (RATR).

    Args:
        cfg: Runtime whose vector queue is reordered in place.
        tsv: Expert topology and local rank.
        dispatch_task_num: Number of dispatch tasks.
        swiglu_task_num: Number of intervening SwiGLU or SwiGLU-grad tasks.
        combine_task_num: Number of combine tasks, possibly using a different split.
    """
    temp = list(cfg.vector_task_indices)
    ep_rank = [(peer + tsv.rank_id) % tsv.ep for peer in range(tsv.ep)]
    if combine_task_num is None:
        combine_task_num = dispatch_task_num
    segments = ((0, dispatch_task_num), (dispatch_task_num + swiglu_task_num, combine_task_num))
    for start, task_count in segments:
        tasks_per_expert = task_count // tsv.all_expert_num
        tasks_per_peer = task_count // tsv.ep
        index = 0
        for expert in range(tsv.single_rank_expert_num):
            for tile in range(tasks_per_expert):
                for peer in ep_rank:
                    source = peer * tasks_per_peer + expert * tasks_per_expert + tile
                    cfg.vector_task_indices[start + index] = temp[start + source]
                    index += 1


def revise_gmm_task_queue_bwd(cfg: RuntimeConfigC, tsv: TaskSplitValue,
                               act_grad_task_num: int,
                               num_cube_cores: int = 24) -> None:
    """
    Backward-only: interleave w2_grad and act_grad experts in cube_task_indices.

    Result pattern: [w2_grad exp0, act_grad exp0, w2_grad exp1, act_grad exp1, ...]
    act_grad start offset = 0; w2_grad start offset = act_grad_task_num.

    Args:
        cfg: Runtime image with the original Cube queue.
        tsv: Expert topology.
        act_grad_task_num: Offset at which weight-gradient tasks begin.
        num_cube_cores: Number of participating Cube workers.
    """
    temp          = list(cfg.cube_task_indices)
    expert_single = tsv.single_rank_expert_num
    changes_num   = 2   # two streams: w2_grad (index=1) and act_grad (index=0)

    for i in range(expert_single * changes_num):
        index = 1 - (i % changes_num)   # alternates: 1, 0, 1, 0, ...
        m     = i // changes_num         # expert block: 0, 0, 1, 1, ...
        for j in range(num_cube_cores):
            dst = i * num_cube_cores + j
            src = index * act_grad_task_num + m * num_cube_cores + j
            cfg.cube_task_indices[dst] = temp[src]
