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
Standalone RuntimeConfig builder.

build_runtime_config() is the generic framework entry point for converting a
ComputeGraph + TaskSplitValue into a binary RuntimeConfigC.  It omits
graph-specific post-processing (add_terminate, revise_task_queue, etc.) — call
build_config_for_rank() from a model's gen_runtime_data.py for full generation.
"""
from hyper_parallel.core.multicore.scheduler.config import (
    TaskSplitValue, init_task_split_value, RuntimeConfigC, QUEUE_CAPACITY,
)
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph


def build_runtime_config(graph: ComputeGraph, tsv: TaskSplitValue,
                         rank_id: int = 0,
                         num_cube_cores: int = 24) -> RuntimeConfigC:
    """
    Build RuntimeConfigC for a single rank from a ComputeGraph.

    Args:
        graph: A fully-built ComputeGraph after propagate_splits() has been called.
        tsv:   TaskSplitValue with TP/EP/seq topology.
        rank_id: Target rank (written to tsv.rank_id before filling).
        num_cube_cores: Number of AIC cube cores (910B=24).

    Returns:
        RuntimeConfigC ready to be serialized with bytes(cfg).
    """
    cfg = RuntimeConfigC()
    cfg.num_workers    = 2 * num_cube_cores
    cfg.queue_capacity = QUEUE_CAPACITY

    init_task_split_value(tsv)
    tsv.rank_id = rank_id

    for op in graph.topological_sort():
        op.fill_config.fill(cfg, op, tsv)

    cfg.task_num = sum(op.task_num for op in graph.topological_sort())
    cfg.atomic_add_values[0] = 1
    return cfg
