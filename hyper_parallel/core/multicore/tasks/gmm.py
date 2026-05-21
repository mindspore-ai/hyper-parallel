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
"""GMM fill config: all GroupedMatMul variants (fwd up_proj/down_proj, bwd act/gate/w1/w2 grad)."""
from dataclasses import dataclass, field
from typing import Set

from hyper_parallel.core.multicore.scheduler.config import (
    TaskDescC, TensorDescC, RuntimeConfigC,
    TaskAiCoreType, TaskType,
    MAX_TENSOR_DIMS,
)
from hyper_parallel.core.multicore.tasks.task_base import FillConfig
from hyper_parallel.core.multicore.tasks.utils import (
    advance_tsv_cube, advance_tsv_cube_only,
)


@dataclass
class GmmFillConfig(FillConfig):
    """
    GMM fill config covering all variants (fwd up_proj/down_proj, bwd act_grad/w1_grad/
    gate_grad/w2_grad).  Tensor attributes (tensor_type, dtype_size, is_dynamic,
    transpose, shape) are read from TensorSpec.

    offset_inputs : Set[int]
        Input indices that receive base_ptr_offset and dynamic_shape.
        activation GMMs (up_proj, down_proj, act_grad, gate_grad): {0}
        weight-grad GMMs (w1_grad, w2_grad): {0, 1}

    rank_in_event : bool
        True  → dependent_event adds single_rank_expert_num * rank_id.
                 GMM1 (fwd/bwd), GMM4/w1_grad (bwd).
        False → no rank offset.
                 GMM2 (fwd/bwd), GMM3/w2_grad (bwd).

    global_trigger : bool
        False → trigger_event = pre_event_num + data_index + 1 (per-expert).
                 GMM1/GMM2 (activation-gradient path).
        True  → trigger_event = all_event_num (global).
                 GMM4/GMM3 (weight-gradient path).

    out_offset : bool
        True  → out.base_ptr_offset = data_index * 4096 * shape[1].
        False → 0 (weight-grad output writes to a dedicated buffer).

    advance : "cube" | "cube_only" | "cube_custom"
        "cube"        — standard cube advance (also advances events).
        "cube_only"   — advances task/cube counters only (no event update).
                        Used by: GMM1 bwd.
        "cube_custom" — manual: pre_task_num/pre_cube_task_num += task_num,
                        pre_pre_event_num = pre_event_num,
                        pre_event_num += event_delta.
                        Used by: GMM4 bwd, GMM3 bwd.

    event_delta : int
        Only used when advance="cube_custom".
        GMM4 bwd: single_rank_expert_num.  GMM3 bwd: 1.

    num_cube_cores : int
        Number of AI Cube cores (910B=24).
    """
    offset_inputs:  Set[int] = field(default_factory=lambda: {0})
    rank_in_event:  bool     = False
    global_trigger: bool     = False
    out_offset:     bool     = True
    advance:        str      = "cube"   # "cube" | "cube_only" | "cube_custom"
    event_delta:    int      = 0        # only used when advance="cube_custom"
    num_cube_cores: int      = 24

    def fill(self, cfg: RuntimeConfigC, op, tsv) -> None:
        task_num = op.task_num
        param    = op.param_positions
        glist_j  = len(op.inputs) - 1   # last input is always group_list

        for i in range(task_num):
            data_index = i // self.num_cube_cores
            task = TaskDescC()
            task.task_type        = TaskType.TASK_GROUPED_MATMUL
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
            task.num_inputs       = len(op.inputs)
            task.num_outputs      = len(op.outputs)

            for j, spec in enumerate(op.inputs):
                td = TensorDescC()
                td.input_position  = param[j]
                td.dynamic_shape   = 0
                td.base_ptr_offset = 0

                if j == glist_j:   # group_list: fixed type=0, dtype int64
                    td.tensor_type = 0
                    td.data_type   = spec.dtype_size
                else:              # x or weight: tensor_type from TensorSpec
                    td.tensor_type = spec.tensor_type
                    td.data_type   = spec.dtype_size

                if j in self.offset_inputs:
                    td.base_ptr_offset = data_index * 4096 * spec.shape[1]
                    td.dynamic_shape   = int(spec.is_dynamic)

                td.transpose_flag = int(spec.transpose)
                for k in range(min(len(spec.shape), MAX_TENSOR_DIMS)):
                    td.dim[k] = spec.shape[k]
                task.inputs[j] = td

            out_spec = op.outputs[0]
            out = TensorDescC()
            out.tensor_type     = out_spec.tensor_type
            out.data_type       = out_spec.dtype_size
            out.input_position  = param[task.num_inputs]
            out.dynamic_shape   = int(out_spec.is_dynamic)
            out.base_ptr_offset = (data_index * 4096 * out_spec.shape[1]
                                   if self.out_offset else 0)
            if self.out_offset:   # C++ fills dims only for activation outputs
                for k in range(min(len(out_spec.shape), MAX_TENSOR_DIMS)):
                    out.dim[k] = out_spec.shape[k]
            task.outputs[0] = out

            dep_extra = (tsv.single_rank_expert_num * tsv.rank_id
                         if self.rank_in_event else 0)
            task.dependent_event = tsv.pre_pre_event_num + dep_extra + data_index + 1
            if self.global_trigger:
                task.trigger_event = tsv.all_event_num
            else:
                task.trigger_event = tsv.pre_event_num + data_index + 1
            cfg.all_event_num_triggers[task.trigger_event] = self.num_cube_cores

            task.task_index           = i
            task.task_split_num       = task_num
            task.task_split_value     = op.split_value
            task.tiling_data_position = op.tiling_position

            cfg.all_tasks[tsv.pre_task_num + i]              = task
            cfg.cube_task_indices[tsv.pre_cube_task_num + i] = tsv.pre_task_num + i

        cfg.task_index_num[0] += task_num
        if self.advance == "cube":
            advance_tsv_cube(tsv, task_num, event_group_size=task_num // self.num_cube_cores)
        elif self.advance == "cube_only":
            advance_tsv_cube_only(tsv, task_num)
        else:   # "cube_custom"
            tsv.pre_task_num      += task_num
            tsv.pre_cube_task_num += task_num
            tsv.pre_pre_event_num  = tsv.pre_event_num
            tsv.pre_event_num     += self.event_delta
