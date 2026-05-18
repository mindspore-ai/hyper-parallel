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
"""SwiGLU fill config: forward (TASK_SWI_GLU) and backward gradient (TASK_SWI_GLU_GRAD)."""
from dataclasses import dataclass

from hyper_parallel.core.multicore.scheduler.config import (
    TaskDescC, TensorDescC, RuntimeConfigC,
    TaskAiCoreType, TaskType,
    MAX_TENSOR_DIMS,
)
from hyper_parallel.core.multicore.scheduler.graph import OpType
from hyper_parallel.core.multicore.tasks.task_base import FillConfig
from hyper_parallel.core.multicore.tasks.utils import advance_tsv_vector


@dataclass
class SwiGLUFillConfig(FillConfig):
    """
    SwiGLU fill config — forward (TASK_SWI_GLU) and backward (TASK_SWI_GLU_GRAD).

    No config fields; task_type is derived from op.op_type.  All input/output
    tensor_type values are always 1 (vector operator convention).
    """

    def fill(self, cfg: RuntimeConfigC, op, tsv) -> None:
        task_type = (TaskType.TASK_SWI_GLU if op.op_type == OpType.SWIGLU
                     else TaskType.TASK_SWI_GLU_GRAD)

        num_triggers = tsv.per_expert_seq // op.split_value
        task_num     = op.task_num
        param        = op.param_positions

        for i in range(task_num):
            task = TaskDescC()
            task.task_type        = task_type
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
            task.num_inputs       = len(op.inputs)
            task.num_outputs      = len(op.outputs)

            for j, spec in enumerate(op.inputs):
                td = TensorDescC()
                td.tensor_type     = 1   # all SwiGLU inputs are tensor lists
                td.data_type       = spec.dtype_size
                td.input_position  = param[j]
                td.base_ptr_offset = i * spec.shape[1] * op.split_value
                td.dynamic_shape   = int(spec.is_dynamic)
                for k in range(min(len(spec.shape), MAX_TENSOR_DIMS)):
                    td.dim[k] = spec.shape[k]
                task.inputs[j] = td

            for j, spec in enumerate(op.outputs):
                td = TensorDescC()
                td.tensor_type     = 1   # all SwiGLU outputs are tensor lists
                td.data_type       = spec.dtype_size
                td.input_position  = param[task.num_inputs + j]
                td.base_ptr_offset = i * spec.shape[1] * op.split_value
                td.dynamic_shape   = int(spec.is_dynamic)
                for k in range(min(len(spec.shape), MAX_TENSOR_DIMS)):
                    td.dim[k] = spec.shape[k]
                task.outputs[j] = td

            ev_idx = i // num_triggers
            task.dependent_event = tsv.pre_pre_event_num + ev_idx + 1
            task.trigger_event   = tsv.pre_event_num + ev_idx + 1
            cfg.all_event_num_triggers[task.trigger_event] = num_triggers

            task.task_index           = i
            task.task_split_num       = task_num
            task.task_split_value     = op.split_value
            task.tiling_data_position = op.tiling_position

            cfg.all_tasks[tsv.pre_task_num + i]               = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + i] = tsv.pre_task_num + i

        cfg.task_index_num[1] += task_num
        advance_tsv_vector(tsv, task_num, event_group_size=tsv.single_rank_expert_num)
