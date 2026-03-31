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
Polymorphic fill configs for mega-kernel task scheduling.

Each operator type has a FillConfig subclass that encapsulates both the
config data and the fill logic (fill method).  OperatorNode.fill_config
holds an instance; gen_runtime_data calls op.fill_config.fill(cfg, op, tsv).

Public fill config classes:
  AllToAllFillConfig — dispatch and combine (fwd + bwd), unified
  GmmFillConfig      — all GMM variants G1/G2/G3/G4 (fwd + bwd)
  SwiGLUFillConfig   — SwiGLU fwd and SwiGLU-grad bwd (no config fields)

Utility functions (called directly from gen_runtime_data):
  add_terminate             — terminate task; caller passes trigger_count int
  add_dynamic_data          — dynamic data record (fwd pos=6, bwd pos=19)
  revise_gmm_task_queue_bwd — backward GMM1/GMM4 interleave in cube_task_indices
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Set

from hyper_parallel.core.multicore.modules.moe_ffn.common.runtime_structs import (
    TaskDescC, TensorDescC, RuntimeConfigC,
    TaskAiCoreType, TaskType, DynamicType,
    MAX_TENSOR_DIMS,
)
from hyper_parallel.core.multicore.modules.moe_ffn.common.task_builder_utils import (
    advance_tsv_vector, advance_tsv_cube,
    advance_tsv_cube_only, advance_tsv_vector_only,
)
from hyper_parallel.core.multicore.modules.moe_ffn.common.compute_graph import OpType


# ── MoE AllToAll type enum ────────────────────────────────────────────────────

class AllToAllType(Enum):
    """
    MoE AllToAll semantic type, determines event wiring.

    DISPATCH — scatter tokens from model-parallel ranks to expert-holding ranks.
        Event wiring: per-expert trigger.
          dependent_event = pre_pre_event_num + 0
          trigger_event   = pre_event_num + (i // per_g_e_num) + 1
          trigger_count   = task_num * ep // all_expert_num

    COMBINE — gather expert results back to the originating model-parallel rank.
        Event wiring: global trigger (wait for all experts before gathering).
          dependent_event = pre_pre_event_num + (i // per_g_e_num) % sre + 1
          trigger_event   = all_event_num
          trigger_count   = task_num

    OTHER
        Reserved for AllToAll patterns outside the MoE dispatch/combine semantic.
    """
    DISPATCH = 1
    COMBINE  = 2
    OTHER    = 3


# ── Abstract base ──────────────────────────────────────────────────────────────

class FillConfig(ABC):
    """Abstract base for all fill configs.  Subclasses hold config data and
    implement fill() with the actual task-building logic."""

    @abstractmethod
    def fill(self, cfg: RuntimeConfigC, op, tsv) -> None:
        """Fill tasks into cfg for the given op using runtime state tsv."""


# ── AllToAll (dispatch + combine, forward + backward) ─────────────────────────

@dataclass
class AllToAllFillConfig(FillConfig):
    """
    AllToAll behaviour config covering dispatch and combine for fwd and bwd.

    moe_type : AllToAllType
        Determines event wiring; see AllToAllType for details.

    advance : "vector" | "vector_only"
        "vector"      — advance_tsv_vector(tsv, task_num, event_group_size=event_group)
                        Advances pre_event_num/pre_pre_event_num/pre_task_num/
                        pre_vector_task_num.
                        Used by: dispatch (fwd/bwd), combine forward.
        "vector_only" — advance_tsv_vector_only(tsv, task_num)
                        Only advances pre_task_num/pre_vector_task_num; does not
                        advance event counters.
                        Used by: combine backward (event advance is deferred to GMM3).

    event_group : int
        Only effective when advance="vector"; passed to advance_tsv_vector.
        dispatch (fwd/bwd): all_expert_num
        combine forward:    1
    """
    moe_type:    AllToAllType = AllToAllType.DISPATCH
    advance:     str          = "vector"   # "vector" | "vector_only"
    event_group: int          = 1          # only used when advance="vector"

    def fill(self, cfg: RuntimeConfigC, op, tsv) -> None:
        task_num    = op.task_num
        per_g_e_num = task_num // tsv.all_expert_num
        param       = op.param_positions

        for i in range(task_num):
            task = TaskDescC()
            task.task_type        = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
            task.num_inputs       = len(op.inputs)
            task.num_outputs      = len(op.outputs)

            for j, spec in enumerate(op.inputs):
                td = TensorDescC()
                td.data_type       = spec.dtype_size
                td.input_position  = param[j]
                td.base_ptr_offset = 0

                if j == 1:   # src data: tensor_type and is_dynamic come from TensorSpec
                    td.tensor_type   = spec.tensor_type
                    td.dynamic_shape = int(spec.is_dynamic)
                else:        # metadata (target_offset / src_offset / size): fixed type=0
                    td.tensor_type     = 0
                    td.base_ptr_offset = i // per_g_e_num   # expert-group index

                task.inputs[j] = td

            out_spec = op.outputs[0]
            out = TensorDescC()
            out.tensor_type     = out_spec.tensor_type
            out.data_type       = out_spec.dtype_size
            out.input_position  = param[task.num_inputs]
            out.base_ptr_offset = 0
            out.dynamic_shape   = int(out_spec.is_dynamic)
            task.outputs[0] = out

            if self.moe_type == AllToAllType.DISPATCH:
                res = i // per_g_e_num
                task.dependent_event = tsv.pre_pre_event_num + 0
                task.trigger_event   = tsv.pre_event_num + res + 1
                cfg.all_event_num_triggers[task.trigger_event] = (
                    task_num * tsv.ep // tsv.all_expert_num
                )
            else:   # COMBINE (or OTHER)
                current_dep = (i // per_g_e_num) % tsv.single_rank_expert_num
                task.dependent_event = tsv.pre_pre_event_num + current_dep + 1
                task.trigger_event   = tsv.all_event_num
                cfg.all_event_num_triggers[task.trigger_event] = task_num

            task.task_index           = i
            task.task_split_num       = task_num
            task.task_split_value     = op.split_value
            task.tiling_data_position = 0xFFFFFFFF

            cfg.all_tasks[tsv.pre_task_num + i] = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + i] = tsv.pre_task_num + i

        cfg.task_index_num[1] += task_num
        if self.advance == "vector":
            advance_tsv_vector(tsv, task_num, event_group_size=self.event_group)
        else:   # "vector_only"
            advance_tsv_vector_only(tsv, task_num)


# ── GMM (up_proj/down_proj/act_grad/gate_grad/w1_grad/w2_grad, fwd + bwd) ─────

@dataclass
class GmmFillConfig(FillConfig):
    """
    GMM behaviour config covering all GMM variants (fwd up_proj/down_proj,
    bwd act_grad/w1_grad/gate_grad/w2_grad).
    Tensor-level attributes (tensor_type, dtype_size, is_dynamic, transpose,
    shape) are read from TensorSpec.

    offset_inputs : Set[int]
        Set of input indices that receive base_ptr_offset and dynamic_shape.
        fwd/bwd activation GMMs (up_proj, down_proj, act_grad, gate_grad): {0}
        bwd weight-grad GMMs (w1_grad, w2_grad): {0, 1}

    rank_in_event : bool
        True  → dependent_event adds single_rank_expert_num * rank_id.
                 Used by: GMM1 (fwd/bwd), GMM4/w1_grad (bwd).
        False → no rank offset.
                 Used by: GMM2 (fwd/bwd), GMM3/w2_grad (bwd).

    global_trigger : bool
        False → trigger_event = pre_event_num + data_index + 1 (per-expert trigger).
                 Used by: GMM1/GMM2 (activation-gradient path).
        True  → trigger_event = all_event_num (global trigger).
                 Used by: GMM4/GMM3 (weight-gradient path).

    out_offset : bool
        True  → out.base_ptr_offset = data_index * 4096 * shape[1].
        False → out.base_ptr_offset = 0 (weight-grad output writes to dedicated buffer).

    advance : "cube" | "cube_only" | "cube_custom"
        "cube"        — advance_tsv_cube(tsv, task_num, event_group_size=task_num//CUBE)
                        Standard cube advance (also advances events).
                        Used by: GMM1 fwd, GMM2 fwd/bwd.
        "cube_only"   — advance_tsv_cube_only(tsv, task_num)
                        Advances task/cube counters only; event advance is deferred
                        to the subsequent GMM4.
                        Used by: GMM1 bwd (GMM1 and GMM4 run in parallel sharing events).
        "cube_custom" — Manual advance: pre_task_num/pre_cube_task_num += task_num,
                        pre_pre_event_num = pre_event_num,
                        pre_event_num += event_delta.
                        Used by: GMM4 bwd (event_delta = sre), GMM3 bwd (event_delta = 1).

    event_delta : int
        Only effective when advance="cube_custom"; increment for pre_event_num.
        GMM4 bwd: single_rank_expert_num (computed and passed at graph declaration time).
        GMM3 bwd: 1.
    """
    offset_inputs:  Set[int] = field(default_factory=lambda: {0})
    rank_in_event:  bool     = False
    global_trigger: bool     = False
    out_offset:     bool     = True
    advance:        str      = "cube"   # "cube" | "cube_only" | "cube_custom"
    event_delta:    int      = 0        # only used when advance="cube_custom"
    num_cube_cores: int      = 24       # number of AI Cube cores (910B=24)

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
                else:              # x or weight: tensor_type comes from TensorSpec
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
            if self.out_offset:   # C++ fills dims only for activation outputs, not weight grads
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

            cfg.all_tasks[tsv.pre_task_num + i]             = task
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


# ── SwiGLU / SwiGLU-grad ──────────────────────────────────────────────────────

@dataclass
class SwiGLUFillConfig(FillConfig):
    """
    SwiGLU fill config — forward (TASK_SWI_GLU) and backward gradient
    (TASK_SWI_GLU_GRAD).

    No config fields; task_type is derived from op.op_type, split_value and
    task_num are read from op.  All input/output tensor_type values are always
    1 (vector operator convention).
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


# ── Utility functions ──────────────────────────────────────────────────────────

def add_terminate(cfg: RuntimeConfigC, tsv, trigger_count: int) -> None:
    """
    Append a terminate task to cfg.

    trigger_count : value written to cfg.all_event_num_triggers[all_event_num].
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
    cfg.dynamic_data.dynamic_type           = DynamicType.DYNAMIC_DSV3_MOE_FFN
    cfg.dynamic_data.dynamic_input_position = dynamic_input_position
    cfg.dynamic_data.dynamic_group_size     = tsv.single_rank_expert_num
    cfg.dynamic_data.dynamic_max_seq_len    = -1


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
