"""
Backward fill functions — one per operator in the backward graph.
Each function exactly mirrors the corresponding C++ task_split_* function
in multicore_moe_ffn_grad/examples/task_register.hpp.

Key backward vs forward differences (see plan doc for full table):
  A1  bwd: j==1 input tensor_type=1; output tensor_type=1, dynamic_shape=1
  GMM1 bwd: all inputs tensor_type=1; weight transpose_flag=1; NO event advance
  GMM4 bwd: j==0,1 offset+dynamic; j==0 transpose_flag=1; trigger=all_event_num;
            post-advance events using gmm_task_num (GMM1's count!)
  A2  bwd: output tensor_type=1; slot=pre_vector_task_num+i; NO event advance
  GMM3 bwd: same shape as GMM4; post-advance pre_event_num += 1
  Terminate bwd: triggers = g4 + g3 + a2/ep*ep
  Dynamic data bwd: dynamic_input_position = 19
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from common.runtime_structs import (
    TaskDescC, TensorDescC, RuntimeConfigC,
    TaskAiCoreType, TaskType,
    MAX_TENSOR_DIMS, NUM_WORKERS_CUBE, TASK_TYPE_INDEX_NUM,
)
from common.task_builder_utils import (
    advance_tsv_vector, advance_tsv_cube,
    advance_tsv_cube_only, advance_tsv_vector_only,
)


# ── AllToAll A1 (backward) ────────────────────────────────────────────────────
# C++: task_split_alltoallv  (bwd, line 106 in bwd task_register.hpp)
# KEY diff from fwd: j==1 tensor_type=1; output tensor_type=1, dynamic_shape=1

def fill_alltoall_a1_bwd(cfg: RuntimeConfigC, op, tsv) -> None:
    task_num    = op.task_num
    per_g_e_num = task_num // tsv.all_expert_num
    event_trig  = (tsv.alltoall_task_num * tsv.ep) // tsv.all_expert_num
    param       = op.param_positions  # [1,2,3,4,0]
    dtype_size  = 2

    for i in range(task_num):
        task = TaskDescC()
        task.task_type        = TaskType.TASK_SHMEM_PUT_MEM_SINGAL
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
        task.num_inputs       = len(op.inputs)
        task.num_outputs      = len(op.outputs)

        for j in range(task.num_inputs):
            td = TensorDescC()
            td.tensor_type     = 0
            td.data_type       = dtype_size
            td.input_position  = param[j]
            td.base_ptr_offset = 0
            if j == 0:                           # target_offset: int64
                td.data_type       = 8
                td.base_ptr_offset = i // per_g_e_num
            elif j == 1:                         # src: bf16, tensor_type=1 (BWD diff)
                td.tensor_type = 1
            elif j == 2:                         # src_offset: int64
                td.data_type       = 8
                td.base_ptr_offset = i // per_g_e_num
            elif j == 3:                         # size: int32
                td.data_type       = 4
                td.base_ptr_offset = i // per_g_e_num
            task.inputs[j] = td

        # output: tensor_type=1  (BWD diff: fwd uses type=0)
        out = TensorDescC()
        out.tensor_type     = 1
        out.data_type       = dtype_size
        out.input_position  = param[task.num_inputs]
        out.base_ptr_offset = 0
        out.dynamic_shape   = int(op.outputs[0].is_dynamic)
        task.outputs[0] = out

        res = i // per_g_e_num
        task.dependent_event = tsv.pre_pre_event_num + 0
        task.trigger_event   = tsv.pre_event_num + res + 1
        cfg.all_event_num_triggers[task.trigger_event] = event_trig

        task.task_index           = i
        task.task_split_num       = task_num
        task.task_split_value     = op.split_value
        task.tiling_data_position = 0xFFFFFFFF

        cfg.all_tasks[tsv.pre_task_num + i]                  = task
        cfg.vector_task_indexs[tsv.pre_vector_task_num + i]  = tsv.pre_task_num + i

    cfg.task_index_num[1] += task_num
    advance_tsv_vector(tsv, task_num, event_group_size=tsv.all_expert_num)


# ── GMM1 (backward) ───────────────────────────────────────────────────────────
# C++: task_split_gmm  (bwd, line 179 in bwd task_register.hpp)
# KEY diffs from fwd:
#   - all inputs start tensor_type=1
#   - j!=0 also gets transpose_flag=1 (weight)
#   - j==2 gets tensor_type=0, int64, transpose_flag=0 (group_list)
#   - tiling_position = 20
#   - NO event advance after (only task/cube counters advance)

def fill_gmm_g1_bwd(cfg: RuntimeConfigC, op, tsv) -> None:
    task_num  = op.task_num
    param     = op.param_positions  # [0, 7, 19, 8]
    dtype_size = 2

    for i in range(task_num):
        data_index = i // NUM_WORKERS_CUBE
        task = TaskDescC()
        task.task_type        = TaskType.TASK_GROUPED_MATMUL
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
        task.num_inputs       = len(op.inputs)
        task.num_outputs      = len(op.outputs)

        for j in range(task.num_inputs):
            td = TensorDescC()
            td.data_type       = dtype_size
            td.input_position  = param[j]
            td.tensor_type     = 1
            td.dynamic_shape   = 0
            td.base_ptr_offset = 0

            if j == 0:   # x
                td.base_ptr_offset = data_index * 4096 * op.inputs[j].shape[1]
                td.dynamic_shape   = int(op.inputs[j].is_dynamic)
            else:        # weight: type=1
                td.tensor_type    = 1
                td.base_ptr_offset = 0

            if j == 2:   # group_list: type=0, int64
                td.tensor_type    = 0
                td.data_type      = 8

            td.transpose_flag = int(op.inputs[j].transpose)

            shape = op.inputs[j].shape
            for k in range(min(len(shape), MAX_TENSOR_DIMS)):
                td.dim[k] = shape[k]
            task.inputs[j] = td

        # output: tensor_type=1
        out_spec = op.outputs[0]
        out = TensorDescC()
        out.data_type       = dtype_size
        out.input_position  = param[task.num_inputs]
        out.tensor_type     = 1
        out.base_ptr_offset = data_index * 4096 * out_spec.shape[1]
        out.dynamic_shape   = int(out_spec.is_dynamic)
        for k in range(min(len(out_spec.shape), MAX_TENSOR_DIMS)):
            out.dim[k] = out_spec.shape[k]
        task.outputs[0] = out

        task.dependent_event = (tsv.pre_pre_event_num
                                + tsv.single_rank_expert_num * tsv.rank_id
                                + data_index + 1)
        task.trigger_event   = tsv.pre_event_num + data_index + 1
        cfg.all_event_num_triggers[task.trigger_event] = NUM_WORKERS_CUBE

        task.task_index           = i
        task.task_split_num       = task_num
        task.task_split_value     = op.split_value
        task.tiling_data_position = op.tiling_position  # 20

        cfg.all_tasks[tsv.pre_task_num + i]             = task
        cfg.cube_task_indexs[tsv.pre_cube_task_num + i] = tsv.pre_task_num + i

    cfg.task_index_num[0] += task_num
    # BWD: NO pre_pre_event_num / pre_event_num update — only task/cube advance
    advance_tsv_cube_only(tsv, task_num)


# ── GMM4 (backward) — weight grad for GMM2 ────────────────────────────────────
# C++: task_split_gmm_g4  (bwd, line 271 in bwd task_register.hpp)
# Inputs: x1^T (transposed), x2, group_list
#   j==0,1: offset+dynamic; j==0 also transpose_flag=1
#   j==2: tensor_type=0, int64
# Output: type=1, offset=0, no dynamic
# trigger = all_event_num
# Post-advance: uses gmm_task_num (GMM1's task count!) for event advance

def fill_gmm_g4_bwd(cfg: RuntimeConfigC, op, tsv) -> None:
    task_num  = op.task_num
    param     = op.param_positions  # [5, 0, 19, 6]
    dtype_size = 2

    for i in range(task_num):
        data_index = i // NUM_WORKERS_CUBE
        task = TaskDescC()
        task.task_type        = TaskType.TASK_GROUPED_MATMUL
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
        task.num_inputs       = len(op.inputs)
        task.num_outputs      = len(op.outputs)

        for j in range(task.num_inputs):
            td = TensorDescC()
            td.data_type       = dtype_size
            td.input_position  = param[j]
            td.tensor_type     = 1
            td.base_ptr_offset = 0
            td.dynamic_shape   = 0

            if j == 0 or j == 1:  # both x1,x2: offset+dynamic
                td.base_ptr_offset = data_index * 4096 * op.inputs[j].shape[1]
                td.dynamic_shape   = int(op.inputs[j].is_dynamic)

            if j == 2:            # group_list: type=0, int64
                td.tensor_type    = 0
                td.data_type      = 8

            td.transpose_flag = int(op.inputs[j].transpose)

            shape = op.inputs[j].shape
            for k in range(min(len(shape), MAX_TENSOR_DIMS)):
                td.dim[k] = shape[k]
            task.inputs[j] = td

        # output: type=1, offset=0, no dynamic
        out_spec = op.outputs[0]
        out = TensorDescC()
        out.data_type       = dtype_size
        out.input_position  = param[task.num_inputs]
        out.tensor_type     = 1
        out.base_ptr_offset = 0
        out.dynamic_shape   = int(out_spec.is_dynamic)
        task.outputs[0] = out

        task.dependent_event = (tsv.pre_pre_event_num
                                + tsv.single_rank_expert_num * tsv.rank_id
                                + data_index + 1)
        task.trigger_event   = tsv.all_event_num
        cfg.all_event_num_triggers[task.trigger_event] = NUM_WORKERS_CUBE

        task.task_index           = i
        task.task_split_num       = task_num
        task.task_split_value     = op.split_value
        task.tiling_data_position = op.tiling_position  # 23

        cfg.all_tasks[tsv.pre_task_num + i]             = task
        cfg.cube_task_indexs[tsv.pre_cube_task_num + i] = tsv.pre_task_num + i

    cfg.task_index_num[0] += task_num
    # Post-advance: uses gmm_task_num (GMM1's task count!), NOT g4's own count
    tsv.pre_task_num        += task_num
    tsv.pre_cube_task_num   += task_num
    tsv.pre_pre_event_num    = tsv.pre_event_num
    tsv.pre_event_num       += tsv.gmm_task_num // NUM_WORKERS_CUBE


# ── SwiGLU-grad (backward) ────────────────────────────────────────────────────
# C++: task_split_swiglu_grad  (bwd, line 353 in bwd task_register.hpp)
# Same structure as forward SwiGLU but:
#   - task_type = TASK_SWI_GLU_GRAD
#   - tiling_position = 24

def fill_swiglu_grad_bwd(cfg: RuntimeConfigC, op, tsv) -> None:
    dtype_size = 2

    # Recalculate using dynamic_shape=1 branch (same logic as C++)
    num_triggers  = tsv.per_expert_seq // tsv.swiglu_split_value
    task_num      = num_triggers * tsv.single_rank_expert_num
    swiglu_ev_num = tsv.single_rank_expert_num
    tsv.swiglu_task_num = task_num

    param = op.param_positions  # [8, 9, 10]

    for i in range(task_num):
        task = TaskDescC()
        task.task_type        = TaskType.TASK_SWI_GLU_GRAD
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.num_inputs       = len(op.inputs)
        task.num_outputs      = len(op.outputs)

        for j in range(task.num_inputs):
            in_spec = op.inputs[j]
            td = TensorDescC()
            td.tensor_type     = 1
            td.data_type       = dtype_size
            td.input_position  = param[j]
            td.base_ptr_offset = i * in_spec.shape[1] * tsv.swiglu_split_value
            td.dynamic_shape   = int(in_spec.is_dynamic)
            shape = in_spec.shape
            for k in range(min(len(shape), MAX_TENSOR_DIMS)):
                td.dim[k] = shape[k]
            task.inputs[j] = td

        for j in range(task.num_outputs):
            out_spec = op.outputs[j]
            td = TensorDescC()
            td.tensor_type     = 1
            td.data_type       = dtype_size
            td.input_position  = param[task.num_inputs + j]
            td.base_ptr_offset = i * out_spec.shape[1] * tsv.swiglu_split_value
            td.dynamic_shape   = int(out_spec.is_dynamic)
            shape = out_spec.shape
            for k in range(min(len(shape), MAX_TENSOR_DIMS)):
                td.dim[k] = shape[k]
            task.outputs[j] = td

        ev_idx = i // num_triggers
        task.dependent_event = tsv.pre_pre_event_num + ev_idx + 1
        task.trigger_event   = tsv.pre_event_num + ev_idx + 1
        cfg.all_event_num_triggers[task.trigger_event] = num_triggers

        task.task_index           = i
        task.task_split_num       = task_num
        task.task_split_value     = op.split_value
        task.tiling_data_position = op.tiling_position  # 24

        cfg.all_tasks[tsv.pre_task_num + i]                  = task
        cfg.vector_task_indexs[tsv.pre_vector_task_num + i]  = tsv.pre_task_num + i

    cfg.task_index_num[1] += task_num
    advance_tsv_vector(tsv, task_num, event_group_size=swiglu_ev_num)


# ── GMM2 (backward) ───────────────────────────────────────────────────────────
# C++: task_split_gmm_g2  (bwd, line 434 in bwd task_register.hpp)
# KEY diffs from fwd GMM2:
#   - j!=0 gets transpose_flag=1 (weight transposed)
#   - j==2: type=0, int64, transpose_flag=0
#   - tiling_position = 21

def fill_gmm_g2_bwd(cfg: RuntimeConfigC, op, tsv) -> None:
    task_num  = op.task_num
    param     = op.param_positions  # [10, 11, 19, 12]
    dtype_size = 2

    for i in range(task_num):
        data_index = i // NUM_WORKERS_CUBE
        task = TaskDescC()
        task.task_type        = TaskType.TASK_GROUPED_MATMUL
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
        task.num_inputs       = len(op.inputs)
        task.num_outputs      = len(op.outputs)

        for j in range(task.num_inputs):
            td = TensorDescC()
            td.data_type       = dtype_size
            td.input_position  = param[j]
            td.tensor_type     = 1
            td.dynamic_shape   = 0

            if j == 0:   # x: dynamic, offset
                td.base_ptr_offset = data_index * 4096 * op.inputs[j].shape[1]
                td.dynamic_shape   = int(op.inputs[j].is_dynamic)
            else:        # weight: type=1
                td.base_ptr_offset = 0

            if j == 2:   # group_list: type=0, int64
                td.tensor_type    = 0
                td.data_type      = 8

            td.transpose_flag = int(op.inputs[j].transpose)

            shape = op.inputs[j].shape
            for k in range(min(len(shape), MAX_TENSOR_DIMS)):
                td.dim[k] = shape[k]
            task.inputs[j] = td

        # output: type=1, dynamic
        out_spec = op.outputs[0]
        out = TensorDescC()
        out.data_type       = dtype_size
        out.input_position  = param[task.num_inputs]
        out.tensor_type     = 1
        out.base_ptr_offset = data_index * 4096 * out_spec.shape[1]
        out.dynamic_shape   = int(out_spec.is_dynamic)
        for k in range(min(len(out_spec.shape), MAX_TENSOR_DIMS)):
            out.dim[k] = out_spec.shape[k]
        task.outputs[0] = out

        task.dependent_event = tsv.pre_pre_event_num + data_index + 1
        task.trigger_event   = tsv.pre_event_num + data_index + 1
        cfg.all_event_num_triggers[task.trigger_event] = NUM_WORKERS_CUBE

        task.task_index           = i
        task.task_split_num       = task_num
        task.task_split_value     = op.split_value
        task.tiling_data_position = op.tiling_position  # 21

        cfg.all_tasks[tsv.pre_task_num + i]             = task
        cfg.cube_task_indexs[tsv.pre_cube_task_num + i] = tsv.pre_task_num + i

    cfg.task_index_num[0] += task_num
    advance_tsv_cube(tsv, task_num, event_group_size=task_num // NUM_WORKERS_CUBE)


# ── AllToAll A2 (backward) ────────────────────────────────────────────────────
# C++: task_split_alltoallv_a2  (bwd, line 526 in bwd task_register.hpp)
# KEY diffs from fwd A2:
#   - output tensor_type=1 (not 0)
#   - vector slot = pre_vector_task_num + i  (not fixed alltoall+swiglu offset)
#   - NO event advance after

def fill_alltoall_a2_bwd(cfg: RuntimeConfigC, op, tsv) -> None:
    task_num    = op.task_num
    per_g_e_num = task_num // tsv.all_expert_num
    param       = op.param_positions  # [14, 12, 15, 16, 13]
    dtype_size  = 2

    for i in range(task_num):
        current_dep = (i // per_g_e_num) % tsv.single_rank_expert_num

        task = TaskDescC()
        task.task_type        = TaskType.TASK_SHMEM_PUT_MEM_SINGAL
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
        task.num_inputs       = len(op.inputs)
        task.num_outputs      = len(op.outputs)

        for j in range(task.num_inputs):
            td = TensorDescC()
            td.tensor_type     = 0
            td.data_type       = dtype_size
            td.input_position  = param[j]
            td.base_ptr_offset = 0

            if j == 0:    # target_offset: int64
                td.data_type       = 8
                td.base_ptr_offset = i // per_g_e_num
            elif j == 1:  # src (gmm2 output): type=1, dynamic
                td.dynamic_shape = int(op.inputs[j].is_dynamic)
                td.tensor_type   = 1
            elif j == 2:  # src_offset: int64
                td.data_type       = 8
                td.base_ptr_offset = i // per_g_e_num
            elif j == 3:  # size: int32
                td.data_type       = 4
                td.base_ptr_offset = i // per_g_e_num
            task.inputs[j] = td

        # output: tensor_type=1 (BWD diff: fwd uses tensor_type=0)
        out = TensorDescC()
        out.tensor_type     = 1
        out.data_type       = dtype_size
        out.input_position  = param[task.num_inputs]
        out.base_ptr_offset = 0
        task.outputs[0] = out

        task.dependent_event = tsv.pre_pre_event_num + current_dep + 1
        task.trigger_event   = tsv.all_event_num
        cfg.all_event_num_triggers[task.trigger_event] = task_num

        task.task_index           = i
        task.task_split_num       = task_num
        task.task_split_value     = op.split_value
        task.tiling_data_position = 0xFFFFFFFF

        cfg.all_tasks[tsv.pre_task_num + i] = task
        # BWD: use pre_vector_task_num + i (NOT the fixed fwd slot)
        cfg.vector_task_indexs[tsv.pre_vector_task_num + i] = tsv.pre_task_num + i

    cfg.task_index_num[1] += task_num
    # BWD: NO event advance
    advance_tsv_vector_only(tsv, task_num)


# ── GMM3 (backward) — weight grad for GMM1 ────────────────────────────────────
# C++: task_split_gmm_g3  (bwd, line 593 in bwd task_register.hpp)
# Same structure as GMM4 but different post-advance: pre_event_num += 1

def fill_gmm_g3_bwd(cfg: RuntimeConfigC, op, tsv) -> None:
    task_num  = op.task_num
    param     = op.param_positions  # [17, 10, 19, 18]
    dtype_size = 2

    for i in range(task_num):
        data_index = i // NUM_WORKERS_CUBE
        task = TaskDescC()
        task.task_type        = TaskType.TASK_GROUPED_MATMUL
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE
        task.num_inputs       = len(op.inputs)
        task.num_outputs      = len(op.outputs)

        for j in range(task.num_inputs):
            td = TensorDescC()
            td.data_type       = dtype_size
            td.input_position  = param[j]
            td.tensor_type     = 1
            td.base_ptr_offset = 0
            td.dynamic_shape   = 0

            if j == 0 or j == 1:  # both x1,x2: offset+dynamic
                td.base_ptr_offset = data_index * 4096 * op.inputs[j].shape[1]
                td.dynamic_shape   = int(op.inputs[j].is_dynamic)

            if j == 2:            # group_list: type=0, int64
                td.tensor_type    = 0
                td.data_type      = 8

            td.transpose_flag = int(op.inputs[j].transpose)

            shape = op.inputs[j].shape
            for k in range(min(len(shape), MAX_TENSOR_DIMS)):
                td.dim[k] = shape[k]
            task.inputs[j] = td

        # output: type=1, offset=0, no dynamic
        out_spec = op.outputs[0]
        out = TensorDescC()
        out.data_type       = dtype_size
        out.input_position  = param[task.num_inputs]
        out.tensor_type     = 1
        out.base_ptr_offset = 0
        out.dynamic_shape   = int(out_spec.is_dynamic)
        task.outputs[0] = out

        task.dependent_event = tsv.pre_pre_event_num + data_index + 1
        task.trigger_event   = tsv.all_event_num
        cfg.all_event_num_triggers[task.trigger_event] = NUM_WORKERS_CUBE

        task.task_index           = i
        task.task_split_num       = task_num
        task.task_split_value     = op.split_value
        task.tiling_data_position = op.tiling_position  # 22

        cfg.all_tasks[tsv.pre_task_num + i]             = task
        cfg.cube_task_indexs[tsv.pre_cube_task_num + i] = tsv.pre_task_num + i

    cfg.task_index_num[0] += task_num
    # Post-advance: pre_event_num += 1 (GMM3 diff from GMM4)
    tsv.pre_task_num        += task_num
    tsv.pre_cube_task_num   += task_num
    tsv.pre_pre_event_num    = tsv.pre_event_num
    tsv.pre_event_num       += 1


# ── Terminate task (backward) ─────────────────────────────────────────────────
# C++: add_terminate_task  (bwd, line 676 in bwd task_register.hpp)
# BWD diff: triggers = g4 + g3 + a2/ep*ep  (fwd only has a2/ep*ep)

def add_terminate_bwd(cfg: RuntimeConfigC, tsv) -> None:
    cfg.all_event_num_triggers[tsv.all_event_num] = (
        tsv.gmm_task_num_g4 + tsv.gmm_task_num_g3
        + tsv.alltoall_task_num_a2 // tsv.ep * tsv.ep
    )
    task = TaskDescC()
    task.task_type       = TaskType.TASK_TERMINATE
    task.dependent_event = tsv.pre_pre_event_num + 1
    task.trigger_event   = tsv.pre_event_num + 1
    cfg.all_event_num_triggers[task.trigger_event] = 1

    cfg.all_tasks[tsv.pre_task_num]                  = task
    cfg.vector_task_indexs[tsv.pre_vector_task_num]  = tsv.pre_task_num
    cfg.task_index_num[1] += 1


# ── revise_gmm_task_queue (backward only) ────────────────────────────────────
# C++: revise_gmm_task_queue  (bwd, line 826 in bwd task_register.hpp)
# Interleaves GMM4 and GMM1 experts in cube_task_indexs:
#   dst pattern: [GMM4 exp0, GMM1 exp0, GMM4 exp1, GMM1 exp1, ...]
# index=1 → position gmm_task_num (= GMM4's start in cube queue)
# index=0 → position 0 (GMM1's start)

def revise_gmm_task_queue_bwd(cfg: RuntimeConfigC, tsv) -> None:
    temp = list(cfg.cube_task_indexs)
    expert_single = tsv.single_rank_expert_num
    changes_num   = 2  # 2 streams: GMM4 (index=1) and GMM1 (index=0)

    for i in range(expert_single * changes_num):
        index = 1 - (i % changes_num)  # alternates: 1, 0, 1, 0, ...
        m     = i // changes_num        # expert block index: 0,0,1,1,...
        for j in range(NUM_WORKERS_CUBE):
            dst = i * NUM_WORKERS_CUBE + j
            src = index * tsv.gmm_task_num + m * NUM_WORKERS_CUBE + j
            cfg.cube_task_indexs[dst] = temp[src]


# ── Dynamic data (backward) ───────────────────────────────────────────────────
# C++: add_dynamic_data  (bwd, line 851 in bwd task_register.hpp)
# BWD: dynamic_input_position = 19  (fwd uses 6)

def add_dynamic_data_bwd(cfg: RuntimeConfigC, tsv) -> None:
    from common.runtime_structs import DynamicType
    cfg.dynamic_data.dynamic_type           = DynamicType.DYNAMIC_DSV3_MOE_FFN
    cfg.dynamic_data.dynamic_input_position = 19   # backward fixed
    cfg.dynamic_data.dynamic_group_size     = tsv.single_rank_expert_num
    cfg.dynamic_data.dynamic_max_seq_len    = -1
