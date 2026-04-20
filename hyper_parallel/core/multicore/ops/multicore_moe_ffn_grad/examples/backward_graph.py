"""
Backward compute graph declaration.
Call build_backward_graph(tsv) to get a ready-to-iterate ComputeGraph.

Operator execution order:
  dispatch → gmm1 (+ gmm4 in parallel) → swiglu_grad → gmm2 → {combine, gmm3}
  (combine and gmm3 both depend on gmm2 and run in parallel)

param_positions (confirmed from C++ main, line 719–750):
  dispatch:    [target_offset=1, src=2, src_offset=3, size=4  | target=0]
  GMM1:        [x=0, weight=7, glist=19 | y=8]
  GMM4:        [x1=5, x2=0, glist=19 | y=6]
  SwiGLU-grad: [x=8, dy=9 | out=10]
  GMM2:        [x=10, weight=11, glist=19 | y=12]
  combine:     [target_offset=14, src=12, src_offset=15, size=16 | target=13]
  GMM3:        [x1=17, x2=10, glist=19 | y=18]

Tiling positions:
  GMM1=20, GMM4=23, SwiGLU-grad=24, GMM2=21, GMM3=22

Backward tensor_type overview (all activations are tensor lists, type=1):
  dispatch output (target) / GMM1 x / GMM4 x2  → type=1
  dispatch src input                             → type=1
  gmm4_x1, gmm4_y, gmm1_weight, gmm1_y         → type=1
  swiglu_dy, swiglu_out                         → type=1
  gmm2_weight, gmm2_y                           → type=1
  combine output, gmm3_x1, gmm3_y              → type=1
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from common.compute_graph import ComputeGraph, OperatorNode, TensorSpec, SplitSpec, OpType
from common.runtime_structs import NUM_WORKERS_CUBE
from common.task_builders import (
    AllToAllFillConfig, AllToAllType, GmmFillConfig, SwiGLUFillConfig,
)


def build_backward_graph(tsv, *,
                         dispatch_sv: int = 128,
                         gmm1_sv:     int = 4096,
                         gmm4_sv:     int = 4096,
                         swiglu_sv:   int = 128,
                         gmm2_sv:     int = 4096,
                         gmm3_sv:     int = 4096,
                         combine_sv:  int = 128,
                         hidden_size:       int = 7168,
                         intermediate_size: int = 4096,
                         dtype_size:        int = 2,
                         num_cube_cores:    int = 24) -> ComputeGraph:
    """
    Construct the backward DAG.
    GMM1 and GMM4 run in parallel (both depend only on dispatch).
    combine and GMM3 run in parallel (both depend only on GMM2).
    """
    sre      = tsv.single_rank_expert_num
    half_int = intermediate_size // 2

    # ── Tensor specs ──────────────────────────────────────────────────────────
    # dispatch 张量（target 同时作为 GMM1 x 和 GMM4 x2）
    target           = TensorSpec("target",           [tsv.per_rank_seq, hidden_size],       param_position=0,  dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    target_offset    = TensorSpec("target_offset",    [tsv.all_expert_num],                  param_position=1,  dtype_size=8)
    src              = TensorSpec("src",              [tsv.per_rank_seq, hidden_size],       param_position=2,  dtype_size=dtype_size, tensor_type=1)
    src_offset       = TensorSpec("src_offset",       [tsv.all_expert_num],                  param_position=3,  dtype_size=8)
    size_d           = TensorSpec("size_d",           [tsv.all_expert_num],                  param_position=4,  dtype_size=4)

    # GMM4 张量（x1 转置输入，x2 = target）
    gmm4_x1          = TensorSpec("gmm4_x1",          [tsv.per_rank_seq, half_int],          param_position=5,  dtype_size=dtype_size, tensor_type=1, is_dynamic=True, transpose=True)
    gmm4_y           = TensorSpec("gmm4_y",            [sre, half_int, hidden_size],          param_position=6,  dtype_size=dtype_size, tensor_type=1)

    # GMM1 张量（x = target）
    gmm1_weight      = TensorSpec("gmm1_weight",       [sre, half_int, hidden_size],          param_position=7,  dtype_size=dtype_size, tensor_type=1, transpose=True)
    gmm1_y           = TensorSpec("gmm1_y",            [tsv.per_rank_seq, half_int],          param_position=8,  dtype_size=dtype_size, tensor_type=1, is_dynamic=True)

    # SwiGLU-grad 张量（fill_swiglu 对所有 tensor 硬编码 type=1）
    swiglu_dy        = TensorSpec("swiglu_dy",         [tsv.per_rank_seq, intermediate_size], param_position=9,  dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    # swiglu_out 共享：swiglu_grad 输出 / GMM2 输入[0] / GMM3 输入[1]
    swiglu_out       = TensorSpec("swiglu_out",        [tsv.per_rank_seq, intermediate_size], param_position=10, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)

    # GMM2 张量
    gmm2_weight      = TensorSpec("gmm2_weight",       [sre, hidden_size, intermediate_size], param_position=11, dtype_size=dtype_size, tensor_type=1, transpose=True)
    gmm2_y           = TensorSpec("gmm2_y",            [tsv.per_rank_seq, hidden_size],       param_position=12, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)

    # combine 张量
    combine_out      = TensorSpec("combine_out",       [tsv.per_rank_seq, hidden_size],       param_position=13, dtype_size=dtype_size, tensor_type=1)
    target_offset_c  = TensorSpec("target_offset_c",   [tsv.all_expert_num],                  param_position=14, dtype_size=8)
    src_offset_c     = TensorSpec("src_offset_c",      [tsv.all_expert_num],                  param_position=15, dtype_size=8)
    size_c           = TensorSpec("size_c",            [tsv.all_expert_num],                  param_position=16, dtype_size=4)

    # GMM3 张量（x1 转置输入，x2 = swiglu_out）
    gmm3_x1          = TensorSpec("gmm3_x1",           [tsv.per_rank_seq, hidden_size],       param_position=17, dtype_size=dtype_size, tensor_type=1, is_dynamic=True, transpose=True)
    gmm3_y           = TensorSpec("gmm3_y",            [sre, hidden_size, intermediate_size], param_position=18, dtype_size=dtype_size, tensor_type=1)

    # 共享 group list（始终 type=0，dtype int64）
    glist            = TensorSpec("glist",             [sre],                                 param_position=19, dtype_size=8)

    # ── Operator nodes ────────────────────────────────────────────────────────
    dispatch = OperatorNode(
        name="dispatch", op_type=OpType.ALLTOALL,
        inputs=[target_offset, src, src_offset, size_d],
        outputs=[target],
        param_positions=[1, 2, 3, 4, 0],
        split_value=dispatch_sv,
        split_spec=SplitSpec(
            split_inputs=None,
            split_output_dims=[0],
            task_num_fn=lambda tsv: (
                tsv.all_expert_num * (tsv.per_expert_seq_to_other // dispatch_sv)
            ),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(
            moe_type=AllToAllType.DISPATCH,
            advance="vector",
            event_group=tsv.all_expert_num,
        ),
    )

    # GMM1.inputs[0] = target（dispatch 输出），共享同一 TensorSpec 对象
    gmm1 = OperatorNode(
        name="gmm1", op_type=OpType.GMM,
        inputs=[target, gmm1_weight, glist],
        outputs=[gmm1_y],
        param_positions=[0, 7, 19, 8],
        split_value=gmm1_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)],   # inputs[0]=target（dispatch 输出，图内连通）
            split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=20,
        fill_config=GmmFillConfig(
            offset_inputs={0},
            rank_in_event=True,
            global_trigger=False,
            out_offset=True,
            advance="cube_only",   # event 推进延迟到 GMM4
            num_cube_cores=num_cube_cores,
        ),
    )

    # GMM4.inputs[1] = target（dispatch 输出，作为 x2）；inputs[0] = gmm4_x1（转置）
    gmm4 = OperatorNode(
        name="gmm4", op_type=OpType.GMM,
        inputs=[gmm4_x1, target, glist],
        outputs=[gmm4_y],
        param_positions=[5, 0, 19, 6],
        split_value=gmm4_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)],   # inputs[1]=target（dispatch 输出，图内连通）
            split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=23,
        fill_config=GmmFillConfig(
            offset_inputs={0, 1},
            rank_in_event=True,
            global_trigger=True,
            out_offset=False,
            advance="cube_custom",
            event_delta=sre,   # = gmm1_task_num // num_cube_cores
            num_cube_cores=num_cube_cores,
        ),
    )

    swiglu_grad = OperatorNode(
        name="swiglu_grad", op_type=OpType.SWIGLU_GRAD,
        inputs=[gmm1_y, swiglu_dy],
        outputs=[swiglu_out],
        param_positions=[8, 9, 10],
        split_value=swiglu_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)],   # inputs[0]=gmm1_y（GMM1 输出，图内连通）
            split_output_dims=[0],
            task_num_fn=lambda tsv: (
                (tsv.per_expert_seq // swiglu_sv) * tsv.single_rank_expert_num
            ),
        ),
        tiling_position=24,
        fill_config=SwiGLUFillConfig(),
    )

    gmm2 = OperatorNode(
        name="gmm2", op_type=OpType.GMM,
        inputs=[swiglu_out, gmm2_weight, glist],
        outputs=[gmm2_y],
        param_positions=[10, 11, 19, 12],
        split_value=gmm2_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)],   # inputs[0]=swiglu_out（swiglu_grad 输出，图内连通）
            split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=21,
        fill_config=GmmFillConfig(
            offset_inputs={0},
            rank_in_event=False,
            global_trigger=False,
            out_offset=True,
            advance="cube",
            num_cube_cores=num_cube_cores,
        ),
    )

    combine = OperatorNode(
        name="combine", op_type=OpType.ALLTOALL,
        inputs=[target_offset_c, gmm2_y, src_offset_c, size_c],
        outputs=[combine_out],
        param_positions=[14, 12, 15, 16, 13],
        split_value=combine_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)],   # inputs[1]=gmm2_y（GMM2 输出，图内连通）
            split_output_dims=[0],
            task_num_fn=lambda tsv: (
                tsv.all_expert_num * (tsv.per_expert_seq_to_other // combine_sv)
            ),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(
            moe_type=AllToAllType.COMBINE,
            advance="vector_only",
            event_group=1,
        ),
    )

    # GMM3.inputs[1] = swiglu_out（swiglu_grad 输出），共享 TensorSpec，保证切分传播
    gmm3 = OperatorNode(
        name="gmm3", op_type=OpType.GMM,
        inputs=[gmm3_x1, swiglu_out, glist],
        outputs=[gmm3_y],
        param_positions=[17, 10, 19, 18],
        split_value=gmm3_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)],   # inputs[1]=swiglu_out（swiglu_grad 输出，图内连通）
            split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=22,
        fill_config=GmmFillConfig(
            offset_inputs={0, 1},
            rank_in_event=False,
            global_trigger=True,
            out_offset=False,
            advance="cube_custom",
            event_delta=1,
            num_cube_cores=num_cube_cores,
        ),
    )

    # ── Build graph ───────────────────────────────────────────────────────────
    g = ComputeGraph()
    (g.add_op(dispatch)
      .add_op(gmm1)
      .add_op(gmm4)
      .add_op(swiglu_grad)
      .add_op(gmm2)
      .add_op(combine)
      .add_op(gmm3)
      .add_edge(dispatch,    gmm1)
      .add_edge(dispatch,    gmm4)
      .add_edge(gmm1,        swiglu_grad)
      .add_edge(swiglu_grad, gmm2)
      .add_edge(gmm2,        combine)
      .add_edge(gmm2,        gmm3))
    return g
