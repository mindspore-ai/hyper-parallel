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
"""MoE-FFN backward compute graph."""

from hyper_parallel.core.multicore.scheduler.graph import (
    ComputeGraph, OperatorNode, TensorSpec, SplitSpec, OpType,
)
from hyper_parallel.core.multicore.tasks.alltoall import AllToAllFillConfig, AllToAllType
from hyper_parallel.core.multicore.tasks.gmm import GmmFillConfig
from hyper_parallel.core.multicore.tasks.swiglu import SwiGLUFillConfig

# Tiling buffer positions in the C++ runtime descriptor array.
_TILING_POS_ACT_GRAD    = 20
_TILING_POS_GATE_GRAD   = 21
_TILING_POS_W1_GRAD     = 22
_TILING_POS_W2_GRAD     = 23
_TILING_POS_SWIGLU_GRAD = 24


def _build_bwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size):
    """Create all TensorSpec objects for the backward graph."""
    sre = tsv.single_rank_expert_num
    target = TensorSpec(
        "target", [tsv.per_rank_seq, hidden_size],
        param_position=0, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    target_offset = TensorSpec(
        "target_offset", [tsv.all_expert_num],
        param_position=1, dtype_size=8)
    src = TensorSpec(
        "src", [tsv.per_rank_seq, hidden_size],
        param_position=2, dtype_size=dtype_size, tensor_type=1)
    src_offset = TensorSpec(
        "src_offset", [tsv.all_expert_num],
        param_position=3, dtype_size=8)
    size_d = TensorSpec(
        "size_d", [tsv.all_expert_num],
        param_position=4, dtype_size=4)
    w2_grad_x1 = TensorSpec(
        "w2_grad_x1", [tsv.per_rank_seq, intermediate_size],
        param_position=5, dtype_size=dtype_size,
        tensor_type=1, is_dynamic=True, transpose=True)
    w2_grad_y = TensorSpec(
        "w2_grad_y", [sre, intermediate_size, hidden_size],
        param_position=6, dtype_size=dtype_size, tensor_type=1)
    act_grad_weight = TensorSpec(
        "act_grad_weight", [sre, intermediate_size, hidden_size],
        param_position=7, dtype_size=dtype_size, tensor_type=1, transpose=True)
    act_grad_y = TensorSpec(
        "act_grad_y", [tsv.per_rank_seq, intermediate_size],
        param_position=8, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    swiglu_dy = TensorSpec(
        "swiglu_dy", [tsv.per_rank_seq, intermediate_size * 2],
        param_position=9, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    swiglu_out = TensorSpec(
        "swiglu_out", [tsv.per_rank_seq, intermediate_size * 2],
        param_position=10, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    gate_grad_weight = TensorSpec(
        "gate_grad_weight", [sre, hidden_size, intermediate_size * 2],
        param_position=11, dtype_size=dtype_size, tensor_type=1, transpose=True)
    gate_grad_y = TensorSpec(
        "gate_grad_y", [tsv.per_rank_seq, hidden_size],
        param_position=12, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    combine_out = TensorSpec(
        "combine_out", [tsv.per_rank_seq, hidden_size],
        param_position=13, dtype_size=dtype_size, tensor_type=1)
    target_offset_c = TensorSpec(
        "target_offset_c", [tsv.all_expert_num],
        param_position=14, dtype_size=8)
    src_offset_c = TensorSpec(
        "src_offset_c", [tsv.all_expert_num],
        param_position=15, dtype_size=8)
    size_c = TensorSpec(
        "size_c", [tsv.all_expert_num],
        param_position=16, dtype_size=4)
    w1_grad_x1 = TensorSpec(
        "w1_grad_x1", [tsv.per_rank_seq, hidden_size],
        param_position=17, dtype_size=dtype_size,
        tensor_type=1, is_dynamic=True, transpose=True)
    w1_grad_y = TensorSpec(
        "w1_grad_y", [sre, hidden_size, intermediate_size * 2],
        param_position=18, dtype_size=dtype_size, tensor_type=1)
    glist = TensorSpec(
        "glist", [sre],
        param_position=19, dtype_size=8)
    return (target, target_offset, src, src_offset, size_d,
            w2_grad_x1, w2_grad_y, act_grad_weight, act_grad_y,
            swiglu_dy, swiglu_out, gate_grad_weight, gate_grad_y,
            combine_out, target_offset_c, src_offset_c, size_c,
            w1_grad_x1, w1_grad_y, glist)


def _build_bwd_ops_first(tsv, specs, *, dispatch_sv, act_grad_sv, w2_grad_sv,
                         swiglu_sv, num_cube_cores):
    """Create dispatch, act_grad, w2_grad, swiglu_grad operator nodes."""
    (target, target_offset, src, src_offset, size_d,
     w2_grad_x1, w2_grad_y, act_grad_weight, act_grad_y,
     swiglu_dy, swiglu_out, *_) = specs
    glist = specs[-1]
    dispatch = OperatorNode(
        name="dispatch", op_type=OpType.ALLTOALL,
        inputs=[target_offset, src, src_offset, size_d], outputs=[target],
        param_positions=[1, 2, 3, 4, 0], split_value=dispatch_sv,
        split_spec=SplitSpec(
            split_inputs=None, split_output_dims=[0],
            task_num_fn=lambda tsv: tsv.all_expert_num * (tsv.per_expert_seq_to_other // dispatch_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(moe_type=AllToAllType.DISPATCH, advance="vector",
                                       event_group=tsv.all_expert_num),
    )
    act_grad = OperatorNode(
        name="act_grad", op_type=OpType.GMM,
        inputs=[target, act_grad_weight, glist], outputs=[act_grad_y],
        param_positions=[0, 7, 19, 8], split_value=act_grad_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_ACT_GRAD,
        fill_config=GmmFillConfig(offset_inputs={0}, rank_in_event=True, global_trigger=False,
                                  out_offset=True, advance="cube_only", num_cube_cores=num_cube_cores),
    )
    w2_grad = OperatorNode(
        name="w2_grad", op_type=OpType.GMM,
        inputs=[w2_grad_x1, target, glist], outputs=[w2_grad_y],
        param_positions=[5, 0, 19, 6], split_value=w2_grad_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_W2_GRAD,
        fill_config=GmmFillConfig(offset_inputs={0, 1}, rank_in_event=True, global_trigger=True,
                                  out_offset=False, advance="cube_custom",
                                  event_delta=tsv.single_rank_expert_num, num_cube_cores=num_cube_cores),
    )
    swiglu_grad = OperatorNode(
        name="swiglu_grad", op_type=OpType.SWIGLU_GRAD,
        inputs=[act_grad_y, swiglu_dy], outputs=[swiglu_out],
        param_positions=[8, 9, 10], split_value=swiglu_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: (tsv.per_expert_seq // swiglu_sv) * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_SWIGLU_GRAD,
        fill_config=SwiGLUFillConfig(),
    )
    return dispatch, act_grad, w2_grad, swiglu_grad


def _build_bwd_ops_second(specs, *, gate_grad_sv, w1_grad_sv, combine_sv, num_cube_cores):
    """Create gate_grad, combine, w1_grad operator nodes."""
    (*_, swiglu_out, gate_grad_weight, gate_grad_y,
     combine_out, target_offset_c, src_offset_c, size_c,
     w1_grad_x1, w1_grad_y, glist) = specs
    gate_grad = OperatorNode(
        name="gate_grad", op_type=OpType.GMM,
        inputs=[swiglu_out, gate_grad_weight, glist], outputs=[gate_grad_y],
        param_positions=[10, 11, 19, 12], split_value=gate_grad_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_GATE_GRAD,
        fill_config=GmmFillConfig(offset_inputs={0}, rank_in_event=False, global_trigger=False,
                                  out_offset=True, advance="cube", num_cube_cores=num_cube_cores),
    )
    combine = OperatorNode(
        name="combine", op_type=OpType.ALLTOALL,
        inputs=[target_offset_c, gate_grad_y, src_offset_c, size_c], outputs=[combine_out],
        param_positions=[14, 12, 15, 16, 13], split_value=combine_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: tsv.all_expert_num * (tsv.per_expert_seq_to_other // combine_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(moe_type=AllToAllType.COMBINE, advance="vector_only", event_group=1),
    )
    w1_grad = OperatorNode(
        name="w1_grad", op_type=OpType.GMM,
        inputs=[w1_grad_x1, swiglu_out, glist], outputs=[w1_grad_y],
        param_positions=[17, 10, 19, 18], split_value=w1_grad_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_W1_GRAD,
        fill_config=GmmFillConfig(offset_inputs={0, 1}, rank_in_event=False, global_trigger=True,
                                  out_offset=False, advance="cube_custom",
                                  event_delta=1, num_cube_cores=num_cube_cores),
    )
    return gate_grad, combine, w1_grad


def build_backward_graph(tsv, *,
                         dispatch_sv:   int = 128,
                         act_grad_sv:   int = 4096,
                         w2_grad_sv:    int = 4096,
                         swiglu_sv:     int = 128,
                         gate_grad_sv:  int = 4096,
                         w1_grad_sv:    int = 4096,
                         combine_sv:    int = 128,
                         hidden_size:       int = 7168,
                         intermediate_size: int = 2048,
                         dtype_size:        int = 2,
                         num_cube_cores:    int = 24) -> ComputeGraph:
    """Build the MoE-FFN backward DAG.

    Execution order:
      dispatch -> {act_grad, w2_grad} -> swiglu_grad -> gate_grad -> {combine, w1_grad}
    act_grad and w2_grad run in parallel (both depend on dispatch).
    combine and w1_grad run in parallel (both depend on gate_grad).

    Param positions (C++ memory slots):
      dispatch:     [target_offset=1, src=2, src_offset=3, size=4 | target=0]
      act_grad:     [x=0, weight=7, glist=19 | y=8]
      w2_grad:      [x1=5, x2=0, glist=19 | y=6]
      swiglu_grad:  [x=8, dy=9 | out=10]
      gate_grad:    [x=10, weight=11, glist=19 | y=12]
      combine:      [target_offset=14, src=12, src_offset=15, size=16 | target=13]
      w1_grad:      [x1=17, x2=10, glist=19 | y=18]

    Args:
        tsv: TaskSplitValue carrying TP/EP/seq partition metadata.
        dispatch_sv / act_grad_sv / w2_grad_sv / swiglu_sv / gate_grad_sv /
        w1_grad_sv / combine_sv: tile sizes per operator.
        hidden_size: model hidden dimension.
        intermediate_size: FFN intermediate dimension after SwiGLU halving.
        dtype_size: bytes per activation element (2=bf16, 4=fp32).
        num_cube_cores: number of AIC cube cores on the target device.

    Returns:
        A fully-connected ComputeGraph ready for propagate_splits().
    """
    specs = _build_bwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size)
    dispatch, act_grad, w2_grad, swiglu_grad = _build_bwd_ops_first(
        tsv, specs,
        dispatch_sv=dispatch_sv, act_grad_sv=act_grad_sv,
        w2_grad_sv=w2_grad_sv, swiglu_sv=swiglu_sv, num_cube_cores=num_cube_cores,
    )
    gate_grad, combine, w1_grad = _build_bwd_ops_second(
        specs,
        gate_grad_sv=gate_grad_sv, w1_grad_sv=w1_grad_sv,
        combine_sv=combine_sv, num_cube_cores=num_cube_cores,
    )
    graph = ComputeGraph()
    (graph.add_op(dispatch).add_op(act_grad).add_op(w2_grad)
          .add_op(swiglu_grad).add_op(gate_grad).add_op(combine).add_op(w1_grad)
          .add_edge(dispatch,    act_grad)
          .add_edge(dispatch,    w2_grad)
          .add_edge(act_grad,    swiglu_grad)
          .add_edge(swiglu_grad, gate_grad)
          .add_edge(gate_grad,   combine)
          .add_edge(gate_grad,   w1_grad))
    return graph
