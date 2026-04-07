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
"""MoE-FFN forward compute graph."""

from hyper_parallel.core.multicore.modules.moe_ffn.common.compute_graph import (
    ComputeGraph, OperatorNode, TensorSpec, SplitSpec, OpType,
)
from hyper_parallel.core.multicore.modules.moe_ffn.common.task_builders import (
    AllToAllFillConfig, AllToAllType, GmmFillConfig, SwiGLUFillConfig,
)

# Tiling buffer positions in the C++ runtime descriptor array.
_TILING_POS_UP_PROJ = 17
_TILING_POS_SWIGLU  = 18
_TILING_POS_DN_PROJ = 19


def _build_fwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size):
    """Create all TensorSpec objects for the forward graph."""
    sre = tsv.single_rank_expert_num
    target = TensorSpec(
        "target", [tsv.per_rank_seq, hidden_size],
        param_position=0, dtype_size=dtype_size, is_dynamic=True)
    target_offset = TensorSpec(
        "target_offset", [tsv.all_expert_num],
        param_position=1, dtype_size=8)
    src = TensorSpec(
        "src", [tsv.per_rank_seq, hidden_size],
        param_position=2, dtype_size=dtype_size)
    src_offset = TensorSpec(
        "src_offset", [tsv.all_expert_num],
        param_position=3, dtype_size=8)
    size_d = TensorSpec(
        "size_d", [tsv.all_expert_num],
        param_position=4, dtype_size=4)
    up_proj_weight = TensorSpec(
        "up_proj_weight", [sre, hidden_size, intermediate_size * 2],
        param_position=5, dtype_size=dtype_size, tensor_type=1)
    up_proj_glist = TensorSpec(
        "up_proj_glist", [sre],
        param_position=6, dtype_size=8)
    up_proj_y = TensorSpec(
        "up_proj_y", [tsv.per_rank_seq, intermediate_size * 2],
        param_position=7, dtype_size=dtype_size, is_dynamic=True)
    swiglu_out = TensorSpec(
        "swiglu_out", [tsv.per_rank_seq, intermediate_size],
        param_position=8, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    down_proj_weight = TensorSpec(
        "down_proj_weight", [sre, intermediate_size, hidden_size],
        param_position=9, dtype_size=dtype_size, tensor_type=1)
    down_proj_glist = TensorSpec(
        "down_proj_glist", [sre],
        param_position=10, dtype_size=8)
    down_proj_y = TensorSpec(
        "down_proj_y", [tsv.per_rank_seq, hidden_size],
        param_position=11, dtype_size=dtype_size, tensor_type=1, is_dynamic=True)
    combine_out = TensorSpec(
        "combine_out", [tsv.per_rank_seq, hidden_size],
        param_position=12, dtype_size=dtype_size)
    target_offset_c = TensorSpec(
        "target_offset_c", [tsv.all_expert_num],
        param_position=13, dtype_size=8)
    src_offset_c = TensorSpec(
        "src_offset_c", [tsv.all_expert_num],
        param_position=14, dtype_size=8)
    size_c = TensorSpec(
        "size_c", [tsv.all_expert_num],
        param_position=15, dtype_size=4)
    return (target, target_offset, src, src_offset, size_d,
            up_proj_weight, up_proj_glist, up_proj_y, swiglu_out,
            down_proj_weight, down_proj_glist, down_proj_y,
            combine_out, target_offset_c, src_offset_c, size_c)


def _build_fwd_ops(tsv, specs, *, dispatch_sv, up_proj_sv, swiglu_sv,
                   down_proj_sv, combine_sv, num_cube_cores):
    """Create all OperatorNode objects for the forward graph."""
    (target, target_offset, src, src_offset, size_d,
     up_proj_weight, up_proj_glist, up_proj_y, swiglu_out,
     down_proj_weight, down_proj_glist, down_proj_y,
     combine_out, target_offset_c, src_offset_c, size_c) = specs
    dispatch = OperatorNode(
        name="dispatch", op_type=OpType.ALLTOALL,
        inputs=[target_offset, src, src_offset, size_d],
        outputs=[target],
        param_positions=[1, 2, 3, 4, 0],
        split_value=dispatch_sv,
        split_spec=SplitSpec(
            split_inputs=None, split_output_dims=[0],
            task_num_fn=lambda tsv: tsv.all_expert_num * (tsv.per_expert_seq_to_other // dispatch_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(moe_type=AllToAllType.DISPATCH, advance="vector",
                                       event_group=tsv.all_expert_num),
    )
    up_proj = OperatorNode(
        name="up_proj", op_type=OpType.GMM,
        inputs=[target, up_proj_weight, up_proj_glist], outputs=[up_proj_y],
        param_positions=[0, 5, 6, 7], split_value=up_proj_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_UP_PROJ,
        fill_config=GmmFillConfig(offset_inputs={0}, rank_in_event=True, global_trigger=False,
                                  out_offset=True, advance="cube", num_cube_cores=num_cube_cores),
    )
    swiglu = OperatorNode(
        name="swiglu", op_type=OpType.SWIGLU,
        inputs=[up_proj_y], outputs=[swiglu_out],
        param_positions=[7, 8], split_value=swiglu_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: (tsv.per_expert_seq // swiglu_sv) * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_SWIGLU,
        fill_config=SwiGLUFillConfig(),
    )
    down_proj = OperatorNode(
        name="down_proj", op_type=OpType.GMM,
        inputs=[swiglu_out, down_proj_weight, down_proj_glist], outputs=[down_proj_y],
        param_positions=[8, 9, 10, 11], split_value=down_proj_sv,
        split_spec=SplitSpec(
            split_inputs=[(0, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv, _ncc=num_cube_cores: _ncc * tsv.single_rank_expert_num,
        ),
        tiling_position=_TILING_POS_DN_PROJ,
        fill_config=GmmFillConfig(offset_inputs={0}, rank_in_event=False, global_trigger=False,
                                  out_offset=True, advance="cube", num_cube_cores=num_cube_cores),
    )
    combine = OperatorNode(
        name="combine", op_type=OpType.ALLTOALL,
        inputs=[target_offset_c, down_proj_y, src_offset_c, size_c], outputs=[combine_out],
        param_positions=[13, 11, 14, 15, 12], split_value=combine_sv,
        split_spec=SplitSpec(
            split_inputs=[(1, 0)], split_output_dims=[0],
            task_num_fn=lambda tsv: tsv.all_expert_num * (tsv.per_expert_seq_to_other // combine_sv),
        ),
        tiling_position=-1,
        fill_config=AllToAllFillConfig(moe_type=AllToAllType.COMBINE, advance="vector", event_group=1),
    )
    return dispatch, up_proj, swiglu, down_proj, combine


def build_forward_graph(tsv, *,
                        dispatch_sv:  int = 128,
                        up_proj_sv:   int = 4096,
                        swiglu_sv:    int = 128,
                        down_proj_sv: int = 4096,
                        combine_sv:   int = 128,
                        hidden_size:       int = 7168,
                        intermediate_size: int = 2048,
                        dtype_size:        int = 2,
                        num_cube_cores:    int = 24) -> ComputeGraph:
    """Build the MoE-FFN forward DAG: dispatch -> up_proj -> swiglu -> down_proj -> combine.

    Operator execution order and param_positions (C++ memory slots):
      dispatch:  [target_offset=1, src=2, src_offset=3, size=4 | target=0]
      up_proj:   [x=0, weight=5, glist=6 | y=7]
      swiglu:    [x=7 | out=8]
      down_proj: [x=8, weight=9, glist=10 | y=11]
      combine:   [target_offset=13, src=11, src_offset=14, size=15 | target=12]

    Args:
        tsv: TaskSplitValue carrying TP/EP/seq partition metadata.
        dispatch_sv: tile size for dispatch AllToAll (vector cores).
        up_proj_sv: tile size for up-projection GMM (cube cores).
        swiglu_sv: tile size for SwiGLU (vector cores).
        down_proj_sv: tile size for down-projection GMM (cube cores).
        combine_sv: tile size for combine AllToAll (vector cores).
        hidden_size: model hidden dimension.
        intermediate_size: FFN intermediate dimension after SwiGLU halving (half of up-proj output).
        dtype_size: bytes per activation element (2=bf16, 4=fp32).
        num_cube_cores: number of AIC cube cores on the target device.

    Returns:
        A fully-connected ComputeGraph ready for propagate_splits().
    """
    specs = _build_fwd_tensor_specs(tsv, hidden_size, intermediate_size, dtype_size)
    dispatch, up_proj, swiglu, down_proj, combine = _build_fwd_ops(
        tsv, specs,
        dispatch_sv=dispatch_sv, up_proj_sv=up_proj_sv, swiglu_sv=swiglu_sv,
        down_proj_sv=down_proj_sv, combine_sv=combine_sv, num_cube_cores=num_cube_cores,
    )
    graph = ComputeGraph()
    (graph.add_op(dispatch).add_op(up_proj).add_op(swiglu).add_op(down_proj).add_op(combine)
          .add_edge(dispatch,  up_proj)
          .add_edge(up_proj,   swiglu)
          .add_edge(swiglu,    down_proj)
          .add_edge(down_proj, combine))
    return graph
