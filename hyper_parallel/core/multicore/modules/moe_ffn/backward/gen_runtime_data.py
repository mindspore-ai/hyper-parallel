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
Backward pass: generate RuntimeConfig binary files and tiling binary files.

Usage:
    python gen_runtime_data.py [--tp 4] [--ep 4] [--seq_size 8192]
                               [--all_expert_num 32] [--top_k 8]
                               [--output_dir mega_kernel_gmm_grad_tp4_ep4_910b]

Outputs (rank-independent):
    <output_dir>/act_grad_tiling.bin         (act_grad, pos 20)
    <output_dir>/gate_grad_tiling.bin        (gate_grad, pos 21)
    <output_dir>/w2_grad_tiling.bin          (w2_grad, pos 22)
    <output_dir>/w1_grad_tiling.bin          (w1_grad, pos 23)
    <output_dir>/swiglu_grad_tiling.bin      (SwiGLU-grad, pos 24)
    <output_dir>/all_event_counters.bin      1024×int32 zeros (4 KB)
    <output_dir>/gmm_workspace.bin           256 MiB zeros

Outputs (per rank):
    <output_dir>/runtime_config_input_rank_<i>.bin
"""
import argparse
import os
import numpy as np

from hyper_parallel.core.multicore.modules.moe_ffn.common.runtime_structs import (
    RuntimeConfigC, QUEUE_CAPACITY,
)
from hyper_parallel.core.multicore.modules.moe_ffn.common.compute_graph import TaskSplitValue, init_task_split_value
from hyper_parallel.core.multicore.modules.moe_ffn.common.task_builder_utils import revise_task_queue
from hyper_parallel.core.multicore.modules.moe_ffn.common.task_builders import (
    add_terminate, add_dynamic_data, revise_gmm_task_queue_bwd,
)
from .tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_swiglu_grad_tiling_bytes,
)
from .backward_graph import build_backward_graph


def parse_args():
    """Parse command-line arguments for backward data generation."""
    p = argparse.ArgumentParser()
    p.add_argument('--tp',                type=int, default=4)
    p.add_argument('--ep',                type=int, default=4)
    p.add_argument('--seq_size',          type=int, default=8192)
    p.add_argument('--all_expert_num',    type=int, default=32)
    p.add_argument('--top_k',             type=int, default=8)
    p.add_argument('--hidden_size',       type=int, default=7168,
                   help='Model hidden dimension (e.g. 7168 for Qwen3-235B)')
    p.add_argument('--intermediate_size', type=int, default=2048,
                   help='FFN intermediate dimension after SwiGLU halving (half of up-proj output, e.g. 2048)')
    p.add_argument('--dtype_size',        type=int, default=2,
                   help='Activation bytes per element: bf16=2, fp32=4')
    p.add_argument('--num_cube_cores',    type=int, default=24,
                   help='Number of AI Cube cores on target hardware (910B=24)')
    p.add_argument('--output_dir',        type=str,
                   default='mega_kernel_gmm_grad_tp4_ep4_910b')
    return p.parse_args()


def build_config_for_rank(graph, tsv: TaskSplitValue, rank_id: int,
                          num_cube_cores: int = 24) -> RuntimeConfigC:
    """Build backward RuntimeConfig for a single rank."""
    cfg = RuntimeConfigC()
    cfg.num_workers    = 2 * num_cube_cores   # NUM_WORKERS_VECTOR = 2 × NUM_WORKERS_CUBE
    cfg.queue_capacity = QUEUE_CAPACITY

    # Reset counters
    init_task_split_value(tsv)
    tsv.rank_id = rank_id

    # Fill tasks in topological order (dispatch, act_grad, w1_grad, swiglu_grad, gate_grad, combine, w2_grad)
    for op in graph.topological_sort():
        op.fill_config.fill(cfg, op, tsv)

    # task_num_all = sum of all op tasks + 1 (terminate)
    task_num_all = sum(op.task_num for op in graph.topological_sort()) + 1

    dispatch_op    = graph.get_op("dispatch")
    swiglu_grad_op = graph.get_op("swiglu_grad")
    act_grad_op    = graph.get_op("act_grad")
    w2_grad_op     = graph.get_op("w2_grad")
    w1_grad_op     = graph.get_op("w1_grad")
    combine_op     = graph.get_op("combine")

    add_terminate(cfg, tsv,
                  w1_grad_op.task_num + w2_grad_op.task_num
                  + combine_op.task_num // tsv.ep * tsv.ep)
    revise_task_queue(cfg, tsv, dispatch_op.task_num, swiglu_grad_op.task_num)
    revise_gmm_task_queue_bwd(cfg, tsv, act_grad_op.task_num, num_cube_cores=num_cube_cores)
    add_dynamic_data(cfg, tsv, dynamic_input_position=19)

    cfg.task_num = task_num_all
    cfg.atomic_add_values[0] = 1
    return cfg


def write_bin(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    print(f"  wrote {len(data):>10,} bytes → {path}")


def main():
    args = parse_args()
    out  = args.output_dir

    tsv   = TaskSplitValue(
        tp=args.tp, ep=args.ep,
        seq_size=args.seq_size,
        all_expert_num=args.all_expert_num,
        top_k=args.top_k,
    )
    num_groups = tsv.single_rank_expert_num
    graph = build_backward_graph(tsv,
                                 dispatch_sv=128,  act_grad_sv=4096,  w1_grad_sv=4096,
                                 swiglu_sv=128,    gate_grad_sv=4096, w2_grad_sv=4096,
                                 combine_sv=128,
                                 hidden_size=args.hidden_size,
                                 intermediate_size=args.intermediate_size,
                                 dtype_size=args.dtype_size,
                                 num_cube_cores=args.num_cube_cores)
    # Compute task_num for each operator via split-axis propagation
    graph.propagate_splits(tsv)

    dispatch_op    = graph.get_op("dispatch")
    act_grad_op    = graph.get_op("act_grad")
    w1_grad_op     = graph.get_op("w1_grad")
    swiglu_grad_op = graph.get_op("swiglu_grad")
    gate_grad_op   = graph.get_op("gate_grad")
    w2_grad_op     = graph.get_op("w2_grad")
    combine_op     = graph.get_op("combine")

    print(f"[bwd] tp={args.tp} ep={args.ep} seq={args.seq_size} "
          f"E={args.all_expert_num} topk={args.top_k}")
    print(f"      dispatch={dispatch_op.task_num}  act_grad={act_grad_op.task_num}  "
          f"w1_grad={w1_grad_op.task_num}  swiglu_grad={swiglu_grad_op.task_num}  "
          f"gate_grad={gate_grad_op.task_num}  w2_grad={w2_grad_op.task_num}  "
          f"combine={combine_op.task_num}")

    # ── Tiling files (rank-independent) ──────────────────────────────────────
    act_grad_bytes    = get_act_grad_tiling_bytes(act_grad_op.split_value,
                                                  hidden_size=args.hidden_size,
                                                  intermediate_size=args.intermediate_size,
                                                  num_groups=num_groups,
                                                  num_cube_cores=args.num_cube_cores)
    gate_grad_bytes   = get_gate_grad_tiling_bytes(gate_grad_op.split_value,
                                                   hidden_size=args.hidden_size,
                                                   intermediate_size=args.intermediate_size,
                                                   num_groups=num_groups,
                                                   num_cube_cores=args.num_cube_cores)
    w2_grad_bytes     = get_w2_grad_tiling_bytes(w2_grad_op.split_value,
                                                 hidden_size=args.hidden_size,
                                                 intermediate_size=args.intermediate_size,
                                                 num_groups=num_groups,
                                                 num_cube_cores=args.num_cube_cores)
    w1_grad_bytes     = get_w1_grad_tiling_bytes(w1_grad_op.split_value,
                                                 hidden_size=args.hidden_size,
                                                 intermediate_size=args.intermediate_size,
                                                 num_groups=num_groups,
                                                 num_cube_cores=args.num_cube_cores)
    swiglu_grad_bytes = get_swiglu_grad_tiling_bytes(swiglu_grad_op.split_value,
                                                     intermediate_size=args.intermediate_size)

    write_bin(os.path.join(out, 'act_grad_tiling.bin'),   act_grad_bytes)
    write_bin(os.path.join(out, 'gate_grad_tiling.bin'),  gate_grad_bytes)
    write_bin(os.path.join(out, 'w2_grad_tiling.bin'),    w2_grad_bytes)
    write_bin(os.path.join(out, 'w1_grad_tiling.bin'),    w1_grad_bytes)
    write_bin(os.path.join(out, 'swiglu_grad_tiling.bin'), swiglu_grad_bytes)

    # ── Event counters + workspace (rank-independent) ─────────────────────────
    # all_event_counters: 1024×int32_t zeros — matches C++ reference gen_data
    write_bin(os.path.join(out, 'all_event_counters.bin'),
              np.zeros(1024, dtype=np.int32).tobytes())
    # gmm_workspace: 256 MiB zeros — kernel-internal scratch buffer
    write_bin(os.path.join(out, 'gmm_workspace.bin'),
              bytes(256 * 1024 * 1024))

    # ── RuntimeConfig files (one per rank) ───────────────────────────────────
    for rank_id in range(args.ep):
        cfg  = build_config_for_rank(graph, tsv, rank_id, num_cube_cores=args.num_cube_cores)
        data = bytes(cfg)
        path = os.path.join(out, f'runtime_config_input_rank_{rank_id}.bin')
        write_bin(path, data)

    print("[bwd] done.")


if __name__ == '__main__':
    main()
