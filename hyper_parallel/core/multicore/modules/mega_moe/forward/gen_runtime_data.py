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
Forward pass: generate RuntimeConfig binary files and tiling binary files.

Usage:
    python gen_runtime_data.py [--tp 4] [--ep 4] [--seq_size 8192]
                               [--all_expert_num 32] [--top_k 8]
                               [--output_dir mega_moe_tp4_ep4_910b]

Outputs (rank-independent):
    <output_dir>/up_proj_tiling.bin
    <output_dir>/swiglu_tiling.bin
    <output_dir>/down_proj_tiling.bin
    <output_dir>/all_event_counters.bin   graph-sized int32 zeros (at least 4 KB)
    <output_dir>/gmm_workspace.bin        256 MiB zeros

Outputs (per rank):
    <output_dir>/runtime_config_input_rank_<i>.bin
"""
import argparse
import os

from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.tiling_tables import (
    get_down_proj_tiling_bytes,
    get_swiglu_tiling_bytes,
    get_up_proj_tiling_bytes,
)
from hyper_parallel.core.multicore.modules.mega_moe.profiling import (
    _configure_mega_moe_profile_metadata,
)
from hyper_parallel.core.multicore.scheduler.builder import allocate_graph_config
from hyper_parallel.core.multicore.scheduler.config import (
    RuntimeConfigC,
    TaskSplitValue,
    configure_ready_handshake,
    event_workspace_bytes,
    init_task_split_value,
    validate_runtime_config,
)
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config
from hyper_parallel.core.multicore.scheduler.scheduler import revise_task_queue
from hyper_parallel.core.multicore.tasks.utils import add_dynamic_data, add_terminate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for forward data generation."""
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
                   default='mega_moe_tp4_ep4_910b')
    return p.parse_args()


def build_config_for_rank(graph: ComputeGraph, tsv: TaskSplitValue, rank_id: int,
                          num_cube_cores: int = 24) -> RuntimeConfigC:
    """Build RuntimeConfig for a single rank.

    Args:
        graph: Graph with propagated task splits.
        tsv: Rank topology and running event offsets.
        rank_id: Expert-parallel rank whose queues are generated.
        num_cube_cores: Number of participating Cube workers.

    Returns:
        Validated runtime image including transport handshakes and profiler metadata.
    """
    cfg = allocate_graph_config(graph, tsv)
    cfg.num_workers    = 2 * num_cube_cores   # NUM_WORKERS_VECTOR = 2 × NUM_WORKERS_CUBE

    init_task_split_value(tsv)
    tsv.rank_id = rank_id   # C++ forward never sets rank_id; all ranks use default 0

    # Fill tasks in topological order
    for op in graph.topological_sort():
        op.fill_config.fill(cfg, op, tsv)

    # task_num_all = sum of all op tasks (no +1 for terminate)
    task_num_all = sum(op.task_num for op in graph.topological_sort())

    dispatch_op = graph.get_op("dispatch")
    swiglu_op   = graph.get_op("swiglu")
    combine_op  = graph.get_op("combine")
    add_terminate(cfg, tsv, combine_op.task_num // tsv.ep * tsv.ep)
    revise_task_queue(cfg, tsv, dispatch_op.task_num, swiglu_op.task_num, combine_op.task_num)
    add_dynamic_data(cfg, tsv, dynamic_input_position=6)

    cfg.task_num = task_num_all
    cfg.atomic_add_values[0] = 1
    configure_ready_handshake(cfg, tsv)
    validate_runtime_config(cfg, tsv, num_cube_cores)
    _configure_mega_moe_profile_metadata(
        cfg,
        graph,
        tsv,
        num_cube_cores=num_cube_cores,
        is_backward=False,
    )
    return cfg


def write_bin(path: str, data: bytes) -> None:
    """Write binary data to a file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    print(f"  wrote {len(data):>10,} bytes → {path}")


def main() -> None:
    """Entry point for forward pass runtime data generation."""
    args = parse_args()
    out  = args.output_dir

    tsv   = TaskSplitValue(
        tp=args.tp, ep=args.ep,
        seq_size=args.seq_size,
        all_expert_num=args.all_expert_num,
        top_k=args.top_k,
    )
    num_groups = tsv.single_rank_expert_num
    graph = build_forward_graph(tsv,
                                dispatch_sv=128,  up_proj_sv=4096,
                                swiglu_sv=128,    down_proj_sv=4096,
                                combine_sv=128,
                                hidden_size=args.hidden_size,
                                intermediate_size=args.intermediate_size,
                                dtype_size=args.dtype_size,
                                num_cube_cores=args.num_cube_cores)
    # Compute task_num for each operator via split-axis propagation
    graph.propagate_splits(tsv)

    dispatch_op  = graph.get_op("dispatch")
    up_proj_op   = graph.get_op("up_proj")
    swiglu_op    = graph.get_op("swiglu")
    down_proj_op = graph.get_op("down_proj")
    combine_op   = graph.get_op("combine")

    print(f"[fwd] tp={args.tp} ep={args.ep} seq={args.seq_size} "
          f"E={args.all_expert_num} topk={args.top_k}")
    print(f"      dispatch={dispatch_op.task_num}  up_proj={up_proj_op.task_num}  "
          f"swiglu={swiglu_op.task_num}  down_proj={down_proj_op.task_num}  "
          f"combine={combine_op.task_num}")

    # ── Tiling files (rank-independent) ──────────────────────────────────────
    up_proj_bytes   = get_up_proj_tiling_bytes(up_proj_op.split_value,
                                               hidden_size=args.hidden_size,
                                               intermediate_size=args.intermediate_size,
                                               num_groups=num_groups,
                                               num_cube_cores=args.num_cube_cores)
    down_proj_bytes = get_down_proj_tiling_bytes(down_proj_op.split_value,
                                                 hidden_size=args.hidden_size,
                                                 intermediate_size=args.intermediate_size,
                                                 num_groups=num_groups,
                                                 num_cube_cores=args.num_cube_cores)
    swiglu_bytes    = get_swiglu_tiling_bytes(swiglu_op.split_value,
                                              intermediate_size=args.intermediate_size)

    write_bin(os.path.join(out, 'up_proj_tiling.bin'),   up_proj_bytes)
    write_bin(os.path.join(out, 'swiglu_tiling.bin'),    swiglu_bytes)
    write_bin(os.path.join(out, 'down_proj_tiling.bin'), down_proj_bytes)

    # ── Event counters + workspace (rank-independent) ─────────────────────────
    # Reserve the same event capacity as the online workspace and runtime.
    write_bin(os.path.join(out, 'all_event_counters.bin'),
              bytes(event_workspace_bytes(tsv.ep, tsv.all_expert_num)))
    # gmm_workspace: 256 MiB zeros — kernel-internal scratch buffer
    write_bin(os.path.join(out, 'gmm_workspace.bin'),
              bytes(256 * 1024 * 1024))

    # ── RuntimeConfig files (one per rank) ───────────────────────────────────
    for rank_id in range(args.ep):
        cfg  = build_config_for_rank(graph, tsv, rank_id, num_cube_cores=args.num_cube_cores)
        data = serialize_runtime_config(cfg)
        path = os.path.join(out, f'runtime_config_input_rank_{rank_id}.bin')
        write_bin(path, data)

    print("[fwd] done.")


if __name__ == '__main__':
    main()
