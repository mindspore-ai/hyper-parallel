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
"""MoE-FFN distributed test cases with precision validation (TP=2 EP=2) — MindSpore.

The GMM and SwiGLU computations can be verified in isolation against
MindSpore reference implementations.  Each checked output must satisfy
  |kernel − ref| ≤ atol + rtol × |ref|   (rtol = atol = 1e-3, i.e. 0.1 %)
on every element.

Reference ops used (all run on Ascend):
  - GroupedMatmul(split_item=3, group_type=0)          (grouped matmul)
  - silu(gate) * up                                     (SwiGLU forward, silu = x * sigmoid(x))
  - ms.ops.auto_generate.gen_ops_prim.SwigluGrad        (SwiGLU backward)

Note: glist tensors passed to the mega kernel must be cumulative sums.
The _ref_gmm helpers accept per-group counts and accumulate internally.

Forward precision test (test_moe_ffn_fwd_tp2ep2):
  dispatch_target (pre-filled) → GMM1 → SwiGLU → GMM2 → down_proj_y
  Checked output: down_proj_y

Backward precision test (test_moe_ffn_bwd_tp2ep2):
  dispatched gradient.  Checked outputs (GMM1 → SwiGLU → GMM2 backward chain):
    act_grad_y  (GMM1 bwd  : dispatch_target @ W2.T)
    grad_gate   (SwiGLU bwd: analytical)
    gate_dx     (GMM2 bwd  : grad_gate @ W1.T)
"""
import os

os.environ.setdefault('SYMMETRIC_MEMORY_HEAP_SIZE', str(1024 * 1024 * 1024))

# Environment variables must be set before mindspore is imported to take effect.
# pylint: disable=wrong-import-position
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.ops.auto_generate import GroupedMatmul

import hyper_parallel.core.multicore as mc
from hyper_parallel.core import symmetric_memory as symm

from hyper_parallel.core.multicore.modules.common.compute_graph import TaskSplitValue
from hyper_parallel.core.multicore.modules.moe_ffn.forward.forward_graph import build_forward_graph
from hyper_parallel.core.multicore.modules.moe_ffn.forward.gen_runtime_data import build_config_for_rank as fwd_config
from hyper_parallel.core.multicore.modules.moe_ffn.forward.tiling_tables import (
    get_up_proj_tiling_bytes,
    get_down_proj_tiling_bytes,
    get_swiglu_tiling_bytes,
)
from hyper_parallel.core.multicore.modules.moe_ffn.backward.backward_graph import build_backward_graph
from hyper_parallel.core.multicore.modules.moe_ffn.backward.gen_runtime_data import build_config_for_rank as bwd_config
from hyper_parallel.core.multicore.modules.moe_ffn.backward.tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_swiglu_grad_tiling_bytes,
)
# pylint: enable=wrong-import-position

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
ms.set_seed(0)
init("hccl")

# ── Test configuration ────────────────────────────────────────────────────────
_TP                = 2
_EP                = 2
_SEQ_SIZE          = 1024
_ALL_EXPERT_NUM    = 16
_TOP_K             = 8
_HIDDEN_SIZE       = 5120
_INTERMEDIATE_SIZE = 2048   # post-SwiGLU half dimension; up_proj output width = intermediate_size * 2

_NUM_CUBE_CORES              = 20
_GMM_WORKSPACE_BYTES         = 32 * 1024 * 1024
_SWIGLU_GRAD_WORKSPACE_BYTES = 16 * 1024 * 1024

# Precision tolerance: 0.1 %.
_RTOL = 1e-3
_ATOL = 1e-3


# ── Reference helpers ─────────────────────────────────────────────────────────

def _ref_gmm(x, w, gl):
    """Grouped matmul via GroupedMatmul(split_item=3, group_type=0).

    x                 : [total, K]  bf16
    w                 : [E, K, N]   bf16
    group_list_counts : [E]  int64  per-group token counts (NOT cumulative)
    Returns           : [total, N]
    """
    return GroupedMatmul(split_item=3, group_type=0)(
        [x], [w], None, None, None, None, None, gl)[0]


def _ref_gmm_t(x, w, gl):
    """Grouped matmul with transposed weight: x @ w[g].T per group.

    x                 : [total, N_w]
    w                 : [E, K_w, N_w]  (stored as (E, out, in))
    group_list_counts : [E]  int64  per-group token counts (NOT cumulative)
    Returns           : [total, K_w]
    """
    w_t = ms.mint.transpose(w, -1, -2)     # [E, N_w, K_w]
    return GroupedMatmul(split_item=3, group_type=0)(
        [x], [w_t], None, None, None, None, None, gl)[0]


def _ref_swiglu(x):
    """SwiGLU forward via ms.ops.auto_generate.gen_ops_prim.Swiglu (axis=-1).

    x       : [N, 2D]  bf16
    Returns : [N, D]   bf16
    """
    return ms.ops.auto_generate.gen_ops_prim.Swiglu()(x, -1)


def _ref_swiglu_bwd(gate_input, grad_out):
    """SwiGLU backward via ms.ops.auto_generate.gen_ops_prim.SwigluGrad.

    gate_input : [N, 2D]  — forward SwiGLU input (up_proj_y)
    grad_out   : [N, D]   — gradient w.r.t. swiglu output (act_grad_y)
    Returns    : [N, 2D]  — gradient w.r.t. gate_input
    """
    return ms.ops.auto_generate.gen_ops_prim.SwigluGrad()(grad_out, gate_input, -1)


# ── Precision assertion ───────────────────────────────────────────────────────

def _check_allclose(name, actual, ref, rank, rtol=_RTOL, atol=_ATOL):
    """Assert ms.mint.allclose on bf16 tensors directly."""
    if bool(ms.mint.allclose(actual, ref, rtol=rtol, atol=atol)):
        return

    # Convert to float32 numpy for diagnostics
    a_np = actual.astype(ms.float32).asnumpy().flatten()
    r_np = ref.astype(ms.float32).asnumpy().flatten()
    diff = np.abs(a_np - r_np)
    tol  = atol + rtol * np.abs(r_np)
    bad  = np.where(diff > tol)[0]

    n_bad  = len(bad)
    n_show = min(n_bad, 20)
    lines  = [
        f"rank={rank}: {name} precision FAILED (rtol={rtol}, atol={atol})",
        f"  shape={tuple(actual.shape)}  mismatched={n_bad}/{a_np.size}"
        f"  ({100.0 * n_bad / max(a_np.size, 1):.2f}%)",
        f"  max_abs_diff={diff.max():.6g}  (at flat idx {diff.argmax()})",
        f"  first {n_show} mismatch(es)  [flat_idx | actual | ref | abs_diff | tol]:",
    ]
    for idx in bad[:n_show]:
        nd_idx = np.unravel_index(idx, actual.shape)
        lines.append(
            f"    {nd_idx}  actual={a_np[idx]:.6g}  ref={r_np[idx]:.6g}"
            f"  diff={diff[idx]:.6g}  tol={tol[idx]:.6g}"
        )
    raise AssertionError("\n".join(lines))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bytes_to_ms(data: bytes) -> ms.Tensor:
    arr = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return ms.ops.cast(ms.Tensor(arr, dtype=ms.uint8), ms.uint8)


def _rand_bf16(rng, *shape):
    """Create a bfloat16 Tensor with uniform random values in [-1, 1)."""
    return ms.Tensor(rng.uniform(-1, 1, shape).astype(np.float32)).astype(ms.bfloat16)


def _rand_weight_bf16(rng, fan_in, *shape):
    """Create a bfloat16 weight Tensor with Xavier uniform init: U[-1/√fan_in, 1/√fan_in].

    Keeps GMM output variance ≈ 1 regardless of fan_in, preventing the
    variance explosion (and SwiGLU positive-bias runaway) that occurs when
    weights are drawn from U[-1, 1] with large fan_in (e.g. hidden_size=5120).
    """
    limit = 1.0 / np.sqrt(fan_in)
    return ms.Tensor(rng.uniform(-limit, limit, shape).astype(np.float32)).astype(ms.bfloat16)


# ── Forward precision test ────────────────────────────────────────────────────

def test_moe_ffn_fwd_tp2ep2():
    """
    Feature: MoE-FFN forward kernel, TP=2 EP=2 (2 ranks).
    Description: Build forward compute graph, generate tiling and runtime_config
                 inline, call moe_ffn_fwd, compare down_proj_y against MindSpore
                 reference (GMM1 → SwiGLU → GMM2) with rtol=atol=1e-3.
    Expectation: Precision PASS on all ranks.
    """
    rank = get_rank()
    rng  = np.random.default_rng(seed=rank)

    tsv = TaskSplitValue(
        tp=_TP, ep=_EP, seq_size=_SEQ_SIZE,
        all_expert_num=_ALL_EXPERT_NUM, top_k=_TOP_K,
    )
    graph = build_forward_graph(
        tsv,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_cube_cores=_NUM_CUBE_CORES,
    )
    graph.propagate_splits(tsv)

    up_proj_op   = graph.get_op('up_proj')
    down_proj_op = graph.get_op('down_proj')
    swiglu_op    = graph.get_op('swiglu')
    num_groups   = tsv.single_rank_expert_num

    out_dir = os.path.join(os.path.dirname(__file__), 'moe_fwd_tiling_data_ms')
    os.makedirs(out_dir, exist_ok=True)
    if rank == 0:
        for fname, data in [
            ('up_proj_tiling.bin',   get_up_proj_tiling_bytes(
                up_proj_op.split_value,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                num_groups=num_groups,
                num_cube_cores=_NUM_CUBE_CORES)),
            ('down_proj_tiling.bin', get_down_proj_tiling_bytes(
                down_proj_op.split_value,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                num_groups=num_groups,
                num_cube_cores=_NUM_CUBE_CORES)),
            ('swiglu_tiling.bin',    get_swiglu_tiling_bytes(
                swiglu_op.split_value,
                intermediate_size=_INTERMEDIATE_SIZE)),
        ]:
            with open(os.path.join(out_dir, fname), 'wb') as f:
                f.write(data)
    ms.mint.distributed.barrier()

    cfg              = fwd_config(graph, tsv, rank, num_cube_cores=_NUM_CUBE_CORES)
    runtime_cfg      = _bytes_to_ms(bytes(cfg))
    with open(os.path.join(out_dir, 'up_proj_tiling.bin'), 'rb') as fh:
        up_proj_tiling   = _bytes_to_ms(fh.read())
    with open(os.path.join(out_dir, 'down_proj_tiling.bin'), 'rb') as fh:
        down_proj_tiling = _bytes_to_ms(fh.read())
    with open(os.path.join(out_dir, 'swiglu_tiling.bin'), 'rb') as fh:
        swiglu_tiling    = _bytes_to_ms(fh.read())

    sre               = tsv.single_rank_expert_num
    int_size          = _INTERMEDIATE_SIZE
    prs               = tsv.per_rank_seq
    tokens_per_expert = prs // sre

    # AllToAll skipped: dispatch/combine sizes are zeros.
    # dispatch_target is pre-filled — GMM1 uses it directly as its token input.
    dispatch_target     = symm.empty((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)
    dispatch_target[:]  = _rand_bf16(rng, prs, _HIDDEN_SIZE)
    dispatch_target_off = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int64)
    dispatch_src        = ms.ops.zeros((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)
    dispatch_src_off    = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int64)
    dispatch_size       = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int32)

    up_proj_weight        = _rand_weight_bf16(rng, _HIDDEN_SIZE, sre, _HIDDEN_SIZE, _INTERMEDIATE_SIZE * 2)
    up_proj_glist_counts = Tensor(np.full((sre,), tokens_per_expert, dtype=np.int32))
    up_proj_glist        = up_proj_glist_counts.cumsum(axis=0)   # kernel expects cumulative sums
    up_proj_glist         = ms.ops.cast(up_proj_glist, ms.int64)
    up_proj_y             = ms.ops.zeros((prs, _INTERMEDIATE_SIZE * 2), dtype=ms.bfloat16)
    swiglu_out            = ms.ops.zeros((prs, int_size), dtype=ms.bfloat16)

    down_proj_weight        = _rand_weight_bf16(rng, int_size, sre, int_size, _HIDDEN_SIZE)
    down_proj_glist_counts = Tensor(np.full((sre,), tokens_per_expert, dtype=np.int32))
    down_proj_glist        = down_proj_glist_counts.cumsum(axis=0)   # kernel expects cumulative sums
    down_proj_glist         = ms.ops.cast(down_proj_glist, ms.int64)
    down_proj_y             = ms.ops.zeros((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)

    combine_target     = symm.empty((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)
    combine_target_off = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int64)
    combine_src_off    = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int64)
    combine_size       = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int32)

    gmm_workspace      = ms.ops.zeros((_GMM_WORKSPACE_BYTES,), dtype=ms.uint8)
    all_event_counters = symm.empty((4096,), dtype=ms.uint8)
    all_event_counters.fill_(0)
    ms.mint.distributed.barrier()
    # ── Kernel call ───────────────────────────────────────────────────────────
    mc.moe_ffn_fwd(
        dispatch_target, dispatch_target_off, dispatch_src,
        dispatch_src_off, dispatch_size,
        up_proj_weight, up_proj_glist, up_proj_y,
        swiglu_out,
        down_proj_weight, down_proj_glist, down_proj_y,
        combine_target, combine_target_off, combine_src_off, combine_size,
        gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
        runtime_cfg, all_event_counters,
        rank, _EP, _ALL_EXPERT_NUM, _HIDDEN_SIZE, _SEQ_SIZE,
    )

    # ── MindSpore reference: GMM1 → SwiGLU → GMM2 ────────────────────────────
    up_proj_y_ref   = _ref_gmm(dispatch_target, up_proj_weight,  up_proj_glist)
    swiglu_out_ref  = _ref_swiglu(up_proj_y_ref)
    down_proj_y_ref = _ref_gmm(swiglu_out_ref,  down_proj_weight, down_proj_glist)

    _check_allclose('down_proj_y (GMM2 fwd)', down_proj_y, down_proj_y_ref, rank)


# ── Backward precision test ───────────────────────────────────────────────────

def test_moe_ffn_bwd_tp2ep2():
    """
    Feature: MoE-FFN backward kernel, TP=2 EP=2 (2 ranks).
    Description: Build backward compute graph, generate tiling and runtime_config
                 inline, call moe_ffn_bwd.  With AllToAll skipped, 'dispatch_target'
                 (slot 0) is pre-filled as the dispatched gradient.  Compare act_grad_y,
                 grad_gate and gate_dx against MindSpore reference with ±0.1 %
                 tolerance (rtol=atol=1e-3).
    Expectation: Precision PASS on all ranks.

    C++ param-slot map (from backward_graph.py):
      act_grad  : x=0(dispatch_target), weight=7(w2), glist=19 → y=8(act_grad_y)
      w1_grad   : x1=5(hidden), x2=0(dispatch_target), glist=19 → y=6(hidden_dw)
      swiglu_grad: x=8(act_grad_y), dy=9(gate) → out=10(grad_gate)
      gate_grad : x=10(grad_gate), weight=11(w1), glist=19 → y=12(gate_dx)
    """
    rank = get_rank()
    rng  = np.random.default_rng(seed=rank + 10)

    tsv = TaskSplitValue(
        tp=_TP, ep=_EP, seq_size=_SEQ_SIZE,
        all_expert_num=_ALL_EXPERT_NUM, top_k=_TOP_K,
    )
    graph = build_backward_graph(
        tsv,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_cube_cores=_NUM_CUBE_CORES,
    )
    graph.propagate_splits(tsv)

    act_grad_op    = graph.get_op('act_grad')
    gate_grad_op   = graph.get_op('gate_grad')
    w2_grad_op     = graph.get_op('w2_grad')
    w1_grad_op     = graph.get_op('w1_grad')
    swiglu_grad_op = graph.get_op('swiglu_grad')
    num_groups     = tsv.single_rank_expert_num

    out_dir = os.path.join(os.path.dirname(__file__), 'moe_bwd_tiling_data_ms')
    os.makedirs(out_dir, exist_ok=True)
    if rank == 0:
        for fname, data in [
            ('act_grad_tiling.bin',    get_act_grad_tiling_bytes(
                act_grad_op.split_value,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                num_groups=num_groups,
                num_cube_cores=_NUM_CUBE_CORES)),
            ('gate_grad_tiling.bin',   get_gate_grad_tiling_bytes(
                gate_grad_op.split_value,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                num_groups=num_groups,
                num_cube_cores=_NUM_CUBE_CORES)),
            ('w2_grad_tiling.bin',     get_w2_grad_tiling_bytes(
                w2_grad_op.split_value,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                num_groups=num_groups,
                num_cube_cores=_NUM_CUBE_CORES)),
            ('w1_grad_tiling.bin',     get_w1_grad_tiling_bytes(
                w1_grad_op.split_value,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                num_groups=num_groups,
                num_cube_cores=_NUM_CUBE_CORES)),
            ('swiglu_grad_tiling.bin', get_swiglu_grad_tiling_bytes(
                swiglu_grad_op.split_value,
                intermediate_size=_INTERMEDIATE_SIZE)),
        ]:
            with open(os.path.join(out_dir, fname), 'wb') as f:
                f.write(data)
    ms.mint.distributed.barrier()

    cfg                = bwd_config(graph, tsv, rank, num_cube_cores=_NUM_CUBE_CORES)
    runtime_cfg        = _bytes_to_ms(bytes(cfg))
    with open(os.path.join(out_dir, 'act_grad_tiling.bin'), 'rb') as fh:
        act_grad_tiling    = _bytes_to_ms(fh.read())
    with open(os.path.join(out_dir, 'gate_grad_tiling.bin'), 'rb') as fh:
        gate_grad_tiling   = _bytes_to_ms(fh.read())
    with open(os.path.join(out_dir, 'w2_grad_tiling.bin'), 'rb') as fh:
        w2_grad_tiling     = _bytes_to_ms(fh.read())
    with open(os.path.join(out_dir, 'w1_grad_tiling.bin'), 'rb') as fh:
        w1_grad_tiling     = _bytes_to_ms(fh.read())
    with open(os.path.join(out_dir, 'swiglu_grad_tiling.bin'), 'rb') as fh:
        swiglu_grad_tiling = _bytes_to_ms(fh.read())

    sre               = tsv.single_rank_expert_num
    int_size          = _INTERMEDIATE_SIZE
    prs               = tsv.per_rank_seq
    tokens_per_expert = prs // sre

    # AllToAll dispatch skipped: dispatch_size = zeros.
    # 'dispatch_target' (slot 0) is pre-filled with the dispatched gradient so that
    # act_grad and w1_grad have non-trivial input data.
    dispatch_target     = symm.empty((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)
    dispatch_target[:]  = _rand_bf16(rng, prs, _HIDDEN_SIZE)
    dispatch_target_off = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int64)
    dy                  = ms.ops.zeros((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)   # unused (AllToAll skipped)
    dispatch_src_off    = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int64)
    dispatch_size       = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int32)

    # Forward-pass saved tensors
    hidden              = _rand_bf16(rng, prs, int_size)              # swiglu_out from fwd
    hidden_dw           = ms.ops.zeros((sre, int_size, _HIDDEN_SIZE), dtype=ms.bfloat16)   # W2 gradient output
    w2                  = _rand_weight_bf16(rng, int_size, sre, int_size, _HIDDEN_SIZE)     # W2 = down_proj_weight
    act_grad_y          = ms.ops.zeros((prs, int_size), dtype=ms.bfloat16)                 # GMM1 bwd output
    gate                = _rand_bf16(rng, prs, _INTERMEDIATE_SIZE * 2)                    # up_proj_y from fwd
    grad_gate           = ms.ops.zeros((prs, _INTERMEDIATE_SIZE * 2), dtype=ms.bfloat16)   # SwiGLU grad output
    w1                  = _rand_weight_bf16(rng, _HIDDEN_SIZE, sre, _HIDDEN_SIZE, _INTERMEDIATE_SIZE * 2)  # W1 = up_proj_weight
    gate_dx             = ms.ops.zeros((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)              # GMM2 bwd output
    grad_x              = symm.empty((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)                # combine output

    combine_target_off = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int64)
    combine_src_off    = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int64)
    combine_size       = ms.ops.zeros((_ALL_EXPERT_NUM,), dtype=ms.int32)

    permute_out        = _rand_bf16(rng, prs, _HIDDEN_SIZE)                               # input tokens x
    gate_dw            = ms.ops.zeros((sre, _HIDDEN_SIZE, _INTERMEDIATE_SIZE * 2), dtype=ms.bfloat16)  # dW1 output

    group_list_counts = Tensor(np.full((sre,), tokens_per_expert, dtype=np.int32))
    group_list        = group_list_counts.cumsum(axis=0)   # kernel expects cumulative sums
    group_list         = ms.ops.cast(group_list, ms.int64)

    gmm_workspace         = ms.ops.zeros((_GMM_WORKSPACE_BYTES,),         dtype=ms.uint8)
    swiglu_grad_workspace = ms.ops.zeros((_SWIGLU_GRAD_WORKSPACE_BYTES,), dtype=ms.uint8)
    all_event_counters    = symm.empty((4096,), dtype=ms.uint8)
    all_event_counters.fill_(0)
    ms.mint.distributed.barrier()
    # ── Kernel call ───────────────────────────────────────────────────────────
    mc.moe_ffn_bwd(
        dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size,
        hidden, hidden_dw,
        w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
        combine_target_off, combine_src_off, combine_size,
        permute_out, gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling,
        swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
        runtime_cfg, all_event_counters,
        rank, _EP, _ALL_EXPERT_NUM, _HIDDEN_SIZE, _SEQ_SIZE,
    )

    act_grad_y_ref = _ref_gmm_t(dispatch_target, w2, group_list)
    grad_gate_ref = _ref_swiglu_bwd(gate, act_grad_y_ref)
    gate_dx_ref = _ref_gmm_t(grad_gate_ref, w1, group_list)

    # ── Precision checks ──────────────────────────────────────────────────────
    _check_allclose('act_grad_y (GMM1 bwd)',   act_grad_y, act_grad_y_ref, rank)
    _check_allclose('grad_gate  (SwiGLU bwd)', grad_gate,  grad_gate_ref,  rank)
    _check_allclose('gate_dx    (GMM2 bwd)',   gate_dx,    gate_dx_ref,    rank)
