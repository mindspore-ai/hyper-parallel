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
"""MegaMoe distributed test cases with precision validation (TP=2 EP=2) — MindSpore.

The GMM and SwiGLU computations can be verified in isolation against
MindSpore reference implementations.  Each checked output must satisfy
  |kernel − ref| ≤ atol + rtol × |ref|   (rtol = atol = 1e-3, i.e., 0.1 %)
on every element.

Reference ops used (all run on Ascend):
  - GroupedMatmul(split_item=3, group_type=0)          (grouped matmul)
  - silu(gate) * up                                     (SwiGLU forward, silu = x * sigmoid(x))
  - ms.ops.auto_generate.gen_ops_prim.SwigluGrad        (SwiGLU backward)

Note: glist tensors passed to the MegaMoe kernel must be cumulative sums.
The _ref_gmm helpers accept per-group counts and accumulate internally.

Forward precision test (test_moe_fwd_tp2ep2):
  dispatch_target (pre-filled) → GMM1 → SwiGLU → GMM2 → down_proj_y
  Checked output: down_proj_y

Backward precision test (test_moe_bwd_tp2ep2):
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

from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import build_forward_graph
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import build_config_for_rank as fwd_config
from hyper_parallel.core.multicore.modules.mega_moe.forward.tiling_tables import (
    get_up_proj_tiling_bytes,
    get_down_proj_tiling_bytes,
    get_swiglu_tiling_bytes,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import build_config_for_rank as bwd_config
from hyper_parallel.core.multicore.modules.mega_moe.backward.tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
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

_NUM_CUBE_CORES              = ms.runtime.get_device_limit(get_rank())["cube_core_num"]
_GMM_WORKSPACE_BYTES         = 32 * 1024 * 1024
_SWIGLU_GRAD_WORKSPACE_BYTES = 16 * 1024 * 1024

# Precision tolerance: 0.1 %.
_RTOL = 1e-3
_ATOL = 1e-3


# ── Reference helpers ─────────────────────────────────────────────────────────

def _ref_gmm(x, w, gl):
    """Grouped matmul via GroupedMatmul(split_item=3, group_type=0).

    x  : [total, K]  bf16
    w  : [E, K, N]   bf16
    gl : [E]  int64  — cumulative group list (gl[-1] == total tokens)
    Returns : [total, N]
    """
    return GroupedMatmul(split_item=3, group_type=0)(
        [x], [w], None, None, None, None, None, gl)[0]


def _ref_gmm_t(x, w, gl):
    """Grouped matmul with transposed weight: x @ w[g].T per group.

    x  : [total, N_w]
    w  : [E, K_w, N_w]  (stored as (E, out, in))
    gl : [E]  int64  — cumulative group list (gl[-1] == total tokens)
    Returns : [total, K_w]
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


def _ref_gmm_dw(x_t, y, gl):
    """Weight-gradient grouped matmul: x_t @ y per expert group.

    Computes the outer product per expert via GroupedMatmul(group_type=2),
    matching the bprop_fn1 pattern in moev3.py.

    x_t : [K, total_tokens]  bf16  — transposed activation (caller does mint.transpose)
    y   : [total_tokens, N]  bf16  — gradient flowing into this weight
    gl  : [E]  int64  — cumulative group list
    Returns: [E, K, N]  — per-expert weight gradient
    """
    return GroupedMatmul(split_item=3, group_type=2)(
        [x_t], [y], None, None, None, None, None, gl)[0]


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


def _compute_dispatch_offsets(flat_tokens_per_expert, ep_group):
    """Compute per-rank dispatch/combine offsets from actual token counts via AlltoAll.

    Reimplements MoEAlltoAllMultiCoreTokenDispatcher.get_onesided_data_offset with
    descriptive variable names.  Three AlltoAll calls exchange token-count information
    so each EP rank derives exactly where dispatched and combined tokens reside.

    flat_tokens_per_expert : [all_expert_num] int32 — tokens this rank sends per expert
                             slot, ordered (rank0_exp0, rank0_exp1, …,
                             rank{ep-1}_exp{sre-1})
    ep_group               : EP communication group name

    Returns (all sizes in TOKEN units, NOT bytes):
        dispatch_src_off    [all_expert_num] int64  — source row-offsets in dispatch_src
        dispatch_target_off [all_expert_num] int64  — target row-offsets in dispatch_target
        dispatch_size       [all_expert_num] int32  — token counts per dispatch slot
        combine_src_off     [all_expert_num] int64  — source row-offsets in down_proj_y/gate_dx
        combine_target_off  [all_expert_num] int64  — target row-offsets in combine_target/grad_x
        combine_size        [all_expert_num] int32  — token counts per combine slot
        send_list           [ep]             int64  — tokens sent to each EP rank (AlltoAllV)
        receive_list        [ep]             int64  — tokens received from each EP rank (AlltoAllV)
    """
    ep  = _EP
    sre = _ALL_EXPERT_NUM // ep
    zero_i32 = ms.Tensor(np.zeros(1, dtype=np.int32))

    # Use float32 for AlltoAll (MindSpore AlltoAll requires floating-point input).
    flat_f32 = flat_tokens_per_expert.astype(ms.float32)

    # Dispatch source offsets: cumulative row-offsets into dispatch_src.
    dispatch_src_off = ms.ops.concat(
        [zero_i32, ms.mint.cumsum(flat_tokens_per_expert, 0)[:-1]])
    dispatch_size    = flat_tokens_per_expert.astype(ms.int32)

    # AlltoAll 1: exchange per-expert token counts across EP ranks.
    #   After AlltoAll: recv_counts[r, e] = tokens rank r sends to THIS rank's local expert e.
    alltoall_ep = ms.ops.AlltoAll(split_count=ep, split_dim=-2, concat_dim=-2, group=ep_group)
    recv_counts = alltoall_ep(flat_f32.reshape(ep, sre)).astype(ms.int32)

    # Token counts for the combine operation (reverse of dispatch).
    combine_size = recv_counts.reshape(-1).astype(ms.int32)

    # Tokens sent / received per EP rank — used by AlltoAllV in reference functions.
    send_list    = flat_tokens_per_expert.reshape(ep, sre).sum(axis=-1).astype(ms.int64)
    receive_list = recv_counts.sum(axis=-1).astype(ms.int64)

    # Cumulative combine-source offsets in expert-first, rank-second order.
    # recv_counts.T[e, r] = tokens from rank r for local expert e.
    recv_expert_rank_flat = recv_counts.T.reshape(-1)   # [exp0_r0, exp0_r1, …, expSre_r{ep-1}]
    combine_src_cumsum    = ms.ops.concat(
        [zero_i32, ms.mint.cumsum(recv_expert_rank_flat, 0)[:-1]])
    # Reshape to [sre, ep] then transpose → [ep, sre] for AlltoAll 2.
    combine_src_ep_sre = combine_src_cumsum.reshape(sre, ep).T   # [ep, sre]
    combine_src_off    = combine_src_ep_sre.reshape(-1).astype(ms.int64)

    # AlltoAll 2: send combine_src offsets to origin ranks → dispatch_target_off.
    #   origin rank r learns where its tokens land in THIS rank's dispatch_target.
    dispatch_target_off = alltoall_ep(
        combine_src_ep_sre.astype(ms.float32)).reshape(-1).astype(ms.int64)

    # AlltoAll 3: send dispatch_src offsets to expert ranks → combine_target_off.
    #   expert rank learns where in origin rank's combine_target to write results.
    alltoall_combine = ms.ops.AlltoAll(split_count=ep, split_dim=-1, concat_dim=-1, group=ep_group)
    combine_target_off = alltoall_combine(
        dispatch_src_off.astype(ms.float32)).astype(ms.int64)

    dispatch_src_off = dispatch_src_off.astype(ms.int64)

    return (dispatch_src_off, dispatch_target_off, dispatch_size,
            combine_src_off, combine_target_off, combine_size,
            send_list, receive_list)


# ── Kernel run helpers ────────────────────────────────────────────────────────

def _run_fwd_kernel(
        dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off, dispatch_size,
        up_proj_weight, up_proj_glist, up_proj_y, swiglu_out,
        down_proj_weight, down_proj_glist, down_proj_y,
        combine_target, combine_target_off, combine_src_off, combine_size,
        gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
        runtime_cfg, all_event_counters, rank):
    """Call mega_moe kernel in-place (fills dispatch_target, down_proj_y, combine_target)."""
    mc.mega_moe(
        dispatch_target,
        dispatch_target_off * _HIDDEN_SIZE,
        dispatch_src,
        dispatch_src_off    * _HIDDEN_SIZE,
        dispatch_size       * _HIDDEN_SIZE,
        up_proj_weight, up_proj_glist, up_proj_y, swiglu_out,
        down_proj_weight, down_proj_glist, down_proj_y,
        combine_target,
        combine_target_off  * _HIDDEN_SIZE,
        combine_src_off     * _HIDDEN_SIZE,
        combine_size        * _HIDDEN_SIZE,
        gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
        runtime_cfg, all_event_counters,
        rank, _EP, _ALL_EXPERT_NUM, _HIDDEN_SIZE, _SEQ_SIZE,
    )


def _run_fwd_reference(
        dispatch_src, up_proj_weight, up_proj_glist,
        down_proj_weight, down_proj_glist,
        sre, peto, send_list, receive_list):
    """AlltoAllV dispatch → MoeTokenPermute → GMM1 → SwiGLU → GMM2
    → MoeTokenUnpermute → AlltoAllV combine (reference).

    The AlltoAllV output is rank-first; MoeTokenPermute sorts tokens to
    expert-first order (matching the kernel's dispatch_target layout).
    MoeTokenUnpermute restores rank-first order before the combine AlltoAllV.

    send_list / receive_list: per-EP-rank token counts (int64 CPU tensors)
        derived from _compute_dispatch_offsets for the correct per-rank values.

    Returns: (dispatch_target_ref, down_proj_y_ref, combine_ref)
    """
    alltoallv = ms.ops.AlltoAllV(group="hccl_world_group", block_size=_HIDDEN_SIZE)

    # 1. AlltoAllV dispatch → rank-first at receiving rank.
    ops_dispatch = alltoallv(dispatch_src.reshape(-1), send_list, receive_list)
    local_x      = ops_dispatch.reshape(-1, _HIDDEN_SIZE)

    # 2. Communicate expert_id: for balanced load the received expert_id is deterministic
    #    (each EP rank sends peto tokens per local expert in expert-slot order).
    local_expert_id = ms.Tensor(
        np.tile(np.repeat(np.arange(sre, dtype=np.int32), peto), _EP))

    # 3. MoeTokenPermute: sort rank-first tokens → expert-first order.
    #    dispatch_target_ref now matches the kernel's dispatch_target layout.
    dispatch_target_ref, unresort_map = ms.ops.moe_token_permute(local_x, local_expert_id)

    # 4. GMM chain on expert-sorted tokens.
    up_proj_y_ref   = _ref_gmm(dispatch_target_ref, up_proj_weight,  up_proj_glist)
    swiglu_out_ref  = _ref_swiglu(up_proj_y_ref)
    down_proj_y_ref = _ref_gmm(swiglu_out_ref,      down_proj_weight, down_proj_glist)

    # 5. MoeTokenUnpermute: restore rank-first order before AlltoAllV combine.
    down_proj_y_rank_first = ms.ops.moe_token_unpermute(down_proj_y_ref, unresort_map)

    # 6. AlltoAllV combine (swap send/receive lists).
    combine_ref = alltoallv(
        down_proj_y_rank_first.reshape(-1), receive_list, send_list
    ).reshape(-1, _HIDDEN_SIZE)

    return dispatch_target_ref, down_proj_y_ref, combine_ref


def _run_bwd_kernel(
        dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size,
        hidden, hidden_dw, w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
        combine_target_off, combine_src_off, combine_size,
        permute_out, gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
        swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
        runtime_cfg, all_event_counters, rank):
    """Call mega_moe_grad kernel in-place (fills dispatch_target, act_grad_y, grad_gate,
    gate_dx, hidden_dw, gate_dw, grad_x)."""
    mc.mega_moe_grad(
        dispatch_target,
        dispatch_target_off * _HIDDEN_SIZE,
        dy,
        dispatch_src_off    * _HIDDEN_SIZE,
        dispatch_size       * _HIDDEN_SIZE,
        ms.mint.transpose(hidden, -1, -2), hidden_dw,
        ms.mint.transpose(w2, -1, -2), act_grad_y, gate, grad_gate,
        ms.mint.transpose(w1, -1, -2), gate_dx, grad_x,
        combine_target_off  * _HIDDEN_SIZE,
        combine_src_off     * _HIDDEN_SIZE,
        combine_size        * _HIDDEN_SIZE,
        ms.mint.transpose(permute_out, -1, -2), gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
        swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
        runtime_cfg, all_event_counters,
        rank, _EP, _ALL_EXPERT_NUM, _HIDDEN_SIZE, _SEQ_SIZE,
    )


def _run_bwd_reference(
        dy, w2, gate, w1, group_list, hidden, permute_out,
        sre, peto, send_list, receive_list):
    """AlltoAllV dispatch dy → moe_token_permute → bwd GMM/SwiGLU chain + dW
    → moe_token_unpermute → AlltoAllV combine (reference).

    Structure mirrors _run_fwd_reference:
      moe_token_permute(dy)      ≡ backward of forward moe_token_unpermute
      moe_token_unpermute(gate_dx) ≡ backward of forward moe_token_permute

    Returns: (dispatch_target_ref, act_grad_y_ref, grad_gate_ref,
              gate_dx_ref, hidden_dw_ref, gate_dw_ref, grad_x_ref)
    """
    alltoallv = ms.ops.AlltoAllV(group="hccl_world_group", block_size=_HIDDEN_SIZE)

    # 1. AlltoAllV dispatch dy → rank-first at receiving rank.
    local_dy = alltoallv(dy.reshape(-1), send_list, receive_list).reshape(-1, _HIDDEN_SIZE)

    # 2. moe_token_permute: rank-first → expert-first (mirrors forward step 3).
    local_expert_id = ms.Tensor(
        np.tile(np.repeat(np.arange(sre, dtype=np.int32), peto), _EP))
    dispatch_target_ref, unresort_map = ms.ops.moe_token_permute(local_dy, local_expert_id)

    # 3. Backward GMM chain (inputs are expert-sorted).
    act_grad_y_ref = _ref_gmm(dispatch_target_ref, ms.mint.transpose(w2, -1, -2), group_list)
    grad_gate_ref  = _ref_swiglu_bwd(gate, act_grad_y_ref)
    gate_dx_ref    = _ref_gmm(grad_gate_ref, ms.mint.transpose(w1, -1, -2), group_list)

    # 4. Weight gradients via GroupedMatmul(group_type=2).
    hidden_dw_ref = _ref_gmm_dw(ms.mint.transpose(hidden,      -1, -2), dispatch_target_ref, group_list)
    gate_dw_ref   = _ref_gmm_dw(ms.mint.transpose(permute_out, -1, -2), grad_gate_ref,       group_list)

    # 5. moe_token_unpermute: expert-first → rank-first (mirrors forward step 5).
    gate_dx_rank_first = ms.ops.moe_token_unpermute(gate_dx_ref, unresort_map)

    # 6. AlltoAllV combine (swap send/receive lists).
    grad_x_ref = alltoallv(
        gate_dx_rank_first.reshape(-1), receive_list, send_list
    ).reshape(-1, _HIDDEN_SIZE)

    return dispatch_target_ref, act_grad_y_ref, grad_gate_ref, gate_dx_ref, hidden_dw_ref, gate_dw_ref, grad_x_ref


# ── Forward precision test ────────────────────────────────────────────────────

def test_mega_moe_tp2ep2():
    """
    Feature: MoE-FFN forward kernel, TP=2 EP=2 (2 ranks).
    Description: Build forward compute graph, generate tiling and runtime_config
                 inline, call moe_fwd, compare down_proj_y against MindSpore
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

    out_dir = os.path.join(os.path.dirname(__file__), 'mega_moe_fwd_tiling_data_ms')
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
    peto = prs // _ALL_EXPERT_NUM    # tokens this rank sends to each expert slot

    # dispatch_src: tokens sorted by expert slot, ready to scatter across EP ranks.
    dispatch_src    = _rand_bf16(rng, prs, _HIDDEN_SIZE)
    dispatch_target = symm.empty((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)

    up_proj_weight   = _rand_weight_bf16(rng, _HIDDEN_SIZE, sre, _HIDDEN_SIZE, _INTERMEDIATE_SIZE * 2)
    up_proj_glist    = ms.ops.cast(
        Tensor(np.full((sre,), tokens_per_expert, dtype=np.int32)).cumsum(axis=0), ms.int64)
    up_proj_y        = ms.ops.zeros((prs, _INTERMEDIATE_SIZE * 2), dtype=ms.bfloat16)
    swiglu_out       = ms.ops.zeros((prs, int_size), dtype=ms.bfloat16)

    down_proj_weight = _rand_weight_bf16(rng, int_size, sre, int_size, _HIDDEN_SIZE)
    down_proj_glist  = ms.ops.cast(
        Tensor(np.full((sre,), tokens_per_expert, dtype=np.int32)).cumsum(axis=0), ms.int64)
    down_proj_y      = ms.ops.zeros((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)

    combine_target = symm.empty((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)

    # Compute per-rank dispatch/combine offsets via AlltoAll.
    # For balanced routing flat_tokens_per_expert is uniform (peto per slot);
    # in general this encodes the actual routing table.
    flat_tokens_per_expert = ms.Tensor(np.full((_ALL_EXPERT_NUM,), peto, dtype=np.int32))
    (dispatch_src_off, dispatch_target_off, dispatch_size,
     combine_src_off, combine_target_off, combine_size,
     send_list, receive_list) = _compute_dispatch_offsets(
        flat_tokens_per_expert, "hccl_world_group")

    gmm_workspace      = ms.ops.zeros((_GMM_WORKSPACE_BYTES,), dtype=ms.uint8)
    all_event_counters = symm.empty((4096,), dtype=ms.uint8)
    all_event_counters.fill_(0)
    ms.mint.distributed.barrier()

    # ── 多核 kernel 运行 ──────────────────────────────────────────────────────
    _run_fwd_kernel(
        dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off, dispatch_size,
        up_proj_weight, up_proj_glist, up_proj_y, swiglu_out,
        down_proj_weight, down_proj_glist, down_proj_y,
        combine_target, combine_target_off, combine_src_off, combine_size,
        gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
        runtime_cfg, all_event_counters, rank,
    )

    # ── 原版参考运行 ──────────────────────────────────────────────────────────
    dispatch_target_ref, down_proj_y_ref, combine_ref = _run_fwd_reference(
        dispatch_src, up_proj_weight, up_proj_glist,
        down_proj_weight, down_proj_glist,
        sre, peto, send_list, receive_list,
    )

    # ── 精度对比 ──────────────────────────────────────────────────────────────
    _check_allclose('dispatch_target (AllToAllV fwd)', dispatch_target, dispatch_target_ref, rank)
    _check_allclose('down_proj_y     (GMM2 fwd)',      down_proj_y,     down_proj_y_ref,     rank)
    _check_allclose('combine_target  (AllToAllV fwd)', combine_target,  combine_ref,          rank)


# ── Backward precision test ───────────────────────────────────────────────────

def test_mega_moe_grad_tp2ep2():
    """
    Feature: MoE-FFN backward kernel, TP=2 EP=2 (2 ranks).
    Description: Build backward compute graph, generate tiling and runtime_config inline.
                 Run ops.AlltoAllV(dy) as reference dispatch, then execute the full
                 backward chain and AlltoAllV combine via mc.mega_moe_grad.  Compare
                 all outputs against the MindSpore reference with ±0.1 % tolerance
                 (rtol=atol=1e-3).
    Expectation: Precision PASS on all ranks.

    C++ param-slot map (from backward_graph.py):
      dispatch  : src=dy(slot 2), target=dispatch_target(slot 0)
      act_grad  : x=0(dispatch_target), weight=7(w2), glist=19 → y=8(act_grad_y)
      w2_grad   : x1=5(hidden), x2=0(dispatch_target), glist=19 → y=6(hidden_dw)
      swiglu_grad: x=8(act_grad_y), dy=9(gate) → out=10(grad_gate)
      gate_grad : x=10(grad_gate), weight=11(w1), glist=19 → y=12(gate_dx)
      combine   : src=gate_dx(slot 12), target=grad_x(slot 13)
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
    w1_grad_op     = graph.get_op('w1_grad')
    w2_grad_op     = graph.get_op('w2_grad')
    swiglu_grad_op = graph.get_op('swiglu_grad')
    num_groups     = tsv.single_rank_expert_num

    out_dir = os.path.join(os.path.dirname(__file__), 'mega_moe_bwd_tiling_data_ms')
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
            ('w1_grad_tiling.bin',     get_w1_grad_tiling_bytes(
                w1_grad_op.split_value,
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
    with open(os.path.join(out_dir, 'w1_grad_tiling.bin'), 'rb') as fh:
        w1_grad_tiling     = _bytes_to_ms(fh.read())
    with open(os.path.join(out_dir, 'w2_grad_tiling.bin'), 'rb') as fh:
        w2_grad_tiling     = _bytes_to_ms(fh.read())
    with open(os.path.join(out_dir, 'swiglu_grad_tiling.bin'), 'rb') as fh:
        swiglu_grad_tiling = _bytes_to_ms(fh.read())

    sre               = tsv.single_rank_expert_num
    int_size          = _INTERMEDIATE_SIZE
    prs               = tsv.per_rank_seq
    tokens_per_expert = prs // sre
    peto = prs // _ALL_EXPERT_NUM    # tokens this rank sends to each expert slot

    # dy: upstream gradient, sorted by expert slot (same layout as dispatch_src in fwd).
    dy              = _rand_bf16(rng, prs, _HIDDEN_SIZE)
    dispatch_target = symm.empty((prs, _HIDDEN_SIZE), dtype=ms.bfloat16)

    # Forward-pass saved tensors
    hidden     = _rand_bf16(rng, prs, int_size)                                               # swiglu_out from fwd
    hidden_dw  = ms.ops.zeros((sre, int_size, _HIDDEN_SIZE),            dtype=ms.bfloat16)    # dW₂ output
    w2         = _rand_weight_bf16(rng, int_size, sre, int_size, _HIDDEN_SIZE)                # down_proj_weight
    act_grad_y = ms.ops.zeros((prs, int_size),                          dtype=ms.bfloat16)    # GMM1 bwd output
    gate       = _rand_bf16(rng, prs, _INTERMEDIATE_SIZE * 2)                                 # up_proj_y from fwd
    grad_gate  = ms.ops.zeros((prs, _INTERMEDIATE_SIZE * 2),            dtype=ms.bfloat16)    # SwiGLU grad output
    w1         = _rand_weight_bf16(rng, _HIDDEN_SIZE, sre, _HIDDEN_SIZE, _INTERMEDIATE_SIZE * 2)  # up_proj_weight
    gate_dx    = ms.ops.zeros((prs, _HIDDEN_SIZE),                      dtype=ms.bfloat16)    # GMM2 bwd output
    grad_x     = symm.empty((prs, _HIDDEN_SIZE),                        dtype=ms.bfloat16)    # combine output

    # Compute per-rank dispatch/combine offsets via AlltoAll (same formula as forward).
    flat_tokens_per_expert = ms.Tensor(np.full((_ALL_EXPERT_NUM,), peto, dtype=np.int32))
    (dispatch_src_off, dispatch_target_off, dispatch_size,
     combine_src_off, combine_target_off, combine_size,
     send_list, receive_list) = _compute_dispatch_offsets(
        flat_tokens_per_expert, "hccl_world_group")

    permute_out = _rand_bf16(rng, prs, _HIDDEN_SIZE)                                          # input tokens x
    gate_dw     = ms.ops.zeros((sre, _HIDDEN_SIZE, _INTERMEDIATE_SIZE * 2), dtype=ms.bfloat16)  # dW₁ output

    group_list = ms.ops.cast(
        Tensor(np.full((sre,), tokens_per_expert, dtype=np.int32)).cumsum(axis=0), ms.int64)

    gmm_workspace         = ms.ops.zeros((_GMM_WORKSPACE_BYTES,),         dtype=ms.uint8)
    swiglu_grad_workspace = ms.ops.zeros((_SWIGLU_GRAD_WORKSPACE_BYTES,), dtype=ms.uint8)
    all_event_counters    = symm.empty((4096,), dtype=ms.uint8)
    all_event_counters.fill_(0)
    ms.mint.distributed.barrier()

    # ── 多核 kernel 运行 ──────────────────────────────────────────────────────
    _run_bwd_kernel(
        dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size,
        hidden, hidden_dw, w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
        combine_target_off, combine_src_off, combine_size,
        permute_out, gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
        swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
        runtime_cfg, all_event_counters, rank,
    )

    # ── 原版参考运行 ──────────────────────────────────────────────────────────
    (dispatch_target_ref, act_grad_y_ref, grad_gate_ref,
     gate_dx_ref, hidden_dw_ref, gate_dw_ref, grad_x_ref) = _run_bwd_reference(
        dy, w2, gate, w1, group_list, hidden, permute_out,
        sre, peto, send_list, receive_list,
    )

    # ── 精度对比 ──────────────────────────────────────────────────────────────
    _check_allclose('dispatch_target (bwd AllToAllV)',  dispatch_target, dispatch_target_ref, rank)
    _check_allclose('act_grad_y (GMM1 bwd)',            act_grad_y,      act_grad_y_ref,      rank)
    _check_allclose('grad_gate  (SwiGLU bwd)',          grad_gate,       grad_gate_ref,        rank)
    _check_allclose('gate_dx    (GMM2 bwd)',            gate_dx,         gate_dx_ref,          rank)
    _check_allclose('hidden_dw  (w2_grad dW2)',         hidden_dw,       hidden_dw_ref,        rank)
    _check_allclose('gate_dw    (w1_grad dW1)',         gate_dw,         gate_dw_ref,          rank)
    _check_allclose('grad_x     (combine AllToAllV)',   grad_x,          grad_x_ref,           rank)
