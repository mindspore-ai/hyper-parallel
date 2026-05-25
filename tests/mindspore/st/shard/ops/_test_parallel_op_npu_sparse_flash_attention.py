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
"""MindSpore distributed ST for npu_sparse_flash_attention.

Pipeline: ops.lightning_indexer(q_idx, k_idx, w) → sparse_indices
          → ops.sparse_flash_attention(q, k, v, sparse_indices, ...).

The two ops use different head_dim projections of the same sequence:
  - lightning_indexer : q_idx/k_idx head_dim = D_IDX = 128 (kernel constraint)
  - sparse_flash_attention : q/k head_dim = D_DIM = 512 (MLA attention_mode=2 constraint)

For BSND+CP the lightning_indexer distributed op slices k_idx to the causal window
``k_idx[:, :S1_local*(split_id+1), :, :]``; the sparse_flash_attention distributed op
applies the same slice to k/v/key_rope, so sparse_indices remain valid.  This mirrors
MindFormers adjust_bsnd_input logic.

Distributed sharding patterns verified against MindFormers dsa_attention.py shard():
  - BSND: q/si sharded on B (dp) and/or S1 (cp); k/v sharded on B (dp) only.
  - TND:  q/si sharded on T1 = dp*cp combined; k/v sharded on T2 = dp only.
"""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor, ops
from mindspore.ops import sparse_flash_attention
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial


def setup_module() -> None:
    """Initialize MindSpore context and communication."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


np.random.seed(42)
ms.set_seed(42)
ms.set_deterministic(True)

# sparse_flash_attention head dims (attention_mode=2 MLA kernel constraints).
N2 = 1
D_DIM = 512       # qk_head_dim for sparse_flash_attention (must be 512)
DR_DIM = 64       # rope head_dim
SCALE_VALUE = 0.135234

# lightning_indexer head dim (kernel constraint: must be 128).
D_IDX = 128

# BSND layout — replicated/DP tests.
B_BSND = 4
S1_BSND = 4
S2_BSND = 1024
N1_BSND = 8

# Number of top-k blocks for lightning_indexer (non-CP BSND).
SPARSE_COUNT_BSND = 8

# BSND layout — CP/dp+cp tests (realistic self-attention scenario).
# S1_BSND_CP=S2_BSND_CP are the GLOBAL sequence lengths (Q and K come from the same
# sequence).  CP=2 splits Q on S1: each rank processes S1_local=S1_BSND_CP//2 query tokens.
# K is Replicate (full length on each rank); the distributed op truncates it to the
# causal window per rank: rank r sees k[:, :S1_local*(r+1), :, :].
# S1=S2 (self-attention) ensures causal truncation is lossless for rank 0.
# SPARSE_COUNT_CP <= S1_local = S1_BSND_CP // 2.
B_BSND_CP = 2
N1_BSND_CP = 32
S1_BSND_CP = 1024
S2_BSND_CP = 1024
SPARSE_COUNT_CP = 128

# TND layout: T1=T2=1024, N1=8.
S_TND = 1024
N1_TND = 8
BATCH_DIM = 128
SPARSE_COUNT_TND = 8

_INPUT_NAMES = ("query", "key", "value", "query_rope", "key_rope")
_FWD_OUTPUT_NAMES = ("attention_out", "softmax_max", "softmax_sum")


def _make_tnd_seq_lens():
    """Build TND actual_seq_lengths tensors (prefix sums, int32, shape (128,))."""
    step = S_TND // BATCH_DIM
    arr = (np.arange(BATCH_DIM, dtype=np.int32) + 1) * step
    t = Tensor(arr, ms.int32)
    return t, Tensor(arr.copy(), ms.int32)


def _make_tnd_local_seq_lens():
    """Build LOCAL TND actual_seq_lengths for DP test (half batch per rank)."""
    local_batch = BATCH_DIM // 2
    local_s = S_TND // 2
    step = local_s // local_batch
    arr = (np.arange(local_batch, dtype=np.int32) + 1) * step
    t = Tensor(arr, ms.int32)
    return t, Tensor(arr.copy(), ms.int32)



def _np_to_bf16(arr):
    """Convert float32 numpy array to MindSpore bfloat16 Tensor."""
    return Tensor(arr, ms.bfloat16)


def _to_f32np(t):
    """Convert a MindSpore tensor (any dtype) to float32 numpy array."""
    return t.astype(ms.float32).asnumpy()


def _generate_inputs(layout='BSND', s1=None, s2=None, n1=None, b=None):
    """Generate random float32 numpy arrays for BSND or TND layout.

    Returns two groups of tensors:
      - SFA group (D_DIM=512): q, k, v, q_rope, k_rope
      - Indexer group (D_IDX=128): q_idx, k_idx, w
        q_idx/k_idx share the same B/S1/S2 dims as q/k but with D_IDX head_dim.
        w has shape (B, S1, N1) for BSND or (T1, N1) for TND.

    Args:
        layout: 'BSND' or 'TND'.
        s1: Query sequence length (BSND only; defaults to S1_BSND).
        s2: Key/value sequence length (BSND only; defaults to S2_BSND).
        n1: Number of query heads (BSND only; defaults to N1_BSND).
        b: Batch size (BSND only; defaults to B_BSND).

    Returns:
        tuple: (q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np)
    """
    if layout == 'BSND':
        s1 = s1 if s1 is not None else S1_BSND
        s2 = s2 if s2 is not None else S2_BSND
        n1 = n1 if n1 is not None else N1_BSND
        b = b if b is not None else B_BSND
        # SFA tensors (D_DIM=512)
        q_np = np.random.randn(b, s1, n1, D_DIM).astype(np.float32)
        k_np = np.random.randn(b, s2, N2, D_DIM).astype(np.float32)
        v_np = k_np.copy()
        q_rope_np = np.random.randn(b, s1, n1, DR_DIM).astype(np.float32)
        k_rope_np = np.random.randn(b, s2, N2, DR_DIM).astype(np.float32)
        # Indexer tensors (D_IDX=128)
        q_idx_np = np.random.randn(b, s1, n1, D_IDX).astype(np.float32)
        k_idx_np = np.random.randn(b, s2, N2, D_IDX).astype(np.float32)
        w_np = np.random.randn(b, s1, n1).astype(np.float32)
    else:  # TND
        # SFA tensors (D_DIM=512)
        q_np = np.random.randn(S_TND, N1_TND, D_DIM).astype(np.float32)
        k_np = np.random.randn(S_TND, N2, D_DIM).astype(np.float32)
        v_np = k_np.copy()
        q_rope_np = np.random.randn(S_TND, N1_TND, DR_DIM).astype(np.float32)
        k_rope_np = np.random.randn(S_TND, N2, DR_DIM).astype(np.float32)
        # Indexer tensors (D_IDX=128)
        q_idx_np = np.random.randn(S_TND, N1_TND, D_IDX).astype(np.float32)
        k_idx_np = np.random.randn(S_TND, N2, D_IDX).astype(np.float32)
        w_np = np.random.randn(S_TND, N1_TND).astype(np.float32)
    return q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np


def _call_sfa(q, k, v, si, q_rope, k_rope,
              actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
              layout='BSND'):
    """Call sparse_flash_attention with an explicit layout string.

    For BSND, actual_seq_lengths_query and actual_seq_lengths_kv should be
    None (kernel infers from tensor shape); for TND they must be passed.
    Omitted parameters use kernel defaults: sparse_mode=3 (rightDownCausal),
    sparse_block_size=1, pre_tokens/next_tokens=INT64_MAX.

    Args:
        q: Query tensor (D_DIM=512).
        k: Key tensor (D_DIM=512).
        v: Value tensor (D_DIM=512).
        si: Sparse indices tensor (int32), generated by lightning_indexer.
        q_rope: Query rope tensor.
        k_rope: Key rope tensor.
        actual_seq_lengths_query: Cumulative query lengths (TND only).
        actual_seq_lengths_kv: Cumulative kv lengths (TND only).
        layout: Layout string ('BSND' or 'TND').

    Returns:
        tuple: (attention_out, softmax_max, softmax_sum)
    """
    return sparse_flash_attention(
        q, k, v, si, SCALE_VALUE,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=q_rope, key_rope=k_rope,
        layout_query=layout, layout_kv=layout,
        attention_mode=2,
        return_softmax_lse=True,
    )


def _get_sparse_indices_bsnd(q_idx, k_idx, w, sparse_count):
    """Run lightning_indexer in BSND mode and return sparse_indices.

    Uses q_idx/k_idx with D_IDX=128 (lightning_indexer kernel constraint).
    sparse_mode defaults to 3 (rightDownCausal); for BSND+CP the CP impl
    truncates k_idx to the causal window before calling the kernel.

    Args:
        q_idx: Query index tensor (B, S1, N1, D_IDX=128).
        k_idx: Key index tensor (B, S2, N2, D_IDX=128).
        w: Weights tensor (B, S1, N1).
        sparse_count: Number of top-k blocks to select.

    Returns:
        sparse_indices tensor (int32), shape (B, S1, N2, sparse_count).
    """
    out = ops.lightning_indexer(q_idx, k_idx, w, sparse_count=sparse_count, return_value=True)
    return out[0]


def _get_sparse_indices_tnd(q_idx, k_idx, w, qlen, klen, sparse_count):
    """Run lightning_indexer in TND mode and return sparse_indices.

    Uses q_idx/k_idx with D_IDX=128 (lightning_indexer kernel constraint).

    Args:
        q_idx: Query index tensor (T1, N1, D_IDX=128).
        k_idx: Key index tensor (T2, N2, D_IDX=128).
        w: Weights tensor (T1, N1).
        qlen: Cumulative query sequence lengths (int32 Tensor).
        klen: Cumulative key sequence lengths (int32 Tensor).
        sparse_count: Number of top-k blocks to select.

    Returns:
        sparse_indices tensor (int32), shape (T1, N2, sparse_count).
    """
    out = ops.lightning_indexer(
        q_idx, k_idx, w,
        actual_seq_lengths_query=qlen,
        actual_seq_lengths_key=klen,
        layout_query='TND', layout_key='TND',
        sparse_count=sparse_count,
        return_value=True,
    )
    return out[0]


def _run_standalone(q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
                    actual_seq_q=None, actual_seq_kv=None, layout='BSND', sparse_count=None):
    """Run standalone lightning_indexer + SFA forward and backward.

    lightning_indexer(q_idx, k_idx, w) → si → sparse_flash_attention(q, k, v, si, ...).

    Args:
        q_np: SFA query numpy array (D_DIM=512).
        k_np: SFA key numpy array (D_DIM=512).
        v_np: SFA value numpy array (D_DIM=512).
        q_idx_np: Indexer query numpy array (D_IDX=128).
        k_idx_np: Indexer key numpy array (D_IDX=128).
        w_np: Indexer weights numpy array.
        q_rope_np: Query rope numpy array.
        k_rope_np: Key rope numpy array.
        actual_seq_q: Sequence lengths for query.
        actual_seq_kv: Sequence lengths for kv.
        layout: 'BSND' or 'TND'.
        sparse_count: Number of top-k blocks.

    Returns:
        tuple: (fwd_outs, grad_outs) — each a tuple of float32 ndarrays.
    """
    q = _np_to_bf16(q_np)
    k = _np_to_bf16(k_np)
    v = _np_to_bf16(v_np)
    q_idx = _np_to_bf16(q_idx_np)
    k_idx = _np_to_bf16(k_idx_np)
    w = _np_to_bf16(w_np)
    q_rope = _np_to_bf16(q_rope_np)
    k_rope = _np_to_bf16(k_rope_np)

    if layout == 'BSND':
        sc = sparse_count or SPARSE_COUNT_BSND
        si = _get_sparse_indices_bsnd(q_idx, k_idx, w, sc)
    else:
        sc = sparse_count or SPARSE_COUNT_TND
        si = _get_sparse_indices_tnd(q_idx, k_idx, w, actual_seq_q, actual_seq_kv, sc)

    out = _call_sfa(q, k, v, si, q_rope, k_rope, actual_seq_q, actual_seq_kv, layout)
    fwd_outs = tuple(_to_f32np(o) for o in out)

    grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        q, k, v, si, q_rope, k_rope, actual_seq_q, actual_seq_kv, layout
    )
    grad_outs = tuple(_to_f32np(g) for g in grads)
    return fwd_outs, grad_outs


def _get_grad_placements(d_inputs: tuple) -> list:
    """Derive the correct DTensor placement for each input's gradient.

    Args:
        d_inputs: Tuple of DTensor inputs.

    Returns:
        list of placement tuples, one per input.
    """
    n_dims = len(d_inputs[0].layout.placements)
    dim_has_shard = [
        any(isinstance(d_inp.layout.placements[d], Shard) for d_inp in d_inputs)
        for d in range(n_dims)
    ]
    result = []
    for d_inp in d_inputs:
        mesh_placements = d_inp.layout.placements
        grad_placements = tuple(
            Partial("sum") if isinstance(p, Replicate) and dim_has_shard[dim_idx] else p
            for dim_idx, p in enumerate(mesh_placements)
        )
        result.append(grad_placements)
    return result


def _assert_fwd(dist_outs, ref_outs, tag):
    """Assert all three forward outputs match reference within tolerance."""
    for i, (name, ref) in enumerate(zip(_FWD_OUTPUT_NAMES, ref_outs)):
        dist_np = _to_f32np(dist_outs[i].full_tensor())
        assert np.allclose(dist_np, ref, atol=1e-3, rtol=1e-3), (
            f"{tag} forward {name} mismatch: "
            f"max_diff={np.abs(dist_np - ref).max()}"
        )


_BWD_TOL = {
    'key': (1e-1, 1e-1),
    'key_rope': (1e-1, 1e-1),
    'value': (1e-2, 1e-2),
}
_BWD_TOL_DEFAULT = (1e-3, 1e-3)


def _assert_bwd(raw_grads, d_inputs, ref_grad, tag, tol=None):
    """Assert backward gradients match reference within tolerance.

    Args:
        tol: Optional per-input tolerance dict mapping input name to (atol, rtol).
             Pass _BWD_TOL for BSND+CP tests; omit for all other tests (uniform 1e-3).
    """
    grad_placements = _get_grad_placements(d_inputs)
    for name, raw_g, d_inp, ref, g_placements in zip(
            _INPUT_NAMES, raw_grads, d_inputs, ref_grad, grad_placements):
        dist_g = (
            DTensor.from_local(raw_g, d_inp.layout.mesh, g_placements)
            .full_tensor()
            .astype(ms.float32)
            .asnumpy()
        )
        atol, rtol = tol.get(name, _BWD_TOL_DEFAULT) if tol else _BWD_TOL_DEFAULT
        assert np.allclose(dist_g, ref, atol=atol, rtol=rtol), (
            f"{tag} backward grad_{name} mismatch: "
            f"max_diff={np.abs(dist_g - ref).max()}"
        )


def test_sfa_bsnd_replicated():
    """
    Feature: npu_sparse_flash_attention (MindSpore) BSND forward/backward, all replicated.
    Description:
        - All inputs replicated on 2-device mesh.
        - sparse_indices from lightning_indexer(q_idx, k_idx, w) with D_IDX=128.
        - SFA called with q/k/v of D_DIM=512 (MLA attention_mode=2).
        - bfloat16, attention_mode=2, v=k.clone().
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np = _generate_inputs(layout='BSND')
    ref_fwd, ref_grad = _run_standalone(
        q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
        sparse_count=SPARSE_COUNT_BSND,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(_np_to_bf16(q_np), mesh, (Replicate(),))
    dk = distribute_tensor(_np_to_bf16(k_np), mesh, (Replicate(),))
    dv = distribute_tensor(_np_to_bf16(v_np), mesh, (Replicate(),))
    dq_idx = distribute_tensor(_np_to_bf16(q_idx_np), mesh, (Replicate(),))
    dk_idx = distribute_tensor(_np_to_bf16(k_idx_np), mesh, (Replicate(),))
    dw = distribute_tensor(_np_to_bf16(w_np), mesh, (Replicate(),))
    dq_rope = distribute_tensor(_np_to_bf16(q_rope_np), mesh, (Replicate(),))
    dk_rope = distribute_tensor(_np_to_bf16(k_rope_np), mesh, (Replicate(),))

    dsi = _get_sparse_indices_bsnd(dq_idx, dk_idx, dw, SPARSE_COUNT_BSND)

    out = _call_sfa(dq, dk, dv, dsi, dq_rope, dk_rope)
    _assert_fwd(out, ref_fwd, "BSND Replicated")

    raw_grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        dq, dk, dv, dsi, dq_rope, dk_rope
    )
    _assert_bwd(raw_grads, (dq, dk, dv, dq_rope, dk_rope), ref_grad, "BSND Replicated")


def test_sfa_bsnd_dp():
    """
    Feature: npu_sparse_flash_attention (MindSpore) BSND forward/backward with B-dim DP.
    Description:
        - 2-device dp mesh; all inputs sharded on B (dim 0).
        - sparse_indices from distributed lightning_indexer(dq_idx, dk_idx, dw).
        - bfloat16, attention_mode=2, v=k.clone().
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np = _generate_inputs(layout='BSND')
    ref_fwd, ref_grad = _run_standalone(
        q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
        sparse_count=SPARSE_COUNT_BSND,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(_np_to_bf16(q_np), mesh, (Shard(0),))
    dk = distribute_tensor(_np_to_bf16(k_np), mesh, (Shard(0),))
    dv = distribute_tensor(_np_to_bf16(v_np), mesh, (Shard(0),))
    dq_idx = distribute_tensor(_np_to_bf16(q_idx_np), mesh, (Shard(0),))
    dk_idx = distribute_tensor(_np_to_bf16(k_idx_np), mesh, (Shard(0),))
    dw = distribute_tensor(_np_to_bf16(w_np), mesh, (Shard(0),))
    dq_rope = distribute_tensor(_np_to_bf16(q_rope_np), mesh, (Shard(0),))
    dk_rope = distribute_tensor(_np_to_bf16(k_rope_np), mesh, (Shard(0),))

    dsi = _get_sparse_indices_bsnd(dq_idx, dk_idx, dw, SPARSE_COUNT_BSND)

    out = _call_sfa(dq, dk, dv, dsi, dq_rope, dk_rope)
    _assert_fwd(out, ref_fwd, "BSND DP")

    raw_grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        dq, dk, dv, dsi, dq_rope, dk_rope
    )
    _assert_bwd(raw_grads, (dq, dk, dv, dq_rope, dk_rope), ref_grad, "BSND DP")


def test_sfa_bsnd_cp():
    """
    Feature: npu_sparse_flash_attention (MindSpore) BSND with S1-dim context parallelism.
    Description:
        - 2-device dp_cp mesh; q/q_idx/w/q_rope sharded on S1 (dim 1);
          k/k_idx/v/k_rope replicated.
        - lightning_indexer distributed op slices k_idx to causal window
          k_idx[:, :S1_local*(split_id+1), :, :] per rank (D_IDX=128).
        - sparse_flash_attention distributed op applies the same slice to k/v/key_rope
          (D_DIM=512), ensuring sparse_indices remain valid.
        - S1=S2 (self-attention) ensures causal truncation is lossless.
        - SPARSE_COUNT_CP <= S1_local = S1_BSND_CP // 2.
        - bfloat16, attention_mode=2, v=k.clone().
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np = _generate_inputs(
        layout='BSND', s1=S1_BSND_CP, s2=S2_BSND_CP, n1=N1_BSND_CP, b=B_BSND_CP
    )
    ref_fwd, ref_grad = _run_standalone(
        q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
        sparse_count=SPARSE_COUNT_CP,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp_cp",))
    dq = distribute_tensor(_np_to_bf16(q_np), mesh, (Shard(1),))
    dk = distribute_tensor(_np_to_bf16(k_np), mesh, (Replicate(),))
    dv = distribute_tensor(_np_to_bf16(v_np), mesh, (Replicate(),))
    dq_idx = distribute_tensor(_np_to_bf16(q_idx_np), mesh, (Shard(1),))
    dk_idx = distribute_tensor(_np_to_bf16(k_idx_np), mesh, (Replicate(),))
    dw = distribute_tensor(_np_to_bf16(w_np), mesh, (Shard(1),))
    dq_rope = distribute_tensor(_np_to_bf16(q_rope_np), mesh, (Shard(1),))
    dk_rope = distribute_tensor(_np_to_bf16(k_rope_np), mesh, (Replicate(),))

    dsi = _get_sparse_indices_bsnd(dq_idx, dk_idx, dw, SPARSE_COUNT_CP)

    out = _call_sfa(dq, dk, dv, dsi, dq_rope, dk_rope)
    _assert_fwd(out, ref_fwd, "BSND CP")

    raw_grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        dq, dk, dv, dsi, dq_rope, dk_rope
    )
    _assert_bwd(raw_grads, (dq, dk, dv, dq_rope, dk_rope), ref_grad, "BSND CP", tol=_BWD_TOL)


def test_sfa_bsnd_dp_cp():
    """
    Feature: npu_sparse_flash_attention (MindSpore) BSND with 2-D dp+cp mesh.
    Description:
        - 4-card 2-D mesh (dp=2, cp=2); q/q_idx/w/q_rope sharded on B (dp) and S1 (cp).
          k/k_idx/v/k_rope sharded on B (dp), S2 replicated.
        - Mirrors MindFormers dsa_attention.py BSND shard():
            q/si = layout("dp", "cp_tp", "None", "None")
            k/v  = layout("dp", "None", "None", "None")
        - lightning_indexer slices k_idx to causal window per (dp, cp) rank.
        - sparse_flash_attention applies the same slice to k/v/key_rope.
        - S1=S2 (self-attention) ensures causal truncation is lossless.
        - Each rank: B_local = B_BSND_CP//dp = 1, S1_local = S1_BSND_CP//cp = S1_BSND_CP//2.
        - bfloat16, attention_mode=2, v=k.clone().
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np = _generate_inputs(
        layout='BSND', s1=S1_BSND_CP, s2=S2_BSND_CP, n1=N1_BSND_CP, b=B_BSND_CP
    )
    ref_fwd, ref_grad = _run_standalone(
        q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
        sparse_count=SPARSE_COUNT_CP,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp"))
    dq = distribute_tensor(_np_to_bf16(q_np), mesh, (Shard(0), Shard(1)))
    dk = distribute_tensor(_np_to_bf16(k_np), mesh, (Shard(0), Replicate()))
    dv = distribute_tensor(_np_to_bf16(v_np), mesh, (Shard(0), Replicate()))
    dq_idx = distribute_tensor(_np_to_bf16(q_idx_np), mesh, (Shard(0), Shard(1)))
    dk_idx = distribute_tensor(_np_to_bf16(k_idx_np), mesh, (Shard(0), Replicate()))
    dw = distribute_tensor(_np_to_bf16(w_np), mesh, (Shard(0), Shard(1)))
    dq_rope = distribute_tensor(_np_to_bf16(q_rope_np), mesh, (Shard(0), Shard(1)))
    dk_rope = distribute_tensor(_np_to_bf16(k_rope_np), mesh, (Shard(0), Replicate()))

    dsi = _get_sparse_indices_bsnd(dq_idx, dk_idx, dw, SPARSE_COUNT_CP)

    out = _call_sfa(dq, dk, dv, dsi, dq_rope, dk_rope)
    _assert_fwd(out, ref_fwd, "BSND dp+cp")

    raw_grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        dq, dk, dv, dsi, dq_rope, dk_rope
    )
    _assert_bwd(raw_grads, (dq, dk, dv, dq_rope, dk_rope), ref_grad, "BSND dp+cp", tol=_BWD_TOL)


def test_sfa_tnd_replicated():
    """
    Feature: npu_sparse_flash_attention (MindSpore) TND forward/backward, all replicated.
    Description:
        - sparse_indices from lightning_indexer(q_idx, k_idx, w) with D_IDX=128.
        - SFA called with q/k/v of D_DIM=512.
        - actual_seq_lengths as int32 Tensor of shape (128,) with prefix sums.
        - bfloat16, attention_mode=2, v=k.clone().
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np = _generate_inputs(layout='TND')
    actual_seq_q, actual_seq_kv = _make_tnd_seq_lens()
    ref_fwd, ref_grad = _run_standalone(
        q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        layout='TND', sparse_count=SPARSE_COUNT_TND,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(_np_to_bf16(q_np), mesh, (Replicate(),))
    dk = distribute_tensor(_np_to_bf16(k_np), mesh, (Replicate(),))
    dv = distribute_tensor(_np_to_bf16(v_np), mesh, (Replicate(),))
    dq_idx = distribute_tensor(_np_to_bf16(q_idx_np), mesh, (Replicate(),))
    dk_idx = distribute_tensor(_np_to_bf16(k_idx_np), mesh, (Replicate(),))
    dw = distribute_tensor(_np_to_bf16(w_np), mesh, (Replicate(),))
    dq_rope = distribute_tensor(_np_to_bf16(q_rope_np), mesh, (Replicate(),))
    dk_rope = distribute_tensor(_np_to_bf16(k_rope_np), mesh, (Replicate(),))

    d_actual_seq_q = distribute_tensor(actual_seq_q, mesh, (Replicate(),))
    d_actual_seq_kv = distribute_tensor(actual_seq_kv, mesh, (Replicate(),))

    dsi = _get_sparse_indices_tnd(dq_idx, dk_idx, dw, d_actual_seq_q, d_actual_seq_kv, SPARSE_COUNT_TND)

    out = _call_sfa(dq, dk, dv, dsi, dq_rope, dk_rope, d_actual_seq_q, d_actual_seq_kv, 'TND')
    _assert_fwd(out, ref_fwd, "TND Replicated")

    raw_grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        dq, dk, dv, dsi, dq_rope, dk_rope, d_actual_seq_q, d_actual_seq_kv, 'TND'
    )
    _assert_bwd(raw_grads, (dq, dk, dv, dq_rope, dk_rope), ref_grad, "TND Replicated")


def test_sfa_tnd_dp():
    """
    Feature: npu_sparse_flash_attention (MindSpore) TND forward/backward with T1-dim DP.
    Description:
        - 2-device dp mesh; all tensors sharded on T1 (dim 0).
        - sparse_indices from distributed lightning_indexer (TND DP mode).
        - Both q and k sharded on T1 — pure DP, no CP seq_len adjustment.
        - bfloat16, attention_mode=2, v=k.clone().
    Expectation: Gathered distributed outputs match standalone within tolerance.
    """
    q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np = _generate_inputs(layout='TND')
    actual_seq_q, actual_seq_kv = _make_tnd_seq_lens()
    ref_fwd, ref_grad = _run_standalone(
        q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        layout='TND', sparse_count=SPARSE_COUNT_TND,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(_np_to_bf16(q_np), mesh, (Shard(0),))
    dk = distribute_tensor(_np_to_bf16(k_np), mesh, (Shard(0),))
    dv = distribute_tensor(_np_to_bf16(v_np), mesh, (Shard(0),))
    dq_idx = distribute_tensor(_np_to_bf16(q_idx_np), mesh, (Shard(0),))
    dk_idx = distribute_tensor(_np_to_bf16(k_idx_np), mesh, (Shard(0),))
    dw = distribute_tensor(_np_to_bf16(w_np), mesh, (Shard(0),))
    dq_rope = distribute_tensor(_np_to_bf16(q_rope_np), mesh, (Shard(0),))
    dk_rope = distribute_tensor(_np_to_bf16(k_rope_np), mesh, (Shard(0),))

    d_actual_seq_q = distribute_tensor(actual_seq_q, mesh, (Replicate(),))
    d_actual_seq_kv = distribute_tensor(actual_seq_kv, mesh, (Replicate(),))

    dsi = _get_sparse_indices_tnd(dq_idx, dk_idx, dw, d_actual_seq_q, d_actual_seq_kv, SPARSE_COUNT_TND)

    out = _call_sfa(dq, dk, dv, dsi, dq_rope, dk_rope, d_actual_seq_q, d_actual_seq_kv, 'TND')
    _assert_fwd(out, ref_fwd, "TND DP")

    raw_grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        dq, dk, dv, dsi, dq_rope, dk_rope, d_actual_seq_q, d_actual_seq_kv, 'TND'
    )
    _assert_bwd(raw_grads, (dq, dk, dv, dq_rope, dk_rope), ref_grad, "TND DP")


def test_sfa_tnd_cp():
    """
    Feature: npu_sparse_flash_attention (MindSpore) TND with context parallelism.
    Description:
        - 2-device dp_cp mesh; q/q_idx/w/q_rope sharded on T1; k/k_idx/v/k_rope replicated.
        - sparse_indices from distributed lightning_indexer (TND CP mode, D_IDX=128).
        - actual_seq_lengths adjusted per rank by _tnd_cp_impl in both ops.
        - bfloat16, attention_mode=2, v=k.clone().
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np = _generate_inputs(layout='TND')
    actual_seq_q, actual_seq_kv = _make_tnd_seq_lens()
    ref_fwd, ref_grad = _run_standalone(
        q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        layout='TND', sparse_count=SPARSE_COUNT_TND,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp_cp",))
    dq = distribute_tensor(_np_to_bf16(q_np), mesh, (Shard(0),))
    dk = distribute_tensor(_np_to_bf16(k_np), mesh, (Replicate(),))
    dv = distribute_tensor(_np_to_bf16(v_np), mesh, (Replicate(),))
    dq_idx = distribute_tensor(_np_to_bf16(q_idx_np), mesh, (Shard(0),))
    dk_idx = distribute_tensor(_np_to_bf16(k_idx_np), mesh, (Replicate(),))
    dw = distribute_tensor(_np_to_bf16(w_np), mesh, (Shard(0),))
    dq_rope = distribute_tensor(_np_to_bf16(q_rope_np), mesh, (Shard(0),))
    dk_rope = distribute_tensor(_np_to_bf16(k_rope_np), mesh, (Replicate(),))

    d_actual_seq_q = distribute_tensor(actual_seq_q, mesh, (Replicate(),))
    d_actual_seq_kv = distribute_tensor(actual_seq_kv, mesh, (Replicate(),))

    dsi = _get_sparse_indices_tnd(dq_idx, dk_idx, dw, d_actual_seq_q, d_actual_seq_kv, SPARSE_COUNT_TND)

    out = _call_sfa(dq, dk, dv, dsi, dq_rope, dk_rope, d_actual_seq_q, d_actual_seq_kv, 'TND')
    _assert_fwd(out, ref_fwd, "TND CP")

    raw_grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        dq, dk, dv, dsi, dq_rope, dk_rope, d_actual_seq_q, d_actual_seq_kv, 'TND'
    )
    _assert_bwd(raw_grads, (dq, dk, dv, dq_rope, dk_rope), ref_grad, "TND CP")


def test_sfa_tnd_dp_cp():
    """
    Feature: npu_sparse_flash_attention (MindSpore) TND with 2-D dp+cp mesh.
    Description:
        - 4-card 2-D mesh (dp=2, cp=2); T1 of q/q_idx/w/q_rope sharded by BOTH dp and cp.
          T2 of k/k_idx/v/k_rope sharded by dp only, cp Replicate.
        - Mirrors MindFormers dsa_attention.py TND shard():
            q/si = layout("dp_cp_tp", "None", "None")
            k/v  = layout("dp", "None", "None")
        - sparse_indices from distributed lightning_indexer (TND DP+CP mode, D_IDX=128).
        - _tnd_cp_impl adjusts actual_seq_lengths per rank in both ops.
        - bfloat16, attention_mode=2, v=k.clone().
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np = _generate_inputs(layout='TND')
    actual_seq_q, actual_seq_kv = _make_tnd_seq_lens()
    ref_fwd, ref_grad = _run_standalone(
        q_np, k_np, v_np, q_idx_np, k_idx_np, w_np, q_rope_np, k_rope_np,
        actual_seq_q=actual_seq_q, actual_seq_kv=actual_seq_kv,
        layout='TND', sparse_count=SPARSE_COUNT_TND,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp"))
    dq = distribute_tensor(_np_to_bf16(q_np), mesh, (Shard(0), Shard(0)))
    dk = distribute_tensor(_np_to_bf16(k_np), mesh, (Shard(0), Replicate()))
    dv = distribute_tensor(_np_to_bf16(v_np), mesh, (Shard(0), Replicate()))
    dq_idx = distribute_tensor(_np_to_bf16(q_idx_np), mesh, (Shard(0), Shard(0)))
    dk_idx = distribute_tensor(_np_to_bf16(k_idx_np), mesh, (Shard(0), Replicate()))
    dw = distribute_tensor(_np_to_bf16(w_np), mesh, (Shard(0), Shard(0)))
    dq_rope = distribute_tensor(_np_to_bf16(q_rope_np), mesh, (Shard(0), Shard(0)))
    dk_rope = distribute_tensor(_np_to_bf16(k_rope_np), mesh, (Shard(0), Replicate()))

    d_actual_seq_q = distribute_tensor(actual_seq_q, mesh, (Replicate(),))
    d_actual_seq_kv = distribute_tensor(actual_seq_kv, mesh, (Replicate(),))

    dsi = _get_sparse_indices_tnd(dq_idx, dk_idx, dw, d_actual_seq_q, d_actual_seq_kv, SPARSE_COUNT_TND)

    out = _call_sfa(dq, dk, dv, dsi, dq_rope, dk_rope, d_actual_seq_q, d_actual_seq_kv, 'TND')
    _assert_fwd(out, ref_fwd, "TND dp+cp")

    raw_grads = ms.grad(_call_sfa, (0, 1, 2, 4, 5))(
        dq, dk, dv, dsi, dq_rope, dk_rope, d_actual_seq_q, d_actual_seq_kv, 'TND'
    )
    _assert_bwd(raw_grads, (dq, dk, dv, dq_rope, dk_rope), ref_grad, "TND dp+cp")
