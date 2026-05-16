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
"""MindSpore ST for npu_sparse_lightning_indexer_grad_kl_loss custom op."""
import numpy as np
import mindspore as ms
from mindspore.ops.operations.nn_ops import FlashAttentionScore
import mindspore.communication.management as D
from mindspore import Tensor
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.custom_ops.experimental import npu_sparse_lightning_indexer_grad_kl_loss


def setup_module():
    """Initialize MindSpore context and communication."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


np.random.seed(42)
ms.set_seed(42)
ms.set_deterministic(True)

BATCH = 4
S1 = 128
S2 = 128
TOPK = 2048
N1 = 64
N1INDEX = 64
N2 = 1
HEAD_DIM = 512
DINDEX = 128
ROPE_DIM = 64
SCALE = 1.0 / (HEAD_DIM ** 0.5)


def _make_seq_len(batches, seq_per_batch):
    """Return cumulative seq_len list with shape (batches,), relative to first batch."""
    return [seq_per_batch * (i + 1) for i in range(batches)]


_GLOBAL_QLEN = _make_seq_len(BATCH, S1)
_GLOBAL_KLEN = _make_seq_len(BATCH, S2)

_FWD_NAMES = ("d_query_index", "d_key_index", "d_weights", "loss")
_BWD_NAMES = ("d_query_index", "d_key_index", "d_weights")


def _generate_inputs(layout="BSND"):
    """Generate random inputs for the op chain."""
    batch_seq1 = (BATCH, S1) if layout == "BSND" else (BATCH * S1,)
    batch_seq2 = (BATCH, S2) if layout == "BSND" else (BATCH * S2,)

    q_np = np.random.randn(*batch_seq1, N1, HEAD_DIM).astype(np.float16)
    k_np = np.random.randn(*batch_seq2, N2, HEAD_DIM).astype(np.float16)
    qi_np = np.random.randn(*batch_seq1, N1INDEX, DINDEX).astype(np.float16)
    ki_np = np.random.randn(*batch_seq2, N2, DINDEX).astype(np.float16)
    w_raw = np.random.randn(*batch_seq1, N1INDEX).astype(np.float16)
    a, b, kk = -0.05, 0.05, 3.0
    w_scale = (b - a) / (2 * kk)
    w_shift = (a + b) / 2
    w_np = (w_raw * w_scale + w_shift).astype(np.float16)
    qr_np = np.random.randn(*batch_seq1, N1, ROPE_DIM).astype(np.float16)
    kr_np = np.random.randn(*batch_seq2, N2, ROPE_DIM).astype(np.float16)

    if layout == "BSND":
        si_np = np.zeros((BATCH, S1, N2, TOPK), dtype=np.int32)
    else:
        si_np = np.zeros((BATCH * S1, N2, TOPK), dtype=np.int32)

    for s1_idx in range(S1):
        s2_real_size = int(S2 - S1 + s1_idx + 1)
        if s2_real_size <= 0:
            s2_real_size = S2
        s2_real_len = min(s2_real_size, TOPK)
        for b_idx in range(BATCH):
            if layout == "BSND":
                row = si_np
                row[b_idx, s1_idx, 0, :s2_real_len] = np.random.randint(
                    0, s2_real_size, size=s2_real_len, dtype=np.int32)
                row[b_idx, s1_idx, 0, s2_real_len:TOPK] = -1
            else:
                idx = b_idx * S1 + s1_idx
                row = si_np
                row[idx, 0, :s2_real_len] = np.random.randint(
                    0, s2_real_size, size=s2_real_len, dtype=np.int32)
                row[idx, 0, s2_real_len:TOPK] = -1
    return q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np


def _get_sm_stats(q_np, k_np, layout):
    """Compute FA softmax max and sum stats, returning shape appropriate for layout."""
    if layout == "TND":
        fa_q = q_np.reshape(BATCH, S1, N1, HEAD_DIM)
        fa_k = k_np.reshape(BATCH, S2, N2, HEAD_DIM)
    else:
        fa_q = q_np
        fa_k = k_np

    fa_op = FlashAttentionScore(head_num=N1, scale_value=SCALE, input_layout="BSND", sparse_mode=3)
    fa_out = fa_op(Tensor(fa_q), Tensor(fa_k), Tensor(np.zeros(fa_k.shape, np.float16)),
                   None, None, None, None)
    sm_max_np = fa_out[0].amax(-1, True).transpose(0, 3, 2, 1).asnumpy()
    sm_sum_np = fa_out[1].sum(dim=-1, keepdim=True).transpose(0, 3, 2, 1).asnumpy()

    if layout == "TND":
        sm_max_np = sm_max_np.transpose(1, 0, 2, 3).reshape(1, BATCH * S1, N1)
        sm_sum_np = sm_sum_np.transpose(1, 0, 2, 3).reshape(1, BATCH * S1, N1)
    return sm_max_np, sm_sum_np


def _foward_fn(q, k, qi, ki, w, si, sm_max, sm_sum,
               qr, kr, **kwargs):
    out = npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, SCALE,
        query_rope=qr, key_rope=kr,
        **kwargs,
    )
    return out[3].sum()




def _run_standalone(q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np,
                    sm_max_np, sm_sum_np, layout="BSND"):
    """Run forward and backward on full tensors.

    Returns:
        tuple: (fwd_out, grad_out) where fwd_out is a 4-tuple and grad_out is a 3-tuple.
    """
    q = Tensor(q_np)
    k = Tensor(k_np)
    qi = Tensor(qi_np)
    ki = Tensor(ki_np)
    w = Tensor(w_np)
    qr = Tensor(qr_np)
    kr = Tensor(kr_np)
    si = Tensor(si_np)
    sm_max = Tensor(sm_max_np)
    sm_sum = Tensor(sm_sum_np)

    actual_qlen = _GLOBAL_QLEN if layout == "TND" else None
    actual_klen = _GLOBAL_KLEN if layout == "TND" else None

    fwd_out = npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, SCALE,
        query_rope=qr, key_rope=kr,
        actual_seq_qlen=actual_qlen, actual_seq_klen=actual_klen, layout=layout,
    )

    if layout == "TND":
        grad = ms.grad(_foward_fn, (2, 3, 4))(
            q, k, qi, ki, w, si, sm_max, sm_sum,
            qr, kr,
            actual_seq_qlen=actual_qlen, actual_seq_klen=actual_klen, layout=layout,
        )
    else:
        grad = ms.grad(_foward_fn, (2, 3, 4))(
            q, k, qi, ki, w, si, sm_max, sm_sum,
            qr, kr,
        )

    fwd_arrays = tuple(o.asnumpy().astype(np.float32) for o in fwd_out)
    grad_arrays = tuple(g.asnumpy().astype(np.float32) for g in grad)
    return fwd_arrays, grad_arrays


def _get_grad_placements(d_inputs):
    """Derive the correct DTensor placement for each input's gradient."""
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


def _assert_fwd(results, ref_out, tag):
    """Assert all forward outputs match the reference within tolerance."""
    for i, (name, ref) in enumerate(zip(_FWD_NAMES, ref_out)):
        dist = results[i].full_tensor().asnumpy().astype(np.float32)
        assert np.allclose(dist, ref, atol=1e-3, rtol=1e-3), (
            f"{tag} forward {name} mismatch: max_diff={np.abs(dist - ref).max()}"
        )


def _assert_bwd(raw_grads, d_inputs, ref_grad, tag):
    """Assert all backward gradients match the reference within tolerance."""
    grad_placements = _get_grad_placements(d_inputs)
    for name, raw_g, d_inp, ref, g_placements in zip(
            _BWD_NAMES, raw_grads, d_inputs, ref_grad, grad_placements):
        dist_g = (
            DTensor.from_local(raw_g, d_inp.layout.mesh, g_placements)
            .full_tensor()
            .asnumpy()
            .astype(np.float32)
        )
        assert np.allclose(dist_g, ref, atol=1e-3, rtol=1e-3), (
            f"{tag} backward {name} mismatch: max_diff={np.abs(dist_g - ref).max()}"
        )




def test_sparse_grad_kl_loss_bsnd_replicated():
    """
    Feature: npu_sparse_lightning_indexer_grad_kl_loss BSND replicated on 2-device mesh.
    Description:
        - All inputs replicated; forward output and backward gradients
          compared against standalone single-card results.
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np = _generate_inputs(layout="BSND")
    sm_max_np, sm_sum_np = _get_sm_stats(q_np, k_np, layout="BSND")
    ref_out, ref_grad = _run_standalone(q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np,
                                        sm_max_np, sm_sum_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    q = distribute_tensor(Tensor(q_np), mesh, (Replicate(),))
    k = distribute_tensor(Tensor(k_np), mesh, (Replicate(),))
    qi = distribute_tensor(Tensor(qi_np), mesh, (Replicate(),))
    ki = distribute_tensor(Tensor(ki_np), mesh, (Replicate(),))
    w = distribute_tensor(Tensor(w_np), mesh, (Replicate(),))
    si = distribute_tensor(Tensor(si_np), mesh, (Replicate(),))
    sm_max = distribute_tensor(Tensor(sm_max_np), mesh, (Replicate(),))
    sm_sum = distribute_tensor(Tensor(sm_sum_np), mesh, (Replicate(),))
    qr = distribute_tensor(Tensor(qr_np), mesh, (Replicate(),))
    kr = distribute_tensor(Tensor(kr_np), mesh, (Replicate(),))

    results = npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, SCALE,
        query_rope=qr, key_rope=kr,
    )
    _assert_fwd(results, ref_out, "BSND/Replicated")

    raw_grads = ms.grad(_foward_fn, (2, 3, 4))(
        q, k, qi, ki, w, si, sm_max, sm_sum,
        qr, kr,
    )
    _assert_bwd(raw_grads, (qi, ki, w), ref_grad, "BSND/Replicated")


def test_sparse_grad_kl_loss_bsnd_dp():
    """
    Feature: npu_sparse_lightning_indexer_grad_kl_loss BSND with batch-dimension DP.
    Description:
        - 1-D dp mesh; forward output and backward gradients compared against standalone.
    Expectation: full_tensor() outputs match standalone within tolerance.
    """
    q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np = _generate_inputs(layout="BSND")
    sm_max_np, sm_sum_np = _get_sm_stats(q_np, k_np, layout="BSND")
    ref_out, ref_grad = _run_standalone(q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np,
                                        sm_max_np, sm_sum_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    q = distribute_tensor(Tensor(q_np), mesh, (Shard(0),))
    k = distribute_tensor(Tensor(k_np), mesh, (Shard(0),))
    qi = distribute_tensor(Tensor(qi_np), mesh, (Shard(0),))
    ki = distribute_tensor(Tensor(ki_np), mesh, (Shard(0),))
    w = distribute_tensor(Tensor(w_np), mesh, (Shard(0),))
    si = distribute_tensor(Tensor(si_np), mesh, (Shard(0),))
    sm_max = distribute_tensor(Tensor(sm_max_np), mesh, (Shard(0),))
    sm_sum = distribute_tensor(Tensor(sm_sum_np), mesh, (Shard(0),))
    qr = distribute_tensor(Tensor(qr_np), mesh, (Shard(0),))
    kr = distribute_tensor(Tensor(kr_np), mesh, (Shard(0),))

    results = npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, SCALE,
        query_rope=qr, key_rope=kr,
    )
    _assert_fwd(results, ref_out, "BSND/DP")

    raw_grads = ms.grad(_foward_fn, (2, 3, 4))(
        q, k, qi, ki, w, si, sm_max, sm_sum,
        qr, kr,
    )
    _assert_bwd(raw_grads, (qi, ki, w), ref_grad, "BSND/DP")


def test_sparse_grad_kl_loss_bsnd_dp_cp():
    """
    Feature: npu_sparse_lightning_indexer_grad_kl_loss BSND with dp=2, cp=2.
    Description:
        - 2-D mesh (2, 2); forward output and backward gradients compared against standalone.
    Expectation: full_tensor() outputs match standalone within tolerance.
    """
    q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np = _generate_inputs(layout="BSND")
    sm_max_np, sm_sum_np = _get_sm_stats(q_np, k_np, layout="BSND")
    ref_out, ref_grad = _run_standalone(q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np,
                                        sm_max_np, sm_sum_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp"))
    q = distribute_tensor(Tensor(q_np), mesh, (Shard(0), Shard(1)))
    k = distribute_tensor(Tensor(k_np), mesh, (Shard(0), Replicate()))
    qi = distribute_tensor(Tensor(qi_np), mesh, (Shard(0), Shard(1)))
    ki = distribute_tensor(Tensor(ki_np), mesh, (Shard(0), Replicate()))
    w = distribute_tensor(Tensor(w_np), mesh, (Shard(0), Shard(1)))
    si = distribute_tensor(Tensor(si_np), mesh, (Shard(0), Shard(1)))
    sm_max = distribute_tensor(Tensor(sm_max_np), mesh, (Shard(0), Shard(2)))
    sm_sum = distribute_tensor(Tensor(sm_sum_np), mesh, (Shard(0), Shard(2)))
    qr = distribute_tensor(Tensor(qr_np), mesh, (Shard(0), Shard(1)))
    kr = distribute_tensor(Tensor(kr_np), mesh, (Shard(0), Replicate()))

    results = npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, SCALE,
        query_rope=qr, key_rope=kr, sparse_mode=3,
    )
    _assert_fwd(results, ref_out, "BSND/DP+CP")

    raw_grads = ms.grad(_foward_fn, (2, 3, 4))(
        q, k, qi, ki, w, si, sm_max, sm_sum,
        qr, kr, sparse_mode=3,
    )
    _assert_bwd(raw_grads, (qi, ki, w), ref_grad, "BSND/DP+CP")


def test_sparse_grad_kl_loss_tnd_dp():
    """
    Feature: npu_sparse_lightning_indexer_grad_kl_loss TND with T-dimension DP.
    Description:
        - 1-D dp mesh; forward output and backward gradients compared against standalone TND.
    Expectation: full_tensor() outputs match standalone TND within tolerance.
    """
    q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np = _generate_inputs(layout="TND")
    sm_max_np, sm_sum_np = _get_sm_stats(q_np, k_np, layout="TND")
    ref_out, ref_grad = _run_standalone(q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np,
                                        sm_max_np, sm_sum_np, layout="TND")

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    q = distribute_tensor(Tensor(q_np), mesh, (Shard(0),))
    k = distribute_tensor(Tensor(k_np), mesh, (Shard(0),))
    qi = distribute_tensor(Tensor(qi_np), mesh, (Shard(0),))
    ki = distribute_tensor(Tensor(ki_np), mesh, (Shard(0),))
    w = distribute_tensor(Tensor(w_np), mesh, (Shard(0),))
    si = distribute_tensor(Tensor(si_np), mesh, (Shard(0),))
    sm_max = distribute_tensor(Tensor(sm_max_np), mesh, (Shard(1),))
    sm_sum = distribute_tensor(Tensor(sm_sum_np), mesh, (Shard(1),))
    qr = distribute_tensor(Tensor(qr_np), mesh, (Shard(0),))
    kr = distribute_tensor(Tensor(kr_np), mesh, (Shard(0),))

    dqlen = distribute_tensor(Tensor(_GLOBAL_QLEN, dtype=ms.int32), mesh, (Replicate(),))
    dklen = distribute_tensor(Tensor(_GLOBAL_KLEN, dtype=ms.int32), mesh, (Replicate(),))

    results = npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, SCALE,
        query_rope=qr, key_rope=kr,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
    )
    _assert_fwd(results, ref_out, "TND/DP")

    raw_grads = ms.grad(_foward_fn, (2, 3, 4))(
        q, k, qi, ki, w, si, sm_max, sm_sum,
        qr, kr,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
    )
    _assert_bwd(raw_grads, (qi, ki, w), ref_grad, "TND/DP")


def test_sparse_grad_kl_loss_tnd_dp_cp():
    """
    Feature: npu_sparse_lightning_indexer_grad_kl_loss TND with dp=2, cp=2.
    Description:
        - 2-D mesh (2, 2); forward output and backward gradients compared against standalone TND.
    Expectation: full_tensor() outputs match standalone TND within tolerance.
    """
    q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np = _generate_inputs(layout="TND")
    sm_max_np, sm_sum_np = _get_sm_stats(q_np, k_np, layout="TND")
    ref_out, ref_grad = _run_standalone(q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np,
                                        sm_max_np, sm_sum_np, layout="TND")

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp"))
    q = distribute_tensor(Tensor(q_np), mesh, (Shard(0), Shard(0)))
    k = distribute_tensor(Tensor(k_np), mesh, (Shard(0), Replicate()))
    qi = distribute_tensor(Tensor(qi_np), mesh, (Shard(0), Shard(0)))
    ki = distribute_tensor(Tensor(ki_np), mesh, (Shard(0), Replicate()))
    w = distribute_tensor(Tensor(w_np), mesh, (Shard(0), Shard(0)))
    si = distribute_tensor(Tensor(si_np), mesh, (Shard(0), Shard(0)))
    sm_max = distribute_tensor(Tensor(sm_max_np), mesh, (Shard(1), Shard(1)))
    sm_sum = distribute_tensor(Tensor(sm_sum_np), mesh, (Shard(1), Shard(1)))
    qr = distribute_tensor(Tensor(qr_np), mesh, (Shard(0), Shard(0)))
    kr = distribute_tensor(Tensor(kr_np), mesh, (Shard(0), Replicate()))

    dqlen = distribute_tensor(Tensor(_GLOBAL_QLEN, dtype=ms.int32), mesh, (Replicate(),))
    dklen = distribute_tensor(Tensor(_GLOBAL_KLEN, dtype=ms.int32), mesh, (Replicate(),))

    results = npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, SCALE,
        query_rope=qr, key_rope=kr,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
        sparse_mode=3,
    )
    _assert_fwd(results, ref_out, "TND/DP+CP")

    raw_grads = ms.grad(_foward_fn, (2, 3, 4))(
        q, k, qi, ki, w, si, sm_max, sm_sum,
        qr, kr,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
        sparse_mode=3,
    )
    _assert_bwd(raw_grads, (qi, ki, w), ref_grad, "TND/DP+CP")




def test_sparse_grad_kl_loss_tnd_replicated():
    """
    Feature: npu_sparse_lightning_indexer_grad_kl_loss TND replicated on 2-device mesh.
    Description:
        - All TND inputs replicated; forward output and backward gradients
          compared against standalone TND.
    Expectation: Distributed outputs match standalone within tolerance.
    """
    q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np = _generate_inputs(layout="TND")
    sm_max_np, sm_sum_np = _get_sm_stats(q_np, k_np, layout="TND")
    ref_out, ref_grad = _run_standalone(q_np, k_np, qi_np, ki_np, w_np, qr_np, kr_np, si_np,
                                        sm_max_np, sm_sum_np, layout="TND")

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    q = distribute_tensor(Tensor(q_np), mesh, (Replicate(),))
    k = distribute_tensor(Tensor(k_np), mesh, (Replicate(),))
    qi = distribute_tensor(Tensor(qi_np), mesh, (Replicate(),))
    ki = distribute_tensor(Tensor(ki_np), mesh, (Replicate(),))
    w = distribute_tensor(Tensor(w_np), mesh, (Replicate(),))
    si = distribute_tensor(Tensor(si_np), mesh, (Replicate(),))
    sm_max = distribute_tensor(Tensor(sm_max_np), mesh, (Replicate(),))
    sm_sum = distribute_tensor(Tensor(sm_sum_np), mesh, (Replicate(),))
    qr = distribute_tensor(Tensor(qr_np), mesh, (Replicate(),))
    kr = distribute_tensor(Tensor(kr_np), mesh, (Replicate(),))

    dqlen = distribute_tensor(Tensor(_GLOBAL_QLEN, dtype=ms.int32), mesh, (Replicate(),))
    dklen = distribute_tensor(Tensor(_GLOBAL_KLEN, dtype=ms.int32), mesh, (Replicate(),))

    results = npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, SCALE,
        query_rope=qr, key_rope=kr,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
    )
    _assert_fwd(results, ref_out, "TND/Replicated")

    raw_grads = ms.grad(_foward_fn, (2, 3, 4))(
        q, k, qi, ki, w, si, sm_max, sm_sum,
        qr, kr,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
    )
    _assert_bwd(raw_grads, (qi, ki, w), ref_grad, "TND/Replicated")
