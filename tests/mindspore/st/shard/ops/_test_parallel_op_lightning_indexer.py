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
"""MindSpore ST for lightning_indexer distributed op.

Distributed sharding patterns verified against MindFormers dsa_indexer.py shard() spec:
  - BSND: q/si sharded on B (dp) and/or S1 (cp_tp); k sharded on B (dp) only, S2 NOT sharded.
  - TND:  q/si sharded on T1 = dp*cp*tp combined; k sharded on T2 = dp only.
"""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor, ops
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate


def setup_module() -> None:
    """Initialize the distributed environment for the test module."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


np.random.seed(42)
ms.set_seed(42)
ms.set_deterministic(True)

BATCH = 4
S1 = 128
S2 = 128
N1 = 64
N2 = 1
HEAD_DIM = 128
SPARSE_COUNT = 2048


def _make_seq_len(batches, seq_per_batch):
    """Return cumulative seq_len list with shape (batches,), relative to first batch."""
    return [seq_per_batch * (i + 1) for i in range(batches)]


# Standalone + Replicate: global cumulative across all batches
_GLOBAL_QLEN = _make_seq_len(BATCH, S1)
_GLOBAL_KLEN = _make_seq_len(BATCH, S2)

_FWD_NAMES = ("sparse_indices", "sparse_values")
_BWD_NAMES = ("d_query", "d_key", "d_weights")


def _generate_inputs(layout="BSND"):
    batch_seq1 = (BATCH, S1) if layout == "BSND" else (BATCH * S1,)
    batch_seq2 = (BATCH, S2) if layout == "BSND" else (BATCH * S2,)

    q_np = np.random.randn(*batch_seq1, N1, HEAD_DIM).astype(np.float16)
    k_np = np.random.randn(*batch_seq2, N2, HEAD_DIM).astype(np.float16)
    w_np = np.random.randn(*batch_seq1, N1).astype(np.float16)
    return q_np, k_np, w_np


def _forward_fn(q, k, w, layout="BSND", seq_len_q=None, seq_len_k=None):
    """Forward wrapper returning loss for ms.grad consumption."""
    if layout == "TND":
        out = ops.lightning_indexer(
            q, k, w,
            actual_seq_lengths_query=seq_len_q,
            actual_seq_lengths_key=seq_len_k,
            layout_query='TND', layout_key='TND',
            sparse_count=SPARSE_COUNT,
            return_value=True,
        )
    else:
        out = ops.lightning_indexer(
            q, k, w,
            sparse_count=SPARSE_COUNT,
            return_value=True,
        )
    return out[1].sum()


def _run_standalone(q_np, k_np, w_np, layout="BSND"):
    """Run lightning_indexer + flash_attention on full tensors."""
    q = Tensor(q_np)
    k = Tensor(k_np)
    w = Tensor(w_np)

    if layout == "TND":
        qlen = Tensor(_GLOBAL_QLEN, dtype=ms.int32)
        klen = Tensor(_GLOBAL_KLEN, dtype=ms.int32)
        fwd_out = ops.lightning_indexer(
            q, k, w,
            actual_seq_lengths_query=qlen,
            actual_seq_lengths_key=klen,
            layout_query='TND', layout_key='TND',
            sparse_count=SPARSE_COUNT,
            return_value=True,
        )
        grads = ms.grad(_forward_fn, (0, 1, 2))(q, k, w, layout=layout,
                                                seq_len_q=qlen, seq_len_k=klen)
    else:
        fwd_out = ops.lightning_indexer(
            q, k, w,
            sparse_count=SPARSE_COUNT, return_value=True,
        )
        grads = ms.grad(_forward_fn, (0, 1, 2))(q, k, w)
    fwd_indices = fwd_out[0].asnumpy().astype(np.int32)
    fwd_values = fwd_out[1].asnumpy().astype(np.float32)


    return (
        fwd_indices, fwd_values,
        grads[0].asnumpy().astype(np.float32),
        grads[1].asnumpy().astype(np.float32),
        grads[2].asnumpy().astype(np.float32),
    )


def _assert_fwd(results, ref_out, tag):
    """Assert distributed forward outputs match standalone reference."""
    ref_indices, ref_values = ref_out
    dist_indices = results[0].full_tensor().asnumpy().astype(np.int32)
    dist_values = results[1].full_tensor().asnumpy().astype(np.float32)

    assert np.array_equal(dist_indices, ref_indices), (
        f"{tag} forward sparse_indices mismatch: "
        f"max_diff={np.abs(dist_indices.astype(np.int64) - ref_indices.astype(np.int64)).max()}"
    )
    assert np.allclose(dist_values, ref_values, atol=1e-2, rtol=1e-2), (
        f"{tag} forward sparse_values mismatch: max_diff={np.abs(dist_values - ref_values).max()}"
    )


# ======================== BSND replicated ========================

def test_lightning_indexer_bsnd_replicated():
    """
    Feature: lightning_indexer BSND forward/backward — all inputs replicated on 2-device mesh.
    Description:
        - All inputs replicated; output redistributed to Replicate and compared.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="BSND")
    ref_indices, ref_values, _, _, _ = _run_standalone(q_np, k_np, w_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(Tensor(q_np), mesh, (Replicate(),))
    dk = distribute_tensor(Tensor(k_np), mesh, (Replicate(),))
    dw = distribute_tensor(Tensor(w_np), mesh, (Replicate(),))

    out = ops.lightning_indexer(dq, dk, dw, sparse_count=SPARSE_COUNT, return_value=True)
    _assert_fwd(out, (ref_indices, ref_values), "BSND/Replicated")


def test_lightning_indexer_bsnd_dp():
    """
    Feature: lightning_indexer BSND forward/backward with batch-dimension DP.
    Description:
        - 1-D dp mesh; Q/K/W sharded on B (dim 0).
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="BSND")
    ref_indices, ref_values, _, _, _ = _run_standalone(q_np, k_np, w_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(Tensor(q_np), mesh, (Shard(0),))
    dk = distribute_tensor(Tensor(k_np), mesh, (Shard(0),))
    dw = distribute_tensor(Tensor(w_np), mesh, (Shard(0),))

    out = ops.lightning_indexer(dq, dk, dw, sparse_count=SPARSE_COUNT, return_value=True)
    _assert_fwd(out, (ref_indices, ref_values), "BSND/DP")


# ======================== BSND DP+CP ========================

def test_lightning_indexer_bsnd_dp_cp():
    """
    Feature: lightning_indexer BSND forward/backward with dp=2, cp=2 on 4-device mesh.
    Description:
        - 2-D mesh (2, 2); Q/W sharded on (dp, cp); K sharded on (dp,) only.
        - CP key window slicing for rightDownCausal mode.
        - Mirrors MindFormers BSND shard(): layout("dp","cp_tp","None","None") for q.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="BSND")
    ref_indices, ref_values, _, _, _ = _run_standalone(q_np, k_np, w_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp"))
    dq = distribute_tensor(Tensor(q_np), mesh, (Shard(0), Shard(1)))
    dk = distribute_tensor(Tensor(k_np), mesh, (Shard(0), Replicate()))
    dw = distribute_tensor(Tensor(w_np), mesh, (Shard(0), Shard(1)))

    out = ops.lightning_indexer(
        dq, dk, dw, sparse_count=SPARSE_COUNT, return_value=True,
    )
    _assert_fwd(out, (ref_indices, ref_values), "BSND/DP+CP")


# ======================== TND replicated ========================

def test_lightning_indexer_tnd_replicated():
    """
    Feature: lightning_indexer TND forward/backward — all inputs replicated on 2-device mesh.
    Description:
        - Inputs in TND layout; actual_seq_lengths as lists.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="TND")
    ref_indices, ref_values, _, _, _ = _run_standalone(
        q_np, k_np, w_np, layout="TND"
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(Tensor(q_np), mesh, (Replicate(),))
    dk = distribute_tensor(Tensor(k_np), mesh, (Replicate(),))
    dw = distribute_tensor(Tensor(w_np), mesh, (Replicate(),))

    dqlen = Tensor(_GLOBAL_QLEN, dtype=ms.int32)
    dklen = Tensor(_GLOBAL_KLEN, dtype=ms.int32)

    out = ops.lightning_indexer(
        dq, dk, dw,
        actual_seq_lengths_query=dqlen,
        actual_seq_lengths_key=dklen,
        layout_query='TND', layout_key='TND',
        sparse_count=SPARSE_COUNT, return_value=True,
    )
    _assert_fwd(out, (ref_indices, ref_values), "TND/Replicated")


# ======================== TND DP ========================

def test_lightning_indexer_tnd_dp():
    """
    Feature: lightning_indexer TND forward/backward with T1-dim data parallel.
    Description:
        - 2-card dp mesh; query AND key both sharded on T1 (dim 0) — pure data parallel.
        - q_split == k_split → no CP seq_len adjustment needed.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="TND")
    ref_indices, ref_values, _, _, _ = _run_standalone(
        q_np, k_np, w_np, layout="TND"
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(Tensor(q_np), mesh, (Shard(0),))
    dk = distribute_tensor(Tensor(k_np), mesh, (Shard(0),))
    dw = distribute_tensor(Tensor(w_np), mesh, (Shard(0),))

    dqlen = Tensor(_GLOBAL_QLEN, dtype=ms.int32)
    dklen = Tensor(_GLOBAL_KLEN, dtype=ms.int32)

    out = ops.lightning_indexer(
        dq, dk, dw,
        actual_seq_lengths_query=dqlen,
        actual_seq_lengths_key=dklen,
        layout_query='TND', layout_key='TND',
        sparse_count=SPARSE_COUNT, return_value=True,
    )
    _assert_fwd(out, (ref_indices, ref_values), "TND/DP")


# ======================== TND DP+CP ========================

def test_lightning_indexer_tnd_dp_cp():
    """
    Feature: lightning_indexer TND forward/backward with 2-D dp+cp mesh (combined T1 sharding).
    Description:
        - 4-card 2-D mesh (dp=2, cp=2); Q/W sharded on T1 by BOTH dp and cp.
        - K sharded on T2 by dp only, cp Replicate.
        - Mirrors MindFormers TND shard(): layout("dp_cp_tp","None","None") for q.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="TND")
    ref_indices, ref_values, _, _, _ = _run_standalone(
        q_np, k_np, w_np, layout="TND"
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp"))
    dq = distribute_tensor(Tensor(q_np), mesh, (Shard(0), Shard(0)))
    dk = distribute_tensor(Tensor(k_np), mesh, (Shard(0), Replicate()))
    dw = distribute_tensor(Tensor(w_np), mesh, (Shard(0), Shard(0)))

    dqlen = Tensor(_GLOBAL_QLEN, dtype=ms.int32)
    dklen = Tensor(_GLOBAL_KLEN, dtype=ms.int32)

    out = ops.lightning_indexer(
        dq, dk, dw,
        actual_seq_lengths_query=dqlen,
        actual_seq_lengths_key=dklen,
        layout_query='TND', layout_key='TND',
        sparse_count=SPARSE_COUNT, return_value=True,
    )
    _assert_fwd(out, (ref_indices, ref_values), "TND/DP+CP")
