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
"""MindFormers ST for npu_dense_lightning_indexer_softmax_lse custom op."""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.custom_ops.experimental import npu_dense_lightning_indexer_softmax_lse


def setup_module() -> None:
    """Initialize MindSpore context and communication."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


np.random.seed(42)
ms.set_seed(42)
ms.set_deterministic(True)

BATCH = 4
S1 = 128
S2 = 128
N1INDEX = 32
N2INDEX = 1
HEAD_DIM = 128

def _make_seq_len(batches, seq_per_batch):
    """Return cumulative seq_len list with shape (batches,), relative to first batch."""
    return [seq_per_batch * (i + 1) for i in range(batches)]


_GLOBAL_QLEN = _make_seq_len(BATCH, S1)
_GLOBAL_KLEN = _make_seq_len(BATCH, S2)


def _generate_inputs(layout="BSND"):
    """Generate random inputs for npu_dense_lightning_indexer_softmax_lse."""
    batch_seq1 = (BATCH, S1) if layout == "BSND" else (BATCH * S1,)
    batch_seq2 = (BATCH, S2) if layout == "BSND" else (BATCH * S2,)

    q_np = np.random.randn(*batch_seq1, N1INDEX, HEAD_DIM).astype(np.float16)
    k_np = np.random.randn(*batch_seq2, N2INDEX, HEAD_DIM).astype(np.float16)
    w_np = np.random.randn(*batch_seq1, N1INDEX).astype(np.float16)
    return q_np, k_np, w_np



def _run_standalone(q_np, k_np, w_np, layout="BSND"):
    """Run npu_dense_lightning_indexer_softmax_lse on full tensors."""
    q = Tensor(q_np)
    k = Tensor(k_np)
    w = Tensor(w_np)
    if layout == "TND":
        _qlen = Tensor(_GLOBAL_QLEN, dtype=ms.int32)  # pylint: disable=invalid-name
        _klen = Tensor(_GLOBAL_KLEN, dtype=ms.int32)  # pylint: disable=invalid-name
        out = npu_dense_lightning_indexer_softmax_lse(
            q, k, w,
            actual_seq_qlen=_qlen, actual_seq_klen=_klen, layout='TND',
        )
    else:
        out = npu_dense_lightning_indexer_softmax_lse(q, k, w)
    return out[0].asnumpy().astype(np.float32), out[1].asnumpy().astype(np.float32)

def test_softmax_lse_bsnd_replicated():
    """
    Feature: npu_dense_lightning_indexer_softmax_lse BSND replicated on 2-device mesh.
    Description:
        - All inputs replicated; output should match standalone single-card result.
    Expectation: Distributed output matches standalone within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="BSND")
    ref_max, ref_sum = _run_standalone(q_np, k_np, w_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(Tensor(q_np), mesh, (Replicate(),))
    dk = distribute_tensor(Tensor(k_np), mesh, (Replicate(),))
    dw = distribute_tensor(Tensor(w_np), mesh, (Replicate(),))

    out = npu_dense_lightning_indexer_softmax_lse(dq, dk, dw)
    dist_max = out[0].full_tensor().asnumpy().astype(np.float32)
    dist_sum = out[1].full_tensor().asnumpy().astype(np.float32)

    assert np.allclose(dist_max, ref_max, atol=1e-3, rtol=1e-3), (
        f"Replicated softmax_max mismatch: max_diff={np.abs(dist_max - ref_max).max()}"
    )
    assert np.allclose(dist_sum, ref_sum, atol=1e-3, rtol=1e-3), (
        f"Replicated softmax_sum mismatch: max_diff={np.abs(dist_sum - ref_sum).max()}"
    )


def test_softmax_lse_bsnd_dp():
    """
    Feature: npu_dense_lightning_indexer_softmax_lse BSND with batch-dimension DP.
    Description:
        - 1-D dp mesh with B sharded; output (B, N2index, S1) with B sharded.
    Expectation: full_tensor() output matches standalone within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="BSND")
    ref_max, ref_sum = _run_standalone(q_np, k_np, w_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(Tensor(q_np), mesh, (Shard(0),))
    dk = distribute_tensor(Tensor(k_np), mesh, (Shard(0),))
    dw = distribute_tensor(Tensor(w_np), mesh, (Shard(0),))

    out = npu_dense_lightning_indexer_softmax_lse(dq, dk, dw)
    dist_max = out[0].full_tensor().asnumpy().astype(np.float32)
    dist_sum = out[1].full_tensor().asnumpy().astype(np.float32)

    assert np.allclose(dist_max, ref_max, atol=1e-3, rtol=1e-3), (
        f"DP softmax_max mismatch: max_diff={np.abs(dist_max - ref_max).max()}"
    )
    assert np.allclose(dist_sum, ref_sum, atol=1e-3, rtol=1e-3), (
        f"DP softmax_sum mismatch: max_diff={np.abs(dist_sum - ref_sum).max()}"
    )


def test_softmax_lse_bsnd_dp_cp():
    """
    Feature: npu_dense_lightning_indexer_softmax_lse BSND with dp=2, cp=2.
    Description:
        - 2-D mesh (2, 2) with dp on B and cp on S1.
    Expectation: full_tensor() output matches standalone within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="BSND")
    ref_max, ref_sum = _run_standalone(q_np, k_np, w_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp"))
    dq = distribute_tensor(Tensor(q_np), mesh, (Shard(0), Shard(1)))
    dk = distribute_tensor(Tensor(k_np), mesh, (Shard(0), Replicate()))
    dw = distribute_tensor(Tensor(w_np), mesh, (Shard(0), Shard(1)))

    out = npu_dense_lightning_indexer_softmax_lse(dq, dk, dw, sparse_mode=3)
    dist_max = out[0].full_tensor().asnumpy().astype(np.float32)
    dist_sum = out[1].full_tensor().asnumpy().astype(np.float32)

    assert np.allclose(dist_max, ref_max, atol=1e-3, rtol=1e-3), (
        f"DP+CP softmax_max mismatch: max_diff={np.abs(dist_max - ref_max).max()}"
    )
    assert np.allclose(dist_sum, ref_sum, atol=1e-3, rtol=1e-3), (
        f"DP+CP softmax_sum mismatch: max_diff={np.abs(dist_sum - ref_sum).max()}"
    )


def test_softmax_lse_tnd_dp():
    """
    Feature: npu_dense_lightning_indexer_softmax_lse TND with T-dimension DP.
    Description:
        - 1-D dp mesh; query/weights sharded on T1 (dim 0); key sharded on T2 (dim 0).
        - actual_seq_qlen/klen wrapped as DTensor with Shard(0) along dp.
    Expectation: full_tensor() output matches standalone within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="TND")
    ref_max, ref_sum = _run_standalone(q_np, k_np, w_np, layout="TND")

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(Tensor(q_np), mesh, (Shard(0),))
    dk = distribute_tensor(Tensor(k_np), mesh, (Shard(0),))
    dw = distribute_tensor(Tensor(w_np), mesh, (Shard(0),))

    dqlen = Tensor(_GLOBAL_QLEN, dtype=ms.int32)
    dklen = Tensor(_GLOBAL_KLEN, dtype=ms.int32)

    out = npu_dense_lightning_indexer_softmax_lse(
        dq, dk, dw,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
    )
    dist_max = out[0].full_tensor().asnumpy().astype(np.float32)
    dist_sum = out[1].full_tensor().asnumpy().astype(np.float32)

    assert np.allclose(dist_max, ref_max, atol=1e-3, rtol=1e-3), (
        f"TND DP softmax_max mismatch: max_diff={np.abs(dist_max - ref_max).max()}"
    )
    assert np.allclose(dist_sum, ref_sum, atol=1e-3, rtol=1e-3), (
        f"TND DP softmax_sum mismatch: max_diff={np.abs(dist_sum - ref_sum).max()}"
    )


def test_softmax_lse_tnd_dp_cp():
    """
    Feature: npu_dense_lightning_indexer_softmax_lse TND with dp=2, cp=2.
    Description:
        - 2-D mesh (2, 2); query/weights sharded on T1 by both axes;
          key sharded on T2 by dp, replicated on cp.
        - actual_seq_qlen/klen wrapped as DTensor with Shard(0) along dp.
    Expectation: full_tensor() output matches standalone within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="TND")
    ref_max, ref_sum = _run_standalone(q_np, k_np, w_np, layout="TND")

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp"))
    dq = distribute_tensor(Tensor(q_np), mesh, (Shard(0), Shard(0)))
    dk = distribute_tensor(Tensor(k_np), mesh, (Shard(0), Replicate()))
    dw = distribute_tensor(Tensor(w_np), mesh, (Shard(0), Shard(0)))

    dqlen = Tensor(_GLOBAL_QLEN, dtype=ms.int32)
    dklen = Tensor(_GLOBAL_KLEN, dtype=ms.int32)

    out = npu_dense_lightning_indexer_softmax_lse(
        dq, dk, dw,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
        sparse_mode=3,
    )
    dist_max = out[0].full_tensor().asnumpy().astype(np.float32)
    dist_sum = out[1].full_tensor().asnumpy().astype(np.float32)

    assert np.allclose(dist_max, ref_max, atol=1e-3, rtol=1e-3), (
        f"TND DP+CP softmax_max mismatch: max_diff={np.abs(dist_max - ref_max).max()}"
    )
    assert np.allclose(dist_sum, ref_sum, atol=1e-3, rtol=1e-3), (
        f"TND DP+CP softmax_sum mismatch: max_diff={np.abs(dist_sum - ref_sum).max()}"
    )

def test_softmax_lse_tnd_replicated():
    """
    Feature: npu_dense_lightning_indexer_softmax_lse TND replicated on 2-device mesh.
    Description:
        - All TND inputs replicated; output should match standalone TND result.
    Expectation: Distributed output matches standalone within tolerance.
    """
    q_np, k_np, w_np = _generate_inputs(layout="TND")
    ref_max, ref_sum = _run_standalone(q_np, k_np, w_np, layout="TND")

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dq = distribute_tensor(Tensor(q_np), mesh, (Replicate(),))
    dk = distribute_tensor(Tensor(k_np), mesh, (Replicate(),))
    dw = distribute_tensor(Tensor(w_np), mesh, (Replicate(),))

    dqlen = Tensor(_GLOBAL_QLEN, dtype=ms.int32)
    dklen = Tensor(_GLOBAL_KLEN, dtype=ms.int32)

    out = npu_dense_lightning_indexer_softmax_lse(
        dq, dk, dw,
        actual_seq_qlen=dqlen, actual_seq_klen=dklen, layout='TND',
    )
    dist_max = out[0].full_tensor().asnumpy().astype(np.float32)
    dist_sum = out[1].full_tensor().asnumpy().astype(np.float32)

    assert np.allclose(dist_max, ref_max, atol=1e-3, rtol=1e-3), (
        f"TND Replicated softmax_max mismatch: max_diff={np.abs(dist_max - ref_max).max()}"
    )
    assert np.allclose(dist_sum, ref_sum, atol=1e-3, rtol=1e-3), (
        f"TND Replicated softmax_sum mismatch: max_diff={np.abs(dist_sum - ref_sum).max()}"
    )
