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
"""Shard ops cases for ``npu_sparse_lightning_indexer_grad_kl_loss``.

si (sparse indices) is a static index tensor (``data=``); sm_max/sm_sum are
derived from q/k via FlashAttentionScore and need global K, so they are declared
via ``derived_inputs`` (computed once on the full tensors, then distributed) —
matching the old test. Forward only, fp16.
"""
import numpy as np
import mindspore as ms
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.custom_ops.experimental import npu_sparse_lightning_indexer_grad_kl_loss
from tests.shard_ops.framework import (
    CompareSpec,
    DerivedSpec,
    InputSpec,
    OpShardCase,
    register,
)

# Shapes mirror the old test constants.
_BATCH, _S, _N1, _N1IDX, _N2, _HEAD_DIM, _DIDX, _ROPE_DIM, _TOPK = 4, 128, 64, 64, 1, 512, 128, 64, 2048
_SCALE = 1.0 / (_HEAD_DIM ** 0.5)
_GLOBAL_QLEN = np.array([128, 256, 384, 512], dtype=np.int32)
_GLOBAL_KLEN = np.array([128, 256, 384, 512], dtype=np.int32)


def _build_si(layout):
    """Sparse indices: valid random idx per row with -1 padding (old test logic)."""
    rs = np.random.RandomState(0)
    shape = (_BATCH, _S, _N2, _TOPK) if layout == "BSND" else (_BATCH * _S, _N2, _TOPK)
    si = np.zeros(shape, dtype=np.int32)
    for s1 in range(_S):
        s2_real = max(int(_S - _S + s1 + 1), _S) if (_S - _S + s1 + 1) <= 0 else int(_S - _S + s1 + 1)
        s2_len = min(s2_real, _TOPK)
        for b in range(_BATCH):
            if layout == "BSND":
                si[b, s1, 0, :s2_len] = rs.randint(0, s2_real, size=s2_len, dtype=np.int32)
                si[b, s1, 0, s2_len:_TOPK] = -1
            else:
                idx = b * _S + s1
                si[idx, 0, :s2_len] = rs.randint(0, s2_real, size=s2_len, dtype=np.int32)
                si[idx, 0, s2_len:_TOPK] = -1
    return si


def _w(shape):
    """w scaled/shifted to a small range, matching the old test."""
    rs = np.random.RandomState(7)
    w_scale = (0.05 - (-0.05)) / (2 * 3.0)
    return (rs.randn(*shape).astype(np.float16) * w_scale).astype(np.float16)


_SI_BSND = _build_si("BSND")
_SI_TND = _build_si("TND")


# --- derived sm_max/sm_sum from FlashAttentionScore (with the old test's permute) ---

def _fa(q, k, layout="BSND"):
    if layout == "TND":
        q = q.reshape(_BATCH, _S, _N1, _HEAD_DIM)
        k = k.reshape(_BATCH, _S, _N2, _HEAD_DIM)
    fa = FlashAttentionScore(head_num=_N1, scale_value=_SCALE,
                             input_layout="BSND", sparse_mode=3)
    return fa(q, k, ms.mint.zeros_like(k), None, None, None, None)


def _sm_max_bsnd(q, k, *_):
    return ms.mint.permute(_fa(q, k)[0].amax(-1, True), (0, 3, 2, 1))


def _sm_sum_bsnd(q, k, *_):
    return ms.mint.permute(_fa(q, k)[1].sum(-1, keepdim=True), (0, 3, 2, 1))


def _sm_max_tnd(q, k, *_):
    out = ms.mint.permute(_fa(q, k, layout="TND")[0].amax(-1, True), (0, 3, 2, 1))
    return ms.mint.permute(out, (1, 0, 2, 3)).reshape(1, _BATCH * _S, _N1)


def _sm_sum_tnd(q, k, *_):
    out = ms.mint.permute(_fa(q, k, layout="TND")[1].sum(-1, keepdim=True), (0, 3, 2, 1))
    return ms.mint.permute(out, (1, 0, 2, 3)).reshape(1, _BATCH * _S, _N1)


# --- op under test: fn(q,k,qi,ki,w,si,qr,kr[,qlen,klen], sm_max,sm_sum) ---

def _grad_kl_loss_bsnd(q, k, qi, ki, w, si, qr, kr, sm_max, sm_sum):
    return npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, _SCALE, query_rope=qr, key_rope=kr)


def _grad_kl_loss_tnd(q, k, qi, ki, w, si, qr, kr, qlen, klen, sm_max, sm_sum):
    return npu_sparse_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, si, sm_max, sm_sum, _SCALE, query_rope=qr, key_rope=kr,
        actual_seq_qlen=qlen, actual_seq_klen=klen, layout='TND')


def _bsnd_inputs():
    return [
        InputSpec(shape=(_BATCH, _S, _N1, _HEAD_DIM), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(_BATCH, _S, _N2, _HEAD_DIM), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(_BATCH, _S, _N1IDX, _DIDX), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(_BATCH, _S, _N2, _DIDX), init="randn", dtype="float16", seed=45),
        InputSpec(shape=(_BATCH, _S, _N1IDX), dtype="float16", data=_w((_BATCH, _S, _N1IDX))),
        InputSpec(shape=(_BATCH, _S, _N2, _TOPK), dtype="int32", data=_SI_BSND),
        InputSpec(shape=(_BATCH, _S, _N1, _ROPE_DIM), init="randn", dtype="float16", seed=51),
        InputSpec(shape=(_BATCH, _S, _N2, _ROPE_DIM), init="randn", dtype="float16", seed=52),
    ]


def _tnd_inputs():
    t = _BATCH * _S
    return [
        InputSpec(shape=(t, _N1, _HEAD_DIM), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(t, _N2, _HEAD_DIM), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(t, _N1IDX, _DIDX), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(t, _N2, _DIDX), init="randn", dtype="float16", seed=45),
        InputSpec(shape=(t, _N1IDX), dtype="float16", data=_w((t, _N1IDX))),
        InputSpec(shape=(t, _N2, _TOPK), dtype="int32", data=_SI_TND),
        InputSpec(shape=(t, _N1, _ROPE_DIM), init="randn", dtype="float16", seed=51),
        InputSpec(shape=(t, _N2, _ROPE_DIM), init="randn", dtype="float16", seed=52),
        InputSpec(shape=(4,), dtype="int32", data=_GLOBAL_QLEN),
        InputSpec(shape=(4,), dtype="int32", data=_GLOBAL_KLEN),
    ]


# Primary placements (q,k,qi,ki,w,si,qr,kr) from the old test.
_BSND_REPL = [(Replicate(),)] * 8
_BSND_DP = [(Shard(0),)] * 8
_BSND_DP_CP = [
    (Shard(0), Shard(1)), (Shard(0), Replicate()), (Shard(0), Shard(1)),
    (Shard(0), Replicate()), (Shard(0), Shard(1)), (Shard(0), Shard(1)),
    (Shard(0), Shard(1)), (Shard(0), Replicate()),
]
_TND_REPL = [(Replicate(),)] * 10
_TND_DP = [(Shard(0),)] * 8 + [(Replicate(),), (Replicate(),)]
_TND_DP_CP = [
    (Shard(0), Shard(0)), (Shard(0), Replicate()), (Shard(0), Shard(0)),
    (Shard(0), Replicate()), (Shard(0), Shard(0)), (Shard(0), Shard(0)),
    (Shard(0), Shard(0)), (Shard(0), Replicate()),
    (Replicate(), Replicate()), (Replicate(), Replicate()),
]


def _derived(sm_max_fn, sm_sum_fn, sm_pl):
    return [DerivedSpec(sm_max_fn, sm_pl), DerivedSpec(sm_sum_fn, sm_pl)]


_CMP = CompareSpec.allclose(rtol=1e-2, atol=1e-2)


def _reg(name, fn, inputs, placements, derived, mesh, names):
    register(OpShardCase(name=name, fn=fn, inputs=inputs, placements=placements,
                         derived_inputs=derived, compare=_CMP, mesh_shape=mesh,
                         mesh_dim_names=names, tags=("npu_level1",)))


_reg("sparse_grad_kl_loss_ops_bsnd_replicated", _grad_kl_loss_bsnd, _bsnd_inputs(),
     _BSND_REPL, _derived(_sm_max_bsnd, _sm_sum_bsnd, (Replicate(),)), (2,), ("dp",))
_reg("sparse_grad_kl_loss_ops_bsnd_dp", _grad_kl_loss_bsnd, _bsnd_inputs(),
     _BSND_DP, _derived(_sm_max_bsnd, _sm_sum_bsnd, (Shard(0),)), (2,), ("dp",))
_reg("sparse_grad_kl_loss_ops_bsnd_dp_cp", _grad_kl_loss_bsnd, _bsnd_inputs(),
     _BSND_DP_CP, _derived(_sm_max_bsnd, _sm_sum_bsnd, (Shard(0), Shard(2))),
     (2, 2), ("dp", "tp"))
_reg("sparse_grad_kl_loss_ops_tnd_replicated", _grad_kl_loss_tnd, _tnd_inputs(),
     _TND_REPL, _derived(_sm_max_tnd, _sm_sum_tnd, (Replicate(),)), (2,), ("dp",))
_reg("sparse_grad_kl_loss_ops_tnd_dp", _grad_kl_loss_tnd, _tnd_inputs(),
     _TND_DP, _derived(_sm_max_tnd, _sm_sum_tnd, (Shard(1),)), (2,), ("dp",))
_reg("sparse_grad_kl_loss_ops_tnd_dp_cp", _grad_kl_loss_tnd, _tnd_inputs(),
     _TND_DP_CP, _derived(_sm_max_tnd, _sm_sum_tnd, (Shard(1), Shard(1))),
     (2, 2), ("dp", "tp"))
