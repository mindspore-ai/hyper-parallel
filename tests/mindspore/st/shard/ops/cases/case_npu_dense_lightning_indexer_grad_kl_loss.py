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
"""Shard ops cases for ``npu_dense_lightning_indexer_grad_kl_loss``.

The op consumes softmax stats (sm_max/sm_sum from FlashAttentionScore) and
indexer stats (sm_max_idx/sm_sum_idx from npu_dense_lightning_indexer_softmax_lse)
that are *derived* from the primary tensors. They need global information (FA over
the full K dim), so they are declared via ``derived_inputs``: the framework
computes them once on the full tensors and distributes the result — matching the
old test that pre-computes stats on full tensors. Forward only, fp16.
"""
import numpy as np
import mindspore as ms
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.custom_ops.experimental import (
    npu_dense_lightning_indexer_softmax_lse,
    npu_dense_lightning_indexer_grad_kl_loss,
)
from tests.shard_ops.framework import (
    CompareSpec,
    DerivedSpec,
    InputSpec,
    OpShardCase,
    register,
)

# Shapes mirror the old test constants.
_BATCH, _S, _N1, _N1IDX, _HEAD_DIM, _ROPE_DIM = 4, 128, 64, 64, 128, 64
_SCALE = 1.0 / (_HEAD_DIM ** 0.5)
_GLOBAL_QLEN = np.array([128, 256, 384, 512], dtype=np.int32)
_GLOBAL_KLEN = np.array([128, 256, 384, 512], dtype=np.int32)
# qlen/klen: plain Tensor (not a DTensor), used directly by the derived precompute
# and the op fn (not distributed).
_QLEN_T = ms.Tensor(_GLOBAL_QLEN, ms.int32)
_KLEN_T = ms.Tensor(_GLOBAL_KLEN, ms.int32)


# --- derived-input helpers: sm_max / sm_sum from FlashAttentionScore ---
# Primary order is (q, k, qi, ki, w, qr, kr); qlen/klen are plain Tensors (globals).

def _fa(q, k, layout="BSND"):
    if layout == "TND":
        q = q.reshape(_BATCH, _S, _N1, _HEAD_DIM)
        k = k.reshape(_BATCH, _S, _N1, _HEAD_DIM)
    fa = FlashAttentionScore(head_num=_N1, scale_value=_SCALE,
                             input_layout="BSND", sparse_mode=3)
    return fa(q, k, ms.mint.zeros_like(k), None, None, None, None)


def _sm_max_bsnd(q, k, *_):
    return _fa(q, k)[0].amax(-1, True)


def _sm_sum_bsnd(q, k, *_):
    return _fa(q, k)[1].sum(-1, keepdim=True)


def _sm_max_tnd(q, k, *_):
    out = _fa(q, k, layout="TND")[0].amax(-1, True)
    return ms.mint.permute(out, (1, 0, 2, 3)).reshape(_N1, _BATCH * _S, 1)


def _sm_sum_tnd(q, k, *_):
    out = _fa(q, k, layout="TND")[1].sum(-1, keepdim=True)
    return ms.mint.permute(out, (1, 0, 2, 3)).reshape(_N1, _BATCH * _S, 1)


def _sm_max_idx_bsnd(_q, _k, qi, ki, w, *_):  # pylint: disable=C0103
    return npu_dense_lightning_indexer_softmax_lse(qi, ki, w)[0]


def _sm_sum_idx_bsnd(_q, _k, qi, ki, w, *_):  # pylint: disable=C0103
    return npu_dense_lightning_indexer_softmax_lse(qi, ki, w)[1]


def _sm_max_idx_tnd(_q, _k, qi, ki, w, _qr, _kr):  # pylint: disable=C0103
    return npu_dense_lightning_indexer_softmax_lse(
        qi, ki, w, actual_seq_qlen=_QLEN_T, actual_seq_klen=_KLEN_T, layout='TND')[0]


def _sm_sum_idx_tnd(_q, _k, qi, ki, w, _qr, _kr):  # pylint: disable=C0103
    return npu_dense_lightning_indexer_softmax_lse(
        qi, ki, w, actual_seq_qlen=_QLEN_T, actual_seq_klen=_KLEN_T, layout='TND')[1]


# --- op under test: fn(q,k,qi,ki,w,qr,kr, sm_max,sm_sum,sm_max_idx,sm_sum_idx) ---

def _grad_kl_loss_bsnd(q, k, qi, ki, w, qr, kr, sm_max, sm_sum, sm_max_idx, sm_sum_idx):
    return npu_dense_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, sm_max, sm_sum, sm_max_idx, sm_sum_idx, _SCALE,
        query_rope=qr, key_rope=kr)


def _grad_kl_loss_tnd(q, k, qi, ki, w, qr, kr,
                      sm_max, sm_sum, sm_max_idx, sm_sum_idx):
    return npu_dense_lightning_indexer_grad_kl_loss(
        q, k, qi, ki, w, sm_max, sm_sum, sm_max_idx, sm_sum_idx, _SCALE,
        query_rope=qr, key_rope=kr,
        actual_seq_qlen=_QLEN_T, actual_seq_klen=_KLEN_T, layout='TND')


def _bsnd_inputs():
    return [
        InputSpec(shape=(_BATCH, _S, _N1, _HEAD_DIM), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(_BATCH, _S, _N1, _HEAD_DIM), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(_BATCH, _S, _N1IDX, _HEAD_DIM), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(_BATCH, _S, 1, _HEAD_DIM), init="randn", dtype="float16", seed=45),
        InputSpec(shape=(_BATCH, _S, _N1IDX), init="randn", dtype="float16", seed=46),
        InputSpec(shape=(_BATCH, _S, _N1, _ROPE_DIM), init="randn", dtype="float16", seed=51),
        InputSpec(shape=(_BATCH, _S, _N1, _ROPE_DIM), init="randn", dtype="float16", seed=52),
    ]


def _tnd_inputs():
    t = _BATCH * _S
    return [
        InputSpec(shape=(t, _N1, _HEAD_DIM), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(t, _N1, _HEAD_DIM), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(t, _N1IDX, _HEAD_DIM), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(t, 1, _HEAD_DIM), init="randn", dtype="float16", seed=45),
        InputSpec(shape=(t, _N1IDX), init="randn", dtype="float16", seed=46),
        InputSpec(shape=(t, _N1, _ROPE_DIM), init="randn", dtype="float16", seed=51),
        InputSpec(shape=(t, _N1, _ROPE_DIM), init="randn", dtype="float16", seed=52),
    ]


# Primary placements for the 7 BSND inputs (q,k,qi,ki,w,qr,kr).
_BSND_REPL = [(Replicate(),)] * 7
_BSND_DP = [(Shard(0),)] * 7
_BSND_DP_CP = [
    (Shard(0), Shard(1)), (Shard(0), Replicate()), (Shard(0), Shard(1)),
    (Shard(0), Replicate()), (Shard(0), Shard(1)),
    (Shard(0), Shard(1)), (Shard(0), Replicate()),
]
# TND primary placements: q..kr (7). qlen/klen are plain Tensors, not distributed.
_TND_REPL = [(Replicate(),)] * 7
_TND_DP = [(Shard(0),)] * 7
_TND_DP_CP = [
    (Shard(0), Shard(0)), (Shard(0), Replicate()), (Shard(0), Shard(0)),
    (Shard(0), Replicate()), (Shard(0), Shard(0)),
    (Shard(0), Shard(0)), (Shard(0), Replicate()),
]


def _derived_bsnd(sm_pl, idx_pl):
    """sm_max, sm_sum, sm_max_idx, sm_sum_idx for BSND."""
    return [
        DerivedSpec(_sm_max_bsnd, sm_pl),
        DerivedSpec(_sm_sum_bsnd, sm_pl),
        DerivedSpec(_sm_max_idx_bsnd, idx_pl),
        DerivedSpec(_sm_sum_idx_bsnd, idx_pl),
    ]


def _derived_tnd(sm_pl, idx_pl):
    return [
        DerivedSpec(_sm_max_tnd, sm_pl),
        DerivedSpec(_sm_sum_tnd, sm_pl),
        DerivedSpec(_sm_max_idx_tnd, idx_pl),
        DerivedSpec(_sm_sum_idx_tnd, idx_pl),
    ]


_CMP = CompareSpec.allclose(rtol=1e-2, atol=1e-2)


def _reg(name, fn, inputs, placements, derived, mesh, names):
    register(OpShardCase(name=name, fn=fn, inputs=inputs, placements=placements,
                         derived_inputs=derived, compare=_CMP, mesh_shape=mesh,
                         mesh_dim_names=names, tags=("npu_level0",)))


# sm_max/sm_sum and the indexer stats share the same placement per config
# (old test: both distributed identically).
_reg("dense_grad_kl_loss_ops_bsnd_replicated", _grad_kl_loss_bsnd, _bsnd_inputs(),
     _BSND_REPL, _derived_bsnd((Replicate(),), (Replicate(),)), (2,), ("dp",))
_reg("dense_grad_kl_loss_ops_bsnd_dp", _grad_kl_loss_bsnd, _bsnd_inputs(),
     _BSND_DP, _derived_bsnd((Shard(0),), (Shard(0),)), (2,), ("dp",))
_reg("dense_grad_kl_loss_ops_bsnd_dp_cp", _grad_kl_loss_bsnd, _bsnd_inputs(),
     _BSND_DP_CP, _derived_bsnd((Shard(0), Shard(2)), (Shard(0), Shard(2))),
     (2, 2), ("dp", "tp"))
_reg("dense_grad_kl_loss_ops_tnd_replicated", _grad_kl_loss_tnd, _tnd_inputs(),
     _TND_REPL, _derived_tnd((Replicate(),), (Replicate(),)), (2,), ("dp",))
_reg("dense_grad_kl_loss_ops_tnd_dp", _grad_kl_loss_tnd, _tnd_inputs(),
     _TND_DP, _derived_tnd((Shard(1),), (Shard(1),)), (2,), ("dp",))
_reg("dense_grad_kl_loss_ops_tnd_dp_cp", _grad_kl_loss_tnd, _tnd_inputs(),
     _TND_DP_CP, _derived_tnd((Shard(1), Shard(1)), (Shard(1), Shard(1))),
     (2, 2), ("dp", "tp"))
