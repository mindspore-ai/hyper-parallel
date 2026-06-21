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
"""Shard ops cases for chained-matmul Partial propagation (MindSpore).

Chained ``(x @ A.T) @ B.T`` produces a Partial intermediate that must propagate
through the second matmul. Inputs are pre-transposed (a_t = A.T, b_t = B.T); the
framework's reference ``fn(full)`` equals the single-card gold ``x @ A.T @ B.T``.
"""
import numpy as np
from mindspore import mint, ops

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _np(seed, rank_dim, in_dim, out_dim, batch):
    """Build (x, a_t, b_t) numpy arrays matching the old test (a/b scaled by 0.1)."""
    rs = np.random.RandomState(seed)
    x = rs.randn(batch, in_dim).astype(np.float32)
    a = (rs.randn(rank_dim, in_dim) * 0.1).astype(np.float32)
    b = (rs.randn(out_dim, rank_dim) * 0.1).astype(np.float32)
    return x, np.ascontiguousarray(a.T), np.ascontiguousarray(b.T)


_X1, _AT1, _BT1 = _np(42, 8, 16, 32, 4)      # chained matmul tp
_X2, _AT2, _BT2 = _np(43, 8, 16, 32, 8)      # dp×tp (batch=8)


def _linear_np(seed, hid_dim, in_dim, out_dim, batch):
    """linear chain: x(batch,in), a_t=(in,hid), w=(out,hid) un-transposed for ops.dense."""
    rs = np.random.RandomState(seed)
    x = rs.randn(batch, in_dim).astype(np.float32)
    a = (rs.randn(hid_dim, in_dim) * 0.1).astype(np.float32)
    w = (rs.randn(out_dim, hid_dim) * 0.1).astype(np.float32)
    return x, np.ascontiguousarray(a.T), w


_X3, _AT3, _W3 = _linear_np(44, 12, 16, 32, 4)


def _chained_matmul(x, a_t, b_t):
    return mint.matmul(mint.matmul(x, a_t), b_t)


def _linear_chain(x, a_t, w):
    # non-mint: ops.dense exercises LinearDistributedOp (Partial propagation from x)
    return ops.dense(mint.matmul(x, a_t), w)


# tp=2 (1D): x shards k (dim1), a_t shards k (dim0), b_t replicate
register(OpShardCase(
    name="matmul_partial_prop_ops_chained_tp",
    fn=_chained_matmul,
    inputs=[
        InputSpec(shape=_X1.shape, dtype="float32", data=_X1),
        InputSpec(shape=_AT1.shape, dtype="float32", data=_AT1),
        InputSpec(shape=_BT1.shape, dtype="float32", data=_BT1),
    ],
    placements=[(Shard(1),), (Shard(0),), (Replicate(),)],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

# dp=2 × tp=2 (2D): x (Shard(0)dp, Shard(1)tp), a_t (Replicate, Shard(0)), b_t replicate
register(OpShardCase(
    name="matmul_partial_prop_ops_chained_dp_tp",
    fn=_chained_matmul,
    inputs=[
        InputSpec(shape=_X2.shape, dtype="float32", data=_X2),
        InputSpec(shape=_AT2.shape, dtype="float32", data=_AT2),
        InputSpec(shape=_BT2.shape, dtype="float32", data=_BT2),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Replicate(), Shard(0)),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

# linear chain tp=2 (1D): matmul -> ops.dense
register(OpShardCase(
    name="matmul_partial_prop_ops_linear_tp",
    fn=_linear_chain,
    inputs=[
        InputSpec(shape=_X3.shape, dtype="float32", data=_X3),
        InputSpec(shape=_AT3.shape, dtype="float32", data=_AT3),
        InputSpec(shape=_W3.shape, dtype="float32", data=_W3),
    ],
    placements=[(Shard(1),), (Shard(0),), (Replicate(),)],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))
