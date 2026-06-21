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
"""Shard ops cases for ``npu_dense_lightning_indexer_softmax_lse``."""
import numpy as np

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.custom_ops.experimental import npu_dense_lightning_indexer_softmax_lse
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

_GLOBAL_QLEN = np.array([128, 256, 384, 512], dtype=np.int32)
_GLOBAL_KLEN = np.array([128, 256, 384, 512], dtype=np.int32)


def _softmax_lse_bsnd(q, k, w):
    return npu_dense_lightning_indexer_softmax_lse(q, k, w)


def _softmax_lse_bsnd_dp_cp(q, k, w):
    return npu_dense_lightning_indexer_softmax_lse(q, k, w, sparse_mode=3)


def _softmax_lse_tnd(q, k, w, qlen, klen):
    return npu_dense_lightning_indexer_softmax_lse(
        q, k, w,
        actual_seq_qlen=qlen, actual_seq_klen=klen, layout='TND',
    )


def _softmax_lse_tnd_dp_cp(q, k, w, qlen, klen):
    return npu_dense_lightning_indexer_softmax_lse(
        q, k, w,
        actual_seq_qlen=qlen, actual_seq_klen=klen, layout='TND',
        sparse_mode=3,
    )


# ======================== BSND ========================

register(OpShardCase(
    name="dense_softmax_lse_ops_bsnd_replicated",
    fn=_softmax_lse_bsnd,
    inputs=[
        InputSpec(shape=(4, 128, 32, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(4, 128, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(4, 128, 32), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="dense_softmax_lse_ops_bsnd_dp",
    fn=_softmax_lse_bsnd,
    inputs=[
        InputSpec(shape=(4, 128, 32, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(4, 128, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(4, 128, 32), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="dense_softmax_lse_ops_bsnd_dp_cp",
    fn=_softmax_lse_bsnd_dp_cp,
    inputs=[
        InputSpec(shape=(4, 128, 32, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(4, 128, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(4, 128, 32), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Replicate()),
        (Shard(0), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# ======================== TND ========================

register(OpShardCase(
    name="dense_softmax_lse_ops_tnd_replicated",
    fn=_softmax_lse_tnd,
    inputs=[
        InputSpec(shape=(512, 32, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(512, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(512, 32), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(4,), dtype="int32", data=_GLOBAL_QLEN),
        InputSpec(shape=(4,), dtype="int32", data=_GLOBAL_KLEN),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="dense_softmax_lse_ops_tnd_dp",
    fn=_softmax_lse_tnd,
    inputs=[
        InputSpec(shape=(512, 32, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(512, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(512, 32), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(4,), dtype="int32", data=_GLOBAL_QLEN),
        InputSpec(shape=(4,), dtype="int32", data=_GLOBAL_KLEN),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Replicate(),),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="dense_softmax_lse_ops_tnd_dp_cp",
    fn=_softmax_lse_tnd_dp_cp,
    inputs=[
        InputSpec(shape=(512, 32, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(512, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(512, 32), init="randn", dtype="float16", seed=44),
        InputSpec(shape=(4,), dtype="int32", data=_GLOBAL_QLEN),
        InputSpec(shape=(4,), dtype="int32", data=_GLOBAL_KLEN),
    ],
    placements=[
        (Shard(0), Shard(0)),
        (Shard(0), Replicate()),
        (Shard(0), Shard(0)),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))
