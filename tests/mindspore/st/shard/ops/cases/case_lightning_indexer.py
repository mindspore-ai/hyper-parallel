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
"""Shard ops cases for ``lightning_indexer``."""
import numpy as np
import mindspore as ms
from mindspore import ops

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

_SPARSE_COUNT = 2048
_GLOBAL_QLEN = np.array([128, 256, 384, 512], dtype=np.int32)
_GLOBAL_KLEN = np.array([128, 256, 384, 512], dtype=np.int32)
# qlen/klen: plain Tensor (not a DTensor), passed via extra_inputs (not distributed).
_QLEN_T = ms.Tensor(_GLOBAL_QLEN, ms.int32)
_KLEN_T = ms.Tensor(_GLOBAL_KLEN, ms.int32)


def _lightning_indexer_bsnd(q, k, w):
    out = ops.lightning_indexer(q, k, w, sparse_count=_SPARSE_COUNT, return_value=True)
    return out


def _lightning_indexer_tnd(q, k, w, qlen, klen):
    out = ops.lightning_indexer(
        q, k, w,
        actual_seq_lengths_query=qlen,
        actual_seq_lengths_key=klen,
        layout_query='TND', layout_key='TND',
        sparse_count=_SPARSE_COUNT, return_value=True,
    )
    return out


# ======================== BSND ========================

register(OpShardCase(
    name="lightning_indexer_ops_bsnd_replicated",
    fn=_lightning_indexer_bsnd,
    inputs=[
        InputSpec(shape=(4, 128, 64, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(4, 128, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(4, 128, 64), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="lightning_indexer_ops_bsnd_dp",
    fn=_lightning_indexer_bsnd,
    inputs=[
        InputSpec(shape=(4, 128, 64, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(4, 128, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(4, 128, 64), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="lightning_indexer_ops_bsnd_dp_cp",
    fn=_lightning_indexer_bsnd,
    inputs=[
        InputSpec(shape=(4, 128, 64, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(4, 128, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(4, 128, 64), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Replicate()),
        (Shard(0), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# ======================== TND ========================

register(OpShardCase(
    name="lightning_indexer_ops_tnd_replicated",
    fn=_lightning_indexer_tnd,
    inputs=[
        InputSpec(shape=(512, 64, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(512, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(512, 64), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    extra_inputs=[_QLEN_T, _KLEN_T],
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="lightning_indexer_ops_tnd_dp",
    fn=_lightning_indexer_tnd,
    inputs=[
        InputSpec(shape=(512, 64, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(512, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(512, 64), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    extra_inputs=[_QLEN_T, _KLEN_T],
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="lightning_indexer_ops_tnd_dp_cp",
    fn=_lightning_indexer_tnd,
    inputs=[
        InputSpec(shape=(512, 64, 128), init="randn", dtype="float16", seed=42),
        InputSpec(shape=(512, 1, 128), init="randn", dtype="float16", seed=43),
        InputSpec(shape=(512, 64), init="randn", dtype="float16", seed=44),
    ],
    placements=[
        (Shard(0), Shard(0)),
        (Shard(0), Replicate()),
        (Shard(0), Shard(0)),
    ],
    extra_inputs=[_QLEN_T, _KLEN_T],
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))
