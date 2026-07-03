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
"""Shard ops cases for ``npu_sparse_flash_attention`` (MindSpore)."""
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import sparse_flash_attention

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

_SCALE_VALUE = 0.135234
_SPARSE_COUNT_BASIC = 8
_SPARSE_COUNT_CP = 128
_TND_SEQ_LENS = np.arange(1, 129, dtype=np.int32) * 8
# actual_seq_len: plain Tensor (not a DTensor), passed via extra_inputs (not distributed).
_SEQ_LEN_T = ms.Tensor(_TND_SEQ_LENS, ms.int32)


def _sfa_bsnd_basic(q, k, v, q_idx, k_idx, w, q_rope, k_rope):
    si = ops.lightning_indexer(q_idx, k_idx, w, sparse_count=_SPARSE_COUNT_BASIC, return_value=True)[0]
    return sparse_flash_attention(
        q, k, v, si, _SCALE_VALUE,
        query_rope=q_rope, key_rope=k_rope,
        attention_mode=2, return_softmax_lse=True,
    )


def _sfa_bsnd_cp(q, k, v, q_idx, k_idx, w, q_rope, k_rope):
    si = ops.lightning_indexer(q_idx, k_idx, w, sparse_count=_SPARSE_COUNT_CP, return_value=True)[0]
    return sparse_flash_attention(
        q, k, v, si, _SCALE_VALUE,
        query_rope=q_rope, key_rope=k_rope,
        attention_mode=2, return_softmax_lse=True,
    )


def _sfa_tnd(q, k, v, q_idx, k_idx, w, q_rope, k_rope, qlen, klen):
    si = ops.lightning_indexer(
        q_idx, k_idx, w,
        actual_seq_lengths_query=qlen,
        actual_seq_lengths_key=klen,
        layout_query='TND', layout_key='TND',
        sparse_count=_SPARSE_COUNT_BASIC, return_value=True,
    )[0]
    return sparse_flash_attention(
        q, k, v, si, _SCALE_VALUE,
        actual_seq_lengths_query=qlen,
        actual_seq_lengths_kv=klen,
        query_rope=q_rope, key_rope=k_rope,
        layout_query='TND', layout_kv='TND',
        attention_mode=2, return_softmax_lse=True,
    )


# ======================== BSND basic (replicated / dp) ========================

register(OpShardCase(
    name="sfa_ops_bsnd_replicated",
    fn=_sfa_bsnd_basic,
    inputs=[
        InputSpec(shape=(4, 4, 8, 512), init="randn", dtype="bfloat16", seed=42),
        InputSpec(shape=(4, 1024, 1, 512), init="randn", dtype="bfloat16", seed=43),
        InputSpec(shape=(4, 1024, 1, 512), init="randn", dtype="bfloat16", seed=44),
        InputSpec(shape=(4, 4, 8, 128), init="randn", dtype="bfloat16", seed=45),
        InputSpec(shape=(4, 1024, 1, 128), init="randn", dtype="bfloat16", seed=46),
        InputSpec(shape=(4, 4, 8), init="randn", dtype="bfloat16", seed=47),
        InputSpec(shape=(4, 4, 8, 64), init="randn", dtype="bfloat16", seed=48),
        InputSpec(shape=(4, 1024, 1, 64), init="randn", dtype="bfloat16", seed=49),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
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
    name="sfa_ops_bsnd_dp",
    fn=_sfa_bsnd_basic,
    inputs=[
        InputSpec(shape=(4, 4, 8, 512), init="randn", dtype="bfloat16", seed=42),
        InputSpec(shape=(4, 1024, 1, 512), init="randn", dtype="bfloat16", seed=43),
        InputSpec(shape=(4, 1024, 1, 512), init="randn", dtype="bfloat16", seed=44),
        InputSpec(shape=(4, 4, 8, 128), init="randn", dtype="bfloat16", seed=45),
        InputSpec(shape=(4, 1024, 1, 128), init="randn", dtype="bfloat16", seed=46),
        InputSpec(shape=(4, 4, 8), init="randn", dtype="bfloat16", seed=47),
        InputSpec(shape=(4, 4, 8, 64), init="randn", dtype="bfloat16", seed=48),
        InputSpec(shape=(4, 1024, 1, 64), init="randn", dtype="bfloat16", seed=49),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

# ======================== BSND CP (cp / dp+cp) ========================

register(OpShardCase(
    name="sfa_ops_bsnd_cp",
    fn=_sfa_bsnd_cp,
    inputs=[
        InputSpec(shape=(2, 1024, 32, 512), init="randn", dtype="bfloat16", seed=42),
        InputSpec(shape=(2, 1024, 1, 512), init="randn", dtype="bfloat16", seed=43),
        InputSpec(shape=(2, 1024, 1, 512), init="randn", dtype="bfloat16", seed=44),
        InputSpec(shape=(2, 1024, 32, 128), init="randn", dtype="bfloat16", seed=45),
        InputSpec(shape=(2, 1024, 1, 128), init="randn", dtype="bfloat16", seed=46),
        InputSpec(shape=(2, 1024, 32), init="randn", dtype="bfloat16", seed=47),
        InputSpec(shape=(2, 1024, 32, 64), init="randn", dtype="bfloat16", seed=48),
        InputSpec(shape=(2, 1024, 1, 64), init="randn", dtype="bfloat16", seed=49),
    ],
    placements=[
        (Shard(1),),
        (Replicate(),),
        (Replicate(),),
        (Shard(1),),
        (Replicate(),),
        (Shard(1),),
        (Shard(1),),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="sfa_ops_bsnd_dp_cp",
    fn=_sfa_bsnd_cp,
    inputs=[
        InputSpec(shape=(2, 1024, 32, 512), init="randn", dtype="bfloat16", seed=42),
        InputSpec(shape=(2, 1024, 1, 512), init="randn", dtype="bfloat16", seed=43),
        InputSpec(shape=(2, 1024, 1, 512), init="randn", dtype="bfloat16", seed=44),
        InputSpec(shape=(2, 1024, 32, 128), init="randn", dtype="bfloat16", seed=45),
        InputSpec(shape=(2, 1024, 1, 128), init="randn", dtype="bfloat16", seed=46),
        InputSpec(shape=(2, 1024, 32), init="randn", dtype="bfloat16", seed=47),
        InputSpec(shape=(2, 1024, 32, 64), init="randn", dtype="bfloat16", seed=48),
        InputSpec(shape=(2, 1024, 1, 64), init="randn", dtype="bfloat16", seed=49),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
        (Shard(0), Shard(1)),
        (Shard(0), Replicate()),
        (Shard(0), Shard(1)),
        (Shard(0), Shard(1)),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# ======================== TND ========================

register(OpShardCase(
    name="sfa_ops_tnd_replicated",
    fn=_sfa_tnd,
    inputs=[
        InputSpec(shape=(1024, 8, 512), init="randn", dtype="bfloat16", seed=42),
        InputSpec(shape=(1024, 1, 512), init="randn", dtype="bfloat16", seed=43),
        InputSpec(shape=(1024, 1, 512), init="randn", dtype="bfloat16", seed=44),
        InputSpec(shape=(1024, 8, 128), init="randn", dtype="bfloat16", seed=45),
        InputSpec(shape=(1024, 1, 128), init="randn", dtype="bfloat16", seed=46),
        InputSpec(shape=(1024, 8), init="randn", dtype="bfloat16", seed=47),
        InputSpec(shape=(1024, 8, 64), init="randn", dtype="bfloat16", seed=48),
        InputSpec(shape=(1024, 1, 64), init="randn", dtype="bfloat16", seed=49),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    extra_inputs=[_SEQ_LEN_T, _SEQ_LEN_T],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="sfa_ops_tnd_dp",
    fn=_sfa_tnd,
    inputs=[
        InputSpec(shape=(1024, 8, 512), init="randn", dtype="bfloat16", seed=42),
        InputSpec(shape=(1024, 1, 512), init="randn", dtype="bfloat16", seed=43),
        InputSpec(shape=(1024, 1, 512), init="randn", dtype="bfloat16", seed=44),
        InputSpec(shape=(1024, 8, 128), init="randn", dtype="bfloat16", seed=45),
        InputSpec(shape=(1024, 1, 128), init="randn", dtype="bfloat16", seed=46),
        InputSpec(shape=(1024, 8), init="randn", dtype="bfloat16", seed=47),
        InputSpec(shape=(1024, 8, 64), init="randn", dtype="bfloat16", seed=48),
        InputSpec(shape=(1024, 1, 64), init="randn", dtype="bfloat16", seed=49),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    extra_inputs=[_SEQ_LEN_T, _SEQ_LEN_T],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="sfa_ops_tnd_cp",
    fn=_sfa_tnd,
    inputs=[
        InputSpec(shape=(1024, 8, 512), init="randn", dtype="bfloat16", seed=42),
        InputSpec(shape=(1024, 1, 512), init="randn", dtype="bfloat16", seed=43),
        InputSpec(shape=(1024, 1, 512), init="randn", dtype="bfloat16", seed=44),
        InputSpec(shape=(1024, 8, 128), init="randn", dtype="bfloat16", seed=45),
        InputSpec(shape=(1024, 1, 128), init="randn", dtype="bfloat16", seed=46),
        InputSpec(shape=(1024, 8), init="randn", dtype="bfloat16", seed=47),
        InputSpec(shape=(1024, 8, 64), init="randn", dtype="bfloat16", seed=48),
        InputSpec(shape=(1024, 1, 64), init="randn", dtype="bfloat16", seed=49),
    ],
    placements=[
        (Shard(0),),
        (Replicate(),),
        (Replicate(),),
        (Shard(0),),
        (Replicate(),),
        (Shard(0),),
        (Shard(0),),
        (Replicate(),),
    ],
    extra_inputs=[_SEQ_LEN_T, _SEQ_LEN_T],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="sfa_ops_tnd_dp_cp",
    fn=_sfa_tnd,
    inputs=[
        InputSpec(shape=(1024, 8, 512), init="randn", dtype="bfloat16", seed=42),
        InputSpec(shape=(1024, 1, 512), init="randn", dtype="bfloat16", seed=43),
        InputSpec(shape=(1024, 1, 512), init="randn", dtype="bfloat16", seed=44),
        InputSpec(shape=(1024, 8, 128), init="randn", dtype="bfloat16", seed=45),
        InputSpec(shape=(1024, 1, 128), init="randn", dtype="bfloat16", seed=46),
        InputSpec(shape=(1024, 8), init="randn", dtype="bfloat16", seed=47),
        InputSpec(shape=(1024, 8, 64), init="randn", dtype="bfloat16", seed=48),
        InputSpec(shape=(1024, 1, 64), init="randn", dtype="bfloat16", seed=49),
    ],
    placements=[
        (Shard(0), Shard(0)),
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
        (Shard(0), Shard(0)),
        (Shard(0), Replicate()),
        (Shard(0), Shard(0)),
        (Shard(0), Shard(0)),
        (Shard(0), Replicate()),
    ],
    extra_inputs=[_SEQ_LEN_T, _SEQ_LEN_T],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))
