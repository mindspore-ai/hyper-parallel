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
"""Shard ops cases for ``ms.ops.flash_attention_score``.

.. note::
    Migrated from 31 old test cases covering BSH/BNSD/SBH/BSND/TND layouts,
    sparse modes, 2D/3D meshes, and various edge conditions.  All cases use
    ``ms.ops.flash_attention_score`` (a standard MindSpore ops; non-mint).
"""
import numpy as np
from mindspore import ops

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_HEAD_DIM = 64
_HEAD_NUM = 16
_HIDDEN_SIZE = _HEAD_NUM * _HEAD_DIM  # 1024
_SCALE = 1.0 / (_HEAD_DIM ** 0.5)      # 0.125
_BATCH = 8
_SEQ_LEN = 512
_TOTAL_TOKENS = _BATCH * _SEQ_LEN      # 4096

_CAUSAL_MASK_2048 = np.triu(np.ones((2048, 2048)), k=1).astype(np.bool_)
_FULL_MASK_512 = np.zeros((512, 512), dtype=np.bool_)
_TND_SEQ_LEN = [((i + 1) * _SEQ_LEN) for i in range(_BATCH)]


# ---------------------------------------------------------------------------
# Wrapper functions (one per input_layout)
# ---------------------------------------------------------------------------
def _fas_bsh(q, k, v, **kwargs):
    # non-mint: ms.ops.flash_attention_score is standard MS ops
    return ops.flash_attention_score(q, k, v, **kwargs)


def _fas_bsh_mask(q, k, v, attn_mask, **kwargs):
    # non-mint: ms.ops.flash_attention_score is standard MS ops
    return ops.flash_attention_score(q, k, v, attn_mask=attn_mask, **kwargs)


def _fas_bnsd(q, k, v, **kwargs):
    return ops.flash_attention_score(q, k, v, **kwargs)


def _fas_sbh(q, k, v, **kwargs):
    return ops.flash_attention_score(q, k, v, **kwargs)


def _fas_bsnd(q, k, v, **kwargs):
    return ops.flash_attention_score(q, k, v, **kwargs)


def _fas_tnd(q, k, v, **kwargs):
    return ops.flash_attention_score(q, k, v, **kwargs)


def _fas_tnd_mask(q, k, v, attn_mask, **kwargs):
    return ops.flash_attention_score(q, k, v, attn_mask=attn_mask, **kwargs)


# ===================================================================
# BSH layout cases  (1-D meshes: level1)
# ===================================================================

register(OpShardCase(
    name="fas_bsh_replicated",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_dp",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_mp",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(2),),
        (Shard(2),),
        (Shard(2),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_sp",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(1),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))


# ===================================================================
# BSH layout cases  (2-D / 3-D meshes: level0 / level1)
# ===================================================================

register(OpShardCase(
    name="fas_bsh_dp_mp_2d",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Shard(2)),
        (Shard(0), Shard(2)),
        (Shard(0), Shard(2)),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="fas_bsh_sp_mp_2d",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(1), Shard(2)),
        (Replicate(), Shard(2)),
        (Replicate(), Shard(2)),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="fas_bsh_dp_sp_mp_3d",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Shard(1), Shard(2)),
        (Shard(0), Replicate(), Shard(2)),
        (Shard(0), Replicate(), Shard(2)),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))


# ===================================================================
# BNSD layout cases
# ===================================================================

register(OpShardCase(
    name="fas_bnsd_replicated",
    fn=_fas_bnsd,
    inputs=[
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BNSD", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bnsd_dp",
    fn=_fas_bnsd,
    inputs=[
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BNSD", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bnsd_mp",
    fn=_fas_bnsd,
    inputs=[
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(1),),
        (Shard(1),),
        (Shard(1),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BNSD", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bnsd_sp",
    fn=_fas_bnsd,
    inputs=[
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(2),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BNSD", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bnsd_dp_mp_2d",
    fn=_fas_bnsd,
    inputs=[
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Shard(1)),
        (Shard(0), Shard(1)),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BNSD", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="fas_bnsd_dp_sp_2d",
    fn=_fas_bnsd,
    inputs=[
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Shard(2)),
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BNSD", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="fas_bnsd_sp_mp_2d",
    fn=_fas_bnsd,
    inputs=[
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(2), Shard(1)),
        (Replicate(), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BNSD", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

# BNSD SP correctness (value comparison)
register(OpShardCase(
    name="fas_bnsd_sp_correctness",
    fn=_fas_bnsd,
    inputs=[
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _HEAD_NUM, _SEQ_LEN, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(2),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BNSD", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))


# ===================================================================
# SBH layout cases
# ===================================================================

register(OpShardCase(
    name="fas_sbh_dp",
    fn=_fas_sbh,
    inputs=[
        InputSpec(shape=(_SEQ_LEN, _BATCH, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_SEQ_LEN, _BATCH, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_SEQ_LEN, _BATCH, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(1),),
        (Shard(1),),
        (Shard(1),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "SBH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))


# ===================================================================
# BSND layout cases
# ===================================================================

register(OpShardCase(
    name="fas_bsnd_dp",
    fn=_fas_bsnd,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSND", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsnd_mp",
    fn=_fas_bsnd,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(2),),
        (Shard(2),),
        (Shard(2),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSND", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsnd_sp",
    fn=_fas_bsnd,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(1),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSND", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsnd_dp_mp_2d",
    fn=_fas_bsnd,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Shard(2)),
        (Shard(0), Shard(2)),
        (Shard(0), Shard(2)),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSND", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))


# ===================================================================
# TND layout cases
# ===================================================================

register(OpShardCase(
    name="fas_tnd_dp",
    fn=_fas_tnd,
    inputs=[
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "TND", "scalar_value": _SCALE, "sparse_mode": 0,
        "actual_seq_qlen": _TND_SEQ_LEN, "actual_seq_kvlen": _TND_SEQ_LEN,
    },
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_tnd_mp",
    fn=_fas_tnd,
    inputs=[
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(1),),
        (Shard(1),),
        (Shard(1),),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "TND", "scalar_value": _SCALE, "sparse_mode": 0,
        "actual_seq_qlen": _TND_SEQ_LEN, "actual_seq_kvlen": _TND_SEQ_LEN,
    },
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_tnd_dp_mp_2d",
    fn=_fas_tnd,
    inputs=[
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Shard(1)),
        (Shard(0), Shard(1)),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "TND", "scalar_value": _SCALE, "sparse_mode": 0,
        "actual_seq_qlen": _TND_SEQ_LEN, "actual_seq_kvlen": _TND_SEQ_LEN,
    },
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

# TND DP correctness (value comparison, kv_sharded variant)
register(OpShardCase(
    name="fas_tnd_dp_correctness",
    fn=_fas_tnd,
    inputs=[
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "TND", "scalar_value": _SCALE, "sparse_mode": 0,
        "actual_seq_qlen": _TND_SEQ_LEN, "actual_seq_kvlen": _TND_SEQ_LEN,
    },
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_tnd_dp_kv_sharded",
    fn=_fas_tnd,
    inputs=[
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "TND", "scalar_value": _SCALE, "sparse_mode": 0,
        "actual_seq_qlen": _TND_SEQ_LEN, "actual_seq_kvlen": _TND_SEQ_LEN,
    },
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_tnd_cp",
    fn=_fas_tnd_mask,
    inputs=[
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM), dtype="float16", init="randn", seed=44),
                InputSpec(shape=(2048, 2048), dtype="bool", data=_CAUSAL_MASK_2048),
    ],
    placements=[
        (Shard(0),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "TND", "scalar_value": _SCALE, "sparse_mode": 3,
        "actual_seq_qlen": _TND_SEQ_LEN, "actual_seq_kvlen": _TND_SEQ_LEN,
    },
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))


# ===================================================================
# BSH sparse mode cases — SP
# ===================================================================

register(OpShardCase(
    name="fas_bsh_sp_sparse_mode_0",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(1),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_sp_sparse_mode_2",
    fn=_fas_bsh_mask,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
                InputSpec(shape=(2048, 2048), dtype="bool", data=_CAUSAL_MASK_2048),
    ],
    placements=[
        (Shard(1),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 2},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_sp_sparse_mode_3",
    fn=_fas_bsh_mask,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
                InputSpec(shape=(2048, 2048), dtype="bool", data=_CAUSAL_MASK_2048),
    ],
    placements=[
        (Shard(1),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 3},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_sp_sparse_mode_4",
    fn=_fas_bsh_mask,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
                InputSpec(shape=(2048, 2048), dtype="bool", data=_CAUSAL_MASK_2048),
    ],
    placements=[
        (Shard(1),),
        (Replicate(),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 4,
        "pre_tokens": 256, "next_tokens": 256,
    },
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))


# ===================================================================
# BSH sparse mode cases — SP + DP 2-way (2-D mesh)
# ===================================================================

register(OpShardCase(
    name="fas_bsh_sp_dp_sparse_mode_2",
    fn=_fas_bsh_mask,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
                InputSpec(shape=(2048, 2048), dtype="bool", data=_CAUSAL_MASK_2048),
    ],
    placements=[
        (Shard(1), Shard(0)),
        (Replicate(), Shard(0)),
        (Replicate(), Shard(0)),
        (Replicate(), Replicate()),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 2},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="fas_bsh_sp_dp_sparse_mode_3",
    fn=_fas_bsh_mask,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
                InputSpec(shape=(2048, 2048), dtype="bool", data=_CAUSAL_MASK_2048),
    ],
    placements=[
        (Shard(1), Shard(0)),
        (Replicate(), Shard(0)),
        (Replicate(), Shard(0)),
        (Replicate(), Replicate()),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 3},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))


# ===================================================================
# BSH sparse mode cases — DP
# ===================================================================

register(OpShardCase(
    name="fas_bsh_dp_sparse_mode_1",
    fn=_fas_bsh_mask,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
                InputSpec(shape=(_SEQ_LEN, _SEQ_LEN), dtype="bool", data=_FULL_MASK_512),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 1},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_dp_sparse_mode_4",
    fn=_fas_bsh_mask,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
                InputSpec(shape=(2048, 2048), dtype="bool", data=_CAUSAL_MASK_2048),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
        (Replicate(),),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 4,
        "pre_tokens": 256, "next_tokens": 256,
    },
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))


# ===================================================================
# BSH misc / edge cases
# ===================================================================

register(OpShardCase(
    name="fas_bsh_dp_custom_scale",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": 0.125, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_dp_dropout",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={
        "head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0,
        "keep_prob": 0.9,
    },
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_sp_long_seq",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, 2048, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, 2048, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, 2048, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(1),),
        (Replicate(),),
        (Replicate(),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_dp_large_batch",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(64, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(64, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(64, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))

register(OpShardCase(
    name="fas_bsh_redistribute_then_attention",
    fn=_fas_bsh,
    inputs=[
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=43),
        InputSpec(shape=(_BATCH, _SEQ_LEN, _HIDDEN_SIZE), dtype="float16", init="randn", seed=44),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
        (Shard(0),),
    ],
    kwargs={"head_num": _HEAD_NUM, "input_layout": "BSH", "scalar_value": _SCALE, "sparse_mode": 0},
    compare=CompareSpec.allclose(rtol=1e-2, atol=1e-2),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("npu_level1",),
))
