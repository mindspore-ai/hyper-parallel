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
"""Shard ops cases for ``ms.mint.nn.functional.embedding``."""
import numpy as np
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

# Indices must be within [0, vocab_size=32). The old test uses
# np.random.randint(0, 32, ...); ``init="arange"`` would produce 0..127
# (out of range) and silently read garbage rows.
_IDX = np.random.RandomState(42).randint(0, 32, size=(8, 16)).astype(np.int64)


def _embedding(indices, weight, **kwargs):
    return ms.mint.nn.functional.embedding(indices, weight, **kwargs)


# Data Parallel: batch sharded on dp, weight replicated
register(OpShardCase(
    name="embedding_ops_dp",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Column Parallel: batch sharded on dp, weight embed_dim sharded on tp
register(OpShardCase(
    name="embedding_ops_cp",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Row Parallel: batch sharded on dp, weight vocab_dim sharded on tp
register(OpShardCase(
    name="embedding_ops_rp",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Shard(0)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Data Parallel + Column Parallel
register(OpShardCase(
    name="embedding_ops_dp_cp",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Data Parallel + Row Parallel
register(OpShardCase(
    name="embedding_ops_dp_rp",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Shard(0)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Sequence Parallel: seq sharded on dp, weight replicated
register(OpShardCase(
    name="embedding_ops_sp",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(1), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Sequence Parallel + Column Parallel
register(OpShardCase(
    name="embedding_ops_sp_cp",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(1), Replicate()),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Sequence Parallel + Row Parallel
register(OpShardCase(
    name="embedding_ops_sp_rp",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(1), Replicate()),
        (Replicate(), Shard(0)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Weight 2D Sharding: input replicated, weight sharded on both dims
register(OpShardCase(
    name="embedding_ops_weight_2d",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Shard(0), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Weight Row Parallel Only: input replicated, weight vocab_dim sharded on tp
register(OpShardCase(
    name="embedding_ops_weight_rp_only",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Shard(0)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Column Parallel with padding_idx & scale_grad_by_freq
register(OpShardCase(
    name="embedding_ops_cp_kwargs",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Shard(1)),
    ],
    kwargs={"padding_idx": 2, "scale_grad_by_freq": True},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Row Parallel with padding_idx
register(OpShardCase(
    name="embedding_ops_rp_padding",
    fn=_embedding,
    inputs=[
        InputSpec(shape=(8, 16), dtype="int64", data=_IDX),
        InputSpec(shape=(32, 64), init="randn", seed=42),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Shard(0)),
    ],
    kwargs={"padding_idx": 10},
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))
