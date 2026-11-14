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
"""Shard-op cases for MindSpore ``mint.tril``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import CompareSpec, InputSpec, OpShardCase, register


def _tril_default(x):
    return ms.mint.tril(x)


def _tril_diagonal_one(x):
    return ms.mint.tril(x, diagonal=1)


def _tril_diagonal_negative_one(x):
    return ms.mint.tril(x, -1)


def _tril_diagonal_nine(x):
    return ms.mint.tril(x, 9)


def _tril_diagonal_negative_seven(x):
    return ms.mint.tril(x, -7)


register(OpShardCase(
    name="tril_ops_replicated",
    fn=_tril_default,
    inputs=[InputSpec(shape=(6, 8), init="randn", seed=41)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="tril_ops_batch_dp",
    fn=_tril_diagonal_one,
    inputs=[InputSpec(shape=(4, 6, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="tril_ops_row_sharded",
    fn=_tril_default,
    inputs=[InputSpec(shape=(6, 8), init="randn", seed=43)],
    placements=[(Shard(-2),)],
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="tril_ops_col_sharded",
    fn=_tril_default,
    inputs=[InputSpec(shape=(6, 8), init="randn", seed=44)],
    placements=[(Shard(-1),)],
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="tril_ops_row_col_sharded",
    fn=_tril_diagonal_negative_one,
    inputs=[InputSpec(shape=(6, 8), init="randn", seed=45)],
    placements=[(Shard(-2), Shard(-1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("row_tp", "col_tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="tril_ops_hybrid_batch_row",
    fn=_tril_diagonal_one,
    inputs=[InputSpec(shape=(4, 6, 8), init="randn", seed=46)],
    placements=[(Shard(0), Shard(-2))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="tril_ops_diagonal_above_width",
    fn=_tril_diagonal_nine,
    inputs=[InputSpec(shape=(6, 8), init="randn", seed=47)],
    placements=[(Shard(-2),)],
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("npu_level0",),
))

register(OpShardCase(
    name="tril_ops_diagonal_below_height",
    fn=_tril_diagonal_negative_seven,
    inputs=[InputSpec(shape=(6, 8), init="randn", seed=48)],
    placements=[(Shard(-1),)],
    compare=CompareSpec.equal(),
    mesh_shape=(2,),
    mesh_dim_names=("tp",),
    tags=("npu_level0",),
))
