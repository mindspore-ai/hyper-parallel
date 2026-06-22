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
"""Shard ops cases for ``min``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _min_elementwise(a, b):
    return torch.min(a, b)


register(OpShardCase(
    name="min_ops_elementwise",
    fn=_min_elementwise,
    inputs=[
        InputSpec(shape=(8, 8), init="randn", seed=42),
        InputSpec(shape=(8, 8), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- dim reduce on replicated dim, keepdim ---
def _min_dim_reduce(x):
    return torch.min(x, dim=1)

def _min_dim_keepdim(x):
    return torch.min(x, dim=0, keepdim=True)

register(OpShardCase(
    name="min_ops_dim_reduce",
    fn=_min_dim_reduce,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="min_ops_dim_keepdim",
    fn=_min_dim_keepdim,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- dim reduce on sharded dim (produces Partial) ---
def _min_dim_sharded(x):
    return torch.min(x, dim=0)

register(OpShardCase(
    name="min_ops_dim_sharded",
    fn=_min_dim_sharded,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- global reduce (scalar output) ---
def _min_global(x):
    return torch.min(x)

register(OpShardCase(
    name="min_ops_global",
    fn=_min_global,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- 3D element-wise ---
register(OpShardCase(
    name="min_ops_3d_elementwise",
    fn=_min_elementwise,
    inputs=[
        InputSpec(shape=(4, 4, 4), init="randn", seed=42),
        InputSpec(shape=(4, 4, 4), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Shard(1)),
    ],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- 3D reduce negative dim ---
def _min_neg_dim(x):
    return torch.min(x, dim=-1)

register(OpShardCase(
    name="min_ops_3d_reduce_neg_dim",
    fn=_min_neg_dim,
    inputs=[InputSpec(shape=(4, 4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- 3D reduce sharded dim ---
def _min_dim1(x):
    return torch.min(x, dim=1)

register(OpShardCase(
    name="min_ops_3d_dim_sharded",
    fn=_min_dim1,
    inputs=[InputSpec(shape=(4, 4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- 1D mesh global reduce ---
register(OpShardCase(
    name="min_ops_1d_global",
    fn=_min_global,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0),)],
    compare=CompareSpec.equal(),
    tags=("cpu_level1", "npu_level1"),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
))

# --- 1D mesh element-wise ---
register(OpShardCase(
    name="min_ops_1d_elementwise",
    fn=_min_elementwise,
    inputs=[
        InputSpec(shape=(8, 8), init="randn", seed=42),
        InputSpec(shape=(8, 8), init="randn", seed=43),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
    ],
    compare=CompareSpec.equal(),
    tags=("cpu_level1", "npu_level1"),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
))

# --- keepdim with negative dim ---
def _min_keepdim_neg(x):
    return torch.min(x, dim=-2, keepdim=True)

register(OpShardCase(
    name="min_ops_keepdim_neg",
    fn=_min_keepdim_neg,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- fully sharded dim reduce ---
register(OpShardCase(
    name="min_ops_fully_sharded_dim",
    fn=_min_dim1,
    inputs=[InputSpec(shape=(4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))
