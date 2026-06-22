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

"""Shard ops cases for ``max``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _max_elementwise(a, b):
    return torch.max(a, b)


register(OpShardCase(
    name="max_ops_elementwise",
    fn=_max_elementwise,
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
def _max_dim_reduce(x):
    return torch.max(x, dim=1)

def _max_dim_keepdim(x):
    return torch.max(x, dim=0, keepdim=True)

register(OpShardCase(
    name="max_ops_dim_reduce",
    fn=_max_dim_reduce,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="max_ops_dim_keepdim",
    fn=_max_dim_keepdim,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

# --- dim reduce on sharded dim (produces Partial) ---
def _max_dim_sharded(x):
    return torch.max(x, dim=0)

register(OpShardCase(
    name="max_ops_dim_sharded",
    fn=_max_dim_sharded,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

# --- global reduce (scalar output) ---
def _max_global(x):
    return torch.max(x)

register(OpShardCase(
    name="max_ops_global",
    fn=_max_global,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
