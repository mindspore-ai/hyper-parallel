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

"""Shard ops cases for ``reduce``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _sum_dim0(x):
    return ms.mint.sum(x, dim=0)


def _sum_dim1(x):
    return ms.mint.sum(x, dim=1)


def _mean_dim1(x):
    return ms.mint.mean(x, dim=1)


def _mean_keepdim(x):
    return ms.mint.mean(x, dim=0, keepdim=True)


register(OpShardCase(
    name="sum_ops_dim0",
    fn=_sum_dim0,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="sum_ops_dim1",
    fn=_sum_dim1,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="mean_ops_dim1",
    fn=_mean_dim1,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="mean_ops_keepdim",
    fn=_mean_keepdim,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

# --- max_dim with compare_outputs=(0,) to skip indices ---
def _max_dim_dp(x):
    return ms.mint.max(x, dim=1, keepdim=True)

def _max_dim_tp(x):
    return ms.mint.max(x, dim=0, keepdim=True)

def _max_dim_neg(x):
    return ms.mint.max(x, dim=-1, keepdim=True)

def _max_dim_no_keepdim(x):
    return ms.mint.max(x, dim=1, keepdim=False)

register(OpShardCase(
    name="max_dim_ops_dp",
    fn=_max_dim_dp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="max_dim_ops_tp",
    fn=_max_dim_tp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="max_dim_ops_neg_dim",
    fn=_max_dim_neg,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="max_dim_ops_no_keepdim",
    fn=_max_dim_no_keepdim,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0,),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
