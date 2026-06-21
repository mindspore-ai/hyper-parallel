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

"""Shard ops cases for ``transpose``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


# transpose/permute return non-contiguous views; .contiguous() so the
# distributed gather (InnerCommAllGather) can all-gather the local shard.
def _permute(x):
    return ms.mint.permute(x, (2, 0, 1)).contiguous()


def _transpose(x):
    return ms.mint.transpose(x, 1, 2).contiguous()


def _transpose_neg_dim(x):
    return ms.mint.transpose(x, 0, -1).contiguous()


register(OpShardCase(
    name="transpose_ops_permute",
    fn=_permute,
    inputs=[InputSpec(shape=(8, 16, 4), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="transpose_ops_transpose",
    fn=_transpose,
    inputs=[InputSpec(shape=(8, 16, 4), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="transpose_ops_neg_dim",
    fn=_transpose_neg_dim,
    inputs=[InputSpec(shape=(8, 16, 4), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
