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

"""Shard ops cases for ``unsqueeze``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _unsqueeze_dim0(x):
    return ms.mint.unsqueeze(x, 0)


def _unsqueeze_dim1(x):
    return ms.mint.unsqueeze(x, 1)


def _unsqueeze_neg_dim(x):
    return ms.mint.unsqueeze(x, -1)


def _unsqueeze_middle(x):
    return ms.mint.unsqueeze(x, 2)


register(OpShardCase(
    name="unsqueeze_ops_dim0",
    fn=_unsqueeze_dim0,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_ops_dim1",
    fn=_unsqueeze_dim1,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_ops_neg_dim",
    fn=_unsqueeze_neg_dim,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_ops_middle",
    fn=_unsqueeze_middle,
    inputs=[InputSpec(shape=(4, 8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(2))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

def _unsqueeze_neg_axis(x):
    return ms.mint.unsqueeze(x, -2)

register(OpShardCase(
    name="unsqueeze_ops_neg_axis",
    fn=_unsqueeze_neg_axis,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_ops_replicated",
    fn=_unsqueeze_dim1,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_view_ops_dim0",
    fn=_unsqueeze_dim0,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=43)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_view_ops_dim1",
    fn=_unsqueeze_dim1,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=43)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_view_ops_hybrid",
    fn=_unsqueeze_dim1,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=43)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_view_ops_middle",
    fn=_unsqueeze_middle,
    inputs=[InputSpec(shape=(4, 8, 16), init="randn", seed=43)],
    placements=[(Shard(0), Replicate(), Shard(2))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_view_ops_neg_axis",
    fn=_unsqueeze_neg_axis,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=43)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="unsqueeze_view_ops_replicated",
    fn=_unsqueeze_dim1,
    inputs=[InputSpec(shape=(16, 256), init="randn", seed=43)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
