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
"""Shard ops cases for ``torch.Tensor.expand`` / ``expand_as``."""
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _expand(x):
    return x.expand(-1, 16)


def _expand_3d(x):
    return x.expand(-1, 10, -1)


def _expand_prepend(x):
    return x.expand(2, 3, -1, 16)


def _expand_scalar(x):
    return x.expand(3, 4, 5)


def _expand_as(x, y):
    return x.expand_as(y)


register(OpShardCase(
    name="expand_ops_basic",
    fn=_expand,
    inputs=[InputSpec(shape=(8, 1), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="expand_ops_3d",
    fn=_expand_3d,
    inputs=[InputSpec(shape=(4, 1, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="expand_ops_prepend_dims",
    fn=_expand_prepend,
    inputs=[InputSpec(shape=(8, 1), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="expand_ops_scalar",
    fn=_expand_scalar,
    inputs=[InputSpec(shape=(1, 1), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="expand_as_ops_basic",
    fn=_expand_as,
    inputs=[
        InputSpec(shape=(8, 1), init="randn", seed=42),
        InputSpec(shape=(8, 16), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="expand_as_ops_3d",
    fn=_expand_as,
    inputs=[
        InputSpec(shape=(4, 1, 6), init="randn", seed=42),
        InputSpec(shape=(4, 10, 6), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate(), Shard(1)), (Shard(0), Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="expand_as_ops_prepend_dims",
    fn=_expand_as,
    inputs=[
        InputSpec(shape=(8, 1), init="randn", seed=42),
        InputSpec(shape=(2, 3, 8, 16), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(2), Replicate(), Shard(1), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="expand_as_ops_scalar_to_tensor",
    fn=_expand_as,
    inputs=[
        InputSpec(shape=(1, 1), init="randn", seed=42),
        InputSpec(shape=(3, 4, 5), init="randn", seed=43),
    ],
    placements=[(Replicate(), Replicate()), (Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
