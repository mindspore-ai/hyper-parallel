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
"""Shard ops cases for ``Tensor.repeat``."""

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _repeat_1_3(x):
    return x.repeat(1, 3)


def _repeat_3d(x):
    return x.repeat(1, 4, 1)


def _repeat_scalar(x):
    return x.repeat(3, 4, 5)


def _repeat_1_2(x):
    return x.repeat(1, 2)


def _repeat_1_0(x):
    return x.repeat(1, 0)


def _repeat_4d(x):
    return x.repeat(1, 2, 1, 3)


def _repeat_2_3_1(x):
    return x.repeat(2, 3, 1)


register(OpShardCase(
    name="repeat_ops_basic_unsharded",
    fn=_repeat_1_3,
    inputs=[InputSpec(shape=(8, 2), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="repeat_ops_3d",
    fn=_repeat_3d,
    inputs=[InputSpec(shape=(4, 3, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="repeat_ops_scalar_tensor",
    fn=_repeat_scalar,
    inputs=[InputSpec(shape=(1, 1), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="repeat_ops_replicated_dim",
    fn=_repeat_1_2,
    inputs=[InputSpec(shape=(4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="repeat_ops_zero_times",
    fn=_repeat_1_0,
    inputs=[InputSpec(shape=(8, 2), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="repeat_ops_4d_input",
    fn=_repeat_4d,
    inputs=[InputSpec(shape=(2, 3, 4, 5), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(1), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="repeat_ops_sharded_dim_repeat_one",
    fn=_repeat_1_3,
    inputs=[InputSpec(shape=(8, 2), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="repeat_ops_all_dims_replicated",
    fn=_repeat_2_3_1,
    inputs=[InputSpec(shape=(2, 3, 4), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
