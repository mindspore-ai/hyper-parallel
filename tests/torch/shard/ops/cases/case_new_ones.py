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
"""Shard ops cases for ``torch.Tensor.new_ones``."""
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _new_ones_tuple(x):
    return x.new_ones((3, 5))


def _new_ones_list(x):
    return x.new_ones([2, 2, 2])


def _new_ones_int(x):
    return x.new_ones(8)


def _new_ones_scalar(x):
    return x.new_ones(())


def _new_ones_shard_ignored(x):
    return x.new_ones((4, 4))


register(OpShardCase(
    name="new_ones_ops_tuple",
    fn=_new_ones_tuple,
    inputs=[InputSpec(shape=(4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="new_ones_ops_list",
    fn=_new_ones_list,
    inputs=[InputSpec(shape=(4, 4), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="new_ones_ops_int",
    fn=_new_ones_int,
    inputs=[InputSpec(shape=(4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="new_ones_ops_scalar",
    fn=_new_ones_scalar,
    inputs=[InputSpec(shape=(4, 4), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="new_ones_ops_shard_ignored",
    fn=_new_ones_shard_ignored,
    inputs=[InputSpec(shape=(4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
