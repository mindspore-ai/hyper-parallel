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
"""Shard ops cases for ``torch.atleast_1d``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _atleast_1d_single(x):
    return torch.atleast_1d(x)


def _atleast_1d_multi(x, y):
    return torch.atleast_1d(x, y)


register(OpShardCase(
    name="atleast_1d_ops_0d",
    fn=_atleast_1d_single,
    inputs=[InputSpec(shape=(), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="atleast_1d_ops_1d",
    fn=_atleast_1d_single,
    inputs=[InputSpec(shape=(8,), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="atleast_1d_ops_2d",
    fn=_atleast_1d_single,
    inputs=[InputSpec(shape=(8, 4), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="atleast_1d_ops_multi",
    fn=_atleast_1d_multi,
    inputs=[
        InputSpec(shape=(), init="randn", seed=42),
        InputSpec(shape=(8,), init="randn", seed=43),
    ],
    placements=[(Replicate(), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    compare_outputs=(0, 1),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
