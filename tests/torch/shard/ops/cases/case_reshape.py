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

"""Shard ops cases for ``reshape``."""
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _reshape_flatten(x):
    return x.reshape(8, 64)


def _reshape_expand(x):
    return x.reshape(8, 4, 4, -1)


def _reshape_view(x):
    return x.view(8, 4, 16)


register(OpShardCase(
    name="reshape_ops_flatten",
    fn=_reshape_flatten,
    inputs=[InputSpec(shape=(8, 4, 4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="reshape_ops_expand",
    fn=_reshape_expand,
    inputs=[InputSpec(shape=(8, 64), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="reshape_ops_view",
    fn=_reshape_view,
    inputs=[InputSpec(shape=(8, 4, 4, 4), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
