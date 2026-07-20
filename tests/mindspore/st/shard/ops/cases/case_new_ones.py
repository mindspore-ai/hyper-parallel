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
"""Shard ops cases for ``Tensor.new_ones``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _new_ones(x, size):
    return x.new_ones(size)


def _new_ones_dtype(x, size):
    return x.new_ones(size, dtype=ms.float16)


register(OpShardCase(
    name="new_ones_ops_replicated",
    fn=_new_ones,
    inputs=[InputSpec(shape=(4, 8), dtype="float32", init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    extra_inputs=[(3, 4, 5)],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="new_ones_ops_sharded",
    fn=_new_ones,
    inputs=[InputSpec(shape=(4, 8), dtype="float32", init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    extra_inputs=[(4, 8)],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="new_ones_ops_scalar",
    fn=_new_ones,
    inputs=[InputSpec(shape=(4, 8), dtype="float32", init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    extra_inputs=[()],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="new_ones_ops_zero_length",
    fn=_new_ones,
    inputs=[InputSpec(shape=(4, 8), dtype="float32", init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    extra_inputs=[(0, 16)],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="new_ones_ops_dtype_override",
    fn=_new_ones_dtype,
    inputs=[InputSpec(shape=(4, 8), dtype="float32", init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    extra_inputs=[(3, 4)],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))
