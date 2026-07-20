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
"""Shard ops cases for ``Tensor.type_as``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _type_as(x, other):
    return x.type_as(other)


def _type_as_plain_other(x):
    """other is a zero-dim plain Tensor.  Using a zero-dim tensor also
    proves that other.shape does not affect the result.
    """
    other = ms.ops.ones((), dtype=ms.float16)
    return x.type_as(other)


# --- 4-card tests (level0) ---

register(OpShardCase(
    name="type_as_ops_replicated",
    fn=_type_as,
    inputs=[
        InputSpec(shape=(8, 16), dtype="float32", init="randn", seed=42),
        InputSpec(shape=(8, 16), dtype="float16", init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="type_as_ops_dp",
    fn=_type_as,
    inputs=[
        InputSpec(shape=(8, 16), dtype="float32", init="randn", seed=42),
        InputSpec(shape=(8, 16), dtype="float16", init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="type_as_ops_tp",
    fn=_type_as,
    inputs=[
        InputSpec(shape=(8, 16), dtype="float32", init="randn", seed=42),
        InputSpec(shape=(8, 16), dtype="float16", init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="type_as_ops_dp_tp",
    fn=_type_as,
    inputs=[
        InputSpec(shape=(8, 16), dtype="float32", init="randn", seed=42),
        InputSpec(shape=(8, 16), dtype="float16", init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="type_as_ops_cross_layout",
    fn=_type_as,
    inputs=[
        InputSpec(shape=(8, 16), dtype="float32", init="randn", seed=42),
        InputSpec(shape=(8, 16), dtype="float16", init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="type_as_ops_plain_other",
    fn=_type_as_plain_other,
    inputs=[
        InputSpec(shape=(8, 16), dtype="float32", init="randn", seed=42),
    ],
    placements=[
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

# --- 2-card test (level1) ---

register(OpShardCase(
    name="type_as_ops_1d_dp",
    fn=_type_as,
    inputs=[
        InputSpec(shape=(8, 16), dtype="float32", init="randn", seed=42),
        InputSpec(shape=(8, 16), dtype="float16", init="randn", seed=43),
    ],
    placements=[
        (Shard(0),),
        (Shard(0),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
))
