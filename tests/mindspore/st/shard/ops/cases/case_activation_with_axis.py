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

"""Shard ops cases for ``softmax`` and ``swiglu``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _softmax_last_axis(x):
    return ms.mint.nn.functional.softmax(x, dim=-1)


def _softmax_axis_0(x):
    return ms.mint.nn.functional.softmax(x, dim=0)


def _softmax_axis_1(x):
    return ms.mint.nn.functional.softmax(x, dim=1)


def _swiglu_last_axis(x):
    # non-mint: ms.mint.swiglu not available
    return ms.ops.swiglu(x, dim=-1)


def _swiglu_axis_0(x):
    # non-mint: ms.mint.swiglu not available
    return ms.ops.swiglu(x, dim=0)


def _swiglu_axis_1(x):
    # non-mint: ms.mint.swiglu not available
    return ms.ops.swiglu(x, dim=1)


register(OpShardCase(
    name="softmax_ops_dp",
    fn=_softmax_last_axis,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="softmax_ops_mp",
    fn=_softmax_axis_0,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Shard(2))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="softmax_ops_hybrid",
    fn=_softmax_axis_1,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(2))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="softmax_ops_replicated",
    fn=_softmax_last_axis,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="softmax_ops_neg_dim",
    fn=_softmax_last_axis,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="swiglu_ops_dp",
    fn=_swiglu_last_axis,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="swiglu_ops_mp",
    fn=_swiglu_axis_0,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Shard(2))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="swiglu_ops_hybrid",
    fn=_swiglu_axis_1,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(2))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="swiglu_ops_replicated",
    fn=_swiglu_last_axis,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="swiglu_ops_neg_dim",
    fn=_swiglu_last_axis,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
