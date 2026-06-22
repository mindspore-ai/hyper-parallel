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

"""Shard ops cases for ``ops.ArgMaxWithValue``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _argmax_with_value_axis1_keepdims(x):
    # non-mint: ms.mint.argmax_with_value not available
    return ms.ops.ArgMaxWithValue(axis=1, keep_dims=True)(x)


def _argmax_with_value_axis0_keepdims(x):
    # non-mint: ms.mint.argmax_with_value not available
    return ms.ops.ArgMaxWithValue(axis=0, keep_dims=True)(x)


def _argmax_with_value_axis_neg1_keepdims(x):
    # non-mint: ms.mint.argmax_with_value not available
    return ms.ops.ArgMaxWithValue(axis=-1, keep_dims=True)(x)


def _argmax_with_value_axis1_nokeepdims(x):
    # non-mint: ms.mint.argmax_with_value not available
    return ms.ops.ArgMaxWithValue(axis=1, keep_dims=False)(x)


register(OpShardCase(
    name="argmax_with_value_ops_data_parallel",
    fn=_argmax_with_value_axis1_keepdims,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="argmax_with_value_ops_model_parallel",
    fn=_argmax_with_value_axis0_keepdims,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="argmax_with_value_ops_hybrid_parallel",
    fn=_argmax_with_value_axis1_keepdims,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="argmax_with_value_ops_replicate_all",
    fn=_argmax_with_value_axis_neg1_keepdims,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="argmax_with_value_ops_negative_dim",
    fn=_argmax_with_value_axis_neg1_keepdims,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="argmax_with_value_ops_keep_dims_false",
    fn=_argmax_with_value_axis1_nokeepdims,
    inputs=[InputSpec(shape=(8, 16, 32), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
