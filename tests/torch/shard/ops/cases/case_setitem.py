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
"""Shard ops cases for ``Tensor.__setitem__``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _setitem_scalar(x):
    x[1:3] = 0.0
    return x


def _setitem_dtensor_replicated(x, value):
    x[1:3] = value
    return x


def _setitem_shard_kept_dim(x, value):
    x[:, 1:3] = value
    return x


def _setitem_advanced(x, value):
    x[[0, 2]] = value
    return x


def _setitem_view_inplace_zero(x):
    y = x[1:3]
    y.zero_()
    return x


def _setitem_view_inplace_add(x):
    x = x.clone()
    x[1:3].add_(100)
    return x


def _setitem_int_view_inplace_zero(x):
    y = x[2]
    y.zero_()
    return x


def _setitem_int_view_inplace_add(x):
    x = x.clone()
    x[2].add_(100)
    return x


def _setitem_broadcast_dtensor_value(x, value):
    x[1:3] = value
    return x


def _setitem_broadcast_dtensor_shard_kept(x, value):
    x[:, 1:3] = value
    return x


def _setitem_dtensor_dim0_shard(x, value):
    x[:, 1:3] = value
    return x


def _setitem_broadcast_dtensor_1d_dim0_shard(x, value):
    x[:, 1:3] = value
    return x


def _setitem_broadcast_dtensor_2d_dim0_shard(x, value):
    x[:, 1:3] = value
    return x


def _setitem_dtensor_dim1_shard(x, value):
    x[1:3, :] = value
    return x


def _setitem_broadcast_dtensor_dim1_shard(x, value):
    x[1:3, :] = value
    return x


def _setitem_mixed_plain_tensor_dim0_shard(x):
    """Assign a global plain Tensor RHS to a dim-0-sharded DTensor LHS."""
    value = torch.ones(8, 2, device=x.device, dtype=x.dtype)
    x[:, 1:3] = value
    return x


register(OpShardCase(
    name="setitem_ops_scalar",
    fn=_setitem_scalar,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_dtensor_replicated",
    fn=_setitem_dtensor_replicated,
    inputs=[
        InputSpec(shape=(8, 5), init="randn", seed=42),
        InputSpec(shape=(2, 5), init="ones"),
    ],
    placements=[(Replicate(), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_shard_kept_dim",
    fn=_setitem_shard_kept_dim,
    inputs=[
        InputSpec(shape=(8, 5), init="randn", seed=42),
        InputSpec(shape=(8, 2), init="zeros"),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_advanced",
    fn=_setitem_advanced,
    inputs=[
        InputSpec(shape=(8, 5), init="randn", seed=42),
        InputSpec(shape=(2, 5), init="ones"),
    ],
    placements=[(Replicate(), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_view_zero",
    fn=_setitem_view_inplace_zero,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_view_add",
    fn=_setitem_view_inplace_add,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_int_view_zero",
    fn=_setitem_int_view_inplace_zero,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_int_view_add",
    fn=_setitem_int_view_inplace_add,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_broadcast_dtensor_value",
    fn=_setitem_broadcast_dtensor_value,
    inputs=[
        InputSpec(shape=(8, 5), init="randn", seed=42),
        InputSpec(shape=(1, 5), init="ones"),
    ],
    placements=[(Replicate(), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_broadcast_dtensor_shard_kept",
    fn=_setitem_broadcast_dtensor_shard_kept,
    inputs=[
        InputSpec(shape=(8, 5), init="randn", seed=42),
        InputSpec(shape=(8, 1), init="zeros"),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_dtensor_dim0_shard",
    fn=_setitem_dtensor_dim0_shard,
    inputs=[
        InputSpec(shape=(8, 5), init="randn", seed=42),
        InputSpec(shape=(8, 2), init="ones"),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_broadcast_dtensor_1d_dim0",
    fn=_setitem_broadcast_dtensor_1d_dim0_shard,
    inputs=[
        InputSpec(shape=(8, 5), init="randn", seed=42),
        InputSpec(shape=(2,), init="ones"),
    ],
    placements=[(Shard(0), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_broadcast_dtensor_2d_dim0",
    fn=_setitem_broadcast_dtensor_2d_dim0_shard,
    inputs=[
        InputSpec(shape=(8, 5), init="randn", seed=42),
        InputSpec(shape=(1, 2), init="ones"),
    ],
    placements=[(Shard(0), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_dtensor_dim1_shard",
    fn=_setitem_dtensor_dim1_shard,
    inputs=[
        InputSpec(shape=(8, 6), init="randn", seed=42),
        InputSpec(shape=(2, 6), init="ones"),
    ],
    placements=[(Replicate(), Shard(1)), (Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_broadcast_dtensor_dim1_shard",
    fn=_setitem_broadcast_dtensor_dim1_shard,
    inputs=[
        InputSpec(shape=(8, 6), init="randn", seed=42),
        InputSpec(shape=(2, 1), init="ones"),
    ],
    placements=[(Replicate(), Shard(1)), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="setitem_ops_mixed_plain_tensor_dim0_shard",
    fn=_setitem_mixed_plain_tensor_dim0_shard,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))
