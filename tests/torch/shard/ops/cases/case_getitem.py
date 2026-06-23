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
"""Shard ops cases for ``Tensor.__getitem__``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _getitem_int_replicated(x):
    return x[2]


def _getitem_slice_keep_dim(x):
    return x[:, 1:3]


def _getitem_newaxis(x):
    return x[:, None, :]


def _getitem_ellipsis(x):
    return x[..., 1:3]


def _getitem_advanced_list(x):
    return x[[0, 2, 1]]


def _getitem_tuple_int(x):
    return x[(0, 1, 2)]


def _getitem_zero_slice(x):
    return x[2:2]


def _getitem_advanced_paired(x):
    return x[[0, 1], [2, 3]]


def _getitem_multi_d_index(x):
    ind = torch.tensor([[0, 1], [2, 3]], device=x.device)
    return x[ind]


def _getitem_advanced_consecutive(x):
    return x[:, [0, 2], [1, 3]]


def _getitem_advanced_split(x):
    return x[[0, 1], :, [2, 3]]


def _getitem_advanced_keep_shard(x):
    return x[[0, 2]]


def _getitem_mixed_basic(x):
    return x[0, ::1, ..., None]


def _getitem_chained(x):
    return x[1:3][0]


register(OpShardCase(
    name="getitem_ops_int_replicated",
    fn=_getitem_int_replicated,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_slice_keep_dim",
    fn=_getitem_slice_keep_dim,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_newaxis",
    fn=_getitem_newaxis,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_ellipsis",
    fn=_getitem_ellipsis,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_advanced_list",
    fn=_getitem_advanced_list,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_tuple_int",
    fn=_getitem_tuple_int,
    inputs=[InputSpec(shape=(8, 5, 4), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_zero_slice",
    fn=_getitem_zero_slice,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_advanced_paired",
    fn=_getitem_advanced_paired,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_multi_d_index",
    fn=_getitem_multi_d_index,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_advanced_consecutive",
    fn=_getitem_advanced_consecutive,
    inputs=[InputSpec(shape=(8, 5, 4), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_advanced_split",
    fn=_getitem_advanced_split,
    inputs=[InputSpec(shape=(8, 5, 4), init="randn", seed=42)],
    placements=[(Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_advanced_keep_shard",
    fn=_getitem_advanced_keep_shard,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_mixed_basic",
    fn=_getitem_mixed_basic,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="getitem_ops_chained",
    fn=_getitem_chained,
    inputs=[InputSpec(shape=(8, 5), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))
