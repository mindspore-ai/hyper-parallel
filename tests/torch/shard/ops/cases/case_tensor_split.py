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
"""Shard ops cases for ``Tensor.tensor_split``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _tensor_split_3_dim1(x):
    return x.tensor_split(3, dim=1)


def _tensor_split_2_dim1(x):
    return x.tensor_split(2, dim=1)


def _tensor_split_2_default(x):
    return x.tensor_split(2)


def _tensor_split_2_neg1(x):
    return x.tensor_split(2, dim=-1)


def _tensor_split_tuple_indices(x):
    return x.tensor_split((1, 4), dim=1)


def _tensor_split_tensor_indices(x):
    return x.tensor_split(torch.tensor([1, 4]), dim=1)


def _tensor_split_oob_indices(x):
    return x.tensor_split((2, 10), dim=1)


def _tensor_split_4d(x):
    return x.tensor_split(2, dim=2)


def _tensor_split_list_indices(x):
    return x.tensor_split([2, 5], dim=1)


def _tensor_split_4_dim0(x):
    return x.tensor_split(4, dim=0)


register(OpShardCase(
    name="tensor_split_ops_by_sections_unsharded",
    fn=_tensor_split_3_dim1,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_by_indices_unsharded",
    fn=_tensor_split_tuple_indices,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_default_dim",
    fn=_tensor_split_2_default,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_negative_dim",
    fn=_tensor_split_2_neg1,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_3d_sections",
    fn=_tensor_split_2_dim1,
    inputs=[InputSpec(shape=(8, 6, 8), init="randn", seed=42)],
    placements=[(Shard(0), Shard(2))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_1d_tensor_indices",
    fn=_tensor_split_tensor_indices,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_uneven_sections",
    fn=_tensor_split_3_dim1,
    inputs=[InputSpec(shape=(8, 7), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_out_of_bounds_indices",
    fn=_tensor_split_oob_indices,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_4d_multi_shard",
    fn=_tensor_split_4d,
    inputs=[InputSpec(shape=(8, 4, 6, 8), init="randn", seed=42)],
    placements=[(Shard(0), Shard(3))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_list_indices",
    fn=_tensor_split_list_indices,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="tensor_split_ops_replicated",
    fn=_tensor_split_4_dim0,
    inputs=[InputSpec(shape=(8, 6), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
