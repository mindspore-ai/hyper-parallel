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
"""Shard ops cases for MindSpore ``mint.sort``.

Uses ``ms.mint.sort`` exclusively per project convention. Returns
``(values, indices)`` like ``torch.sort``; the framework handles tuple
outputs in its gather/assert step.
"""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _sort_last_dim(x):
    return ms.mint.sort(x, dim=-1)


def _sort_last_dim_descending(x):
    return ms.mint.sort(x, dim=-1, descending=True)


def _sort_middle_dim(x):
    return ms.mint.sort(x, dim=1)


register(OpShardCase(
    name="sort_ops_2d_dp_last_dim",
    fn=_sort_last_dim,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="sort_ops_2d_dp_descending",
    fn=_sort_last_dim_descending,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="sort_ops_3d_dp_tp_middle_dim",
    fn=_sort_middle_dim,
    inputs=[InputSpec(shape=(4, 8, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
