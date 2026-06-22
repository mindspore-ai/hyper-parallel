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

"""Shard ops cases for ``argsort``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _argsort_dp(x):
    return ms.mint.argsort(x, dim=1, descending=False)


def _argsort_tp(x):
    return ms.mint.argsort(x, dim=0, descending=False)


def _argsort_neg_dim(x):
    return ms.mint.argsort(x, dim=-1, descending=False)


def _argsort_descending(x):
    return ms.mint.argsort(x, dim=1, descending=True)


register(OpShardCase(
    name="argsort_ops_dp",
    fn=_argsort_dp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="argsort_ops_tp",
    fn=_argsort_tp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="argsort_ops_neg_dim",
    fn=_argsort_neg_dim,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="argsort_ops_descending",
    fn=_argsort_descending,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
