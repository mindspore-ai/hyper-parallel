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
"""Shard ops cases for Partial-aware MindSpore add and sub."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import CompareSpec, InputSpec, OpShardCase, register


def _partial_matmul(x, w):
    """Produce a Partial(sum) tensor by sharding the contracting dimension."""
    return ms.mint.matmul(x, w)


def _add_partial_replicate(x, w, bias):
    return ms.mint.add(_partial_matmul(x, w), bias)


def _sub_replicate_partial(x, w, bias):
    return ms.mint.sub(bias, _partial_matmul(x, w))


_INPUTS = [
    InputSpec(shape=(8, 4), init="randn", seed=41),
    InputSpec(shape=(4, 6), init="randn", seed=42),
    InputSpec(shape=(8, 6), init="randn", seed=43),
]
_PLACEMENTS = [
    (Shard(0), Shard(1)),
    (Replicate(), Shard(0)),
    (Shard(0), Replicate()),
]


register(OpShardCase(
    name="add_ops_partial_replicate",
    fn=_add_partial_replicate,
    inputs=_INPUTS,
    placements=_PLACEMENTS,
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="sub_ops_replicate_partial",
    fn=_sub_replicate_partial,
    inputs=_INPUTS,
    placements=_PLACEMENTS,
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))
