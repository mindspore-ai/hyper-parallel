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

"""Shard ops cases for ``zeros_like``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _zeros_like_basic(x):
    return ms.mint.zeros_like(x)


def _zeros_like_dtype(x):
    return ms.mint.zeros_like(x, dtype=ms.float16)


register(OpShardCase(
    name="zeros_like_ops_basic",
    fn=_zeros_like_basic,
    inputs=[InputSpec(shape=(16, 64), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="zeros_like_ops_with_dtype",
    fn=_zeros_like_dtype,
    inputs=[InputSpec(shape=(4, 8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(2))],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
