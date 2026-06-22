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
"""Shard ops cases for ``bmm``."""
import mindspore as ms
from mindspore import ops

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _bmm_dp(x, w):
    return ms.mint.bmm(x, w)


register(OpShardCase(
    name="bmm_ops_dp",
    fn=_bmm_dp,
    inputs=[
        InputSpec(shape=(8, 16, 32), init="randn", seed=42),
        InputSpec(shape=(8, 32, 16), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate(), Replicate()),
        (Shard(0), Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level0",),
))

# Note: the original MS test also covers bmm with 3D mesh (2,2,2) and
# BatchMatMul transpose — deferred pending Partial output handling in
# the new framework and 8-card NPU hardware availability.

def _bmm_transpose(x, w):
    # non-mint: ms.mint.bmm with transpose_a is not available
    return ops.BatchMatMul(transpose_a=True)(x, w)

register(OpShardCase(
    name="bmm_ops_transpose",
    fn=_bmm_transpose,
    inputs=[
        InputSpec(shape=(8, 128, 256), init="randn", seed=42),
        InputSpec(shape=(8, 128, 64), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate(), Replicate()),
        (Shard(0), Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
