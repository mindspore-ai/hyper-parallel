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

"""Shard ops cases for ``chunk_view``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


# chunk_view returns a tuple of non-contiguous views; .contiguous() each so
# the distributed gather (InnerCommAllGather) can all-gather them.
def _chunk_view_4_dim1(x):
    return tuple(c.contiguous() for c in ms.ops.auto_generate.chunk_view_op(x, 4, 1))


def _chunk_view_2_dim0(x):
    return tuple(c.contiguous() for c in ms.ops.auto_generate.chunk_view_op(x, 2, 0))


def _chunk_view_2_neg_dim(x):
    return tuple(c.contiguous() for c in ms.ops.auto_generate.chunk_view_op(x, 2, -1))


register(OpShardCase(
    name="chunk_ops_dp",
    fn=_chunk_view_4_dim1,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    compare_outputs=(0, 1, 2, 3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="chunk_ops_mp",
    fn=_chunk_view_2_dim0,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=43)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    compare_outputs=(0, 1),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="chunk_ops_neg_dim",
    fn=_chunk_view_2_neg_dim,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=44)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    compare_outputs=(0, 1),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
