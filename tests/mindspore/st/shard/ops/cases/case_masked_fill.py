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

"""Shard ops cases for ``masked_fill``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _masked_fill(x, mask, value):
    return ms.ops.auto_generate.masked_fill_scalar_op(x, mask, value)


register(OpShardCase(
    name="masked_fill_ops_same_shape",
    fn=_masked_fill,
    inputs=[
        InputSpec(shape=(16, 256), init="randn", seed=42),
        InputSpec(shape=(16, 256), dtype="bool", init="randn", seed=43),
    ],
    extra_inputs=[0.5],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="masked_fill_ops_broadcast_dim0",
    fn=_masked_fill,
    inputs=[
        InputSpec(shape=(16, 256), init="randn", seed=42),
        InputSpec(shape=(1, 256), dtype="bool", init="randn", seed=43),
    ],
    extra_inputs=[-1.0],
    placements=[
        (Shard(0), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="masked_fill_ops_broadcast_dim1",
    fn=_masked_fill,
    inputs=[
        InputSpec(shape=(16, 256), init="randn", seed=42),
        InputSpec(shape=(16, 1), dtype="bool", init="randn", seed=43),
    ],
    extra_inputs=[2.0],
    placements=[
        (Shard(0), Shard(1)),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="masked_fill_ops_partial_shard",
    fn=_masked_fill,
    inputs=[
        InputSpec(shape=(16, 256), init="randn", seed=42),
        InputSpec(shape=(16, 256), dtype="bool", init="randn", seed=43),
    ],
    extra_inputs=[1.5],
    placements=[
        (Replicate(), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
