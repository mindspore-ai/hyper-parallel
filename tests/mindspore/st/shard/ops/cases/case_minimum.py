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

"""Shard ops cases for ``minimum``."""
import numpy as np
import mindspore as ms
from mindspore import ops

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _minimum(x, y):
    return ms.mint.minimum(x, y)

register(OpShardCase(
    name="minimum_ops_same_shape",
    fn=_minimum,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 128), init="randn", seed=43),
    ],
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
    name="minimum_ops_broadcast",
    fn=_minimum,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(1, 128), init="randn", seed=43),
    ],
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
    name="minimum_ops_partial_shard",
    fn=_minimum,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 128), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

# --- less_equal ---
def _less_equal(x, y):
    return ms.mint.less_equal(x, y)

register(OpShardCase(
    name="less_equal_ops_same_shape",
    fn=_less_equal,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 128), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1)), (Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

# --- greater_equal ---
def _greater_equal(x, y):
    return ms.mint.greater_equal(x, y)

register(OpShardCase(
    name="greater_equal_ops_same_shape",
    fn=_greater_equal,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 128), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1)), (Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

# --- logical_or ---
def _logical_or(x, y):
    return ms.mint.logical_or(x, y)

register(OpShardCase(
    name="logical_or_ops_same_shape",
    fn=_logical_or,
    inputs=[
        InputSpec(shape=(16, 128), dtype="bool", init="ones", seed=42),
        InputSpec(shape=(16, 128), dtype="bool", init="zeros", seed=43),
    ],
    placements=[(Shard(0), Shard(1)), (Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

def _mod_same_shape(x, y):
    # non-mint: ms.mint.mod not available
    return ops.Mod()(x, y)

def _mod_broadcast(x, y):
    # non-mint: ms.mint.mod not available
    return ops.Mod()(x, y)

register(OpShardCase(
    name="mod_ops_same_shape",
    fn=_mod_same_shape,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 128), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1)), (Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="minimum_ops_broadcast_dim1",
    fn=_minimum,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 1), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1)), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="minimum_ops_broadcast_dim2",
    fn=_minimum,
    inputs=[
        InputSpec(shape=(16, 128, 64), init="randn", seed=42),
        InputSpec(shape=(16, 128, 1), init="randn", seed=43),
    ],
    # y dim2 is broadcast (size 1); shard x dim2 on tp, replicate y on tp
    placements=[(Shard(0), Shard(2)), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="minimum_ops_broadcast_rank_mismatch",
    fn=_minimum,
    inputs=[
        InputSpec(shape=(8, 16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 128), init="randn", seed=43),  # 2D -> 3D broadcasting
    ],
    # placement length == mesh ndim (2). x shards dim0/dim2; y (broadcast, no
    # dim0) replicates on dp and shards its dim1 (=x dim2) on tp.
    placements=[(Shard(0), Shard(2)), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="minimum_ops_broadcast_scalar",
    fn=_minimum,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(), data=np.array(0.5, dtype=np.float32), dtype="float32"),
    ],
    placements=[(Shard(0), Shard(1)), (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="less_equal_ops_broadcast",
    fn=_less_equal,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(1, 128), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1)), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="mod_ops_broadcast_dim0",
    fn=_mod_broadcast,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(1, 128), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1)), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="mod_ops_broadcast_dim1",
    fn=_mod_broadcast,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 1), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1)), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="mod_ops_broadcast_rank_mismatch",
    fn=_mod_broadcast,
    inputs=[
        InputSpec(shape=(8, 16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 128), init="randn", seed=43),  # 2D -> 3D broadcasting
    ],
    placements=[(Shard(0), Shard(2)), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="mod_ops_scalar",
    fn=_mod_broadcast,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(), data=np.array(3.0, dtype=np.float32), dtype="float32"),
    ],
    placements=[(Shard(0), Shard(1)), (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="mod_ops_partial_shard",
    fn=_mod_broadcast,
    inputs=[
        InputSpec(shape=(16, 128), init="randn", seed=42),
        InputSpec(shape=(16, 128), init="randn", seed=43),
    ],
    placements=[(Replicate(), Shard(1)), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="mod_ops_broadcast_dim2",
    fn=_mod_broadcast,
    inputs=[
        InputSpec(shape=(8, 64, 32), init="randn", seed=42),
        InputSpec(shape=(8, 64, 1), init="randn", seed=43),
    ],
    # y dim2 is broadcast (size 1); shard x dim2 on tp, replicate y on tp
    placements=[(Shard(0), Shard(2)), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
