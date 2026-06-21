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
"""Shard ops cases for ``gather_d`` and ``gather_nd``."""
import numpy as np
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

# gather indices must be in range. ``init="arange"`` produced out-of-range
# values (silent garbage / mismatch); the old test uses np.random.randint
# bounded by the gathered dimension. gather_d: index[..] in [0, x.shape[dim]).
# gather_nd: indices[..., j] in [0, x.shape[j]).
_RS = np.random.RandomState(1)
# gather_d index arrays (same shape as x), valued within the gather dim:
_GD_IDX0_3D = _RS.randint(0, 16, size=(16, 32, 64)).astype(np.int32)   # gather dim0
_GD_IDX1_3D = _RS.randint(0, 32, size=(16, 32, 64)).astype(np.int32)   # gather dim1
_GD_IDX0_2D = _RS.randint(0, 16, size=(16, 32)).astype(np.int32)       # gather dim0 (2D)


def _nd_idx(rows, ranges):
    """Build gather_nd indices (rows, k); column j in [0, ranges[j])."""
    cols = [_RS.randint(0, r, size=(rows, 1)) for r in ranges]
    return np.concatenate(cols, axis=1).astype(np.int32)


_GND_2_16_64 = _nd_idx(16, (16, 64))      # k=2 coords into x(16, 64)
_GND_K1 = _nd_idx(16, (16,))              # k=1 coord into x(16, 8, 32)
_GND_K2 = _nd_idx(16, (16, 8))            # k=2 coords into x(16, 8, 32)
_GND_K3 = _nd_idx(16, (16, 8, 32))        # k=3 coords into x(16, 8, 32)


# --- gather_d helpers ---

def _gatherd_dim0(x, index):
    # non-mint: ms.mint.gather_d not available
    return ms.ops.gather_d(x, 0, index)


def _gatherd_dim1(x, index):
    # non-mint: ms.mint.gather_d not available
    return ms.ops.gather_d(x, 1, index)


# --- gather_nd helper ---

def _gathernd(x, indices):
    # non-mint: ms.mint.gather_nd not available
    return ms.ops.gather_nd(x, indices)


# =============================================================================
# gather_d cases
# =============================================================================

# Data parallel dim=0: x sharded on batch, index replicated
register(OpShardCase(
    name="gatherd_ops_dp_dim0",
    fn=_gatherd_dim0,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int32", data=_GD_IDX0_3D),
    ],
    placements=[
        (Shard(0), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Data parallel dim=1: x sharded on seq, index replicated
register(OpShardCase(
    name="gatherd_ops_dp_dim1",
    fn=_gatherd_dim1,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int32", data=_GD_IDX1_3D),
    ],
    placements=[
        (Replicate(), Shard(1), Replicate()),
        (Replicate(), Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Row parallel dim=0: x sharded and index co-sharded on batch dim
register(OpShardCase(
    name="gatherd_ops_rp_dim0",
    fn=_gatherd_dim0,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int32", data=_GD_IDX0_3D),
    ],
    placements=[
        (Shard(0), Replicate(), Replicate()),
        (Replicate(), Shard(0), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# Input multi-shard dim=0: x sharded on both axes, index matches non-gather axis
register(OpShardCase(
    name="gatherd_ops_multi_shard",
    fn=_gatherd_dim0,
    inputs=[
        InputSpec(shape=(16, 32), init="randn", seed=1),
        InputSpec(shape=(16, 32), dtype="int32", data=_GD_IDX0_2D),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("npu_level1",),
))

# 3D mesh dim=1: x and index sharded on different non-gather axes
register(OpShardCase(
    name="gatherd_ops_3d_mesh_dim1",
    fn=_gatherd_dim1,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int32", data=_GD_IDX1_3D),
    ],
    placements=[
        (Replicate(), Shard(1), Replicate()),
        (Replicate(), Replicate(), Shard(1)),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))

# 3D mesh matched non-gather axes
register(OpShardCase(
    name="gatherd_ops_3d_mesh_matched",
    fn=_gatherd_dim1,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int32", data=_GD_IDX1_3D),
    ],
    placements=[
        (Shard(0), Shard(1), Replicate()),
        (Shard(0), Replicate(), Shard(1)),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))

# =============================================================================
# gather_nd cases (all 3D mesh, level1)
# =============================================================================

# Partial model parallel: params replicated, indices sharded on last mesh dim
register(OpShardCase(
    name="gathernd_ops_partial_mp",
    fn=_gathernd,
    inputs=[
        InputSpec(shape=(16, 64), init="randn", seed=1),
        InputSpec(shape=(16, 2), dtype="int32", data=_GND_2_16_64),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))

# Partial data parallel: params replicated, indices sharded on dp
register(OpShardCase(
    name="gathernd_ops_partial_dp",
    fn=_gathernd,
    inputs=[
        InputSpec(shape=(16, 64), init="randn", seed=1),
        InputSpec(shape=(16, 2), dtype="int32", data=_GND_2_16_64),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))

# Params plain tensor: params replicated, indices sharded on tp
register(OpShardCase(
    name="gathernd_ops_params_plain",
    fn=_gathernd,
    inputs=[
        InputSpec(shape=(16, 64), init="randn", seed=1),
        InputSpec(shape=(16, 2), dtype="int32", data=_GND_2_16_64),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))

# K=1 trailing dims shard: params sharded on trailing dims, indices sharded on non-k dim
register(OpShardCase(
    name="gathernd_ops_k1_trailing",
    fn=_gathernd,
    inputs=[
        InputSpec(shape=(16, 8, 32), init="randn", seed=1),
        InputSpec(shape=(16, 1), dtype="int32", data=_GND_K1),
    ],
    placements=[
        (Replicate(), Shard(1), Shard(2)),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))

# K=2 trailing dim shard: params sharded on trailing dim, indices sharded on non-k dim
register(OpShardCase(
    name="gathernd_ops_k2_trailing",
    fn=_gathernd,
    inputs=[
        InputSpec(shape=(16, 8, 32), init="randn", seed=1),
        InputSpec(shape=(16, 2), dtype="int32", data=_GND_K2),
    ],
    placements=[
        (Replicate(), Replicate(), Shard(2)),
        (Replicate(), Shard(0)),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))

# K=3 no trailing dims: params replicated, indices sharded on non-k dim
register(OpShardCase(
    name="gathernd_ops_k3_no_trailing",
    fn=_gathernd,
    inputs=[
        InputSpec(shape=(16, 8, 32), init="randn", seed=1),
        InputSpec(shape=(16, 3), dtype="int32", data=_GND_K3),
    ],
    placements=[
        (Replicate(), Replicate(), Replicate()),
        (Replicate(), Shard(0)),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("npu_level1",),
))
