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
"""Shard ops cases for ``torch.gather``."""
import numpy as np
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

# gather indices must be within [0, x.shape[dim]).
# Build fixed index arrays with np.random.randint bounded by the gathered dimension.
_RS = np.random.RandomState(1)

# 3D input: (16, 32, 64)
# gather dim=0: index values in [0, 16)
_GD_IDX0_3D = _RS.randint(0, 16, size=(16, 32, 64)).astype(np.int64)
# gather dim=1: index values in [0, 32)
_GD_IDX1_3D = _RS.randint(0, 32, size=(16, 32, 64)).astype(np.int64)
# gather dim=2: index values in [0, 64)
_GD_IDX2_3D = _RS.randint(0, 64, size=(16, 32, 64)).astype(np.int64)

# 2D input: (16, 32)
_GD_IDX0_2D = _RS.randint(0, 16, size=(16, 32)).astype(np.int64)
_GD_IDX1_2D = _RS.randint(0, 32, size=(16, 32)).astype(np.int64)

# Small/special index patterns for edge cases
_GD_IDX_SINGLE = _RS.randint(0, 16, size=(1, 32)).astype(np.int64)
_GD_IDX_DUP = np.tile(np.array([1, 3, 5, 7], dtype=np.int64).reshape(1, 4), (2, 8))  # (2,32)
_GD_IDX_OOO = np.tile(np.array([7, 3, 1, 9], dtype=np.int64).reshape(1, 4), (2, 8))  # (2,32)


# --- torch.gather helpers ---

def _gather_dim0(x, index):
    """torch.gather along dim=0."""
    return torch.gather(x, 0, index)


def _gather_dim0_sparse_grad(x, index):
    """torch.gather dim=0, verify explicit sparse_grad=False is accepted."""
    return torch.gather(x, 0, index, sparse_grad=False)


def _gather_dim1(x, index):
    """torch.gather along dim=1."""
    return torch.gather(x, 1, index)


def _gather_dim2(x, index):
    """torch.gather along dim=2."""
    return torch.gather(x, 2, index)


def _gather_neg1(x, index):
    """torch.gather along dim=-1."""
    return torch.gather(x, -1, index)


def _gather_neg2(x, index):
    """torch.gather along dim=-2."""
    return torch.gather(x, -2, index)


# =============================================================================
# Basic cases: replicated, simple sharding patterns
# =============================================================================

# Data parallel dim=0: x sharded on batch, index replicated
register(OpShardCase(
    name="gather_ops_dp_dim0",
    fn=_gather_dim0,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX0_3D),
    ],
    placements=[
        (Shard(0), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# Data parallel dim=1: x sharded on seq, index replicated
register(OpShardCase(
    name="gather_ops_dp_dim1",
    fn=_gather_dim1,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX1_3D),
    ],
    placements=[
        (Replicate(), Shard(1), Replicate()),
        (Replicate(), Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# Data parallel dim=2: x sharded on feature, index replicated
register(OpShardCase(
    name="gather_ops_dp_dim2",
    fn=_gather_dim2,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX2_3D),
    ],
    placements=[
        (Replicate(), Replicate(), Shard(1)),
        (Replicate(), Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# Row parallel dim=0: x sharded and index co-sharded on non-gather axis
register(OpShardCase(
    name="gather_ops_rp_dim0",
    fn=_gather_dim0,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX0_3D),
    ],
    placements=[
        (Shard(0), Replicate(), Replicate()),
        (Replicate(), Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# Explicit sparse_grad=False: verify keyword args are forwarded correctly
register(OpShardCase(
    name="gather_ops_sparse_grad",
    fn=_gather_dim0_sparse_grad,
    inputs=[
        InputSpec(shape=(16, 32), init="randn", seed=1),
        InputSpec(shape=(16, 32), dtype="int64", data=_GD_IDX0_2D),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# =============================================================================
# 2D input cases
# =============================================================================

# 2D gather dim=0 with DP on dim=0
register(OpShardCase(
    name="gather_ops_2d_dim0",
    fn=_gather_dim0,
    inputs=[
        InputSpec(shape=(16, 32), init="randn", seed=1),
        InputSpec(shape=(16, 32), dtype="int64", data=_GD_IDX0_2D),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# 2D gather dim=1 with DP on dim=1
register(OpShardCase(
    name="gather_ops_2d_dim1",
    fn=_gather_dim1,
    inputs=[
        InputSpec(shape=(16, 32), init="randn", seed=1),
        InputSpec(shape=(16, 32), dtype="int64", data=_GD_IDX1_2D),
    ],
    placements=[
        (Replicate(), Shard(1)),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# =============================================================================
# Multi-shard cases
# =============================================================================

# Input multi-shard: x sharded on both axes, index matches non-gather axis
register(OpShardCase(
    name="gather_ops_multi_shard",
    fn=_gather_dim0,
    inputs=[
        InputSpec(shape=(16, 32), init="randn", seed=1),
        InputSpec(shape=(16, 32), dtype="int64", data=_GD_IDX0_2D),
    ],
    placements=[
        (Shard(0), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# =============================================================================
# Replicated case
# =============================================================================

register(OpShardCase(
    name="gather_ops_replicated",
    fn=_gather_dim1,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX1_3D),
    ],
    placements=[
        (Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# =============================================================================
# Negative dim cases
# =============================================================================

# Negative dim, fully replicated: dim=-1 (dim=2), all non-gather axes match.
register(OpShardCase(
    name="gather_ops_neg_dim",
    fn=_gather_neg1,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX2_3D),
    ],
    placements=[
        (Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# Negative dim with sharding on non-gather axes:
# dim=-2 (dim=1), non-gather axes (0, 2) must match.
register(OpShardCase(
    name="gather_ops_neg_dim_sharded",
    fn=_gather_neg2,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX1_3D),
    ],
    placements=[
        (Shard(0), Replicate(), Shard(1)),
        (Shard(0), Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# =============================================================================
# Special index patterns
# =============================================================================

# Single element gather
register(OpShardCase(
    name="gather_ops_single_elem",
    fn=_gather_dim0,
    inputs=[
        InputSpec(shape=(16, 32), init="randn", seed=1),
        InputSpec(shape=(1, 32), dtype="int64", data=_GD_IDX_SINGLE),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# Duplicate indices: verify gather with repeated index values
register(OpShardCase(
    name="gather_ops_duplicate_idx",
    fn=_gather_dim0,
    inputs=[
        InputSpec(shape=(16, 32), init="randn", seed=1),
        InputSpec(shape=(2, 32), dtype="int64", data=_GD_IDX_DUP),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# Out-of-order indices: verify gather with shuffled index order
register(OpShardCase(
    name="gather_ops_out_of_order",
    fn=_gather_dim0,
    inputs=[
        InputSpec(shape=(16, 32), init="randn", seed=1),
        InputSpec(shape=(2, 32), dtype="int64", data=_GD_IDX_OOO),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# =============================================================================
# 3D mesh cases — higher parallelism
# =============================================================================

# 3D mesh dim=1: x and index sharded on different non-gather axes
register(OpShardCase(
    name="gather_ops_3d_mesh_dim1",
    fn=_gather_dim1,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX1_3D),
    ],
    placements=[
        (Replicate(), Shard(1), Replicate()),
        (Replicate(), Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

# 3D mesh matched non-gather axes
register(OpShardCase(
    name="gather_ops_3d_mesh_matched",
    fn=_gather_dim1,
    inputs=[
        InputSpec(shape=(16, 32, 64), init="randn", seed=1),
        InputSpec(shape=(16, 32, 64), dtype="int64", data=_GD_IDX1_3D),
    ],
    placements=[
        (Shard(0), Shard(1), Replicate()),
        (Shard(0), Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))
