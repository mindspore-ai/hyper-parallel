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
"""Shard ops cases for ``torch.vstack``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _vstack_2d(x, y):
    return torch.vstack((x, y))


def _vstack_3d(x, y):
    return torch.vstack((x, y))


def _vstack_0d(x, y):
    return torch.vstack((x, y))


def _vstack_1d(x, y):
    return torch.vstack((x, y))


def _vstack_mixed_0d_2d(x, y):
    return torch.vstack((x, y))


def _vstack_mixed_1d_2d(x, y):
    return torch.vstack((x, y))


def _vstack_multi(x, y, z):
    return torch.vstack((x, y, z))


def _vstack_single(x):
    return torch.vstack((x,))


# --- 2D cases ---

register(OpShardCase(
    name="vstack_ops_2d_replicated",
    fn=_vstack_2d,
    inputs=[
        InputSpec(shape=(4, 8), init="randn", seed=42),
        InputSpec(shape=(3, 8), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="vstack_ops_2d_non_cat_dim",
    fn=_vstack_2d,
    inputs=[
        InputSpec(shape=(4, 8), init="randn", seed=42),
        InputSpec(shape=(3, 8), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

# --- 3D on 2D mesh ---

register(OpShardCase(
    name="vstack_ops_3d",
    fn=_vstack_3d,
    inputs=[
        InputSpec(shape=(4, 8, 6), init="randn", seed=42),
        InputSpec(shape=(2, 8, 6), init="randn", seed=43),
    ],
    placements=[
        (Shard(1), Shard(2)),
        (Shard(1), Shard(2)),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

# --- 0D scalar ---

register(OpShardCase(
    name="vstack_ops_0d",
    fn=_vstack_0d,
    inputs=[
        InputSpec(shape=(), init="randn", seed=42),
        InputSpec(shape=(), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

# --- 1D cases ---

register(OpShardCase(
    name="vstack_ops_1d",
    fn=_vstack_1d,
    inputs=[
        InputSpec(shape=(8,), init="randn", seed=42),
        InputSpec(shape=(8,), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="vstack_ops_1d_replicated",
    fn=_vstack_1d,
    inputs=[
        InputSpec(shape=(8,), init="randn", seed=42),
        InputSpec(shape=(8,), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

# --- Mixed ndim ---

register(OpShardCase(
    name="vstack_ops_mixed_0d_2d",
    fn=_vstack_mixed_0d_2d,
    inputs=[
        InputSpec(shape=(), init="randn", seed=42),
        InputSpec(shape=(1, 1), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="vstack_ops_mixed_1d_2d",
    fn=_vstack_mixed_1d_2d,
    inputs=[
        InputSpec(shape=(8,), init="randn", seed=42),
        InputSpec(shape=(1, 8), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Shard(1), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

# --- Multi / single ---

register(OpShardCase(
    name="vstack_ops_multi",
    fn=_vstack_multi,
    inputs=[
        InputSpec(shape=(4, 8), init="randn", seed=41),
        InputSpec(shape=(3, 8), init="randn", seed=42),
        InputSpec(shape=(2, 8), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="vstack_ops_single",
    fn=_vstack_single,
    inputs=[
        InputSpec(shape=(4, 8), init="randn", seed=42),
    ],
    placements=[
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

# --- 8-card 3D mesh ---

register(OpShardCase(
    name="vstack_ops_3d_mesh",
    fn=_vstack_3d,
    inputs=[
        InputSpec(shape=(4, 8, 6), init="randn", seed=42),
        InputSpec(shape=(2, 8, 6), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Shard(1), Shard(2)),
        (Replicate(), Shard(1), Shard(2)),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))
