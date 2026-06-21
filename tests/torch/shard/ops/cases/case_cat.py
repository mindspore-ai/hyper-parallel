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
"""Shard ops cases for ``cat``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _cat2d(x, y):
    return torch.cat((x, y), dim=1)


def _cat3d(x, y):
    return torch.cat((x, y), dim=2)


def _cat_multi(x, y, z):
    return torch.cat((x, y, z), dim=1)


def _cat4d(x, y):
    return torch.cat((x, y), dim=2)


def _cat5d(x, y):
    return torch.cat((x, y), dim=4)


def _cat_1d_dim0(x, y):
    return torch.cat((x, y), dim=0)


register(OpShardCase(
    name="cat_ops_basic_2d",
    fn=_cat2d,
    inputs=[
        InputSpec(shape=(8, 16), init="randn", seed=42),
        InputSpec(shape=(8, 16), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="cat_ops_3d",
    fn=_cat3d,
    inputs=[
        InputSpec(shape=(4, 8, 16), init="randn", seed=42),
        InputSpec(shape=(4, 8, 16), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1), Replicate()), (Shard(0), Shard(1), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="cat_ops_multi_tensor",
    fn=_cat_multi,
    inputs=[
        InputSpec(shape=(8, 16), init="randn", seed=41),
        InputSpec(shape=(8, 16), init="randn", seed=42),
        InputSpec(shape=(8, 16), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="cat_ops_mismatched",
    fn=_cat2d,
    inputs=[
        InputSpec(shape=(4, 8), init="randn", seed=42),
        InputSpec(shape=(4, 16), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="cat_ops_4d",
    fn=_cat4d,
    inputs=[
        InputSpec(shape=(4, 4, 8, 8), init="randn", seed=42),
        InputSpec(shape=(4, 4, 16, 8), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate(), Replicate(), Replicate()),
                (Shard(0), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="cat_ops_5d_mixed",
    fn=_cat5d,
    inputs=[
        InputSpec(shape=(2, 4, 8, 16, 32), init="randn", seed=42),
        InputSpec(shape=(2, 4, 8, 16, 32), init="randn", seed=43),
    ],
    placements=[(Replicate(), Replicate(), Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="cat_ops_singleton",
    fn=_cat2d,
    inputs=[
        InputSpec(shape=(8, 1), init="randn", seed=42),
        InputSpec(shape=(8, 1), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="cat_ops_3d_mesh",
    fn=_cat3d,
    inputs=[
        InputSpec(shape=(4, 8, 16), init="randn", seed=42),
        InputSpec(shape=(4, 8, 16), init="randn", seed=43),
    ],
    placements=[(Shard(0), Shard(1), Replicate()), (Shard(0), Shard(1), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="tmp_cat_placeholder",
    fn=_cat_1d_dim0,
    inputs=[
        InputSpec(shape=(8, 16), init="randn", seed=42),
        InputSpec(shape=(8, 16), init="randn", seed=43),
    ],
    placements=[(Shard(1),), (Shard(1),)],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))
