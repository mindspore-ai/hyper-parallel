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

"""Shard ops cases for ``stack``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _stack_dim0(x, y):
    return torch.stack((x, y), dim=0)


def _stack_dim1(x, y):
    return torch.stack((x, y), dim=1)


def _stack_neg_dim(x, y):
    return torch.stack((x, y), dim=-1)


def _stack_multi(x, y, z):
    return torch.stack((x, y, z), dim=1)


def _stack_3d(x, y):
    return torch.stack((x, y), dim=2)


register(OpShardCase(
    name="stack_ops_dim0_dp",
    fn=_stack_dim0,
    inputs=[
        InputSpec(shape=(8, 6), init="randn", seed=42),
        InputSpec(shape=(8, 6), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="stack_ops_dim1",
    fn=_stack_dim1,
    inputs=[
        InputSpec(shape=(8, 6), init="randn", seed=42),
        InputSpec(shape=(8, 6), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="stack_ops_neg_dim",
    fn=_stack_neg_dim,
    inputs=[
        InputSpec(shape=(8, 6), init="randn", seed=42),
        InputSpec(shape=(8, 6), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Shard(1)),
        (Replicate(), Shard(1)),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="stack_ops_multi",
    fn=_stack_multi,
    inputs=[
        InputSpec(shape=(8, 6), init="randn", seed=41),
        InputSpec(shape=(8, 6), init="randn", seed=42),
        InputSpec(shape=(8, 6), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
        (Shard(0), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="stack_ops_3d",
    fn=_stack_3d,
    inputs=[
        InputSpec(shape=(4, 2, 6), init="randn", seed=42),
        InputSpec(shape=(4, 2, 6), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Shard(1), Replicate()),
        (Shard(0), Shard(1), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="stack_ops_replicated",
    fn=_stack_dim0,
    inputs=[
        InputSpec(shape=(8, 6), init="randn", seed=42),
        InputSpec(shape=(8, 6), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

# --- scalars → stack into 1D ---
def _stack_scalars(x, y):
    return torch.stack((x, y), dim=0)

register(OpShardCase(
    name="stack_ops_scalars",
    fn=_stack_scalars,
    inputs=[
        InputSpec(shape=(), init="randn", seed=42),
        InputSpec(shape=(), init="randn", seed=43),
    ],
    placements=[
        (),
        (),
    ],
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
))
