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
"""Shard ops cases for ``softmax``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _softmax_dp(x):
    return torch.softmax(x, dim=1)


def _softmax_tp(x):
    return torch.softmax(x, dim=0)


def _softmax_replicated(x):
    return torch.softmax(x, dim=-1)


register(OpShardCase(
    name="softmax_ops_dp",
    fn=_softmax_dp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="softmax_ops_tp",
    fn=_softmax_tp,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="softmax_ops_replicated",
    fn=_softmax_replicated,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="softmax_ops_neg_dim",
    fn=_softmax_replicated,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="softmax_ops_3d_hybrid",
    fn=_softmax_dp,
    inputs=[InputSpec(shape=(4, 4, 4), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(2))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "cp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))
