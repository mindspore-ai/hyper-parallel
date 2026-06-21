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
"""Shard ops cases for ``torch.argsort``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _argsort_last_dim(x):
    return torch.argsort(x, dim=-1)


def _argsort_dim0(x):
    return torch.argsort(x, dim=0)


def _argsort_descending(x):
    return torch.argsort(x, dim=1, descending=True)


register(OpShardCase(
    name="argsort_ops_last_dim",
    fn=_argsort_last_dim,
    inputs=[InputSpec(shape=(8, 4), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="argsort_ops_dim0",
    fn=_argsort_dim0,
    inputs=[InputSpec(shape=(8, 4), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="argsort_ops_descending",
    fn=_argsort_descending,
    inputs=[InputSpec(shape=(4, 8, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
