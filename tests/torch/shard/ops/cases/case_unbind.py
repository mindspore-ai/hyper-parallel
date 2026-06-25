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
"""Shard ops cases for ``torch.unbind``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _unbind_dim0(x):
    return torch.unbind(x, dim=0)


def _unbind_dim1(x):
    return torch.unbind(x, dim=1)


def _unbind_neg1(x):
    return torch.unbind(x, dim=-1)


register(OpShardCase(
    name="unbind_ops_dim0",
    fn=_unbind_dim0,
    inputs=[InputSpec(shape=(4, 8), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="unbind_ops_dim1",
    fn=_unbind_dim1,
    inputs=[InputSpec(shape=(4, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="unbind_ops_negative_dim",
    fn=_unbind_neg1,
    inputs=[InputSpec(shape=(2, 4, 8), init="randn", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
