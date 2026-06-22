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

"""Shard ops cases for ``split``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _split_default_dim(x):
    return torch.split(x, 4)


def _split_dim1(x):
    return torch.split(x, 4, dim=1)


def _split_list(x):
    return torch.split(x, (8, 12), dim=1)


register(OpShardCase(
    name="split_ops_default_dim",
    fn=_split_default_dim,
    inputs=[InputSpec(shape=(16, 20), init="randn", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="split_ops_dim1",
    fn=_split_dim1,
    inputs=[InputSpec(shape=(16, 20), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))


register(OpShardCase(
    name="split_ops_list",
    fn=_split_list,
    inputs=[InputSpec(shape=(16, 20), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    tags=("cpu_level0", "npu_level0"),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
