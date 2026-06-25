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
"""Shard ops cases for ``torch.isin``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _isin(x, y):
    return torch.isin(x, y)


def _isin_invert(x, y):
    return torch.isin(x, y, invert=True)


def _isin_assume_unique(x, y):
    return torch.isin(x, y, assume_unique=True)


register(OpShardCase(
    name="isin_ops_basic",
    fn=_isin,
    inputs=[
        InputSpec(shape=(8, 16), init="arange", seed=42, dtype="int32"),
        InputSpec(shape=(20,), init="arange", seed=43, dtype="int32"),
    ],
    placements=[(Shard(0), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="isin_ops_invert",
    fn=_isin_invert,
    inputs=[
        InputSpec(shape=(8, 16), init="arange", seed=42, dtype="int32"),
        InputSpec(shape=(20,), init="arange", seed=43, dtype="int32"),
    ],
    placements=[(Shard(0), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="isin_ops_assume_unique",
    fn=_isin_assume_unique,
    inputs=[
        InputSpec(shape=(8, 16), init="arange", seed=42, dtype="int32"),
        InputSpec(shape=(20,), init="arange", seed=43, dtype="int32"),
    ],
    placements=[(Shard(0), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="isin_ops_3d_mixed",
    fn=_isin,
    inputs=[
        InputSpec(shape=(4, 6, 8), init="arange", seed=44, dtype="int32"),
        InputSpec(shape=(20,), init="arange", seed=43, dtype="int32"),
    ],
    placements=[(Shard(0), Shard(2)), (Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
