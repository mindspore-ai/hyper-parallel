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
"""Shard ops cases for ``torch.nn.functional.pad``."""
import torch.nn.functional as F

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _pad_constant(x):
    return F.pad(x, (1, 1, 2, 2), mode='constant', value=0.5)


def _pad_zero_sharded(x):
    return F.pad(x, (1, 1, 0, 0, 0, 0, 0, 0), mode='constant', value=0.0)


register(OpShardCase(
    name="pad_ops_basic_unsharded",
    fn=_pad_constant,
    inputs=[InputSpec(shape=(4, 4, 8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="pad_ops_zero_on_sharded_dim",
    fn=_pad_zero_sharded,
    inputs=[InputSpec(shape=(4, 4, 8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
