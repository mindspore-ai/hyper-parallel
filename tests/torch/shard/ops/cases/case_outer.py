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
"""Shard ops cases for ``torch.outer``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _outer(v1, v2):
    return torch.outer(v1, v2)


register(OpShardCase(
    name="outer_ops_both_replicated",
    fn=_outer,
    inputs=[
        InputSpec(shape=(8,), init="randn", seed=42),
        InputSpec(shape=(16,), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="outer_ops_both_sharded",
    fn=_outer,
    inputs=[
        InputSpec(shape=(8,), init="randn", seed=42),
        InputSpec(shape=(16,), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Shard(0)),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="outer_ops_first_sharded",
    fn=_outer,
    inputs=[
        InputSpec(shape=(8,), init="randn", seed=42),
        InputSpec(shape=(16,), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
