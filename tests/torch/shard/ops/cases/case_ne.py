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
"""Shard ops cases for ``torch.ne``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _ne(x, y):
    return torch.ne(x, y)


register(OpShardCase(
    name="ne_ops_basic",
    fn=_ne,
    inputs=[
        InputSpec(shape=(8, 4), init="randn", seed=42),
        InputSpec(shape=(8, 4), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="ne_ops_scalar",
    fn=_ne,
    inputs=[
        InputSpec(shape=(8, 4), init="randn", seed=42),
    ],
    # Scalar compared against — a plain Python float passed through, matching
    # the old test ``torch.ne(a, scalar_val)`` (extra_inputs, not a DTensor).
    extra_inputs=[0.5],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))
