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
"""Shard ops cases for ``torch.mul``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _mul(x, y):
    return torch.mul(x, y)


def _mul_scalar(x, s):
    return torch.mul(x, s)


register(OpShardCase(
    name="mul_ops_identical_shard",
    fn=_mul,
    inputs=[
        InputSpec(shape=(8, 16), init="randn", seed=42),
        InputSpec(shape=(8, 16), init="randn", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="mul_ops_broadcast",
    fn=_mul,
    inputs=[
        InputSpec(shape=(4, 8, 6), init="randn", seed=42),
        InputSpec(shape=(1, 8, 6), init="randn", seed=43),
    ],
    placements=[
        (Shard(0), Shard(1), Replicate()),
        (Replicate(), Shard(1), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="mul_ops_scalar",
    fn=_mul_scalar,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    extra_inputs=[3.14],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
