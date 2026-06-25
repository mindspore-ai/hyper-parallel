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
"""Shard ops cases for ``torch.remainder``."""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _remainder(x, y):
    return torch.remainder(x, y)


def _remainder_scalar(x):
    return torch.remainder(x, 3.0)


def _remainder_operator(x, y):
    return x % y


register(OpShardCase(
    name="remainder_ops_basic",
    fn=_remainder,
    inputs=[
        InputSpec(shape=(8, 4), init="uniform", seed=42),
        InputSpec(shape=(8, 4), init="uniform", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="remainder_ops_scalar",
    fn=_remainder_scalar,
    inputs=[InputSpec(shape=(8, 4), init="uniform", seed=42)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="remainder_ops_broadcast",
    fn=_remainder,
    inputs=[
        InputSpec(shape=(8, 4), init="uniform", seed=42),
        InputSpec(shape=(1, 4), init="uniform", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="remainder_ops_operator_overload",
    fn=_remainder_operator,
    inputs=[
        InputSpec(shape=(8, 4), init="uniform", seed=42),
        InputSpec(shape=(8, 4), init="uniform", seed=43),
    ],
    placements=[(Shard(0), Replicate()), (Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
