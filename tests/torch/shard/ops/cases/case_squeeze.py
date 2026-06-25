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
"""Shard ops cases for ``Tensor.squeeze``."""

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _squeeze_dim1(x):
    return x.squeeze(1)


def _squeeze_all(x):
    return x.squeeze()


def _squeeze_neg2(x):
    return x.squeeze(-2)


register(OpShardCase(
    name="squeeze_ops_basic",
    fn=_squeeze_dim1,
    inputs=[InputSpec(shape=(8, 1), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="squeeze_ops_no_args_all_dims",
    fn=_squeeze_all,
    inputs=[InputSpec(shape=(1, 4, 1, 8), init="randn", seed=42)],
    placements=[(Shard(1), Shard(3))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="squeeze_ops_specific_axis_negative",
    fn=_squeeze_neg2,
    inputs=[InputSpec(shape=(4, 1, 8), init="randn", seed=42)],
    placements=[(Shard(0), Shard(2))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="squeeze_ops_scalar_like",
    fn=_squeeze_all,
    inputs=[InputSpec(shape=(1, 1), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
