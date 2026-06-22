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
"""Shard ops cases for ``torch.nn.functional.dropout``."""
import torch.nn.functional as F

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _dropout_p0(x):
    return F.dropout(x, p=0.0, training=True)


def _dropout_p05(x):
    return F.dropout(x, p=0.5, training=True)


register(OpShardCase(
    name="dropout_ops_basic",
    fn=_dropout_p05,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    # p>0 dropout is random per-rank; verify shape/dtype only (matches old test)
    compare=CompareSpec.shape(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="dropout_ops_p0",
    fn=_dropout_p0,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="dropout_ops_3d",
    fn=_dropout_p0,
    inputs=[InputSpec(shape=(4, 4, 6), init="randn", seed=42)],
    placements=[(Shard(0), Replicate(), Shard(1))],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="dropout_ops_replicated",
    fn=_dropout_p0,
    inputs=[InputSpec(shape=(8, 8), init="randn", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
