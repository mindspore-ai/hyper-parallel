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
"""Shard ops cases for ``torch.nn.functional.layer_norm``."""
import torch.nn.functional as F

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _layer_norm_2d(x, w, b):
    return F.layer_norm(x, (16,), w, b)


def _layer_norm_3d(x, w, b):
    return F.layer_norm(x, (32,), w, b)


register(OpShardCase(
    name="layernorm_ops_data_parallel",
    fn=_layer_norm_2d,
    inputs=[
        InputSpec(shape=(8, 16), init="randn", seed=42),
        InputSpec(shape=(16,), init="ones"),
        InputSpec(shape=(16,), init="zeros"),
    ],
    placements=[
        (Shard(0), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="layernorm_ops_model_parallel",
    fn=_layer_norm_3d,
    inputs=[
        InputSpec(shape=(8, 16, 32), init="randn", seed=42),
        InputSpec(shape=(32,), init="ones"),
        InputSpec(shape=(32,), init="zeros"),
    ],
    placements=[
        (Replicate(), Shard(1), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="layernorm_ops_hybrid_parallel",
    fn=_layer_norm_3d,
    inputs=[
        InputSpec(shape=(8, 16, 32), init="randn", seed=42),
        InputSpec(shape=(32,), init="ones"),
        InputSpec(shape=(32,), init="zeros"),
    ],
    placements=[
        (Shard(0), Shard(1), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="layernorm_ops_all_replicated",
    fn=_layer_norm_2d,
    inputs=[
        InputSpec(shape=(8, 16), init="randn", seed=42),
        InputSpec(shape=(16,), init="ones"),
        InputSpec(shape=(16,), init="zeros"),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
