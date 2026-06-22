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
"""Shard ops cases for ``torch.nn.functional.embedding``."""
import numpy as np
import torch.nn.functional as F

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

np.random.seed(42)
_EMBED_INPUT_NP = np.random.randint(0, 32, size=(8, 16)).astype(np.int64)
_EMBED_WEIGHT_NP = np.random.randn(32, 64).astype(np.float32)

_INPUT_SPEC = InputSpec(shape=(8, 16), data=_EMBED_INPUT_NP, dtype="int64")
_WEIGHT_SPEC = InputSpec(shape=(32, 64), data=_EMBED_WEIGHT_NP, dtype="float32")


def _embedding(x, w):
    return F.embedding(x, w)


def _embedding_cp_padding_scale(x, w):
    return F.embedding(x, w, padding_idx=2, scale_grad_by_freq=True)


def _embedding_rp_padding(x, w):
    return F.embedding(x, w, padding_idx=10)


def _embedding_positional(x, w, *args):
    return F.embedding(x, w, *args)


register(OpShardCase(
    name="embedding_ops_dp",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_cp",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_rp",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_dp_cp",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_dp_rp",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_sp",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(1), Replicate()), (Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_sp_cp",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(1), Replicate()), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_sp_rp",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(1), Replicate()), (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_weight_2d",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Replicate(), Replicate()), (Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_weight_rp_only",
    fn=_embedding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Replicate(), Replicate()), (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_cp_padding_scale",
    fn=_embedding_cp_padding_scale,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_rp_padding",
    fn=_embedding_rp_padding,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Shard(0))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_cp_positional",
    fn=_embedding_positional,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Shard(1))],
    extra_inputs=[2, None, 2.0, True],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="embedding_ops_rp_positional",
    fn=_embedding_positional,
    inputs=[_INPUT_SPEC, _WEIGHT_SPEC],
    placements=[(Shard(0), Replicate()), (Replicate(), Shard(0))],
    extra_inputs=[10, None, 2.0, False],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
