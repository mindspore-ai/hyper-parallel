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
"""Shard ops cases for ``torch.nn.functional.scaled_dot_product_attention``."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

SCALE = 1.0 / math.sqrt(64)

np.random.seed(42)
_Q_NP = np.random.randn(8, 16, 256, 64).astype(np.float16)
_K_NP = np.random.randn(8, 16, 256, 64).astype(np.float16)
_V_NP = np.random.randn(8, 16, 256, 64).astype(np.float16)

_Q_SPEC = InputSpec(shape=(8, 16, 256, 64), data=_Q_NP, dtype="float16")
_K_SPEC = InputSpec(shape=(8, 16, 256, 64), data=_K_NP, dtype="float16")
_V_SPEC = InputSpec(shape=(8, 16, 256, 64), data=_V_NP, dtype="float16")


def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, scale=SCALE)


def _sdpa_causal(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=SCALE)


def _sdpa_explicit_mask(q, k, v):
    mask = torch.ones(256, 256, dtype=torch.bool, device=q.device).tril(diagonal=0)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=SCALE)


def _sdpa_custom_scale(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, scale=0.125)


register(OpShardCase(
    name="sdpa_ops_replicate",
    fn=_sdpa,
    inputs=[_Q_SPEC, _K_SPEC, _V_SPEC],
    placements=[(Replicate(),), (Replicate(),), (Replicate(),)],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(1,),
    mesh_dim_names=("dp",),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="sdpa_ops_dp",
    fn=_sdpa,
    inputs=[_Q_SPEC, _K_SPEC, _V_SPEC],
    placements=[(Shard(0),), (Shard(0),), (Shard(0),)],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="sdpa_ops_mp",
    fn=_sdpa,
    inputs=[_Q_SPEC, _K_SPEC, _V_SPEC],
    placements=[(Shard(1),), (Shard(1),), (Shard(1),)],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="sdpa_ops_dp_mp_2d",
    fn=_sdpa,
    inputs=[_Q_SPEC, _K_SPEC, _V_SPEC],
    placements=[(Shard(0), Shard(1)), (Shard(0), Shard(1)), (Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="sdpa_ops_sp_causal",
    fn=_sdpa_causal,
    inputs=[_Q_SPEC, _K_SPEC, _V_SPEC],
    placements=[(Shard(2),), (Replicate(),), (Replicate(),)],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="sdpa_ops_sp_explicit_mask",
    fn=_sdpa_explicit_mask,
    inputs=[_Q_SPEC, _K_SPEC, _V_SPEC],
    placements=[(Shard(2),), (Replicate(),), (Replicate(),)],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="sdpa_ops_custom_scale",
    fn=_sdpa_custom_scale,
    inputs=[_Q_SPEC, _K_SPEC, _V_SPEC],
    placements=[(Shard(0),), (Shard(0),), (Shard(0),)],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="sdpa_ops_sp_correctness",
    fn=_sdpa,
    inputs=[_Q_SPEC, _K_SPEC, _V_SPEC],
    placements=[(Shard(2),), (Replicate(),), (Replicate(),)],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    mesh_shape=(2,),
    mesh_dim_names=("dp",),
    tags=("cpu_level1", "npu_level1"),
))
