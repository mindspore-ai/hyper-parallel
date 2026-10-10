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
"""Shard ops cases for ``torch.nn.functional.softplus``."""
import torch.nn.functional as F

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _call_operator(func, *args, **kwargs):
    return func(*args, **kwargs)


def _softplus(x, beta=1, threshold=20):
    return _call_operator(F.softplus, x, beta=beta, threshold=threshold)


register(OpShardCase(
    name="softplus_ops_data_parallel",
    fn=_softplus,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="softplus_ops_model_parallel",
    fn=_softplus,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=43)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="softplus_ops_hybrid_parallel",
    fn=_softplus,
    inputs=[InputSpec(shape=(4, 8, 16), init="randn", seed=44)],
    placements=[(Shard(0), Shard(2))],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="softplus_ops_replicated",
    fn=_softplus,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=45)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="softplus_ops_non_default_params",
    fn=_softplus,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=46)],
    placements=[(Shard(0), Replicate())],
    kwargs={"beta": 2, "threshold": 1},
    compare=CompareSpec.allclose(rtol=1e-4, atol=1e-4),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
