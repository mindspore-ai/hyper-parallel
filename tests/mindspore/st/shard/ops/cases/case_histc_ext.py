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

"""Shard ops cases for ``mint.histc``."""
import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _histc_bins50(x):
    return ms.mint.histc(x, bins=50, min=-3.0, max=3.0)


def _histc_bins100_min0_max5(x):
    return ms.mint.histc(x, bins=100, min=0.0, max=5.0)


def _histc_bins75(x):
    return ms.mint.histc(x, bins=75, min=-5.0, max=5.0)


def _histc_bins100_min0_max3(x):
    return ms.mint.histc(x, bins=100, min=0.0, max=3.0)


register(OpShardCase(
    name="histc_ext_ops_data_parallel",
    fn=_histc_bins50,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="histc_ext_ops_model_parallel",
    fn=_histc_bins100_min0_max5,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=43)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="histc_ext_ops_hybrid_parallel",
    fn=_histc_bins75,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=44)],
    placements=[(Shard(0), Shard(1))],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))

register(OpShardCase(
    name="histc_ext_ops_replicate_all",
    fn=_histc_bins100_min0_max3,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=45)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level1",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    ))
