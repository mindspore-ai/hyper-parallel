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
"""Shard ops cases for ``torch.Tensor.masked_scatter``."""
import numpy as np

from hyper_parallel.core.dtensor.placement_types import Replicate
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _masked_scatter(input_t, mask, source):
    return input_t.masked_scatter(mask, source)


register(OpShardCase(
    name="masked_scatter_ops_basic_replicated",
    fn=_masked_scatter,
    inputs=[
        InputSpec(shape=(8, 8), init="randn", seed=42),
        InputSpec(shape=(8, 8), dtype="bool", data=np.random.RandomState(42).rand(8, 8).astype(np.float32) > 0.5),
        InputSpec(shape=(42,), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="masked_scatter_ops_1d_replicated",
    fn=_masked_scatter,
    inputs=[
        InputSpec(shape=(16,), init="randn", seed=42),
        InputSpec(shape=(16,), dtype="bool", data=np.random.RandomState(44).rand(16).astype(np.float32) > 0.5),
        InputSpec(shape=(13,), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="masked_scatter_ops_3d_broadcast",
    fn=_masked_scatter,
    inputs=[
        InputSpec(shape=(2, 4, 4), init="randn", seed=42),
        InputSpec(shape=(4, 4), dtype="bool", data=np.random.RandomState(44).rand(4, 4).astype(np.float32) > 0.5),
        InputSpec(shape=(26,), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="masked_scatter_ops_oversized_source",
    fn=_masked_scatter,
    inputs=[
        InputSpec(shape=(4, 4), init="randn", seed=42),
        InputSpec(shape=(4, 4), dtype="bool", data=np.random.RandomState(44).rand(4, 4).astype(np.float32) > 0.5),
        InputSpec(shape=(130,), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))

register(OpShardCase(
    name="masked_scatter_ops_all_false_mask",
    fn=_masked_scatter,
    inputs=[
        InputSpec(shape=(8, 8), init="randn", seed=42),
        InputSpec(shape=(8, 8), dtype="bool", init="zeros"),
        InputSpec(shape=(10,), init="randn", seed=43),
    ],
    placements=[
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
        (Replicate(), Replicate()),
    ],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level0"),
))
