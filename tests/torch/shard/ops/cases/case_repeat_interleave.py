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
"""Shard ops cases for ``torch.repeat_interleave``."""
import numpy as np
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)

np.random.seed(42)
_REPEATS_TENSOR_NP = np.random.randint(1, 4, size=(16,)).astype(np.int64)
_REPEATS_TENSOR = torch.from_numpy(_REPEATS_TENSOR_NP)


def _repeat_interleave_dim_minus1(x):
    return torch.repeat_interleave(x, 3, dim=-1)


def _repeat_interleave_with_tensor(x):
    # repeats must live on the same device as x (CPU gloo vs NPU hccl)
    return torch.repeat_interleave(x, _REPEATS_TENSOR.to(x.device), dim=1)


def _repeat_interleave_dim_none(x):
    return torch.repeat_interleave(x, 3)


register(OpShardCase(
    name="repeat_interleave_ops_layout_inference",
    fn=_repeat_interleave_dim_minus1,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="repeat_interleave_ops_with_tensor",
    fn=_repeat_interleave_with_tensor,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="repeat_interleave_ops_dim_none",
    fn=_repeat_interleave_dim_none,
    inputs=[InputSpec(shape=(8, 16), init="randn", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.allclose(rtol=1e-5, atol=1e-5),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
