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
"""Shard ops cases for ``torch.empty_like``.

.. note::
    ``empty_like`` creates tensors with uninitialised memory, so standalone
    and distributed runs produce different random values.  The old tests only
    validated shape and layout.  These cases use zeros(init) instead to make
    the output deterministic for comparison.
"""
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _empty_like(x):
    return torch.empty_like(x)


register(OpShardCase(
    name="empty_like_ops_dp",
    fn=_empty_like,
    inputs=[InputSpec(shape=(4, 8), init="zeros", seed=42)],
    placements=[(Shard(0), Replicate())],
    compare=CompareSpec.shape(),  # uninitialized memory; verify shape/dtype only
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))

register(OpShardCase(
    name="empty_like_ops_tp",
    fn=_empty_like,
    inputs=[InputSpec(shape=(4, 8), init="zeros", seed=42)],
    placements=[(Replicate(), Shard(1))],
    compare=CompareSpec.shape(),  # uninitialized memory; verify shape/dtype only
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level1", "npu_level1"),
))
