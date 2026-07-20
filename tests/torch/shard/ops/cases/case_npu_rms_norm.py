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
"""Shard ops cases for ``torch_npu.npu_rms_norm``.

.. note::
    Ascend-specific operator (``npu_*``); all cases run on NPU only
    (``npu_level0``). ``npu_rms_norm`` returns ``(output, rstd)``;
    only ``output[0]`` (the normalized result) is compared.

    Uses 3-D input ``(batch, seq, hidden)`` so that MP shards the
    non-normalized ``seq`` dimension — the last dim (hidden) is the
    normalization axis and must remain replicated.
"""
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _rms_norm(x, gamma):
    import torch_npu  # pylint: disable=C0415
    return torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)[0]


register(OpShardCase(
    name="npu_rms_norm_ops_dp",
    fn=_rms_norm,
    inputs=[
        InputSpec(shape=(8, 16, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(32,), dtype="float16", init="ones"),
    ],
    placements=[
        (Shard(0), Replicate(), Replicate()),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="npu_rms_norm_ops_mp",
    fn=_rms_norm,
    inputs=[
        InputSpec(shape=(8, 16, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(32,), dtype="float16", init="ones"),
    ],
    placements=[
        (Replicate(), Shard(1), Replicate()),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="npu_rms_norm_ops_hybrid",
    fn=_rms_norm,
    inputs=[
        InputSpec(shape=(8, 16, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(32,), dtype="float16", init="ones"),
    ],
    placements=[
        (Shard(0), Shard(1), Replicate()),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))

register(OpShardCase(
    name="npu_rms_norm_ops_replicated",
    fn=_rms_norm,
    inputs=[
        InputSpec(shape=(8, 16, 32), dtype="float16", init="randn", seed=42),
        InputSpec(shape=(32,), dtype="float16", init="ones"),
    ],
    placements=[
        (Replicate(), Replicate(), Replicate()),
        (Replicate(),),
    ],
    compare=CompareSpec.allclose(rtol=1e-3, atol=1e-3),
    tags=("npu_level0",),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
))
