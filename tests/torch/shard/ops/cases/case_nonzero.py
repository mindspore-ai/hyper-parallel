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
"""Shard ops cases for ``torch.Tensor.nonzero``.
.. note::
    ``nonzero`` produces output whose shape depends on the number of non-zero
    elements.  When inputs are sharded the number of non-zero elements on each
    rank may differ, making value comparison challenging.  The old tests only
    used Replicated placement to avoid this, so all cases here also use
    Replicated.
"""

from hyper_parallel.core.dtensor.placement_types import Replicate
from tests.shard_ops.framework import (
    CompareSpec,
    InputSpec,
    OpShardCase,
    register,
)


def _nonzero(x):
    return x.nonzero()


register(OpShardCase(
    name="nonzero_ops_basic",
    fn=_nonzero,
    inputs=[InputSpec(shape=(3, 3), init="arange", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))

register(OpShardCase(
    name="nonzero_ops_as_tuple",
    fn=_nonzero,
    inputs=[InputSpec(shape=(3, 3), init="arange", seed=42)],
    placements=[(Replicate(), Replicate())],
    compare=CompareSpec.equal(),
    mesh_shape=(2, 2),
    mesh_dim_names=("dp", "tp"),
    tags=("cpu_level0", "npu_level1"),
))
