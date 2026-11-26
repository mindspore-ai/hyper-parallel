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
"""In-place log values, storage and wrapper identity on replicated and sharded DTensors."""
import numpy as np
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.shard_ops.framework import CompareSpec, InputSpec, OpShardCase, register


def _log_inplace(value):
    """Compare in-place log with out-of-place log followed by copy, including return identity."""
    # The runner reuses its inputs between reference and parallel calls.
    source = value.clone().requires_grad_()
    value = source.clone()
    # Equivalence with log()+copy_() applies to no_grad initialization.
    with torch.no_grad():
        expected = value.clone()
        expected.copy_(expected.log())
    local = value.to_local() if hasattr(value, "to_local") else value
    storage = local.data_ptr()
    result = value.log_()
    assert result is value, f"Expected the input object {id(value)}, got {id(result)}"
    actual_local = result.to_local() if hasattr(result, "to_local") else result
    expected_local = expected.to_local() if hasattr(expected, "to_local") else expected
    assert actual_local.data_ptr() == storage, (
        f"Expected storage {storage}, got {actual_local.data_ptr()}"
    )
    torch.testing.assert_close(actual_local, expected_local, rtol=0, atol=0)
    source_local = source.to_local() if hasattr(source, "to_local") else source
    gradient, = torch.autograd.grad(actual_local.sum(), source_local)
    torch.testing.assert_close(gradient, source_local.reciprocal(), rtol=1e-6, atol=1e-6)
    return result


for _name, _placement in (("replicate", Replicate()), ("dp", Shard(0)), ("tp", Shard(-1))):
    register(OpShardCase(
        name=f"log_inplace_ops_{_name}",
        fn=_log_inplace,
        inputs=[InputSpec(shape=(8, 4), data=np.linspace(.01, 16, 32, dtype=np.float32).reshape(8, 4))],
        placements=[(_placement,)],
        compare=CompareSpec.equal(),
        mesh_shape=(2,),
        mesh_dim_names=("dp",),
        tags=("cpu_level0", "npu_level0"),
    ))
