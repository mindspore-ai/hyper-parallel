# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test torch dtensor with distributed prod"""
import numpy as np
import torch
from hyper_parallel import Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

# Generate input data using numpy at file header
# Use uniform distribution [0.5, 1.5] to avoid 0 and maintain numerical stability for product
np.random.seed(42)
standalone_input_np = np.random.uniform(0.5, 1.5, (8, 16)).astype(np.float32)

def test_distributed_prod_unsharded_dim():
    """
    Feature: dtensor + torch.prod layout inference (unsharded dim)
    Description:
        ▪ Test layout inference for torch.prod when reducing along an unsharded dimension.
        ▪ Ensure that:
            a) The output layout is correctly inferred (rank reduction).
            b) Results are consistent between standalone and distributed.
    Expectation: Success.
    """
    init_dist()
    dim = 1

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_result = torch.prod(standalone_input, dim=dim)

    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    # Shard dim 0 (dp), keep dim 1 (unsharded/None)
    x_layout = layout("dp", "None")

    dist_input = global_to_local(standalone_input, x_layout)

    # FIX: Use positional arguments for dim and keepdim.
    # The layout inference mechanism relies on *args to capture dim.
    # Passing kwargs (dim=dim) causes the dispatcher to miss the argument,
    # leading to incorrect layout inference (scalar instead of vector).
    # Signature: torch.prod(input, dim, keepdim=False)
    dist_result = torch.prod(dist_input, dim, False)

    # Layout correctness check
    # Input: [dp, None]. Reduce dim 1 (None).
    # Output should be [dp] (1D tensor sharded on dp).
    expected_layout = layout("dp")
    assert dist_result.layout == expected_layout, \
        f"Layout mismatch. Got {dist_result.layout}, expected {expected_layout}"

    # Ensure no partial state since we reduced a replicated dimension
    assert not dist_result.layout.is_partial(), "Result should not be partial for unsharded dim reduction"

    # Gather distributed results back to global view
    gathered_result = local_to_global(dist_result)

    assert torch.allclose(
        standalone_result, gathered_result, atol=1e-5
    ), "Prod values mismatch between standalone and distributed (unsharded dim)"


def test_distributed_prod_sharded_dim():
    """
    Feature: dtensor + torch.prod layout inference (sharded dim)
    Description:
        ▪ Test layout inference for torch.prod when reducing along a sharded dimension.
        ▪ Ensure that:
            a) The output layout is marked as PARTIAL (with op 'prod').
            b) After resolving partial (reduce_partial), results match standalone.
    Expectation: Success.
    """
    init_dist()
    dim = 1

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_result = torch.prod(standalone_input, dim=dim)

    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    # Shard dim 0 (dp) AND dim 1 (tp)
    x_layout = layout("dp", "tp")

    dist_input = global_to_local(standalone_input, x_layout)

    # FIX: Use positional arguments to ensure correct layout inference
    dist_result = torch.prod(dist_input, dim, False)

    # Layout correctness check
    # We reduced dim 1 which corresponds to 'tp'.
    # The result should be partial on 'tp'.
    assert dist_result.layout.is_partial(), "Result should be partial when reducing sharded dim"
    assert dist_result.layout.get_partial_by_dev_id("tp") == "prod", \
        "Partial operator should be 'prod' for tp axis"

    # Resolve partial state (triggers communication/allreduce)
    # WARNING: If this step calculates a Sum instead of Prod, check tensor_redistribution.py
    final_result = dist_result.reduce_partial()

    # Gather distributed results back to global view
    gathered_result = local_to_global(final_result)

    # If this assertion fails with gathered_result being much larger than standalone_result,
    # it implies the backend executed Sum instead of Prod.
    assert torch.allclose(
        standalone_result, gathered_result, atol=1e-5
    ), "Prod values mismatch between standalone and distributed (sharded dim)"
