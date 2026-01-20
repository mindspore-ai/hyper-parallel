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
"""test torch dtensor with distributed softmax"""
import numpy as np
import torch
from hyper_parallel import Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

# Generate input data using numpy at file header
np.random.seed(42)
# Shape (8, 16), float32
standalone_input_np = np.random.randn(8, 16).astype(np.float32)

def test_distributed_softmax_layout_inference():
    """
    Feature: dtensor + torch.softmax layout inference
    Description:
        ▪ Test layout inference for torch.softmax in distributed setting.
        ▪ Ensure that:
            a) The output layout matches input layout (ActivationWithAxis logic).
            b) Results are consistent between standalone and distributed.
            c) Sharding occurs on non-softmax dimensions.
    Expectation: Success.
    """
    init_dist()
    dim = -1  # Softmax along the last dimension (dim=1)

    # Standalone (single-device) execution for ground truth
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    # Using positional argument for dim to ensure it is captured by _op_dispatch
    standalone_output = torch.softmax(standalone_input, dim)

    # Distributed setup
    # Mesh: (2, 4), Alias: ("dp", "tp")
    layout = Layout((2, 4), ("dp", "tp"))

    # Strategy: Shard on dim=0 ("dp"), Keep dim=1 ("None").
    # Softmax is performed on dim=1, so dim=1 must NOT be sharded.
    x_layout = layout("dp", "None")

    # Convert global view to local shard
    dist_input = global_to_local(standalone_input, x_layout)

    # Execute distributed op
    # 【Fix】: Must use positional argument for 'dim' because _op_dispatch.py
    # does not extract kwargs into extra_args for layout inference.
    # Wrong: torch.softmax(dist_input, dim=dim)
    # Right: torch.softmax(dist_input, dim)
    dist_output = torch.softmax(dist_input, dim)

    # Layout correctness check
    assert dist_output.layout == x_layout, "Torch Softmax: output layout must match input layout"

    # Gather distributed results back to global view
    gathered_output = local_to_global(dist_output)

    # Value correctness check
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "Softmax output mismatch between standalone and distributed"


def test_distributed_softmax_sharded_dim_error():
    """
    Feature: dtensor + torch.softmax error on sharded dim
    Description:
        ▪ Attempt to perform softmax along a sharded dimension.
        ▪ According to ActivationWithAxisDistributedOp.check_layout, this must fail.
    Expectation: Raise ValueError.
    """
    init_dist()
    dim = 1  # Softmax along dim 1

    layout = Layout((2, 4), ("dp", "tp"))

    # Strategy: Keep dim=0 ("dp"), Shard dim=1 ("tp").
    # We will try to Softmax on dim=1, which is sharded.
    x_layout = layout("dp", "tp")

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    dist_input = global_to_local(standalone_input, x_layout)

    try:
        # 【Fix】: Use positional argument for 'dim' here as well
        torch.softmax(dist_input, dim)
        assert False, "Expected ValueError when performing softmax along sharded dimension"
    except ValueError as e:
        # Verify the error message format from ActivationWithAxisDistributedOp
        assert "requires the reduction axis to be un-sharded" in str(e), \
            f"Unexpected error message: {str(e)}"
