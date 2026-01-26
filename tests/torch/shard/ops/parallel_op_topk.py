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
"""test torch dtensor with distributed topk"""

import numpy as np
import torch
from hyper_parallel import Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

# Generate input data using numpy at file header
np.random.seed(42)
standalone_input_np = np.random.randn(8, 16).astype(np.float32)

def test_distributed_topk_layout_inference():
    """
    Feature: dtensor + torch.topk layout inference
    Description:
        - Test layout inference for torch.topk in distributed setting.
        - Ensure that:
            a) The output layouts (values, indices) match input layout.
            b) Results are consistent between standalone and distributed.
    Expectation: Success.
    """
    init_dist()
    k = 5
    dim = -1  # last dimension

    # Standalone (single-device)
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_values, standalone_indices = torch.topk(standalone_input, k, dim=dim)

    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "None") # shard on dim=0 (dp), keep dim=1 unsharded

    dist_input = global_to_local(standalone_input, x_layout)
    dist_values, dist_indices = torch.topk(dist_input, k, dim=dim)

    # Layout correctness check
    assert dist_values.layout == x_layout, "Torch TopK: output values layout mismatch input"
    assert dist_indices.layout == x_layout, "Torch TopK: output indices layout mismatch input"

    # Gather distributed results back to global view
    gathered_values = local_to_global(dist_values)
    gathered_indices = local_to_global(dist_indices)

    assert torch.allclose(
        standalone_values, gathered_values, atol=1e-5
    ), "Topk values mismatch between standalone and distributed"

    assert torch.equal(
        standalone_indices, gathered_indices
    ), "Topk indices mismatch between standalone and distributed"


def test_distributed_topk_sharded_dim_error():
    """
    Feature: dtensor + torch.topk error on sharded dim
    Description:
        - Attempt to perform topk along a sharded dimension.
        - Should raise ValueError during layout inference.
    Expectation: Raise ValueError.
    """
    init_dist()
    k = 3
    dim = 1  # this dim will be sharded

    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "tp")

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    dist_input = global_to_local(standalone_input, x_layout)

    try:
        torch.topk(dist_input, k, dim=dim)
        assert False, "Expected ValueError when topk along sharded dimension"
    except ValueError as e:
        assert "Cannot perform sharding on params along the chosen dim" in str(e)
