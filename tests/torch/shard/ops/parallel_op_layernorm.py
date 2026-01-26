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
"""test torch dtensor with distributed layer_norm"""

import numpy as np
import torch
from hyper_parallel import Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

# Generate input data using numpy at file header
np.random.seed(42)
standalone_input_np = np.random.randn(8, 16).astype(np.float32)

def test_distributed_layer_norm_layout_inference():
    """
    Feature: dtensor + torch.nn.functional.layer_norm layout inference
    Description:
        - Test layout inference for layer_norm in distributed setting.
        - Ensure that:
            a) Output layout matches input layout when normalized dims are unsharded.
            b) Results are consistent between standalone and distributed.
    Expectation: Success.
    """
    init_dist()
    normalized_shape = (16,)  # normalize over last dimension

    # Standalone (single-device)
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    weight = torch.ones(normalized_shape).npu()
    bias = torch.zeros(normalized_shape).npu()
    standalone_output = torch.nn.functional.layer_norm(
        standalone_input, normalized_shape, weight, bias
    )

    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "None")

    dist_input = global_to_local(standalone_input, x_layout)
    # Weight and bias must be replicated (same on all devices)
    dist_weight = global_to_local(weight, layout("None",))
    dist_bias = global_to_local(bias, layout("None",))

    dist_output = torch.nn.functional.layer_norm(
        dist_input, normalized_shape, dist_weight, dist_bias
    )

    assert dist_output.layout == x_layout, "LayerNorm: output layout mismatch input"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "LayerNorm output mismatch between standalone and distributed"


def test_distributed_layer_norm_sharded_normalized_dim_error():
    """
    Feature: dtensor + layer_norm error on sharded normalized dimension
    Description:
        - Attempt to run layer_norm where normalized_shape includes a sharded dimension.
        - Should raise ValueError during layout inference.
    Expectation: Raise ValueError.
    """
    init_dist()
    normalized_shape = (16,)

    layout = Layout((2, 4), ("dp", "tp"))
    # WRONG: shard on dim=1, which is part of normalized_shape → invalid
    x_layout = layout("dp", "tp")

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    dist_input = global_to_local(standalone_input, x_layout)
    dist_weight = global_to_local(torch.ones(normalized_shape).npu(), layout("None",))
    dist_bias = global_to_local(torch.zeros(normalized_shape).npu(), layout("None",))

    try:
        torch.nn.functional.layer_norm(dist_input, normalized_shape, dist_weight, dist_bias)
        assert False, "Expected ValueError when normalized dimension is sharded"
    except ValueError as e:
        assert "Cannot perform sharding on normalized dimension" in str(e)
        assert "dimension 1" in str(e) or "dimension -1" in str(e)
