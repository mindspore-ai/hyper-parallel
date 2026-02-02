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
"""test torch dtensor with distributed repeat_interleave"""

import numpy as np
import torch
from hyper_parallel import Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local
# Generate input data using numpy at file header
np.random.seed(42)
standalone_input_np = np.random.randn(8, 16).astype(np.float32)
def test_distributed_repeat_interleave_layout_inference():
    """
    Feature: dtensor + torch.repeat_interleave layout inference
    Description:
        - Test layout inference for torch.repeat_interleave in distributed setting.
        - Ensure that:
            a) The output layout matches input layout.
            b) Results are consistent between standalone and distributed.
    Expectation: Success.
    """
    init_dist()
    # Simple repeat with integer repeats
    repeats = 3
    dim = -1  # last dimension
    # Standalone (single-device)
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_output = torch.repeat_interleave(standalone_input, repeats, dim=dim)
    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "None")  # shard on dim=0 (dp), keep dim=1 unsharded
    
    dist_input = global_to_local(standalone_input, x_layout)
    dist_output = torch.repeat_interleave(dist_input, repeats, dim=dim)
    
    # Layout correctness check
    assert dist_output.layout == x_layout, "Torch repeat_interleave: output layout mismatch input"
    
    # Gather distributed results back to global view
    gathered_output = local_to_global(dist_output) 
    
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "repeat_interleave output mismatch between standalone and distributed"

def test_distributed_repeat_interleave_with_tensor():
    """
    Feature: dtensor + torch.repeat_interleave with repeats_tensor
    Description:
        - Test layout inference for torch.repeat_interleave in distributed setting.
        - Ensure that:
            a) The output layout matches input layout.
            b) Results are consistent between standalone and distributed.
    Expectation: Success.
    """
    init_dist()
    
    # Create repeats tensor
    repeats_tensor_np = np.random.randint(1, 4, size=(16,)).astype(np.int64)
    repeats_tensor = torch.from_numpy(repeats_tensor_np).npu()
    
    dim = 1  # second dimension
    
    # Standalone (single-device)
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_output = torch.repeat_interleave(standalone_input, repeats_tensor, dim=dim)
    
    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "None")  # shard on dim=0 (dp), keep dim=1 unsharded
    
    dist_input = global_to_local(standalone_input, x_layout)
    dist_output = torch.repeat_interleave(dist_input, repeats_tensor, dim=dim)
    
    # Gather distributed results back to global view
    gathered_output = local_to_global(dist_output)
    
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "repeat_interleave_with_tensor output mismatch between standalone and distributed"

def test_distributed_repeat_interleave_dim_none():
    """
    Feature: dtensor + torch.repeat_interleave dim None
    Description:
        - Test layout inference for torch.repeat_interleave in distributed setting.
        - Ensure that:
            Results are consistent between standalone and distributed.
    Expectation: Success.
    """
    init_dist()

    repeats = 3
    
    # Standalone (single-device)
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_output = torch.repeat_interleave(standalone_input, repeats)
    
    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "None")  # shard on dim=0 (dp), keep dim=1 unsharded
    
    dist_input = global_to_local(standalone_input, x_layout)
    dist_output = torch.repeat_interleave(dist_input, repeats)
    
    # # Layout correctness check
    # assert dist_output.layout == x_layout, "Torch repeat_interleave: output layout mismatch input"
    
    # Gather distributed results back to global view
    gathered_output = local_to_global(dist_output) 
    
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "repeat_interleave_dim_None output mismatch between standalone and distributed"

def test_distributed_repeat_interleave_sharded_dim_error():
    """
    Feature: dtensor + torch.repeat_interleave error on sharded dim
    Description:
        - Attempt to perform repeat_interleave along a sharded dimension.
        - Should raise ValueError during layout inference.
    Expectation: Raise ValueError.
    """
    init_dist()
    repeats = 2
    dim = 1  
    
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "tp")  # shard on both dimensions
    
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    dist_input = global_to_local(standalone_input, x_layout)
    
    try:
        torch.repeat_interleave(dist_input, repeats, dim=dim)
        assert False, "Expected ValueError when repeat_interleave along sharded dimension"
    except ValueError as e:
        assert "Cannot perform sharding on params along the chosen dim" in str(e)
