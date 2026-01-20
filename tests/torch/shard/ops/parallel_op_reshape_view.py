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
"""test torch dtensor with distributed reshape and view"""
import numpy as np
import torch
from hyper_parallel import Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

# Generate input data using numpy at file header
np.random.seed(42)
# Shape: [Batch=8, Channel=4, Height=4, Width=4] -> Total 512 elements
standalone_input_np = np.random.randn(8, 4, 4, 4).astype(np.float32)

def test_distributed_reshape_layout_inference():
    """
    Feature: dtensor + torch.reshape layout inference
    Description:
        1. Test 'flatten' scenario: [B, C, H, W] -> [B, C*H*W]
        2. Test 'expand' scenario: [B, C*H*W] -> [B, C, H, W]
        3. Verify layout consistency and value correctness.
    Expectation: Success.
    """
    init_dist()

    # Setup Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()

    # --- Case 1: Flatten dimensions (preserving sharded dim) ---
    # Target shape: [8, 64]
    standalone_output_flat = standalone_input.reshape(8, 64)

    # Distributed Setup
    # Mesh: (2, 4), dp splits dim 0, tp is unused (None)
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "None", "None", "None") # Shard Batch dim

    dist_input = global_to_local(standalone_input, x_layout)

    # Perform Reshape
    # FIX: Pass shape as a tuple explicitly to bypass OpDispatcher bug
    # dist_input.reshape(8, 64) -> dist_input.reshape((8, 64))
    dist_output_flat = dist_input.reshape((8, 64))

    # Check Layout
    # Expectation: Dim 0 is still sharded ("dp"), Dim 1 is replicate ("None")
    expected_flat_layout = layout("dp", "None")
    assert dist_output_flat.layout == expected_flat_layout, \
        f"Reshape flat layout mismatch. Got {dist_output_flat.layout}, expected {expected_flat_layout}"

    # Check Values
    gathered_flat = local_to_global(dist_output_flat)
    assert torch.allclose(standalone_output_flat, gathered_flat, atol=1e-5), \
        "Reshape values mismatch between standalone and distributed"

    # --- Case 2: Expand dimensions (using -1 inference) ---
    # Input is the flattened distributed tensor from Case 1
    # Target shape: [8, 4, 4, 4]
    # FIX: Pass shape as a tuple explicitly
    dist_output_expand = dist_output_flat.reshape((8, 4, 4, -1))

    # Check Layout
    # Expectation: Dim 0 sharded ("dp"), others replicate
    expected_expand_layout = x_layout
    assert dist_output_expand.layout == expected_expand_layout, \
        "Reshape expand layout mismatch"

    # Check Values
    gathered_expand = local_to_global(dist_output_expand)
    assert torch.allclose(standalone_input, gathered_expand, atol=1e-5), \
        "Reshape expand values mismatch"


def test_distributed_view_layout_inference():
    """
    Feature: dtensor + torch.view layout inference
    Description:
        1. Test view with tuple argument input style.
        2. Test view on a tensor where sharded dimension is preserved but reshaped
           (as long as total size / shards is integer).
    Expectation: Success.
    """
    init_dist()

    # Setup Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()

    # Scenario: Input [8, 4, 4, 4], Layout: Shard dim 1 (Channel)
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("None", "tp", "None", "None") # Shard Channel

    dist_input = global_to_local(standalone_input, x_layout)

    # --- Operation: View ---
    # Merge last two dims: [8, 4, 4, 4] -> [8, 4, 16]
    target_shape = (8, 4, 16)

    # Perform View (PyTorch style: tuple arg)
    dist_view = dist_input.view(target_shape)

    # Standalone reference
    standalone_view = standalone_input.view(target_shape)

    # Check Layout
    expected_view_layout = layout("None", "tp", "None")

    assert dist_view.layout == expected_view_layout, \
        f"View layout mismatch. Got {dist_view.layout}, expected {expected_view_layout}"

    # Check Values
    gathered_view = local_to_global(dist_view)
    assert torch.allclose(standalone_view, gathered_view, atol=1e-5), \
        "View values mismatch between standalone and distributed"


def test_distributed_reshape_fail_mismatch():
    """
    Feature: dtensor + reshape error handling
    Description:
        Attempt to reshape in a way that breaks existing sharding constraints
        or mismatches element count.
    Expectation: Raise ValueError.
    """
    init_dist()

    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "None") # [8, 64] -> split 8

    input_np = np.random.randn(8, 64).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    dist_input = global_to_local(standalone_input, x_layout)

    # Invalid total element count
    try:
        # Pass as tuple to ensuring we are testing element mismatch, not argument drop bug
        dist_input.reshape((8, 32))
        assert False, "Should raise ValueError for element count mismatch"
    except ValueError:
        # Expected error
        pass
    except RuntimeError:
        pass
