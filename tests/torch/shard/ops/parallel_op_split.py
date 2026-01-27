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
"""test torch dtensor with distributed split"""
import numpy as np
import torch
from hyper_parallel import DTensor, Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import global_to_local, local_to_global

# Generate input data using numpy at file header
np.random.seed(42)
standalone_input_np = np.random.randn(16, 20).astype(np.float32)

def test_distributed_split_layout_inference_default_dim():
    """
    Feature: dtensor + torch.split layout inference
    Description:
        - Test layout inference for torch.split in distributed setting.
        - Ensure that:
            a) Output layouts match input layout when splitting on unsharded dim.
            b) Results are consistent with standalone execution.
    Expectation: Success.
    """
    init_dist()
    split_size = 4

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_outputs = torch.split(standalone_input, split_size)
    assert len(standalone_outputs) == 4  # 16 / 4 = 4 chunks

    # Distributed setup
    layout = Layout((4, 2), ("dp", "tp"))
    # Shard only on dim=0
    x_layout = layout("None", "dp")

    dist_input = global_to_local(standalone_input, x_layout)

    # Perform split
    dist_outputs = torch.split(dist_input, split_size)

    assert isinstance(dist_outputs, tuple)
    assert len(dist_outputs) == 4
    for out in dist_outputs:
        assert isinstance(out, DTensor)
        assert out.layout == x_layout, f"Output layout {out.layout} != input layout {x_layout}"

    for i, (ref, dist_out) in enumerate(zip(standalone_outputs, dist_outputs)):
        gathered = local_to_global(dist_out)
        assert torch.allclose(ref, gathered, atol=1e-5), f"Chunk {i} mismatch"


def test_distributed_split_layout_inference():
    """
    Feature: dtensor + torch.split layout inference
    Description:
        - Test layout inference for torch.split in distributed setting.
        - Ensure that:
            a) Output layouts match input layout when splitting on unsharded dim.
            b) Results are consistent with standalone execution.
    Expectation: Success.
    """
    init_dist()
    split_size = 4
    axis = 1  # split along axis 1

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_outputs = torch.split(standalone_input, split_size, dim=axis)
    assert len(standalone_outputs) == 5  # 20 / 4 = 5 chunks

    # Distributed setup
    layout = Layout((4, 2), ("dp", "tp"))
    # Shard only on dim=0
    x_layout = layout("dp", "None")

    dist_input = global_to_local(standalone_input, x_layout)

    # Perform split
    dist_outputs = torch.split(dist_input, split_size, dim=axis)

    assert isinstance(dist_outputs, tuple)
    assert len(dist_outputs) == 5
    for out in dist_outputs:
        assert isinstance(out, DTensor)
        assert out.layout == x_layout, f"Output layout {out.layout} != input layout {x_layout}"

    for i, (ref, dist_out) in enumerate(zip(standalone_outputs, dist_outputs)):
        gathered = local_to_global(dist_out)
        assert torch.allclose(ref, gathered, atol=1e-5), f"Chunk {i} mismatch"


def test_distributed_split_layout_inference_split_list():
    """
    Feature: dtensor + torch.split layout inference
    Description:
        - Test layout inference for torch.split in distributed setting.
        - Ensure that:
            a) Output layouts match input layout when splitting on unsharded dim.
            b) Results are consistent with standalone execution.
    Expectation: Success.
    """
    init_dist()
    split_size = (8, 12)
    axis = 1  # split along axis 1

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_outputs = torch.split(standalone_input, split_size, dim=axis)
    assert len(standalone_outputs) == 2

    # Distributed setup
    layout = Layout((4, 2), ("dp", "tp"))
    # Shard only on dim=0
    x_layout = layout("dp", "None")

    dist_input = global_to_local(standalone_input, x_layout)

    # Perform split
    dist_outputs = torch.split(dist_input, split_size, dim=axis)

    assert isinstance(dist_outputs, tuple)
    assert len(dist_outputs) == 2
    for out in dist_outputs:
        assert isinstance(out, DTensor)
        assert out.layout == x_layout, f"Output layout {out.layout} != input layout {x_layout}"

    for i, (ref, dist_out) in enumerate(zip(standalone_outputs, dist_outputs)):
        gathered = local_to_global(dist_out)
        assert torch.allclose(ref, gathered, atol=1e-5), f"Chunk {i} mismatch"


def test_distributed_split_on_sharded_dim_error():
    """
    Feature: dtensor + torch.split error on sharded axis
    Description:
        - Attempt to split along a sharded dimension.
        - Should raise ValueError during layout inference.
    Expectation: Raise ValueError with expected message.
    """
    init_dist()
    split_size = 4
    axis = 0  # this dim is sharded

    layout = Layout((4, 2), ("dp", "tp"))
    x_layout = layout("dp", "tp")

    local_tensor = torch.randn(4, 12).npu()
    dist_input = DTensor.from_local(local_tensor, x_layout)

    try:
        torch.split(dist_input, split_size, dim=axis)
        assert False, "Expected ValueError when splitting along sharded dimension"
    except ValueError as e:
        assert "Can not split tensor at sharded axis" in str(e)
        assert "layout:" in str(e)
