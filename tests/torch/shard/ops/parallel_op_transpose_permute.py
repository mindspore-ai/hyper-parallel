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
"""test torch dtensor with distributed transpose and permute"""

import numpy as np
import torch
from hyper_parallel import Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

# Generate input data using numpy at file header
# Shape: (8, 16, 4) - distinct dimensions to easily detect swapping errors
np.random.seed(42)
standalone_input_np = np.random.randn(8, 16, 4).astype(np.float32)

def test_distributed_permute_layout_inference():
    """
    Feature: dtensor + torch.permute layout inference
    Description:
        ▪ Test layout inference for torch.permute in distributed setting.
        ▪ Scenario:
            - Input Layout: ("dp", "tp", "None") on a (8, 16, 4) tensor.
            - Permute dims: (2, 0, 1).
        ▪ Ensure that:
            a) The output layout correctly reorders the tensor map.
            b) The data values match standalone execution.
    Expectation: Success.
    """
    init_dist()

    # Permutation order
    dims = (2, 0, 1)

    # Standalone (single-device global view)
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_output = torch.permute(standalone_input, dims)

    # Distributed setup
    # Mesh: 2x4 (8 devices)
    # Layout strategy: dim0 -> dp (shard), dim1 -> tp (shard), dim2 -> None (replicate)
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "tp", "None")

    dist_input = global_to_local(standalone_input, x_layout)

    # Execute distributed op
    dist_output = torch.permute(dist_input, dims)

    # Layout correctness check
    # Input map: (dp, tp, None). Permute (2, 0, 1) -> Output map should be (None, dp, tp)
    expected_layout = layout("None", "dp", "tp")
    assert dist_output.layout == expected_layout, \
        f"Torch permute: layout mismatch. Expected {expected_layout.to_dict()}, got {dist_output.layout.to_dict()}"

    # Gather distributed results back to global view
    gathered_output = local_to_global(dist_output)

    # Data correctness check
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "Permute values mismatch between standalone and distributed"

    assert gathered_output.shape == standalone_output.shape, \
        "Permute shape mismatch between standalone and distributed"


def test_distributed_transpose_layout_inference():
    """
    Feature: dtensor + torch.transpose layout inference
    Description:
        ▪ Test layout inference for torch.transpose in distributed setting.
        ▪ Scenario:
            - Input Layout: ("dp", "tp", "None").
            - Transpose dims: (1, 2). Swapping a sharded dim with a replicated dim.
        ▪ Ensure that:
            a) The output layout correctly swaps the tensor map entries.
            b) The data values match standalone execution.
    Expectation: Success.
    """
    init_dist()

    # Transpose dimensions
    dim0, dim1 = 1, 2

    # Standalone (single-device global view)
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_output = torch.transpose(standalone_input, dim0, dim1)

    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "tp", "None")

    dist_input = global_to_local(standalone_input, x_layout)

    # Execute distributed op
    dist_output = torch.transpose(dist_input, dim0, dim1)

    # Layout correctness check
    # Input map: (dp, tp, None). Swap 1 and 2 -> Output map should be (dp, None, tp)
    expected_layout = layout("dp", "None", "tp")

    assert dist_output.layout == expected_layout, \
        f"Torch transpose: layout mismatch. Expected {expected_layout.to_dict()}, got {dist_output.layout.to_dict()}"

    # Gather distributed results back to global view
    gathered_output = local_to_global(dist_output)

    # Data correctness check
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "Transpose values mismatch between standalone and distributed"


def test_distributed_transpose_negative_dim():
    """
    Feature: dtensor + torch.transpose with negative dimensions
    Description:
        ▪ Test layout inference for torch.transpose using negative indices.
        ▪ Scenario: Transpose dim 0 and dim -1 (last dim).
    Expectation: Success.
    """
    init_dist()

    dim0, dim1 = 0, -1

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_output = torch.transpose(standalone_input, dim0, dim1)

    # Distributed setup
    layout = Layout((2, 4), ("dp", "tp"))
    x_layout = layout("dp", "tp", "None")

    dist_input = global_to_local(standalone_input, x_layout)
    dist_output = torch.transpose(dist_input, dim0, dim1)

    # Layout correctness check
    # Input map: (dp, tp, None). Swap 0 and -1(2) -> Output map should be (None, tp, dp)
    expected_layout = layout("None", "tp", "dp")

    assert dist_output.layout == expected_layout, (
        f"Torch transpose (neg dim): layout mismatch. "
        f"Expected {expected_layout.to_dict()}, "
        f"got {dist_output.layout.to_dict()}"
    )

    # Verify data
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Transpose (neg dim) values mismatch"
