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
"""test torch dtensor with distributed conv3d"""

import numpy as np
import torch
import torch.nn.functional as F
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)

def test_distributed_conv3d_data_parallel():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Data Parallel)
    Description:
        - Input is sharded on batch dimension (N). Weight is replicated.
        - Verify result matches standalone convolution.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    # Input (8, 2, 4, 4, 4), Weight (4, 2, 2, 2, 2)
    input_np = np.random.randn(8, 2, 4, 4, 4).astype(np.float32)
    weight_np = np.random.randn(4, 2, 2, 2, 2).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight)

    # Distributed: Input sharded on batch (dp), Weight replicated
    in_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight)

    # Validation
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Data Parallel Conv3d output mismatch"

def test_distributed_conv3d_column_parallel():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Column Parallel)
    Description:
        - Weight is sharded on output channel dimension (C_out).
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    input_np = np.random.randn(4, 2, 4, 4, 4).astype(np.float32)
    weight_np = np.random.randn(4, 2, 2, 2, 2).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight)

    # Distributed: Weight sharded on C_out (0)
    in_placements = (Replicate(), Replicate())
    w_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Column Parallel Conv3d output mismatch"
def test_distributed_conv3d_spatial_parallel():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Spatial Parallel)
    Description:
        - Input is sharded on the Depth dimension (D).
        - Verify that sharding on spatial axes is preserved through convolution.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    # Input (4, 2, 8, 4, 4), Weight (4, 2, 2, 2, 2)
    # Shard Input on Depth dim=2
    input_np = np.random.randn(4, 2, 8, 4, 4).astype(np.float32)
    weight_np = np.random.randn(4, 2, 2, 2, 2).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight)

    # Distributed: Input sharded on depth (Shard dim=2)
    in_placements = (Replicate(), Replicate(), Shard(0), Replicate(), Replicate())
    w_placements = (Replicate(), Replicate(), Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Spatial Parallel Conv3d output mismatch"

def test_distributed_conv3d_with_bias():
    """
    Feature: dtensor + torch.nn.functional.conv3d (with Bias)
    Description:
        - Verify Conv3d with bias term when output channels are sharded.
    Expectation: Success, bias should be aligned and correctly added.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    input_np = np.random.randn(4, 2, 4, 4, 4).astype(np.float32)
    weight_np = np.random.randn(4, 2, 2, 2, 2).astype(np.float32)
    bias_np = np.random.randn(4).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_bias = to_device(torch.from_numpy(bias_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight, bias=standalone_bias)

    # Distributed: Shard Weight and Bias on C_out (0)
    w_placements = (Shard(0), Replicate(), Replicate(), Replicate(), Replicate())
    b_placements = (Shard(0),)
    in_placements = (Replicate(), Replicate(), Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)
    dist_bias = distribute_tensor(standalone_bias, mesh, b_placements)

    dist_output = F.conv3d(dist_input, dist_weight, bias=dist_bias)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Conv3d with Bias output mismatch"
def test_distributed_conv3d_row_parallel():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Row Parallel)
    Description:
        - Input is sharded on the Input Channel dimension (C_in).
        - Weight is sharded on the Input Channel dimension (C_in).
        - Generates Partial sums which are automatically reduced by the framework.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    # Input (2, 4, 4, 4, 4), Weight (4, 4, 2, 2, 2)
    input_np = np.random.randn(2, 4, 4, 4, 4).astype(np.float32)
    weight_np = np.random.randn(4, 4, 2, 2, 2).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight)

    # Distributed: Shard Input and Weight on C_in (dim=1) along the 'tp' mesh dimension
    in_placements = (Replicate(), Shard(1))
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Row Parallel Conv3d output mismatch"


def test_distributed_conv3d_dp_cp():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Data + Column Parallel 2D)
    Description:
        - Input is sharded on Batch (N) along 'dp'.
        - Weight is sharded on Output Channel (C_out) along 'tp'.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    input_np = np.random.randn(4, 2, 4, 4, 4).astype(np.float32)
    weight_np = np.random.randn(4, 2, 2, 2, 2).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight)

    # Distributed: Input sharded on N (dp), Weight sharded on C_out (tp)
    in_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Data + Column Parallel Conv3d output mismatch"


def test_distributed_conv3d_dp_rp():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Data + Row Parallel 2D)
    Description:
        - Input is sharded on Batch (N) along 'dp' and Input Channel (C_in) along 'tp'.
        - Weight is sharded on Input Channel (C_in) along 'tp'.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    input_np = np.random.randn(4, 4, 4, 4, 4).astype(np.float32)
    weight_np = np.random.randn(2, 4, 2, 2, 2).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight)

    # Distributed: Input sharded on N (dp) and C_in (tp)
    in_placements = (Shard(0), Shard(1))
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Data + Row Parallel Conv3d output mismatch"


def test_distributed_conv3d_spatial_h():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Spatial Parallel on Height)
    Description:
        - Input is sharded on the Height dimension (H, dim=3).
        - Kernel size is 1x1x1 to ensure mathematical equivalence without halo exchange.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    # Shard Input on Height dim=3
    input_np = np.random.randn(2, 2, 4, 8, 4).astype(np.float32)
    weight_np = np.random.randn(4, 2, 1, 1, 1).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight)

    # Distributed: Input sharded on height (Shard dim=3) along 'tp'
    in_placements = (Replicate(), Shard(3))
    w_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Spatial Parallel (H) Conv3d output mismatch"


def test_distributed_conv3d_spatial_w():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Spatial Parallel on Width)
    Description:
        - Input is sharded on the Width dimension (W, dim=4).
        - Kernel size is 1x1x1 to ensure mathematical equivalence without halo exchange.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    # Shard Input on Width dim=4
    input_np = np.random.randn(2, 2, 4, 4, 8).astype(np.float32)
    weight_np = np.random.randn(4, 2, 1, 1, 1).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight)

    # Distributed: Input sharded on width (Shard dim=4) along 'tp'
    in_placements = (Replicate(), Shard(4))
    w_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Spatial Parallel (W) Conv3d output mismatch"


def test_distributed_conv3d_groups_dp():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Groups > 1, Data Parallel)
    Description:
        - Input is sharded on the Batch dimension (N).
        - Convolution uses groups=2.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    input_np = np.random.randn(4, 4, 4, 4, 4).astype(np.float32)
    # C_out=4, C_in/groups=2
    weight_np = np.random.randn(4, 2, 2, 2, 2).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight, groups=2)

    in_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight, groups=2)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Groups DP Conv3d output mismatch"


def test_distributed_conv3d_groups_cp():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Groups > 1, Column Parallel)
    Description:
        - Weight is sharded on the Output Channel dimension (C_out).
        - Convolution uses groups=2.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    input_np = np.random.randn(2, 4, 4, 4, 4).astype(np.float32)
    # C_out=4, C_in/groups=2
    weight_np = np.random.randn(4, 2, 2, 2, 2).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight, groups=2)

    in_placements = (Replicate(), Replicate())
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.conv3d(dist_input, dist_weight, groups=2)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Groups CP Conv3d output mismatch"


def test_distributed_conv3d_groups_cp_with_bias():
    """
    Feature: dtensor + torch.nn.functional.conv3d (Groups > 1, Column Parallel + Bias)
    Description:
        - Weight and Bias are sharded on the Output Channel dimension (C_out).
        - Convolution uses groups=2.
    Expectation: Success with numerical equivalence.
    """
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    input_np = np.random.randn(2, 4, 4, 4, 4).astype(np.float32)
    # C_out=4, C_in/groups=2
    weight_np = np.random.randn(4, 2, 2, 2, 2).astype(np.float32)
    bias_np = np.random.randn(4).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_weight = to_device(torch.from_numpy(weight_np), _DEVICE_TYPE)
    standalone_bias = to_device(torch.from_numpy(bias_np), _DEVICE_TYPE)
    standalone_output = F.conv3d(standalone_input, standalone_weight, bias=standalone_bias, groups=2)

    in_placements = (Replicate(), Replicate())
    w_placements = (Replicate(), Shard(0))
    b_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, in_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)
    dist_bias = distribute_tensor(standalone_bias, mesh, b_placements)

    dist_output = F.conv3d(dist_input, dist_weight, bias=dist_bias, groups=2)

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), \
        "Groups CP Conv3d with Bias output mismatch"
