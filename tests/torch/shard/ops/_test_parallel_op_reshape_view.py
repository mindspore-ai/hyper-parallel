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
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


# Generate input data using numpy at file header
np.random.seed(42)
# Shape: [Batch=8, Channel=4, Height=4, Width=4] -> Total 512 elements
standalone_input_np = np.random.randn(8, 4, 4, 4).astype(np.float32)


def test_reshape_layout_inference() -> None:
    """
    Test torch.reshape layout inference.

    Description:
        1. Test 'flatten' scenario: [B, C, H, W] -> [B, C*H*W]
        2. Test 'expand' scenario: [B, C*H*W] -> [B, C, H, W]
        3. Verify layout consistency and value correctness.
    """
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)

    # --- Case 1: Flatten dimensions (preserving sharded dim) ---
    standalone_output_flat = standalone_input.reshape(8, 64)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output_flat = dist_input.reshape((8, 64))

    expected_flat_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output_flat.layout == expected_flat_layout, (
        f"Reshape flat layout mismatch: "
        f"expected={expected_flat_layout}, got={dist_output_flat.layout}"
    )

    gathered_flat = local_to_global(dist_output_flat)
    assert torch.allclose(standalone_output_flat, gathered_flat, atol=1e-5), (
        "Reshape values mismatch between standalone and distributed"
    )

    # --- Case 2: Expand dimensions (using -1 inference) ---
    dist_output_expand = dist_output_flat.reshape((8, 4, 4, -1))

    expected_expand_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate(), Replicate()), 4)
    assert dist_output_expand.layout == expected_expand_layout, (
        f"Reshape expand layout mismatch: "
        f"expected={expected_expand_layout}, got={dist_output_expand.layout}"
    )

    gathered_expand = local_to_global(dist_output_expand)
    assert torch.allclose(standalone_input, gathered_expand, atol=1e-5), (
        "Reshape expand values mismatch"
    )


def test_view_layout_inference() -> None:
    """
    Test torch.view layout inference.

    Description:
        1. Test view with tuple argument input style.
        2. Test view on a tensor where sharded dimension is preserved but reshaped
           (as long as total size / shards is integer).
    """
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    target_shape = (8, 4, 16)
    dist_view = dist_input.view(target_shape)
    standalone_view = standalone_input.view(target_shape)

    expected_view_layout = _build_layout(mesh, (Replicate(), Shard(1), Replicate()), 3)
    assert dist_view.layout == expected_view_layout, (
        f"View layout mismatch: "
        f"expected={expected_view_layout}, got={dist_view.layout}"
    )

    gathered_view = local_to_global(dist_view)
    assert torch.allclose(standalone_view, gathered_view, atol=1e-5), (
        "View values mismatch between standalone and distributed"
    )
