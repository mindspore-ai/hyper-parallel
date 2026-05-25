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
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 16).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 4, 4).astype(np.float32)


def test_softmax_data_parallel() -> None:
    """Test softmax with data parallel."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.softmax(standalone_input, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.softmax(dist_input, dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        (f"Softmax data parallel layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, rtol=1e-5, atol=1e-5), \
        (f"Softmax data parallel output mismatch: "
         f"standalone={standalone_output}, parallel={gathered_output}")


def test_softmax_model_parallel() -> None:
    """Test softmax with model parallel."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.softmax(standalone_input, dim=0)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.softmax(dist_input, dim=0)

    expected_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        (f"Softmax model parallel layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, rtol=1e-5, atol=1e-5), \
        (f"Softmax model parallel output mismatch: "
         f"standalone={standalone_output}, parallel={gathered_output}")


def test_softmax_hybrid_parallel() -> None:
    """Test softmax with hybrid parallel."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    standalone_output = torch.softmax(standalone_input, dim=1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(2))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.softmax(dist_input, dim=1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        (f"Softmax hybrid parallel layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, rtol=1e-5, atol=1e-5), \
        (f"Softmax hybrid parallel output mismatch: "
         f"standalone={standalone_output}, parallel={gathered_output}")


def test_softmax_all_replicated() -> None:
    """Test softmax with all replicated."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.softmax(standalone_input, dim=-1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.softmax(dist_input, dim=-1)

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        (f"Softmax all replicated layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, rtol=1e-5, atol=1e-5), \
        (f"Softmax all replicated output mismatch: "
         f"standalone={standalone_output}, parallel={gathered_output}")


def test_softmax_negative_dim() -> None:
    """Test softmax with negative dim."""
    init_backend(_DEVICE_TYPE)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    standalone_output = torch.softmax(standalone_input, dim=-1)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.softmax(dist_input, dim=-1)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        (f"Softmax negative dim layout mismatch: "
         f"expected={expected_layout}, got={dist_output.layout}")

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, rtol=1e-5, atol=1e-5), \
        (f"Softmax negative dim output mismatch: "
         f"standalone={standalone_output}, parallel={gathered_output}")
