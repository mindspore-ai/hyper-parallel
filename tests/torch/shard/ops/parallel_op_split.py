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
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
standalone_input_np = np.random.randn(16, 20).astype(np.float32)


def test_split_layout_inference_default_dim() -> None:
    """Test torch.split layout inference (default dim)."""
    init_backend(_DEVICE_TYPE)
    split_size = 4

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_outputs = torch.split(standalone_input, split_size)
    assert len(standalone_outputs) == 4

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = torch.split(dist_input, split_size)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert isinstance(dist_outputs, tuple)
    assert len(dist_outputs) == 4
    for out in dist_outputs:
        assert isinstance(out, DTensor), f"Expected DTensor, got {type(out)}"
        assert out.layout == expected_layout, (
            f"Output layout {out.layout} != expected {expected_layout}"
        )

    for i, (ref, dist_out) in enumerate(zip(standalone_outputs, dist_outputs)):
        gathered = local_to_global(dist_out)
        assert torch.allclose(ref, gathered, atol=1e-5), f"Chunk {i} mismatch"


def test_split_layout_inference() -> None:
    """Test torch.split layout inference with explicit dim."""
    init_backend(_DEVICE_TYPE)
    split_size = 4
    axis = 1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_outputs = torch.split(standalone_input, split_size, dim=axis)
    assert len(standalone_outputs) == 5

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = torch.split(dist_input, split_size, dim=axis)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert isinstance(dist_outputs, tuple)
    assert len(dist_outputs) == 5
    for out in dist_outputs:
        assert isinstance(out, DTensor), f"Expected DTensor, got {type(out)}"
        assert out.layout == expected_layout, (
            f"Output layout {out.layout} != expected {expected_layout}"
        )

    for i, (ref, dist_out) in enumerate(zip(standalone_outputs, dist_outputs)):
        gathered = local_to_global(dist_out)
        assert torch.allclose(ref, gathered, atol=1e-5), f"Chunk {i} mismatch"


def test_split_layout_inference_split_list() -> None:
    """Test torch.split layout inference with list of split sizes."""
    init_backend(_DEVICE_TYPE)
    split_size = (8, 12)
    axis = 1

    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_outputs = torch.split(standalone_input, split_size, dim=axis)
    assert len(standalone_outputs) == 2

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_outputs = torch.split(dist_input, split_size, dim=axis)

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert isinstance(dist_outputs, tuple)
    assert len(dist_outputs) == 2
    for out in dist_outputs:
        assert isinstance(out, DTensor), f"Expected DTensor, got {type(out)}"
        assert out.layout == expected_layout, (
            f"Output layout {out.layout} != expected {expected_layout}"
        )

    for i, (ref, dist_out) in enumerate(zip(standalone_outputs, dist_outputs)):
        gathered = local_to_global(dist_out)
        assert torch.allclose(ref, gathered, atol=1e-5), f"Chunk {i} mismatch"
