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
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device
from tests.torch.shard.utils import local_to_global, global_to_local

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 16).astype(np.float32)
standalone_input_3d_np = np.random.randn(8, 16, 32).astype(np.float32)


def test_layernorm_data_parallel() -> None:
    """Test layer_norm with data parallel."""
    init_backend(_DEVICE_TYPE)
    normalized_shape = (16,)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    weight = to_device(torch.ones(normalized_shape), _DEVICE_TYPE)
    bias = to_device(torch.zeros(normalized_shape), _DEVICE_TYPE)
    standalone_output = torch.nn.functional.layer_norm(
        standalone_input, normalized_shape, weight, bias
    )

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    dist_input = global_to_local(standalone_input, x_layout)
    dist_weight = global_to_local(weight, _build_layout(mesh, (Replicate(), Replicate()), 1))
    dist_bias = global_to_local(bias, _build_layout(mesh, (Replicate(), Replicate()), 1))

    dist_output = torch.nn.functional.layer_norm(
        dist_input, normalized_shape, dist_weight, dist_bias
    )

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, (
        f"LayerNorm data parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), (
        f"LayerNorm data parallel output mismatch: "
        f"standalone={standalone_output}, distributed={gathered_output}"
    )


def test_layernorm_model_parallel() -> None:
    """Test layer_norm with model parallel."""
    init_backend(_DEVICE_TYPE)
    normalized_shape = (32,)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    weight = to_device(torch.ones(normalized_shape), _DEVICE_TYPE)
    bias = to_device(torch.zeros(normalized_shape), _DEVICE_TYPE)
    standalone_output = torch.nn.functional.layer_norm(
        standalone_input, normalized_shape, weight, bias
    )

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    dist_input = global_to_local(standalone_input, x_layout)
    dist_weight = global_to_local(weight, _build_layout(mesh, (Replicate(), Replicate()), 1))
    dist_bias = global_to_local(bias, _build_layout(mesh, (Replicate(), Replicate()), 1))

    dist_output = torch.nn.functional.layer_norm(
        dist_input, normalized_shape, dist_weight, dist_bias
    )

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert dist_output.layout == expected_layout, (
        f"LayerNorm model parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), (
        f"LayerNorm model parallel output mismatch: "
        f"standalone={standalone_output}, distributed={gathered_output}"
    )


def test_layernorm_hybrid_parallel() -> None:
    """Test layer_norm with hybrid parallel."""
    init_backend(_DEVICE_TYPE)
    normalized_shape = (32,)

    standalone_input = to_device(torch.from_numpy(standalone_input_3d_np), _DEVICE_TYPE)
    weight = to_device(torch.ones(normalized_shape), _DEVICE_TYPE)
    bias = to_device(torch.zeros(normalized_shape), _DEVICE_TYPE)
    standalone_output = torch.nn.functional.layer_norm(
        standalone_input, normalized_shape, weight, bias
    )

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    dist_input = global_to_local(standalone_input, x_layout)
    dist_weight = global_to_local(weight, _build_layout(mesh, (Replicate(), Replicate()), 1))
    dist_bias = global_to_local(bias, _build_layout(mesh, (Replicate(), Replicate()), 1))

    dist_output = torch.nn.functional.layer_norm(
        dist_input, normalized_shape, dist_weight, dist_bias
    )

    expected_layout = _build_layout(mesh, x_placements, 3)
    assert dist_output.layout == expected_layout, (
        f"LayerNorm hybrid parallel layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), (
        f"LayerNorm hybrid parallel output mismatch: "
        f"standalone={standalone_output}, distributed={gathered_output}"
    )


def test_layernorm_all_replicated() -> None:
    """Test layer_norm with all replicated."""
    init_backend(_DEVICE_TYPE)
    normalized_shape = (16,)

    standalone_input = to_device(torch.from_numpy(standalone_input_2d_np), _DEVICE_TYPE)
    weight = to_device(torch.ones(normalized_shape), _DEVICE_TYPE)
    bias = to_device(torch.zeros(normalized_shape), _DEVICE_TYPE)
    standalone_output = torch.nn.functional.layer_norm(
        standalone_input, normalized_shape, weight, bias
    )

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    dist_input = global_to_local(standalone_input, x_layout)
    dist_weight = global_to_local(weight, _build_layout(mesh, (Replicate(), Replicate()), 1))
    dist_bias = global_to_local(bias, _build_layout(mesh, (Replicate(), Replicate()), 1))

    dist_output = torch.nn.functional.layer_norm(
        dist_input, normalized_shape, dist_weight, dist_bias
    )

    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, (
        f"LayerNorm all replicated layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-5), (
        f"LayerNorm all replicated output mismatch: "
        f"standalone={standalone_output}, distributed={gathered_output}"
    )
