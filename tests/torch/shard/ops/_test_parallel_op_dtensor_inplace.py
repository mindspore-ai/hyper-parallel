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
"""Distributed worker tests for ``DTensor.copy_`` / ``zero_`` / ``fill_``.

Verifies that the autograd-invisible in-place ops on ``DTensor`` produce
results numerically equivalent to the same operations on a standalone
single-device tensor."""

# pylint: disable=protected-access

import numpy as np
import torch

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.torch.shard.utils import local_to_global
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device


np.random.seed(42)
_DST_2D_NP = np.zeros((8, 4), dtype=np.float32)
_SRC_2D_NP = np.random.randn(8, 4).astype(np.float32)
_SRC_2D_FP16_NP = np.random.randn(8, 4).astype(np.float16)


def test_dtensor_copy_same_placement():
    """copy_ between two same-shape same-placement DTensors matches standalone."""
    init_backend(_DEVICE_TYPE)

    standalone_dst = to_device(torch.from_numpy(_DST_2D_NP.copy()), _DEVICE_TYPE)
    standalone_src = to_device(torch.from_numpy(_SRC_2D_NP), _DEVICE_TYPE)
    standalone_dst.copy_(standalone_src)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dist_dst = distribute_tensor(
        to_device(torch.from_numpy(_DST_2D_NP.copy()), _DEVICE_TYPE), mesh, placements
    )
    dist_src = distribute_tensor(
        to_device(torch.from_numpy(_SRC_2D_NP), _DEVICE_TYPE), mesh, placements
    )
    dist_dst.copy_(dist_src)

    gathered = local_to_global(dist_dst)
    assert torch.allclose(standalone_dst, gathered, rtol=1e-5, atol=1e-6), (
        f"DTensor.copy_ same-placement mismatch: standalone={standalone_dst}, parallel={gathered}"
    )


def test_dtensor_copy_dtype_cast():
    """fp16 src copied into fp32 dst preserves dst dtype and casts values."""
    init_backend(_DEVICE_TYPE)

    standalone_dst = to_device(torch.from_numpy(_DST_2D_NP.copy()), _DEVICE_TYPE)
    standalone_src = to_device(torch.from_numpy(_SRC_2D_FP16_NP), _DEVICE_TYPE)
    standalone_dst.copy_(standalone_src)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dist_dst = distribute_tensor(
        to_device(torch.from_numpy(_DST_2D_NP.copy()), _DEVICE_TYPE), mesh, placements
    )
    dist_src = distribute_tensor(
        to_device(torch.from_numpy(_SRC_2D_FP16_NP), _DEVICE_TYPE), mesh, placements
    )
    dist_dst.copy_(dist_src)

    assert dist_dst._local_tensor.dtype == torch.float32, (
        f"copy_ should preserve dst dtype: expected float32, got {dist_dst._local_tensor.dtype}"
    )
    gathered = local_to_global(dist_dst)
    assert torch.allclose(standalone_dst, gathered, rtol=1e-3, atol=1e-3), (
        f"DTensor.copy_ dtype-cast mismatch: standalone={standalone_dst}, parallel={gathered}"
    )


def test_dtensor_copy_scalar_broadcast():
    """0-d Replicate src broadcasts into Shard dst (placement relaxation case)."""
    init_backend(_DEVICE_TYPE)

    scalar_val = 7.0
    standalone_dst = to_device(torch.from_numpy(_DST_2D_NP.copy()), _DEVICE_TYPE)
    standalone_dst.copy_(torch.tensor(scalar_val, device=standalone_dst.device, dtype=torch.float32))

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2,), mesh_dim_names=("dp",))
    dist_dst = distribute_tensor(
        to_device(torch.from_numpy(_DST_2D_NP.copy()), _DEVICE_TYPE), mesh, (Shard(0),)
    )
    dist_scalar = distribute_tensor(
        to_device(torch.tensor(scalar_val, dtype=torch.float32), _DEVICE_TYPE),
        mesh, (Replicate(),)
    )
    dist_dst.copy_(dist_scalar)

    gathered = local_to_global(dist_dst)
    assert torch.allclose(standalone_dst, gathered, rtol=1e-5, atol=1e-6), (
        f"DTensor.copy_ scalar-broadcast mismatch: standalone={standalone_dst}, parallel={gathered}"
    )


def test_dtensor_zero():
    """zero_ resets local shard; full tensor matches standalone zero_."""
    init_backend(_DEVICE_TYPE)

    standalone = to_device(torch.from_numpy(_SRC_2D_NP.copy()), _DEVICE_TYPE)
    standalone.zero_()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2,), mesh_dim_names=("dp",))
    dist_tensor = distribute_tensor(
        to_device(torch.from_numpy(_SRC_2D_NP.copy()), _DEVICE_TYPE), mesh, (Shard(0),)
    )
    dist_tensor.zero_()

    gathered = local_to_global(dist_tensor)
    assert torch.allclose(standalone, gathered, rtol=1e-6, atol=1e-7), (
        f"DTensor.zero_ mismatch: standalone={standalone}, parallel={gathered}"
    )


def test_dtensor_fill():
    """fill_ writes value into local shard; full tensor matches standalone fill_."""
    init_backend(_DEVICE_TYPE)

    fill_val = 4.25
    standalone = to_device(torch.from_numpy(_DST_2D_NP.copy()), _DEVICE_TYPE)
    standalone.fill_(fill_val)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2,), mesh_dim_names=("dp",))
    dist_tensor = distribute_tensor(
        to_device(torch.from_numpy(_DST_2D_NP.copy()), _DEVICE_TYPE), mesh, (Replicate(),)
    )
    dist_tensor.fill_(fill_val)

    gathered = local_to_global(dist_tensor)
    assert torch.allclose(standalone, gathered, rtol=1e-6, atol=1e-7), (
        f"DTensor.fill_ mismatch: standalone={standalone}, parallel={gathered}"
    )
