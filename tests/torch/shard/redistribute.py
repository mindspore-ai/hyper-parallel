# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""test torch redistribute"""
import numpy as np
import torch
import pytest
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device


def test_shard_to_replicate():
    '''
    Feature: shard to replicate.
    Description:
    Expectation: Run success.
    '''
    init_backend(_DEVICE_TYPE)

    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2),
        mesh_dim_names=("dp", "tp")
    )

    # Define placements using Placement format
    x_placements = (Replicate(), Shard(1))
    dst_placements = (Replicate(), Replicate())

    dist_x = DTensor.from_local(to_device(torch.ones(8, 4), _DEVICE_TYPE), mesh, x_placements)
    out = dist_x.redistribute(mesh, dst_placements)

    expect_out = torch.ones(8, 8)

    assert np.allclose(expect_out.cpu().detach().numpy(),
                       out.to_local().cpu().detach().numpy(),  # use to_local()
                       0.001, 0.001)


def test_replicate_to_shard():
    '''
    Feature: replicate to shard.
    Description:
    Expectation: Run success.
    '''
    init_backend(_DEVICE_TYPE)

    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2),
        mesh_dim_names=("dp", "tp")
    )

    # Define placements using Placement format
    dst_placements = (Replicate(), Shard(1))
    x_placements = (Replicate(), Replicate())

    dist_x = DTensor.from_local(to_device(torch.ones(8, 8), _DEVICE_TYPE), mesh, x_placements)
    out = dist_x.redistribute(mesh, dst_placements)

    expect_out = torch.ones(8, 4)

    assert np.allclose(expect_out.cpu().detach().numpy(),
                       out.to_local().cpu().detach().numpy(),  # use to_local()
                       0.001, 0.001)


def test_different_mesh():
    '''
    Feature: different mesh, unsupported.
    Description:
    Expectation: Run success.
    '''
    init_backend(_DEVICE_TYPE)

    # Create DeviceMesh with 3 dimensions
    mesh_3d = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2, 1),
        mesh_dim_names=("dp", "tp", "sp")
    )

    # Note: For 3D mesh, "tp" maps to dim 1
    dst_placements = (Replicate(), Shard(1), Replicate())

    # Create DeviceMesh with 2 dimensions
    mesh_2d = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2),
        mesh_dim_names=("dp", "tp")
    )

    x_placements = (Replicate(), Replicate())

    dist_x = DTensor.from_local(to_device(torch.ones(8, 8), _DEVICE_TYPE), mesh_2d, x_placements)

    # Redistributing between different meshes should raise RuntimeError
    with pytest.raises(RuntimeError):
        dist_x.redistribute(mesh_3d, dst_placements)


def test_non_contiguous_redistribute():
    '''
    Feature: redistribute a DTensor whose local tensor is non-contiguous.

    A transpose makes the local storage non-contiguous.  The redistribution
    path must internally ensure contiguity before calling communication ops.
    Description:
    Expectation: Run success — the redistributed result matches the expected full tensor.
    '''
    init_backend(_DEVICE_TYPE)

    mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2),
        mesh_dim_names=("dp", "tp")
    )

    src_placements = (Replicate(), Shard(1))
    dst_placements = (Replicate(), Replicate())

    # Non-contiguous local shard via transpose.
    data = torch.ones(2, 2)
    non_contig_local = data.T  # shape (2, 2), non-contiguous
    assert not non_contig_local.is_contiguous()

    dist_x = DTensor.from_local(to_device(non_contig_local, _DEVICE_TYPE), mesh, src_placements)
    out = dist_x.redistribute(mesh, dst_placements)

    expect_out = torch.ones(2, 4)
    assert np.allclose(expect_out.cpu().detach().numpy(),
                       out.to_local().cpu().detach().numpy(),
                       0.001, 0.001)
