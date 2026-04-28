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
"""MindSpore STs for DeviceMesh.concatenate."""

import numpy as np
import pytest

import mindspore as ms
from mindspore.communication import get_rank, init

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh


def _assert_mesh_matches(mesh: DeviceMesh, expected_shape, expected_names, expected_ranks):
    """Assert basic DeviceMesh properties that should agree across all ranks."""
    assert mesh.mesh_shape == expected_shape, (
        f"mesh_shape mismatch: expected {expected_shape}, got {mesh.mesh_shape}"
    )
    assert mesh.mesh_dim_names == expected_names, (
        f"mesh_dim_names mismatch: expected {expected_names}, got {mesh.mesh_dim_names}"
    )
    assert tuple(mesh.rank_list) == tuple(expected_ranks), (
        f"rank_list mismatch: expected {tuple(expected_ranks)}, got {tuple(mesh.rank_list)}"
    )
    assert mesh.rank in mesh.rank_list, (
        f"Current rank {mesh.rank} should participate in concatenated mesh {mesh.rank_list}"
    )


def test_ms_device_mesh_concatenate_supports_root_and_flattened_dims():
    """
    Feature: MindSpore DeviceMesh.concatenate with root and flattened dims.
    Description: Build a 2D (fsdp, tp) root mesh, then concatenate ``fsdp`` with
        both the original ``tp`` mesh and the flattened ``tp_flat`` mesh.
    Expectation: Concatenated meshes have the expected shape, names, and rank lists.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank = get_rank()

    root_mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("fsdp", "tp"))
    fsdp_mesh = root_mesh["fsdp"]
    tp_mesh = root_mesh["tp"]
    tp_flat_mesh = tp_mesh.flatten("tp_flat")

    concat_root_mesh = DeviceMesh.concatenate([fsdp_mesh, tp_mesh])
    _assert_mesh_matches(concat_root_mesh, (2, 2), ("fsdp", "tp"), root_mesh.rank_list)
    assert concat_root_mesh.to_hash() == root_mesh.to_hash(), (
        f"Concatenating root dims should reconstruct root mesh on rank {rank}"
    )

    concat_flat_mesh = DeviceMesh.concatenate([fsdp_mesh, tp_flat_mesh])
    _assert_mesh_matches(concat_flat_mesh, (2, 2), ("fsdp", "tp_flat"), root_mesh.rank_list)
    assert np.array_equal(
        root_mesh.mesh.asnumpy(),
        concat_flat_mesh.mesh.asnumpy(),
    ), f"Flattened concatenate should preserve root mesh tensor on rank {rank}"


def test_ms_device_mesh_concatenate_rejects_out_of_root_order():
    """
    Feature: MindSpore DeviceMesh.concatenate validates original root order.
    Description: Build a root mesh with dim order ``(tp, fsdp)`` and try to
        concatenate ``fsdp`` before ``tp``.
    Expectation: Concatenate raises ValueError for violating root mesh order.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()

    root_mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("tp", "fsdp"))
    fsdp_mesh = root_mesh["fsdp"]
    tp_mesh = root_mesh["tp"]

    with pytest.raises(ValueError, match="follow the root mesh order"):
        DeviceMesh.concatenate([fsdp_mesh, tp_mesh])
