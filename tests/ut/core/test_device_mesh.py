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
"""Unit tests for DeviceMesh functionality.

This module provides comprehensive unit tests for the DeviceMesh class and
related functions in the hyper_parallel framework. All tests use mocking to avoid
dependencies on actual hardware or distributed communication.

Note:
    All test methods decorated with ``@patch("hyper_parallel.core.dtensor.device_mesh._utils")``
    receive a ``mock_utils`` argument injected by the patch decorator.
"""
import functools
import unittest
from unittest.mock import patch, MagicMock

from hyper_parallel.core.dtensor.device_mesh import (
    DeviceMesh,
    _DEVICE_MESH_MAP,
    _create_device_mesh,
    init_device_mesh,
)
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS


_ACTIVE_DIST_PATCHERS = []


def _patch_device_mesh_backend(mock_utils, world_size=8):
    """Patch ``device_mesh.dist`` and mirror its rank helpers onto ``mock_utils``.

    ``device_mesh`` reads rank information from ``torch.distributed`` while group
    creation stays in ``device_mesh._utils``; both are exposed through the same
    ``mock_utils`` handle that the ``@patch`` decorator injects. The patch is
    registered for teardown (``unittest`` ``addCleanup`` when available) so it
    never leaks into a sibling test.
    """
    del _ACTIVE_DIST_PATCHERS[:]
    patcher = patch("hyper_parallel.core.dtensor.device_mesh.dist")
    mock_dist = patcher.start()
    _ACTIVE_DIST_PATCHERS.append(patcher)
    mock_utils.get_rank = mock_dist.get_rank
    mock_utils.get_world_size = mock_dist.get_world_size
    mock_utils.get_process_group_ranks = mock_dist.get_process_group_ranks
    mock_dist.get_rank.return_value = 0
    mock_dist.get_world_size.return_value = world_size
    mock_utils.split_group.return_value = MagicMock(name="group")
    return mock_utils


def _patch_backend_for(fn):
    """Wrap a module-level test so its ``device_mesh.dist`` patch is torn down after it."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            _stop_device_mesh_backend_patches()

    return wrapper


def _setup_device_mesh_mock(mock_utils, world_size=8):
    """Configure a hardware-free mock of the ``device_mesh`` module helpers."""
    _patch_device_mesh_backend(mock_utils, world_size)


def _stop_device_mesh_backend_patches():
    """Stop any ``device_mesh.dist`` patches started by the mock helpers."""
    while _ACTIVE_DIST_PATCHERS:
        _ACTIVE_DIST_PATCHERS.pop().stop()


@_patch_backend_for
@patch("hyper_parallel.core.dtensor.device_mesh._utils")
def test_device_mesh_view_cache_prefers_rooted_mesh_for_layout_rebuild(mock_utils):
    """
    Feature: DeviceMesh global cache for rooted views.
    Description: Rebuild a Layout from an unflatten/sliced view after a rootless cache entry exists.
    Expectation: Layout reuses the rooted view mesh so later concatenate sees a shared root.
    """
    _setup_device_mesh_mock(mock_utils, world_size=8)
    EXISTING_COMM_GROUPS.clear()
    _DEVICE_MESH_MAP.clear()

    mesh_shape = (1, 1, 2, 4)
    mesh_dim_names = ("pp", "dp_replicate", "fsdp", "tp")
    rootless_dense_mesh = _create_device_mesh(
        "npu",
        mesh_shape,
        mesh_dim_names=mesh_dim_names,
        rank_list=tuple(range(8)),
        init_backend=False,
    )
    rootless_tp_mesh = _create_device_mesh(
        "npu",
        (4,),
        mesh_dim_names=("tp",),
        rank_list=(0, 1, 2, 3),
        init_backend=False,
    )
    world_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("world",))
    dense_mesh = world_mesh._unflatten(  # pylint: disable=protected-access
        0,
        mesh_shape,
        mesh_dim_names,
        backend_override={"pp": "fake", "tp": "fake"},
    )
    tp_mesh = dense_mesh["tp"]

    rebuilt_dense_layout = Layout(
        dense_mesh.mesh_shape,
        dense_mesh.mesh_dim_names,
        rank_list=dense_mesh.rank_list,
        init_backend=False,
    )
    rebuilt_layout = Layout(
        tp_mesh.mesh_shape,
        tp_mesh.mesh_dim_names,
        rank_list=tp_mesh.rank_list,
        init_backend=False,
    )
    fsdp_tp_mesh = DeviceMesh.concatenate([dense_mesh["fsdp"], rebuilt_layout.mesh])

    assert rootless_dense_mesh.root_mesh is None
    assert rootless_tp_mesh.root_mesh is None
    assert rebuilt_dense_layout.mesh is dense_mesh
    assert rebuilt_dense_layout.mesh.root_mesh is world_mesh
    assert rebuilt_layout.mesh is tp_mesh
    assert rebuilt_layout.mesh.root_mesh is world_mesh
    assert fsdp_tp_mesh.mesh_dim_names == ("fsdp", "tp")
    assert fsdp_tp_mesh.to_hash() == dense_mesh[("fsdp", "tp")].to_hash()


@_patch_backend_for
@patch("hyper_parallel.core.dtensor.device_mesh._utils")
def test_device_mesh_view_cache_keeps_root_mesh_when_view_matches_root_key(mock_utils):
    """
    Feature: DeviceMesh global cache for root-equivalent views.
    Description: Concatenate all root dimensions and then rebuild the root Layout.
    Expectation: The original root mesh remains the cached mesh for the root key.
    """
    _setup_device_mesh_mock(mock_utils, world_size=4)
    EXISTING_COMM_GROUPS.clear()
    _DEVICE_MESH_MAP.clear()

    root_mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"))
    concat_mesh = DeviceMesh.concatenate([root_mesh["dp"], root_mesh["tp"]])
    rebuilt_layout = Layout(
        root_mesh.mesh_shape,
        root_mesh.mesh_dim_names,
        rank_list=root_mesh.rank_list,
        init_backend=False,
    )

    assert concat_mesh.root_mesh is root_mesh
    assert concat_mesh.to_hash() == root_mesh.to_hash()
    assert rebuilt_layout.mesh is root_mesh


if __name__ == "__main__":
    unittest.main()
