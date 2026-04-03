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
"""Distributed NPU workers for :class:`DeviceMesh` (torchrun, 8 ranks).

Launched from ``test_device_mesh.py`` via ``torchrun_case`` with ``num_proc=8`` (single-node 8-card),
same pattern as ``tests/torch/context_parallel/test_cp_npu.py``.
"""
import torch.distributed as dist

import torch_npu  # noqa: F401  # pylint: disable=unused-import  # side effect: register Ascend NPU

from hyper_parallel import DeviceMesh, init_device_mesh
from hyper_parallel.core.dtensor.device_mesh import _mesh_resources
from tests.torch.utils import init_dist


def _assert_world_size_eight() -> None:
    ws = dist.get_world_size()
    assert ws == 8, f"these cases expect single-node 8-card world_size=8, got {ws}"


def _make_1d_mesh_all_ranks() -> DeviceMesh:
    _assert_world_size_eight()
    ws = dist.get_world_size()
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(ws,),
        mesh_dim_names=("tp",),
    )


def _verify_init_device_mesh_1d_eight_ranks_npu() -> None:
    """
    Feature: init_device_mesh 1-D mesh on 8-rank NPU group
    Description: build mesh_shape (8,) over default process group
    Expectation: rank_list (0..7); current rank in mesh; coordinate matches 1-D index
    """
    mesh = _make_1d_mesh_all_ranks()
    ws = dist.get_world_size()
    assert mesh.ndim == 1
    assert mesh.mesh_shape == (ws,)
    assert mesh.rank_list == tuple(range(ws))
    rk = dist.get_rank()
    assert rk in mesh.rank_list
    coord = mesh.get_coordinate()
    assert coord is not None
    assert len(coord) == 1
    assert coord[0] == rk


def _verify_get_current_mesh_raises_without_context_npu() -> None:
    """
    Feature: get_current_mesh without active ``with mesh`` on distributed rank
    Description: ensure stack empty; call get_current_mesh
    Expectation: RuntimeError on every rank
    """
    _assert_world_size_eight()
    _mesh_resources.mesh_stack.clear()
    try:
        _mesh_resources.get_current_mesh()
    except RuntimeError as e:
        assert "device mesh" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError when no mesh context is active")


def _verify_with_mesh_get_current_mesh_identity_npu() -> None:
    """
    Feature: ``with mesh`` sets thread-local current mesh under torchrun (8 ranks)
    Description: init 1-D mesh over world, enter context on each rank
    Expectation: get_current_mesh() is the same object as mesh; stack empty after exit
    """
    mesh = _make_1d_mesh_all_ranks()
    with mesh:
        assert _mesh_resources.get_current_mesh() is mesh
        assert len(_mesh_resources.mesh_stack) == 1
    assert len(_mesh_resources.mesh_stack) == 0


def _verify_nested_with_get_current_mesh_npu() -> None:
    """
    Feature: nested ``with`` mesh stacks on 8-rank NPU
    Description: outer 1-D mesh (8,); inner 2-D mesh (2, 4) over the same eight ranks
    Expectation: inner block sees inner mesh; after inner exit outer is current
    """
    outer = _make_1d_mesh_all_ranks()
    ws = dist.get_world_size()
    rank_list = tuple(range(ws))
    inner = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, ws // 2),
        mesh_dim_names=("outer", "inner"),
        rank_list=rank_list,
    )
    with outer:
        assert _mesh_resources.get_current_mesh() is outer
        with inner:
            assert _mesh_resources.get_current_mesh() is inner
            assert len(_mesh_resources.mesh_stack) == 2
        assert _mesh_resources.get_current_mesh() is outer
        assert len(_mesh_resources.mesh_stack) == 1
    assert len(_mesh_resources.mesh_stack) == 0


def test_device_mesh_init_1d_eight_ranks_npu():
    """
    Feature: init_device_mesh 1-D mesh on 8-rank NPU (single torchrun job)
    Description: see :func:`_verify_init_device_mesh_1d_eight_ranks_npu`
    Expectation: all assertions pass on every rank
    """
    init_dist()
    _verify_init_device_mesh_1d_eight_ranks_npu()


def test_device_mesh_get_current_mesh_raises_without_context_npu():
    """
    Feature: get_current_mesh without active mesh context on 8-rank NPU
    Description: see :func:`_verify_get_current_mesh_raises_without_context_npu`
    Expectation: all assertions pass on every rank
    """
    init_dist()
    _verify_get_current_mesh_raises_without_context_npu()


def test_device_mesh_with_mesh_current_mesh_identity_npu():
    """
    Feature: ``with mesh`` current mesh identity on 8-rank NPU
    Description: see :func:`_verify_with_mesh_get_current_mesh_identity_npu`
    Expectation: all assertions pass on every rank
    """
    init_dist()
    _verify_with_mesh_get_current_mesh_identity_npu()


def test_device_mesh_nested_with_get_current_mesh_npu():
    """
    Feature: nested ``with`` mesh stack on 8-rank NPU
    Description: see :func:`_verify_nested_with_get_current_mesh_npu`
    Expectation: all assertions pass on every rank
    """
    init_dist()
    _verify_nested_with_get_current_mesh_npu()
