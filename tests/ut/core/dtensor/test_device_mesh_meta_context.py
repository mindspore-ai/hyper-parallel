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
"""Regression tests: DeviceMesh bookkeeping must survive an ambient ``ms.DeviceCtx("meta")``.

``fully_shard(mesh=None)`` lazily builds the default device mesh *inside* a
``ms.DeviceCtx("meta")`` block. The rank/mesh bookkeeping tensors are read back via
``asnumpy()`` (in ``_refresh_mesh_view`` / ``_build_dim_split_ranks``); if they inherit
the meta device they crash with "Not support copy between src:meta and dst:CPU".
``platform.from_numpy`` keeps them host-resident, which these tests guard.
"""
import os

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_device_mesh,
)

ensure_mindspore_platform_for_device_mesh()

import mindspore as ms  # noqa: E402
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _DEVICE_MESH_MAP  # noqa: E402
from hyper_parallel.platform import get_platform  # noqa: E402


@pytest.fixture(autouse=True)
def _mindspore_platform_and_clean_cache():
    """Keep device_mesh bound to MindSpore (other UTs switch it to Torch) and clear caches."""
    ensure_mindspore_platform_for_device_mesh()
    _DEVICE_MESH_MAP.clear()
    yield
    _DEVICE_MESH_MAP.clear()


class TestDeviceMeshUnderMetaContext:
    """Guard the host-resident rank/mesh bookkeeping under a meta device context."""

    def test_platform_from_numpy_is_host_resident_under_meta(self):
        """
        Feature: platform.from_numpy.
        Description: Build a tensor from numpy inside ms.DeviceCtx("meta").
        Expectation: result is host-resident and round-trips back via tensor_to_numpy.
        """
        plat = get_platform()
        arr = np.array([3, 1, 4, 1], dtype=np.int32)
        with ms.DeviceCtx("meta"):
            tensor = plat.from_numpy(arr)
        assert plat.tensor_to_numpy(tensor).tolist() == [3, 1, 4, 1]

    def test_convert_rank_map_list_branch_under_meta(self):
        """
        Feature: DeviceMesh._convert_rank_map_to_tensor list/tuple branch.
        Description: A rank map built from a python list under ms.DeviceCtx("meta").
        Expectation: the rank map stays asnumpy-able (read back in _build_dim_split_ranks).
        """
        with ms.DeviceCtx("meta"):
            rank_map = DeviceMesh._convert_rank_map_to_tensor([0, 1, 2, 3])
        assert get_platform().tensor_to_numpy(rank_map).tolist() == [0, 1, 2, 3]

    def test_build_device_mesh_under_meta(self):
        """
        Feature: DeviceMesh construction under ms.DeviceCtx("meta").
        Description: Construct a 1-D mesh inside the meta context, as fully_shard(mesh=None) does.
        Expectation: construction succeeds and the rank bookkeeping is host-resident.
        """
        with ms.DeviceCtx("meta"):
            mesh = DeviceMesh(
                "npu", np.array([0, 1]), mesh_dim_names=("fsdp",), _init_backend=False
            )
        assert mesh.rank_list == (0, 1)
        assert mesh._flatten_rank_map == (0, 1)
        np.testing.assert_array_equal(
            get_platform().tensor_to_numpy(mesh._rank_map),
            np.array([0, 1], dtype=np.int32),
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
