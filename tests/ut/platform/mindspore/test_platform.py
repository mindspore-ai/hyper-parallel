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
"""Unit tests for MindSporePlatform communication behavior."""
from unittest import mock

import pytest

ms = pytest.importorskip("mindspore")
nn = pytest.importorskip("mindspore.nn")

from hyper_parallel.platform.mindspore.platform import MindSporePlatform  # pylint: disable=wrong-import-position


def test_prepare_batch_p2p_group_does_not_synchronize():
    """MindSpore batch P2P preparation must not introduce a group barrier."""
    with mock.patch("hyper_parallel.platform.mindspore.platform.dist.barrier") as barrier:
        result = MindSporePlatform.prepare_batch_p2p_group(mock.sentinel.pp_group)

    barrier.assert_not_called()
    assert result is None


def test_buffers_dict_includes_all_registered_buffers():
    """MindSpore buffer enumeration includes persistent and non-persistent buffers."""
    cell = nn.Cell()
    cell.register_buffer("persistent", ms.Tensor([1.0]))
    cell.register_buffer("scratch", ms.Tensor([2.0]), persistent=False)

    buffers = dict(MindSporePlatform.buffers_dict(cell))

    assert set(buffers) == {"persistent", "scratch"}
    assert buffers["persistent"] is cell.persistent
    assert buffers["scratch"] is cell.scratch
