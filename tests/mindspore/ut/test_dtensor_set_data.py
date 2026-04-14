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
"""Unit tests for MindSpore DTensor set_data compatibility."""

from types import SimpleNamespace
from unittest.mock import Mock

import mindspore as ms

from hyper_parallel.platform.mindspore.dtensor import DTensorBase


def test_dtensor_set_data_accepts_slice_shape_for_compatibility():
    """`slice_shape` should be accepted and ignored for MindSpore API compatibility."""
    local_update_data = Mock()
    dtensor_update_data = Mock()
    fake_local_tensor = SimpleNamespace(_update_data=local_update_data)
    fake_dtensor = SimpleNamespace(
        _local_tensor=fake_local_tensor,
        _update_data=dtensor_update_data,
    )
    data = ms.Tensor([1, 2, 3], ms.float32)

    DTensorBase.set_data(fake_dtensor, data, slice_shape=True)

    local_update_data.assert_called_once_with(data)
    dtensor_update_data.assert_called_once_with(data)
