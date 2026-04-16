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
from unittest.mock import Mock, patch

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


def test_dtensor_data_setter_updates_wrapper_and_local_tensor():
    """Assigning ``dtensor.data = x`` should synchronize both wrapper and local tensor payloads."""
    base_setter = Mock()
    local_setter = Mock()
    fake_dtensor = SimpleNamespace(
        _set_base_data=base_setter,
        _set_local_tensor_data=local_setter,
    )
    data = ms.Tensor([1, 2, 3], ms.float32)

    DTensorBase.data.fset(fake_dtensor, data)

    base_setter.assert_called_once_with(data)
    local_setter.assert_called_once_with(data)


def test_dtensor_data_setter_uses_local_tensor_for_dtensor_input():
    """Assigning another DTensor should propagate its local shard payload."""
    base_setter = Mock()
    local_setter = Mock()
    input_dtensor = SimpleNamespace(to_local=Mock(return_value="local-shard"))
    fake_dtensor = SimpleNamespace(
        _set_base_data=base_setter,
        _set_local_tensor_data=local_setter,
    )

    original_isinstance = isinstance

    def fake_isinstance(obj, cls):
        if cls is DTensorBase:
            return obj is input_dtensor
        return original_isinstance(obj, cls)

    with patch("builtins.isinstance", side_effect=fake_isinstance):
        DTensorBase.data.fset(fake_dtensor, input_dtensor)

    input_dtensor.to_local.assert_called_once_with()
    base_setter.assert_called_once_with("local-shard")
    local_setter.assert_called_once_with("local-shard")
