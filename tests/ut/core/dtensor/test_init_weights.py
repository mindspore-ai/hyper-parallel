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
"""Unit tests for hyper_parallel.core.dtensor.init_weights."""

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.dtensor import init_weights as init_weights_module

_INIT_ON_DEVICE_TARGET = "hyper_parallel.core.dtensor.init_weights._init_on_device"


@contextmanager
def _fake_init_on_device(device, include_buffers=False):
    """Stand-in for the Torch ``init_on_device`` context manager."""
    del device, include_buffers
    yield


class TestInitEmptyWeights(unittest.TestCase):
    """Tests for InitEmptyWeights."""

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_default_include_buffers_false(self):
        """Test default include buffers false."""
        from hyper_parallel.core.dtensor.init_weights import init_empty_weights

        with init_empty_weights():
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        from hyper_parallel.core.dtensor.init_weights import init_empty_weights

        with init_empty_weights(include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=True
        )


class TestInitOnDevice(unittest.TestCase):
    """Tests for InitOnDevice."""

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_default(self):
        """Test default."""
        from hyper_parallel.core.dtensor.init_weights import init_on_device

        my_device = "npu:0"
        with init_on_device(my_device):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        from hyper_parallel.core.dtensor.init_weights import init_on_device

        my_device = "cpu"
        with init_on_device(my_device, include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=True
        )


if __name__ == "__main__":
    unittest.main()
