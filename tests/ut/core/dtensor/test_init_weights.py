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
from unittest.mock import patch, MagicMock


class TestInitEmptyWeights(unittest.TestCase):
    """Tests for InitEmptyWeights."""
    @patch("hyper_parallel.core.dtensor.init_weights.platform")
    def test_default_include_buffers_false(self, mock_platform):
        """Test default include buffers false."""
        mock_meta_device = MagicMock(name="meta_device")
        mock_platform.meta_device = mock_meta_device

        @contextmanager
        def fake_init_on_device(device, include_buffers=False):
            """Fake init on device."""
            yield

        mock_platform.init_on_device = MagicMock(side_effect=fake_init_on_device)

        from hyper_parallel.core.dtensor.init_weights import init_empty_weights
        with init_empty_weights():
            pass

        mock_platform.init_on_device.assert_called_once_with(mock_meta_device, include_buffers=False)

    @patch("hyper_parallel.core.dtensor.init_weights.platform")
    def test_include_buffers_true(self, mock_platform):
        """Test include buffers true."""
        mock_meta_device = MagicMock(name="meta_device")
        mock_platform.meta_device = mock_meta_device

        @contextmanager
        def fake_init_on_device(device, include_buffers=False):
            """Fake init on device."""
            yield

        mock_platform.init_on_device = MagicMock(side_effect=fake_init_on_device)

        from hyper_parallel.core.dtensor.init_weights import init_empty_weights
        with init_empty_weights(include_buffers=True):
            pass

        mock_platform.init_on_device.assert_called_once_with(mock_meta_device, include_buffers=True)


class TestInitOnDevice(unittest.TestCase):
    """Tests for InitOnDevice."""
    @patch("hyper_parallel.core.dtensor.init_weights.platform")
    def test_default(self, mock_platform):
        """Test default."""
        @contextmanager
        def fake_init_on_device(device, include_buffers=False):
            """Fake init on device."""
            yield

        mock_platform.init_on_device = MagicMock(side_effect=fake_init_on_device)

        from hyper_parallel.core.dtensor.init_weights import init_on_device
        my_device = "npu:0"
        with init_on_device(my_device):
            pass

        mock_platform.init_on_device.assert_called_once_with(my_device, include_buffers=False)

    @patch("hyper_parallel.core.dtensor.init_weights.platform")
    def test_include_buffers_true(self, mock_platform):
        """Test include buffers true."""
        @contextmanager
        def fake_init_on_device(device, include_buffers=False):
            """Fake init on device."""
            yield

        mock_platform.init_on_device = MagicMock(side_effect=fake_init_on_device)

        from hyper_parallel.core.dtensor.init_weights import init_on_device
        my_device = "cpu"
        with init_on_device(my_device, include_buffers=True):
            pass

        mock_platform.init_on_device.assert_called_once_with(my_device, include_buffers=True)


if __name__ == "__main__":
    unittest.main()
