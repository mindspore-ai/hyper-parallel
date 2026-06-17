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
"""Unit tests for hyper_parallel.core.dtensor.parameter_init."""

import unittest
from unittest.mock import patch, MagicMock


class TestInitParameters(unittest.TestCase):
    """Tests for InitParameters."""
    @patch("hyper_parallel.platform.get_platform")
    def test_default_stage_index(self, mock_get_platform):
        """Test default stage index."""
        mock_platform = MagicMock()
        mock_get_platform.return_value = mock_platform

        from hyper_parallel.core.dtensor.parameter_init import init_parameters
        module = MagicMock(name="module")
        init_parameters(module)

        mock_platform.init_parameters.assert_called_once_with(module, 0)

    @patch("hyper_parallel.platform.get_platform")
    def test_custom_stage_index(self, mock_get_platform):
        """Test custom stage index."""
        mock_platform = MagicMock()
        mock_get_platform.return_value = mock_platform

        from hyper_parallel.core.dtensor.parameter_init import init_parameters
        module = MagicMock(name="module")
        init_parameters(module, stage_index=2)

        mock_platform.init_parameters.assert_called_once_with(module, 2)

    @patch("hyper_parallel.platform.get_platform")
    def test_returns_platform_result(self, mock_get_platform):
        """Test returns platform result."""
        mock_platform = MagicMock()
        mock_platform.init_parameters.return_value = "result"
        mock_get_platform.return_value = mock_platform

        from hyper_parallel.core.dtensor.parameter_init import init_parameters
        module = MagicMock()
        result = init_parameters(module)

        self.assertEqual(result, "result")


if __name__ == "__main__":
    unittest.main()
