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
"""Unit tests for :class:`AsyncHandle` in ``hyper_parallel.platform``."""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.platform import AsyncHandle


class TestAsyncHandle(unittest.TestCase):
    """Unit tests for :class:`AsyncHandle`."""

    def test_wait_calls_platform_once(self):
        """First wait calls get_platform().wait_async_tensor; second wait is a no-op."""
        mock_tensor = MagicMock()
        handle = AsyncHandle(mock_tensor)

        with patch("hyper_parallel.platform.platform.get_platform") as mock_get_plat:
            mock_plat = mock_get_plat.return_value
            mock_plat.wait_async_tensor.return_value = mock_tensor
            result1 = handle.wait()
            result2 = handle.wait()

        self.assertEqual(mock_plat.wait_async_tensor.call_count, 1,
                         "wait_async_tensor should be called exactly once")
        self.assertIs(result1, mock_tensor)
        self.assertIs(result2, mock_tensor)

    def test_wait_returns_tensor(self):
        """wait returns the wrapped async tensor after materialisation."""
        real_tensor = torch.randn(4)
        handle = AsyncHandle(real_tensor)

        with patch("hyper_parallel.platform.platform.get_platform") as mock_get_plat:
            mock_plat = mock_get_plat.return_value
            mock_plat.wait_async_tensor.side_effect = lambda t: t
            result = handle.wait()

        self.assertIs(result, real_tensor)

    def test_initial_state(self):
        """Newly created AsyncHandle has _waited=False."""
        mock_tensor = MagicMock()
        handle = AsyncHandle(mock_tensor)
        self.assertFalse(handle._waited)

    def test_waited_state_after_wait(self):
        """After wait(), _waited becomes True."""
        mock_tensor = MagicMock()
        handle = AsyncHandle(mock_tensor)

        with patch("hyper_parallel.platform.platform.get_platform") as mock_get_plat:
            mock_plat = mock_get_plat.return_value
            mock_plat.wait_async_tensor.return_value = mock_tensor
            handle.wait()

        self.assertTrue(handle._waited)


if __name__ == "__main__":
    unittest.main()
