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
"""Unit tests for DeviceTimer."""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from hyper_parallel.auto_parallel.hyper_offload.runtime.timer import DeviceTimer


class TestDeviceTimer(unittest.TestCase):
    """DeviceTimer behaviour."""

    def test_start_no_accelerator_raises(self) -> None:
        """start() raises RuntimeError when no accelerator is available."""
        with patch("torch.accelerator.is_available", return_value=False):
            timer = DeviceTimer()
            with self.assertRaises(RuntimeError):
                timer.start()

    def test_stop_without_start_raises(self) -> None:
        """stop() raises RuntimeError when start() was never called."""
        timer = DeviceTimer()
        with self.assertRaises(RuntimeError):
            timer.stop()

    def test_stop_after_start_no_accelerator_raises(self) -> None:
        """stop() raises RuntimeError when accelerator disappears between
        start() and stop()."""
        timer = DeviceTimer()
        timer._start_event = MagicMock()

        # Accelerator now unavailable
        with patch("torch.accelerator.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                timer.stop()
