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
"""Unit tests for profile_transfer_bandwidth."""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from hyper_parallel.auto_parallel.hyper_offload.runtime.bandwidth import (
    profile_transfer_bandwidth,
)


class TestProfileTransferBandwidth(unittest.TestCase):
    """profile_transfer_bandwidth behaviour."""

    def test_no_accelerator_raises(self) -> None:
        """When accelerator is not available, should raise RuntimeError."""
        with patch("torch.accelerator.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                profile_transfer_bandwidth()

    def test_runtime_error_propagates(self) -> None:
        """When profiling raises RuntimeError, it should propagate."""
        with patch("torch.accelerator.is_available", return_value=True):
            with patch(
                "torch.accelerator.current_accelerator",
                side_effect=RuntimeError("no device"),
            ):
                with self.assertRaises(RuntimeError):
                    profile_transfer_bandwidth()
