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
"""Unit tests for OffloadConfig."""

from __future__ import annotations

import unittest

from hyper_parallel.auto_parallel.hyper_offload.api.config import OffloadConfig


class TestOffloadConfig(unittest.TestCase):
    """OffloadConfig construction and defaults."""

    def test_default_values(self) -> None:
        """Default config should set reasonable defaults."""
        config = OffloadConfig()
        self.assertEqual(config.max_resident_activation_mb, 1024)
        self.assertEqual(config.max_offload_activation_mb, 65536)
        self.assertIsNone(config.planner)

    def test_custom_max_resident(self) -> None:
        """Custom max_resident_activation_mb should be reflected."""
        config = OffloadConfig(max_resident_activation_mb=0)
        self.assertEqual(config.max_resident_activation_mb, 0)

    def test_custom_max_offload(self) -> None:
        """Custom max_offload_activation_mb should be reflected."""
        config = OffloadConfig(max_offload_activation_mb=128)
        self.assertEqual(config.max_offload_activation_mb, 128)

    def test_zero_resident_means_no_device_hold(self) -> None:
        """max_resident_activation_mb=0 means the planner gets a 0 limit."""
        config = OffloadConfig(max_resident_activation_mb=0)
        self.assertEqual(config.max_resident_activation_mb, 0)

    def test_large_offload_pool(self) -> None:
        """A large offload pool should be accepted."""
        config = OffloadConfig(max_offload_activation_mb=131072)
        self.assertEqual(config.max_offload_activation_mb, 131072)

    def test_config_is_dataclass(self) -> None:
        """OffloadConfig should be a dataclass with proper repr."""
        config = OffloadConfig()
        self.assertTrue(repr(config).startswith("OffloadConfig("))
        self.assertIn("max_resident_activation_mb=1024", repr(config))

    def test_config_equality(self) -> None:
        """Two identical configs should be equal (dataclass)."""
        c1 = OffloadConfig()
        c2 = OffloadConfig()
        self.assertEqual(c1, c2)

    def test_config_inequality(self) -> None:
        """Configs with different values should not be equal."""
        c1 = OffloadConfig(max_resident_activation_mb=1)
        c2 = OffloadConfig(max_resident_activation_mb=2)
        self.assertNotEqual(c1, c2)
