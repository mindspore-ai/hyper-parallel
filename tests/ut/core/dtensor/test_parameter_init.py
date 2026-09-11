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
from unittest.mock import MagicMock

from hyper_parallel.core.dtensor.parameter_init import init_parameters


class TestInitParameters(unittest.TestCase):
    """Tests for init_parameters."""

    def test_accepts_module_and_default_stage_index(self):
        """Test default stage index."""
        init_parameters(MagicMock(name="module"))

    def test_accepts_custom_stage_index(self):
        """Test custom stage index."""
        init_parameters(MagicMock(name="module"), stage_index=2)

    def test_rejects_none_module(self):
        """Test that a None module is rejected."""
        with self.assertRaises(ValueError):
            init_parameters(None)

    def test_rejects_negative_stage_index(self):
        """Test that a negative stage index is rejected."""
        with self.assertRaises(ValueError):
            init_parameters(MagicMock(name="module"), stage_index=-1)


if __name__ == "__main__":
    unittest.main()
