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
"""Unit tests for top-level hyper_parallel DTensor API exports."""
from __future__ import annotations

import importlib
import os
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"


class TestHyperParallelPublicExports(unittest.TestCase):
    """Verify DTensor helpers are exported from ``hyper_parallel``."""

    EXPECTED = (
        "distribute_tensor",
        "distribute_module",
        "ones",
        "zeros",
        "empty",
        "full",
        "rand",
        "randn",
        "Shard",
        "Replicate",
        "Partial",
        "Placement",
    )

    def test_all_contains_dtensor_symbols(self):
        import hyper_parallel as hp
        for name in self.EXPECTED:
            self.assertIn(name, hp.__all__, f"{name} missing from hyper_parallel.__all__")

    def test_top_level_imports(self):
        hp = importlib.import_module("hyper_parallel")
        for name in self.EXPECTED:
            self.assertTrue(hasattr(hp, name), f"hyper_parallel.{name} not importable")
            self.assertIsNotNone(getattr(hp, name))


if __name__ == "__main__":
    unittest.main()
