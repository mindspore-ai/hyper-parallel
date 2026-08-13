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
"""Unit tests for PPOptimizer — YAML+JSON dual-file architecture."""

import os
import unittest

from hyper_parallel.auto_parallel.sapp_ppb.pp_modeling.pp_structs import PPStrategyResult
from hyper_parallel.auto_parallel.sapp_ppb.pp_optimizer import PPOptimizer

_DEMO_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixture_pp8_32layers.yaml",
)

_DEMO_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixture_profile_32layers.json",
)


class TestPPOptimizer(unittest.TestCase):
    """Test PPOptimizer class — YAML+JSON dual-file architecture."""

    def test_optimizer_creation(self) -> None:
        """Test creating optimizer instance."""
        optimizer = PPOptimizer()
        self.assertIsNotNone(optimizer)

    def test_optimize_yaml_json_driven(self) -> None:
        """Test YAML+JSON driven optimization."""
        from hyper_parallel.auto_parallel.sapp_ppb.pp_config_builder.layer_loader import SAPP_PPB_AVAILABLE  # pylint: disable=C0415

        if not SAPP_PPB_AVAILABLE:
            self.skipTest("sapp_ppb not available")

        optimizer = PPOptimizer()
        result = optimizer.optimize(yaml_path=_DEMO_YAML, json_path=_DEMO_JSON)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, PPStrategyResult)
        self.assertGreater(result.pp_degree, 0)
        self.assertGreater(result.micro_batch_num, 0)

    def test_optimize_empty_yaml_path_raises(self) -> None:
        """Test optimization with empty yaml_path raises ValueError."""
        optimizer = PPOptimizer()
        with self.assertRaises(ValueError) as ctx:
            optimizer.optimize(yaml_path="", json_path=_DEMO_JSON)
        self.assertIn("yaml_path", str(ctx.exception))

    def test_optimize_empty_json_path_raises(self) -> None:
        """Test optimization with empty json_path raises ValueError."""
        optimizer = PPOptimizer()
        with self.assertRaises(ValueError) as ctx:
            optimizer.optimize(yaml_path=_DEMO_YAML, json_path="")
        self.assertIn("json_path", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
