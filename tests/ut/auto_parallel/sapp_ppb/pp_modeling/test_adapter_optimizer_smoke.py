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
"""Smoke tests for P0 and P1 fixes — YAML+JSON dual-file architecture.

Tests import paths and PPOptimizer validation with empty paths.
"""

import unittest

from hyper_parallel.auto_parallel.sapp_ppb.pp_optimizer import PPOptimizer


class TestP0P1Fixes(unittest.TestCase):
    """Test P0 and P1 fixes."""

    def test_p0_2_sapp_ppb_import_path(self) -> None:
        """Test P0-2: sapp_ppb modules should be importable from hyper_parallel path."""
        try:
            from hyper_parallel.auto_parallel.sapp_ppb.sapp.sapp_pipeline import SappPipeline  # pylint: disable=C0415
            from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import Layer  # pylint: disable=C0415
            from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute  # pylint: disable=C0415
            from hyper_parallel.auto_parallel.sapp_ppb.simulator.pp_simulator import PipelineSimulator  # pylint: disable=C0415
            self.assertIsNotNone(SappPipeline)
            self.assertIsNotNone(Layer)
            self.assertIsNotNone(Recompute)
            self.assertIsNotNone(PipelineSimulator)
        except ImportError as e:
            import traceback  # pylint: disable=C0415
            traceback.print_exc()
            self.fail(f"P0-2 fix failed: Import error - {e}")

    def test_pp_optimizer_empty_yaml_path_raises(self) -> None:
        """PPOptimizer.optimize with empty yaml_path should raise ValueError."""
        optimizer = PPOptimizer()
        with self.assertRaises(ValueError, msg="yaml_path"):
            optimizer.optimize(yaml_path="", json_path="/tmp/dummy.json")

    def test_pp_optimizer_empty_json_path_raises(self) -> None:
        """PPOptimizer.optimize with empty json_path should raise ValueError."""
        optimizer = PPOptimizer()
        with self.assertRaises(ValueError, msg="json_path"):
            optimizer.optimize(yaml_path="/tmp/dummy.yaml", json_path="")


if __name__ == '__main__':
    unittest.main()
