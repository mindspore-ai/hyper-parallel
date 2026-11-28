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
"""Verify that counter-based target planning uses the counter's exact window."""

from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[3] / ".agent/skills/mfu-riser/scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    CONTRACT = runpy.run_path(str(SCRIPTS / "riser_contract.py"))


class TestCounterWindow(unittest.TestCase):
    """Exercise measured and historical or estimated receipt formats."""

    def test_projection_uses_only_sampled_model_work(self):
        """Use the sampled steps when useful work and timing vary across the window."""
        model_samples = [
            {"step": 9, "time_s": 1.0, "model_flops": 20e12},
            {"step": 10, "time_s": 2.0, "model_flops": 20e12},
            {"step": 11, "time_s": 2.0, "model_flops": 40e12},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "counters.json"
            path.write_text("{}", encoding="utf-8")
            receipt = {
                "method": "hardware_counters",
                "coverage": "Cube and Vector",
                "evidence": path.name,
                "sha256": CONTRACT["hash_file"](path),
                "samples": [
                    {"step": 11, "executed_flops": 90e12},
                    {"step": 9, "executed_flops": 30e12},
                ],
            }
            metrics = CONTRACT["_executed_work_metrics"](receipt, model_samples, 100e12, root, {})
        self.assertEqual(metrics["hfu_steps"], [9, 11])
        self.assertAlmostEqual(metrics["hfu_percent"], 40.0)
        self.assertAlmostEqual(metrics["model_mfu_in_hfu_window_percent"], 20.0)
        self.assertAlmostEqual(CONTRACT["target_hfu_at_fixed_work_ratio"](metrics, 50), 100.0)

    def test_legacy_and_estimated_metrics_have_no_counter_projection(self):
        """Handle persisted results without a matched counter window."""
        project = CONTRACT["target_hfu_at_fixed_work_ratio"]
        self.assertIsNone(project({"hfu_method": "hardware_counters", "hfu_percent": 60.0}, 50))
        self.assertIsNone(project({"hfu_method": "estimate", "hfu_percent": 60.0,
                                   "model_mfu_in_hfu_window_percent": 30.0}, 50))


if __name__ == "__main__":
    unittest.main()
