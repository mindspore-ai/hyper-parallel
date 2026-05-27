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
"""Unit tests for SAPP-ND memory estimation.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_memory_estimation.py
"""
import os
import tempfile
import unittest
from unittest.mock import patch

from tests.common.mark_utils import arg_mark

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

WORK_PATH = os.path.dirname(os.path.abspath(__file__))


class TestSappNDMemoryEstimation(unittest.TestCase):
    """A test class for SAPP-ND memory estimation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_memory_estimation_smoke(self):
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Test SAPP-ND memory estimation with a committed model config.
        Expectation: The evaluator parses config and returns valid memory results.
        """
        config_path = os.path.join(WORK_PATH, "mx_test.yaml")

        # Redirect matplotlib config dir to a temp path to avoid polluting $HOME.
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            evaluator = EvaluatorV2(config_path, log_level=0)

            self.assertEqual(
                evaluator.ccfg.model_name, "mixtral-8x7b",
                (f"unexpected model_name: got {evaluator.ccfg.model_name!r}, "
                 f"expected 'mixtral-8x7b'"),
            )
            self.assertEqual(
                set(evaluator.ctx.node_eval.keys()), set(LayerType),
                (f"node_eval keys mismatch: got {set(evaluator.ctx.node_eval.keys())}, "
                 f"expected {set(LayerType)}"),
            )

            strategy = evaluator.get_strategy()
            for key in ("dp", "tp", "pp", "ep"):
                self.assertGreater(
                    strategy[key], 0,
                    f"strategy[{key!r}] must be > 0, got {strategy[key]}",
                )

            peak_mem = evaluator.estimate_peak()
            self.assertGreater(peak_mem, 0, f"peak_mem must be > 0, got {peak_mem}")
            self.assertTrue(
                evaluator.mem_fit(peak_mem),
                f"mem_fit returned False for peak_mem={peak_mem}",
            )

            stage_static_mem = evaluator.static_mem_stage(1)
            stage_dynamic_mem = evaluator.dynamic_mem_stage(1)
            self.assertTrue(
                0 < stage_static_mem < peak_mem,
                f"static_mem_stage(1)={stage_static_mem} not in (0, peak={peak_mem})",
            )
            self.assertTrue(
                0 < stage_dynamic_mem < peak_mem,
                f"dynamic_mem_stage(1)={stage_dynamic_mem} not in (0, peak={peak_mem})",
            )
