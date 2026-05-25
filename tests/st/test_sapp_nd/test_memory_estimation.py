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
"""
Test module for SAPP-ND memory estimation.
How to run this:
pytest tests/st/test_sapp_nd/test_memory_estimation.py
"""
# pylint: disable=import-outside-toplevel
import os
import sys

from tests.common.mark_utils import arg_mark


SAPP_ND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../hyper_parallel/auto_parallel/SAPP-ND")
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
sys.path.insert(0, SAPP_ND_ROOT)


class TestSappNDMemoryEstimation:
    """A test class for SAPP-ND memory estimation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_memory_estimation_smoke(self):
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Test SAPP-ND memory estimation with a committed model config.
        Expectation: The evaluator parses config and returns valid memory results.
        """
        from memory_estimation.estimate_v2 import EvaluatorV2
        from paradise.common.layer_type import LayerType

        config_path = os.path.join(SAPP_ND_ROOT, "memory_estimation", "tests", "mx_test.yaml")
        evaluator = EvaluatorV2(config_path, log_level=0)

        assert evaluator.ccfg.model_name == "mixtral-8x7b"
        assert set(evaluator.ctx.node_eval.keys()) == set(LayerType)

        strategy = evaluator.get_strategy()
        assert strategy["dp"] > 0
        assert strategy["tp"] > 0
        assert strategy["pp"] > 0
        assert strategy["ep"] > 0

        peak_mem = evaluator.estimate_peak()
        assert peak_mem > 0
        assert evaluator.mem_fit(peak_mem)

        stage_static_mem = evaluator.static_mem_stage(1)
        stage_dynamic_mem = evaluator.dynamic_mem_stage(1)
        assert 0 < stage_static_mem < peak_mem
        assert 0 < stage_dynamic_mem < peak_mem
