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
"""Tests that one search candidate, or one search, cannot change the next.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_search_state.py -v
"""
import os
import tempfile
import unittest

import yaml

from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level

DEEPSEEK_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek.yaml")


class TestSearchState(unittest.TestCase):
    """Each candidate and each search starts from its own state."""

    def tearDown(self) -> None:
        """No bounds left over for the next test."""
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()

    def test_neither_a_search_nor_a_candidate_inherits_state(self):
        """
        Feature: the state a search keeps between searches and between candidates.
        Description: An earlier search in the process, on a dense model, left
            EP bounded to 1.  Start a search on the DeepSeek test config cut
            to 7 layers and 16 experts, then set and estimate one candidate
            three times, as the search does for every candidate.  The layer
            hooks used to run on the search's own config, so each estimate
            started from the state the previous one left behind.
        Expectation: EP is bounded by this model's 16 experts, not by the
            tighter bound left behind, and the candidate's peak is the same
            every time.
        """
        set_verbose_level(1)
        with open(DEEPSEEK_YAML, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["model"]["model_config"].update(num_layers=7, offset=0)
        config["parallel_config"].update(data_parallel=4, model_parallel=2, pipeline_stage=2,
                                         expert_parallel=2, micro_batch_num=4)
        config["moe_config"]["expert_num"] = 16
        config["recompute_config"]["recompute"] = True
        Dim.EP.set_bound(1)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "deepseek_small.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)
            search = Par.Parallelize("mindformers", path, Hard.Machine(16, "A2"), global_batch_size=16,
                                     dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.EP, Dim.OP]).instance
            self.assertEqual(Dim.EP.get_bound(), 16)
            candidate = search.config.make_parallel_config_args(dp=16, mp=1, pp=1, ep=1, op=1, mb=1)
            peaks = []
            for _ in range(3):
                self.assertTrue(search.config.set_parallel_config(candidate))
                peaks.append(search.memory_estim())
        self.assertEqual(peaks, [peaks[0]] * 3)


if __name__ == "__main__":
    unittest.main()
