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
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level

DEEPSEEK_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek.yaml")


class TestSearchState(unittest.TestCase):
    """Each candidate and each search starts from its own state."""

    def setUp(self) -> None:
        """A scratch folder, and no bounds left over for the next test."""
        set_verbose_level(1)
        folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.addCleanup(self._reset_bounds)

    @staticmethod
    def _reset_bounds() -> None:
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()

    def _small_deepseek(self) -> str:
        """The DeepSeek test config cut to 7 layers and 16 experts."""
        with open(DEEPSEEK_YAML, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["model"]["model_config"].update(num_layers=7, offset=0)
        config["parallel_config"].update(data_parallel=4, model_parallel=2, pipeline_stage=2,
                                         expert_parallel=2, micro_batch_num=4)
        config["moe_config"]["expert_num"] = 16
        config["recompute_config"]["recompute"] = True
        path = os.path.join(self.folder, "deepseek_small.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        return path

    def _search(self, dims) -> Par.ParallelizeLayer:
        """A search over *dims* for the small DeepSeek on 16 devices."""
        runner = Par.Parallelize("mindformers", self._small_deepseek(), Hard.Machine(16, "A2"),
                                 global_batch_size=16, dimensions=list(dims))
        return runner.instance

    def test_a_candidate_estimates_the_same_every_time(self):
        """
        Feature: memory estimate inside the search.
        Description: Set and estimate one candidate three times, as the
            search does for every candidate.  The layer hooks used to run on
            the search's own config, so each estimate started from the state
            the previous one left behind.
        Expectation: The same peak every time.
        """
        search = self._search([Dim.DP, Dim.TP, Dim.PP, Dim.EP, Dim.OP])
        candidate = search.config.make_parallel_config_args(dp=16, mp=1, pp=1, ep=1, op=1, mb=1)
        peaks = []
        for _ in range(3):
            self.assertTrue(search.config.set_parallel_config(candidate))
            peaks.append(search.memory_estim())
        self.assertEqual(peaks, [peaks[0]] * 3)

    def test_an_interleaved_search_runs(self):
        """
        Feature: search over the interleave degree.
        Description: Search VPP along with DP, TP and PP.  The backward
            overhead used to read a config the previous candidate had left,
            with its own interleave degree, and ran off its chunk list.
        Expectation: The search completes and keeps interleaved candidates.
        """
        search = self._search([Dim.DP, Dim.TP, Dim.PP, Dim.VPP])
        results, _ = search.device_loops(({}, 0), None)
        self.assertIn(2, {candidate.val(Dim.VPP) for candidate in results})

    def test_a_memory_cap_becomes_the_capacity(self):
        """
        Feature: the -M memory cap.
        Description: Start a search with a 50 GB cap, as ``run_nd -M 50GB``
            does.  It used to raise on a ``Memory.set`` that did not exist.
        Expectation: The capacity the search fits candidates against is 50 GB.
        """
        runner = Par.Parallelize("mindformers", self._small_deepseek(), Hard.Machine(16, "A2"),
                                 global_batch_size=16, dimensions=[Dim.DP, Dim.TP, Dim.PP],
                                 max_mem=Memory.from_string("50GB"))
        self.assertEqual(runner.instance.mem_eval.ccfg.device_capacity.to_gb().size, 50)

    def test_building_a_candidate_leaves_the_bounds_alone(self):
        """
        Feature: search-space bounds.
        Description: Bound TP, then build a candidate that has TP.
        Expectation: The bound still holds; it belongs to the search.
        """
        Dim.TP.set_bound(2)
        Dim.Dimensions([(Dim.TP, 4)], all_dims=[Dim.TP])
        self.assertEqual(Dim.TP.get_bound(), 2)

    def test_a_search_does_not_inherit_bounds(self):
        """
        Feature: search-space bounds.
        Description: An earlier search in the process, on a dense model,
            left EP bounded to 1.  Start a search on a 16-expert model.
        Expectation: EP is bounded by this model's 16 experts, not by the
            tighter bound left behind.
        """
        Dim.EP.set_bound(1)
        self._search([Dim.DP, Dim.TP, Dim.PP, Dim.EP])
        self.assertEqual(Dim.EP.get_bound(), 16)

    def test_a_dense_model_searches_no_expert_parallelism(self):
        """
        Feature: search-space bounds.
        Description: Search EP on a dense model.  Building the first
            candidate used to clear every bound, so only the first loop
            iteration was bounded.
        Expectation: Every candidate has EP 1.
        """
        dense = SimpleNamespace(
            model_type="llama", hidden_size=1024, num_hidden_layers=8, num_attention_heads=8,
            num_key_value_heads=8, intermediate_size=2816, vocab_size=32000,
            max_position_embeddings=4096,
        )
        train = {
            "model": {"pretrained_model_name_or_path": "local/unit", "torch_dtype": "bfloat16"},
            "training": {"global_batch_size": 16, "micro_batch_size": 1},
            "accelerator": {"tp_size": 1, "pp_size": 1},
            "fsdp_config": {"dp_shard_size": 16},
            "dataset": {"data_transform": {"max_seq_len": 2048}},
            "context": {"max_device_memory": "64GB", "device_num": 16},
        }
        path = os.path.join(self.folder, "dense.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(train, handle)
        with patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config", return_value=dense):
            runner = Par.Parallelize("hyper_v2", path, Hard.Machine(16, "A2"), global_batch_size=16,
                                     dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.EP])
            results, _ = runner.instance.device_loops(({}, 0), None)
        self.assertTrue(results)
        self.assertEqual({candidate.val(Dim.EP) for candidate in results}, {1})


if __name__ == "__main__":
    unittest.main()
