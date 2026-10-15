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
"""Tests for handing a ranked configuration to pipeline balancing.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_parallelize.py -v
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory


class _Evaluator:
    """Memory evaluator double: records what to_ppb asks of it."""

    def __init__(self) -> None:
        """Nothing asked yet."""
        self.config = None
        self.request = None

    def set_config(self, ccfg: Any) -> None:
        """The config the search just set the candidate on."""
        self.config = ccfg

    def estimate_layer_memory(self, **kwargs: Any) -> dict:
        """A one-layer description."""
        self.request = kwargs
        return {"layers_description": [{"name": "BODY_0", "nb_layer": 4}]}


class TestToPPB(unittest.TestCase):
    """The k-th configuration's layer description, priced, on disk."""

    def test_writes_the_priced_description_of_rank_k(self):
        """
        Feature: ParallelizeLayer.to_ppb.
        Description: Hand rank 1 of a two-entry space to pipeline balancing.
        Expectation: The candidate is set on the config and the memory
            evaluator, the description is priced on the search's device, and
            it is written under the given folder.
        """
        chosen = []
        config = SimpleNamespace(
            set_parallel_config=chosen.append,
            dim_val=lambda dim, _: {Dim.PP: 2, Dim.MBN: 8, Dim.VPP: 1}.get(dim),
            ccfg=SimpleNamespace(device_capacity=Memory.from_string("56GB")),
        )
        runner = object.__new__(Par.ParallelizeLayer)
        runner.config = config
        runner.mem_eval = _Evaluator()
        runner.machine = SimpleNamespace(device=Hard.Device_A2)
        with tempfile.TemporaryDirectory() as folder:
            path = runner.to_ppb([("first", 1, 1.0), ("second", 1, 2.0)], 1, "unit", folder=folder)
            self.assertEqual(path, os.path.join(os.path.abspath(folder), "unit_nd_to_ppb_1.json"))
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"layers_description": [{"name": "BODY_0", "nb_layer": 4}]})
        self.assertEqual(chosen, ["second"])
        self.assertIs(runner.mem_eval.config, config.ccfg)
        self.assertIs(runner.mem_eval.request["device_type"], Hard.Device_A2)


if __name__ == "__main__":
    unittest.main()
