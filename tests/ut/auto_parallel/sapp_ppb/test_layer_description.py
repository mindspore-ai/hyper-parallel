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
"""Tests for how the pipeline balancer reads a layer description.

How to run this:
pytest tests/ut/auto_parallel/sapp_ppb/test_layer_description.py
"""
import json
import os
import tempfile
import unittest

import hyper_parallel.auto_parallel.sapp_ppb.utils.recompute as Recompute
from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import generate_layers_list

# A body layer whose stated backward times all differ from the fallback PPB
# derives from its time: 2 * 10 = 20 without recompute, 20.8 selective and
# 30 fully recomputed.
_BODY = {
    "name": "body",
    "type": "BODY",
    "nb_layer": 4,
    "time": 10.0,
    "forward_time": 10.0,
    "backward_time": 24.0,
    "select_rec_time": 26.0,
    "recompute_time": 37.0,
    "memory_parameter": 100,
    "memory_activation": 50,
    "memory_select_rec": 40,
    "memory_recompute": 5,
}


class TestLayerDescription(unittest.TestCase):
    """The keys the balancer reads are the keys a description writes."""

    def _read(self, description):
        """The one layer ``generate_layers_list`` reads from *description*."""
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "model.json"), "w", encoding="utf-8") as fh:
                json.dump({"layers_description": [description]}, fh)
            layers = generate_layers_list(folder, "model")
        self.assertEqual(len(layers), 1)
        return layers[0]

    def test_every_option_takes_the_backward_time_it_states(self):
        """
        Feature: generate_layers_list.
        Description: A body layer states its backward time without recompute,
            selectively recomputed and fully recomputed.
        Expectation: Each option takes the time stated for it rather than the
            fixed fraction of the layer's time PPB falls back to.
        """
        times = self._read(_BODY).backward_time_rec_
        self.assertEqual(times[Recompute.TYPE.NONE], 24.0)
        self.assertEqual(times[Recompute.TYPE.SLCT], 26.0)
        self.assertEqual(times[Recompute.TYPE.FULL], 37.0)


if __name__ == "__main__":
    unittest.main()
