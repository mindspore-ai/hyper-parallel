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
"""Regression tests for compact TP request checks and exception-safe shutdown."""
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from rl.rollout import vllm


class TestVLLMMemoryBounds(unittest.TestCase):
    """Preserve request mismatch detection and always close weight resources."""

    def test_tp_contract_is_compact_and_detects_value_and_shape_changes(self) -> None:
        """Equal-shaped token differences and differently shaped tensors stay distinct."""
        request = SimpleNamespace(input_ids=torch.ones(2, 10000, dtype=torch.long),
                                  attention_mask=torch.ones(2, 10000, dtype=torch.long),
                                  row_seeds=(1, 2), settings="settings")
        engine = SimpleNamespace(_trainer_tp_size=2, _trainer_tp_group="group")
        contracts = []

        def gather(output, contract, group):
            self.assertEqual(group, "group")
            contracts.append(contract)
            output[:] = [contract, contract]

        with patch.object(vllm.dist, "all_gather_object", side_effect=gather):
            for change in ("none", "token", "mask", "shape"):
                if change == "token":
                    request.input_ids[0, 3] = 2
                elif change == "mask":
                    request.attention_mask[0, 3] = 0
                elif change == "shape":
                    request.input_ids = request.input_ids.reshape(1, -1)
                vllm.VLLMGenerationEngine._validate_tp_request(engine, request)
        self.assertTrue(all(len(pickle.dumps(contract)) < 512 for contract in contracts))
        self.assertEqual(len(set(contracts)), 4)
        with patch.object(vllm.dist, "all_gather_object",
                          side_effect=lambda output, *_args: output.__setitem__(slice(None), contracts[:2])):
            with self.assertRaisesRegex(RuntimeError, "different"):
                vllm.VLLMGenerationEngine._validate_tp_request(engine, request)

    def test_shutdown_attempts_weight_cleanup_and_preserves_primary_failure(self) -> None:
        """A second cleanup failure cannot hide the original server shutdown error."""
        for weight_fails in (False, True):
            with self.subTest(weight_fails=weight_fails):
                calls = []
                client = vllm._VLLMHTTPClient.__new__(vllm._VLLMHTTPClient)

                def server_close():
                    raise RuntimeError("server failure")

                def weight_close():
                    calls.append("weight")
                    if weight_fails:
                        raise ValueError("weight failure")

                client.close = server_close
                engine = SimpleNamespace(_client=client, _weight_sync=SimpleNamespace(close=weight_close))
                with patch.object(vllm, "synchronized_call", side_effect=lambda _label, action: action()):
                    with self.assertRaisesRegex(RuntimeError, "server failure"):
                        vllm.VLLMGenerationEngine.close(engine)
                self.assertEqual(calls, ["weight"])
