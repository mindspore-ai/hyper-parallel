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
"""Recovery cleanup preserves parameters and clears each shard scheduler once."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from hyper_parallel.core.fault_tolerance.adapters import ModelRecoveryAdapter


class TestModelRecoveryAdapter(unittest.TestCase):
    """Exercise shard cleanup without distributed setup or accelerator hardware."""

    def test_repeated_cleanup_deduplicates_schedulers_and_preserves_parameters(self):
        module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        first = SimpleNamespace(hsdp_state=SimpleNamespace(offload_policy=None), abort_iter_state=Mock())
        second = SimpleNamespace(hsdp_state=SimpleNamespace(offload_policy=None), abort_iter_state=Mock())
        module.hsdp_scheduler = module[0].hsdp_scheduler = first
        module[1].hsdp_scheduler = second
        parameters = list(module.parameters())
        adapter = ModelRecoveryAdapter(module)
        for _ in range(2):
            adapter.abort_iteration()
            for expected, actual in zip(parameters, module.parameters()):
                self.assertIs(actual, expected)
        self.assertEqual(first.abort_iter_state.call_count, 2)
        self.assertEqual(second.abort_iter_state.call_count, 2)

    def test_unsupported_offload_is_rejected_before_cleanup(self):
        module = torch.nn.Linear(4, 4)
        module.hsdp_scheduler = SimpleNamespace(hsdp_state=SimpleNamespace(offload_policy=object()))
        with self.assertRaisesRegex(ValueError, "offload"):
            ModelRecoveryAdapter(module)
