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
"""CPU lifetime checks for retiring completed phase scratch before optimizer work."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from hyper_parallel.distributed.expert_parallel.moonep.fused.executor import SparseMoonEPRuntime
from tests.common.mark_utils import arg_mark


_REGIONS = ("selections", "incoming", "down_gradient", "gate_gradient", "activation",
            "project", "forward_activation")


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestFinishStepScratch(unittest.TestCase):
    """Use real CPU storage and mocked communication-free completion markers."""

    def setUp(self) -> None:
        """Create an idle runtime without initializing device communication."""
        self.expert = SimpleNamespace(owner=torch.nn.Module())
        self.runtime = SparseMoonEPRuntime([self.expert])
        self.runtime.forward_vector_workers = 8
        self.runtime.arena = PhaseArena("cpu")
        self.runtime.weight_domain = Mock()
        self.runtime.gradient_domain = Mock()

    def _populate(self) -> dict[str, torch.Tensor]:
        token = self.runtime.arena.begin()
        values = {}
        for name in _REGIONS:
            dtype = torch.float32 if name.endswith("gradient") else torch.bfloat16
            value = self.runtime.arena.allocate(name, (3,), dtype)
            value.fill_(7)
            values[name] = value
        self.runtime.arena.end(token)
        return values

    def test_finish_retires_idle_phase_regions_and_keeps_aliases(self) -> None:
        """Released arena ownership cannot resize saved queued-command aliases."""
        values = self._populate()
        capacity = dict(self.runtime.arena._capacities)
        self.runtime.finish_step()
        self.assertEqual(self.runtime.version, 1)
        self.assertFalse(self.runtime.arena._regions)
        self.assertEqual(self.runtime.arena._capacities, capacity)
        self.assertEqual(self.runtime.statistics["scratch_reserved_bytes"], 0)
        for value in values.values():
            torch.testing.assert_close(value, torch.full_like(value, 7))
        self.runtime.weight_domain.retire_storage.assert_not_called()
        self.runtime.weight_domain.drain.assert_not_called()
        self.runtime.gradient_domain.drain.assert_not_called()


    def test_unfinished_states_reject_before_release(self) -> None:
        """Finish cannot clear a live graph, packet, token, or borrowed G view."""
        fields = ("trace", "executions", "tickets", "backward_layers", "gradient_roles",
                  "gradient_outputs", "gradient_accumulators", "gradient_staging",
                  "backward_active", "backward_token", "poisoned")
        for name in fields:
            with self.subTest(name=name):
                original = getattr(self.runtime, name)
                setattr(self.runtime, name, True)
                with patch.object(self.runtime.arena, "release_regions") as release:
                    with self.assertRaisesRegex(RuntimeError, "live or failed"):
                        self.runtime.finish_step()
                    release.assert_not_called()
                self.assertEqual(self.runtime.version, 0)
                setattr(self.runtime, name, original)
