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
"""CPU checks for temporary retirement between sequential accumulated F/B pairs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts import arena as arena_module
from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from hyper_parallel.distributed.expert_parallel.moonep.fused import executor as execution
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestMicrobatchRetirement(unittest.TestCase):
    """Keep actual CPU storage while mocking stream events and communication."""

    def setUp(self) -> None:
        """Create two registered executors without initializing native resources."""
        with patch.object(execution.dist, "get_rank", return_value=0):
            self.experts = [execution.SparseMoonEPExecutor(torch.nn.Module(), Mock(), ep_group=object(),
                            ep_size=2, num_experts=4, top_k=2) for _ in range(2)]
        self.runtime = execution.SparseMoonEPRuntime(self.experts)
        self.runtime.workspace = SimpleNamespace(metadata={}, spec=SimpleNamespace(block_rows=256))
        self.runtime.forward_vector_workers = 8

    def test_second_pair_retires_regions_without_advancing_optimizer_version(self) -> None:
        """A second original F drops prior B regions while queued views retain valid bytes."""
        arena = self.runtime.arena = PhaseArena("cpu")
        regions = (("selections", torch.bfloat16), ("incoming", torch.bfloat16),
                   ("down_gradient", torch.float32), ("gate_gradient", torch.float32),
                   ("activation", torch.bfloat16))
        previous = []
        for pair in range(2):
            keys = []
            for expert in self.experts:
                key = self.runtime.new_key(expert)
                if expert.layer == 0:
                    self.assertEqual(arena.reserved_bytes, 0)
                    self.assertEqual(self.runtime.version, 0)
                    for tensor in previous:
                        torch.testing.assert_close(tensor, torch.full_like(tensor, 7))
                self.runtime.executions[key] = expert, SimpleNamespace(route=object())
                keys.append(key)
            token = arena.begin()
            current = []
            for name, dtype in regions:
                tensor = arena.allocate(name, (3,), dtype)
                tensor.fill_(7)
                current.append(tensor)
            arena.end(token)
            self.assertEqual(arena.reserved_bytes, len(regions) * 2 * 1024 * 1024)
            if pair:
                for old, new in zip(previous, current):
                    self.assertNotEqual(old.data_ptr(), new.data_ptr())
            previous = current
            for key in reversed(keys):
                self.runtime.complete(key)
        self.runtime.finish_step()
        self.assertEqual(self.runtime.version, 1)
        arena.close()

    def test_live_graph_ticket_or_backward_never_retires_regions(self) -> None:
        """Unsupported overlap must reach its order checks without dropping prior storage."""
        values = (("trace", [object()]), ("executions", {object(): object()}),
                  ("tickets", {object(): object()}), ("backward_active", True))
        for name, value in values:
            with self.subTest(state=name):
                runtime = execution.SparseMoonEPRuntime(self.experts)
                runtime.arena = Mock()
                setattr(runtime, name, value)
                runtime.new_key(self.experts[0])
                runtime.arena.release_regions.assert_not_called()
        runtime = execution.SparseMoonEPRuntime(self.experts)
        runtime.arena = Mock()
        runtime.new_key(self.experts[1])
        runtime.arena.release_regions.assert_not_called()

    def test_active_phase_blocks_retirement_without_freeing_or_advancing_key(self) -> None:
        """Logical phase ownership must end before reference retirement is legal."""
        arena = self.runtime.arena = PhaseArena("cpu")
        token = arena.begin()
        value = arena.allocate("incoming", (3,), torch.bfloat16)
        value.fill_(5)
        before = arena.reserved_bytes
        with self.assertRaisesRegex(RuntimeError, "active phase"):
            self.runtime.new_key(self.experts[0])
        self.assertEqual(self.runtime.sequence, 0)
        self.assertEqual(self.runtime.trace, [])
        self.assertEqual(arena.reserved_bytes, before)
        torch.testing.assert_close(value, torch.full_like(value, 5))
        arena.end(token)
        arena.close()

    def test_retirement_keeps_pending_events_without_waiting_or_overwriting(self) -> None:
        """Dropping root refs preserves the same queued phase events and host-in-flight bound."""
        stream = object()
        events = [Mock(), Mock(), Mock()]
        for event in events:
            event.query.return_value = False
        backend = SimpleNamespace(current_stream=Mock(return_value=stream), Event=Mock(side_effect=events))
        device = SimpleNamespace(type="npu")
        original_empty = torch.empty

        def _cpu_empty(elements: int, *, dtype: torch.dtype, device: Any) -> torch.Tensor:
            self.assertIs(device, arena.device)
            return original_empty(elements, dtype=dtype, device="cpu")

        with (patch.object(arena_module.torch, "device", return_value=device),
              patch.object(arena_module.torch, "npu", backend, create=True),
              patch.object(arena_module.torch, "empty", side_effect=_cpu_empty)):
            arena = self.runtime.arena = PhaseArena("npu:0")
            token = arena.begin()
            captured = arena.allocate("incoming", (3,), torch.bfloat16)
            captured.fill_(9)
            arena.end(token)
            token = arena.begin()
            arena.end(token)
            self.runtime.new_key(self.experts[0])
            self.assertEqual(arena.reserved_bytes, 0)
            self.assertEqual(tuple(arena._pending), tuple(events[:2]))
            for event in events:
                event.query.assert_not_called()
                event.synchronize.assert_not_called()
            torch.testing.assert_close(captured, torch.full_like(captured, 9))
            token = arena.begin()
            events[0].synchronize.assert_called_once_with()
            events[1].synchronize.assert_not_called()
            arena.end(token)
            arena.close()
