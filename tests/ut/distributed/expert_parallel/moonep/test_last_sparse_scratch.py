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
"""CPU lifetime checks for retiring scratch before the remaining dense backward."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from hyper_parallel.distributed.expert_parallel.moonep.fused import executor as execution
from tests.common.mark_utils import arg_mark


_BACKWARD = ("selections", "incoming", "down_gradient", "gate_gradient", "activation")
_FORWARD = ("project", "forward_activation")
_RETIRED = (*_BACKWARD, *_FORWARD)
_GRADIENT_MAPS = ("gradient_roles", "gradient_outputs", "gradient_accumulators", "gradient_staging")


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestLastSparseScratch(unittest.TestCase):
    """Keep real CPU storage while mocking communication and completion markers."""

    def setUp(self) -> None:
        """Create two registered experts without initializing native resources."""
        self.experts = [SimpleNamespace(owner=torch.nn.Module(), ranks=1) for _ in range(2)]
        self.runtime = execution.SparseMoonEPRuntime(self.experts)
        self.runtime.workspace = SimpleNamespace(metadata={}, control_group=object(),
                                                spec=SimpleNamespace(block_rows=256))
        self.runtime.weight_domain = Mock(parked=False)
        self.runtime.gradient_domain = Mock()
        self.runtime.arena = PhaseArena("cpu")
        self.runtime.forward_vector_workers = 8

    def _trace(self) -> tuple[execution.ExecutionKey, ...]:
        keys = []
        for expert in self.experts:
            key = self.runtime.new_key(expert)
            route = object()
            self.runtime.executions[key] = expert, SimpleNamespace(route=route)
            for phase in (False, True, "readiness"):
                self.runtime.workspace.metadata[route, phase] = object()
            self.runtime.workspace.metadata[route, "host_transport", 256] = object()
            keys.append(key)
        return tuple(keys)

    def _admit(self, keys: tuple[execution.ExecutionKey, ...]) -> None:
        self.runtime.backward_token = tuple(reversed(keys))
        self.runtime.backward_position = 0
        self.runtime.backward_active = True

    def _populate(self, *, elements: int = 3) -> dict[str, torch.Tensor]:
        token = self.runtime.arena.begin()
        values = {}
        for name in (*_BACKWARD, *_FORWARD):
            dtype = torch.float32 if name.endswith("gradient") else torch.bfloat16
            values[name] = self.runtime.arena.allocate(name, (elements,), dtype)
            values[name].fill_(7)
        self.runtime.arena.end(token)
        return values

    def _complete(self, key: execution.ExecutionKey) -> None:
        self.runtime.backward_layers.add(key.layer)
        for name in _GRADIENT_MAPS:
            getattr(self.runtime, name)[key] = object()
        self.runtime.complete(key)

    def test_last_completion_retires_after_cleanup_without_finishing_backward(self) -> None:
        """The last sparse node releases scratch while the global token stays active."""
        keys = self._trace()
        values = self._populate()
        self._admit(keys)
        token = self.runtime.backward_token
        original_release = self.runtime.arena.release_regions

        def _release(names: tuple[str, ...]) -> int:
            for name in ("trace", "executions", "tickets", "backward_layers", *_GRADIENT_MAPS):
                self.assertFalse(getattr(self.runtime, name), name)
            self.assertFalse(self.runtime.workspace.metadata)
            self.assertIsNone(self.runtime.arena._active)
            return original_release(names)

        with patch.object(self.runtime.arena, "release_regions", side_effect=_release) as release:
            self._complete(keys[-1])
            release.assert_not_called()
            self.assertEqual(set(self.runtime.arena._regions), set((*_BACKWARD, *_FORWARD)))
            self._complete(keys[0])
            release.assert_called_once_with(_RETIRED)
        self.assertTrue(self.runtime.backward_active)
        self.assertIs(self.runtime.backward_token, token)
        self.assertEqual(self.runtime.backward_position, len(keys))
        self.assertEqual(self.runtime.version, 0)
        self.assertEqual(self.runtime.sequence, len(keys))
        self.assertFalse(self.runtime.arena._regions)
        self.assertEqual(self.runtime.statistics["scratch_reserved_bytes"], 0)
        for value in values.values():
            torch.testing.assert_close(value, torch.full_like(value, 7))
        self.runtime.weight_domain.retire_storage.assert_not_called()
        self.runtime.weight_domain.drain.assert_not_called()
        self.runtime.gradient_domain.drain.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "live or failed"):
            self.runtime.finish_step()


    def test_residual_ownership_prevents_retirement(self) -> None:
        """A remaining lease, execution, layer, or borrowed gradient view keeps scratch."""
        for name in ("trace", "executions", "tickets", "backward_layers", *_GRADIENT_MAPS):
            with self.subTest(field=name):
                keys = self._trace()
                self._admit(keys)
                self._complete(keys[-1])
                container = getattr(self.runtime, name)
                if name == "trace":
                    self.runtime.trace.insert(0, object())
                elif name == "backward_layers":
                    self.runtime.backward_layers.add(-1)
                else:
                    remaining = dict(container)
                    remaining[object()] = object()
                    setattr(self.runtime, name, remaining)
                with patch.object(self.runtime.arena, "release_regions") as release:
                    self._complete(keys[0])
                    release.assert_not_called()
                getattr(self.runtime, name).clear()
                self.runtime.backward_active = False
                self.runtime.backward_token = None
                self.runtime.backward_position = 0
