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
"""CPU failure injection for terminal sparse resource initialization state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep.fused import executor as execution
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestResourceInitialization(unittest.TestCase):
    """Replace device factories and collectives before exercising startup."""

    def setUp(self) -> None:
        """Create metadata-only admitted owners and trap every real device operation."""
        self.experts = [SimpleNamespace(owner=torch.nn.Linear(4, 4, bias=False), ranks=2, experts=4,
                                       home_count=2, top_k=2, replica_capacity=2, rank=0, group=object())]
        self.home = tuple(SimpleNamespace(dtype=torch.bfloat16, device=torch.device("npu"),
                                          shape=shape, stride=Mock(return_value=stride))
                          for shape, stride in (((2, 16, 32), (512, 32, 1)), ((2, 16, 16), (256, 16, 1))))
        self.runtime = execution.SparseMoonEPRuntime(self.experts)
        for name, value in (("is_initialized", True), ("get_world_size", 2), ("get_rank", 0),
                            ("get_process_group_ranks", [0, 1])):
            self.enterContext(patch.object(execution.dist, name, return_value=value))
        self.gather = self.enterContext(patch.object(execution.dist, "all_gather_object", side_effect=self._gather))
        self.groups = self.enterContext(patch.object(execution.dist, "new_group", return_value=object()))
        self.destroy = self.enterContext(patch.object(execution.dist, "destroy_process_group"))
        self.workspace = Mock()
        self.workspace.check_preflight.side_effect = self._preflight
        self.workspace_factory = self.enterContext(patch.object(execution, "RingWorkspace",
                                                               return_value=self.workspace))
        self.arena = Mock()
        self.arena_factory = self.enterContext(patch.object(execution, "PhaseArena", return_value=self.arena))
        self.weight, self.gradient = Mock(), Mock()
        self.domains = self.enterContext(patch.object(execution, "TransferDomain",
                                                      side_effect=[self.weight, self.gradient]))
        for name in ("Stream", "Event", "current_stream", "synchronize", "init"):
            self.enterContext(patch.object(torch.npu, name, side_effect=AssertionError("NPU operation invoked")))

    @staticmethod
    def _gather(output: list, value: Any, **_kwargs: Any) -> None:
        output[:] = [value, value]

    @staticmethod
    def _preflight(error: Exception | None) -> None:
        if error is not None:
            raise RuntimeError("mock shared admission failed") from error

    def _assert_terminal_retry(self) -> None:
        self.assertTrue(self.runtime.poisoned)
        self.assertTrue(self.runtime._initialization_failed)
        calls = tuple(mock.call_count for mock in
                      (self.gather, self.groups, self.workspace_factory, self.arena_factory, self.domains))
        with self.assertRaisesRegex(RuntimeError, "poisoned.*whole-job termination"):
            self.runtime.initialize(self.experts[0], self.home)
        self.assertEqual(calls, tuple(mock.call_count for mock in
                                    (self.gather, self.groups, self.workspace_factory,
                                     self.arena_factory, self.domains)))

    def test_constructor_failures_preserve_cause_and_forbid_retry(self) -> None:
        """Every group, workspace, arena, and domain construction failure is terminal."""
        for stage in ("first_group", "control_group", "workspace", "arena", "weight", "gradient"):
            with self.subTest(stage=stage):
                self.runtime = execution.SparseMoonEPRuntime(self.experts)
                failure = RuntimeError("injected " + stage)
                self.groups.side_effect = None
                self.workspace_factory.side_effect = None
                self.arena_factory.side_effect = None
                self.domains.side_effect = [self.weight, self.gradient]
                if stage == "first_group":
                    self.groups.side_effect = failure
                elif stage == "control_group":
                    self.groups.side_effect = [object(), object(), failure]
                elif stage == "workspace":
                    self.workspace_factory.side_effect = failure
                elif stage == "arena":
                    self.arena_factory.side_effect = failure
                elif stage == "weight":
                    self.domains.side_effect = failure
                else:
                    self.domains.side_effect = [self.weight, failure]
                with self.assertRaises(RuntimeError) as caught:
                    self.runtime.initialize(self.experts[0], self.home)
                self.assertIs(caught.exception, failure)
                self._assert_terminal_retry()
                self.weight.materialize.assert_not_called()
                self.gradient.return_gradient.assert_not_called()

    def test_local_admission_failure_uses_existing_preflight_before_poison(self) -> None:
        """The resource boundary preserves local allocation failure convergence."""
        failure = MemoryError("injected replica allocation")
        self.weight.admit.side_effect = failure
        with self.assertRaisesRegex(RuntimeError, "shared admission failed") as caught:
            self.runtime.initialize(self.experts[0], self.home)
        self.assertIs(caught.exception.__cause__, failure)
        self.workspace.check_preflight.assert_called_once_with(failure)
        self.gradient.admit.assert_not_called()
        self._assert_terminal_retry()

    def test_remote_admission_failure_also_forbids_locally_complete_retry(self) -> None:
        """A valid local allocation cannot be reused after another rank rejects admission."""
        self.workspace.check_preflight.side_effect = RuntimeError("remote preflight failure")
        with self.assertRaisesRegex(RuntimeError, "remote preflight failure"):
            self.runtime.initialize(self.experts[0], self.home)
        self.weight.admit.assert_called_once()
        self.gradient.admit.assert_called_once()
        self.workspace.check_preflight.assert_called_once_with(None)
        self._assert_terminal_retry()

    def test_non_exception_resource_abort_marks_terminal_state(self) -> None:
        """Cancellation also prevents subsequent reuse of a partial workspace."""
        self.domains.side_effect = KeyboardInterrupt("injected interruption")
        with self.assertRaises(KeyboardInterrupt):
            self.runtime.initialize(self.experts[0], self.home)
        self._assert_terminal_retry()

    def test_partial_close_drops_only_unsubmitted_local_references(self) -> None:
        """A local startup failure must not enter collective free, group destroy, or drains."""
        self.domains.side_effect = [self.weight, RuntimeError("gradient constructor failed")]
        with self.assertRaisesRegex(RuntimeError, "gradient constructor failed"):
            self.runtime.initialize(self.experts[0], self.home)
        groups = self.runtime.groups
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "collective resources require whole-job termination"):
                self.runtime.close()
            self.assertIsNone(self.runtime.weight_domain)
            self.assertIsNone(self.runtime.gradient_domain)
            self.assertIsNone(self.runtime.arena)
            self.assertIs(self.runtime.workspace, self.workspace)
            self.assertIs(self.runtime.groups, groups)
            self.assertTrue(self.runtime.poisoned)
            self.assertFalse(self.runtime.closed)
        self.weight.drain.assert_not_called()
        self.gradient.drain.assert_not_called()
        self.workspace.close.assert_not_called()
        self.arena.close.assert_not_called()
        self.destroy.assert_not_called()

    def test_success_keeps_idempotent_initialization_and_collective_close(self) -> None:
        """A fully initialized runtime retains its existing drain and free sequence."""
        self.runtime.initialize(self.experts[0], self.home)
        self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()
        self.assertFalse(self.runtime._initialization_failed)
        self.runtime.close()
        self.runtime.close()
        self.weight.drain.assert_called_once_with()
        self.gradient.drain.assert_called_once_with()
        self.workspace.close.assert_called_once_with()
        self.arena.close.assert_called_once_with()
        self.assertEqual(self.destroy.call_count, 5)
        self.assertTrue(self.runtime.closed)

    def test_never_initialized_close_creates_no_resources(self) -> None:
        """Closing before the first forward remains a local idempotent operation."""
        self.runtime.close()
        self.runtime.close()
        self.gather.assert_not_called()
        self.groups.assert_not_called()
        self.workspace_factory.assert_not_called()
        self.domains.assert_not_called()
        self.destroy.assert_not_called()
        self.assertTrue(self.runtime.closed)
