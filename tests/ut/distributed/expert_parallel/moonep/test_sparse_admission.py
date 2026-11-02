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
"""CPU-only admission checks for the fixed sparse MoonEP execution schedule."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor
from hyper_parallel.distributed.expert_parallel.moonep import checkpoint
from hyper_parallel.distributed.expert_parallel.moonep.fused import executor as execution
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestSparseAdmission(unittest.TestCase):
    """Mock communication boundaries while retaining header/trace protocol logic."""

    def setUp(self) -> None:
        """Build two registered CPU-only executors and an admitted control group."""
        self.group = object()
        with patch.object(execution.dist, "get_rank", return_value=0):
            self.executors = [execution.SparseMoonEPExecutor(
                torch.nn.Module(), Mock(), ep_group=self.group, ep_size=2, num_experts=4, top_k=2)
                for _ in range(2)]
        self.runtime = execution.SparseMoonEPRuntime(self.executors)
        self.runtime.weight_domain = SimpleNamespace(parked=False, retire_storage=Mock(return_value=0),
                                                     quota_elements=32 * 1024 * 1024)
        self.runtime.gradient_domain = SimpleNamespace(quota_elements=16 * 1024 * 1024)
        self.runtime.workspace = SimpleNamespace(control_group=object(), metadata={},
                                                 spec=SimpleNamespace(block_rows=256))
        for expert in self.executors:
            expert.shapes = (16, 32), (16, 16)
        self.ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
        self.route = RouteDescriptor(0, 2, 0, 0, (), (), (), ())
        self.enterContext(patch.object(execution, "upload_metadata",
                                       side_effect=lambda value, device: value.to(device)))
        self.compute = self.enterContext(patch.object(execution, "NativeExpertCompute"))
        self.count_gather = self.enterContext(patch.object(
            execution.dist, "all_gather_into_tensor", side_effect=self._copy_counts))
        self.trace_gather = self.enterContext(patch.object(
            execution.dist, "all_gather_object", side_effect=self._copy_trace))

    @staticmethod
    def _copy_counts(output, value, **_kwargs):
        output.view(2, -1).copy_(value.expand(2, -1))

    @staticmethod
    def _copy_trace(output, value, **_kwargs):
        output[:] = [value, value]

    def _forward_trace(self, cycles=1):
        for _ in range(cycles):
            for expert in self.executors:
                key = self.runtime.new_key(expert)
                self.runtime.executions[key] = expert, SimpleNamespace(route=self.route)
        return tuple(self.runtime.trace)

    def _complete_admitted(self, gradient):
        for key in tuple(reversed(self.runtime.trace)):
            self.runtime.validate_backward(key)
            self.runtime.complete(key)
        return gradient


    def test_forward_header_uses_existing_count_collective(self):
        """One count gather carries both histogram and the fixed call identity."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        with patch.object(execution, "select_allocation", wraps=execution.select_allocation) as select:
            plan = expert.plan(self.ids, key=key)

        self.count_gather.assert_called_once()
        self.trace_gather.assert_not_called()
        payload = self.count_gather.call_args.args[1]
        torch.testing.assert_close(payload[:4], torch.ones(4, dtype=torch.int64))
        self.assertEqual(payload[4:].tolist(), [0, 1, 0, 2, 0, 0])
        self.assertEqual(select.call_args.args[0].device.type, "cpu")
        self.assertEqual(select.call_args.args[0].shape, (2, 4))
        self.assertEqual(plan.route.source_rows, 4)
        self.assertFalse(self.runtime.poisoned)


    def test_local_forward_error_is_carried_to_peers(self):
        """A local contract error participates once with a failed header."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)

        with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
            expert.plan(self.ids, key=key, error=ValueError("invalid score layout"))

        self.count_gather.assert_called_once()
        self.assertEqual(self.count_gather.call_args.args[1][-1].item(), 1)
        self.compute.assert_not_called()


    def test_backward_complete_trace_is_admitted_once(self):
        """Every registered layer consumes one immutable reverse token position."""
        self._forward_trace()
        loss = torch.ones((), requires_grad=True)
        loss.register_hook(self._complete_admitted)

        self.runtime.backward(loss)

        self.trace_gather.assert_called_once()
        self.assertIs(self.trace_gather.call_args.kwargs["group"], self.runtime.workspace.control_group)
        self.assertEqual(self.runtime.trace, [])
        self.assertEqual(self.runtime.executions, {})
        self.assertIsNone(self.runtime.backward_token)
        self.assertFalse(self.runtime.backward_active)
        self.assertFalse(self.runtime.poisoned)


    def test_sequential_forward_backward_pairs_keep_accumulation_supported(self):
        """Multiple complete F/B pairs may precede one optimizer version change."""
        for _ in range(2):
            self._forward_trace()
            loss = torch.ones((), requires_grad=True)
            loss.register_hook(self._complete_admitted)
            self.runtime.backward(loss)

        self.assertEqual(self.trace_gather.call_count, 2)
        self.assertEqual(self.runtime.version, 0)
        self.runtime.finish_step()
        self.assertEqual(self.runtime.version, 1)


    def test_remote_backward_error_is_not_ignored(self):
        """A peer's unsupported option rejects locally valid backward entry too."""
        def _remote_error(output, value, **_kwargs):
            output[:] = [value, (value[0], ("create_graph must be False",))]

        self.trace_gather.side_effect = _remote_error
        self._forward_trace()
        with self.assertRaisesRegex(RuntimeError, "admission failed"):
            self.runtime.backward(torch.ones((), requires_grad=True))

        self.assertTrue(self.runtime.poisoned)
        self.assertIsNone(self.runtime.backward_token)


    def test_finish_step_cannot_overtake_active_dense_backward(self):
        """Even after every sparse node completes, active autograd blocks version advance."""
        self.runtime.backward_active = True

        with self.assertRaisesRegex(RuntimeError, "live or failed execution state"):
            self.runtime.finish_step()

        self.assertEqual(self.runtime.version, 0)


    def test_backward_scratch_retires_at_each_original_microbatch_start(self) -> None:
        """Every completed F/B pair may retire scratch before the next F at the same version."""
        arena = Mock(reserved_bytes=123)
        self.runtime.arena = arena
        for _ in range(2):
            keys = self._forward_trace()
            for key in reversed(keys):
                self.runtime.complete(key)
        self.assertEqual(arena.release_regions.call_count, 2)
        arena.release_regions.assert_called_with(("selections", "incoming", "down_gradient", "gate_gradient",
                                                  "activation", "project", "forward_activation"))
        self.runtime.finish_step()
        self.assertEqual(arena.release_regions.call_count, 3)
        self._forward_trace()
        self.assertEqual(arena.release_regions.call_count, 4)
        self.assertEqual(self.runtime.statistics["scratch_reserved_bytes"], 123)

    def test_replay_reuses_key_without_retiring_backward_storage(self):
        """The checkpoint replay path never invokes the original-forward key factory."""
        arena = Mock(reserved_bytes=0)
        self.runtime.arena = arena
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        scope = SimpleNamespace(replay=True, cursor=0, tape=SimpleNamespace(executions=[(expert, key)]))
        factory = Mock(side_effect=lambda: self.runtime.new_key(expert))
        with patch.object(checkpoint, "_SCOPE", SimpleNamespace(get=lambda: scope)):
            replay_key = checkpoint.bind_execution(expert, factory)
        self.assertIs(replay_key, key)
        factory.assert_not_called()
        arena.release_regions.assert_called_once()


    def test_post_count_planning_failures_poison_without_another_collective(self):
        """Late local allocation/compiler failures require job abort, without claiming peer convergence."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        stages = ((execution, "select_allocation"), (execution, "compile_local_route"),
                  (torch, "argsort"), (torch, "empty_like"), (execution, "NativeExpertCompute"),
                  (execution, "compile_transfer_rounds"))
        for owner, name in stages:
            with self.subTest(stage=name):
                self.runtime.poisoned = False
                self.count_gather.reset_mock()
                failure = MemoryError("injected " + name)
                with patch.object(owner, name, side_effect=failure), \
                        patch.object(self.runtime, "prepare") as materialize, \
                        patch.object(execution.dist, "all_reduce") as status:
                    with self.assertRaisesRegex(RuntimeError, "whole-job termination") as raised:
                        expert.plan(self.ids, key=key)
                self.assertIs(raised.exception.__cause__, failure)
                self.assertTrue(self.runtime.poisoned)
                self.count_gather.assert_called_once()
                self.assertEqual(self.count_gather.call_args.args[1][-1].item(), 0)
                self.trace_gather.assert_not_called()
                materialize.assert_not_called()
                status.assert_not_called()
                self.assertNotIn(key, self.runtime.executions)


    def test_complete_retires_host_and_device_metadata_only_for_its_route(self):
        """Completing an execution releases templates without erasing live generations."""
        keys = self._forward_trace()
        workspace = self.runtime.workspace
        other = RouteDescriptor(1, 2, 0, 0, (), (), (), ())
        retired = [(self.route, phase) for phase in (False, True, "readiness")]
        retired.append((self.route, "host_transport", workspace.spec.block_rows))
        retained = [(other, "host_transport", workspace.spec.block_rows), ("transport_generation", 0)]
        markers = {key: object() for key in retired + retained}
        workspace.metadata.update(markers)

        self.runtime.complete(keys[-1])

        for key in retired:
            self.assertNotIn(key, workspace.metadata)
        for key in retained:
            self.assertIs(workspace.metadata[key], markers[key])
