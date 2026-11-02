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
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
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

    def test_final_forward_retires_only_temporary_regions(self):
        """Loss does not retain projection scratch or invalidate independent saves."""
        key = self._forward_trace()[-1]
        arena = self.runtime.arena = PhaseArena("cpu")
        token = arena.begin()
        project = arena.allocate("project", (3, 16), torch.bfloat16)
        project.fill_(2)
        arena.allocate("forward_activation", (3, 8), torch.bfloat16)
        activation = arena.allocate("activation", (3, 8), torch.bfloat16)
        activation.fill_(3)
        saved = project.clone()
        arena.end(token)
        before = arena.reserved_bytes

        self.runtime.retire_forward_scratch(key)

        self.assertEqual(arena.reserved_bytes, before // 3)
        self.assertEqual(self.runtime.statistics["scratch_reserved_bytes"], arena.reserved_bytes)
        torch.testing.assert_close(project, saved)
        torch.testing.assert_close(activation, torch.full_like(activation, 3))
        token = arena.begin()
        replacement = arena.allocate("project", (3, 16), torch.bfloat16)
        self.assertNotEqual(replacement.data_ptr(), project.data_ptr())
        arena.end(token)

    def test_forward_retirement_skips_replay_and_nonfinal_layers(self):
        """Replayed activations and intermediate original phases keep their arena."""
        keys = self._forward_trace()
        self.runtime.arena = Mock()
        self.runtime.retire_forward_scratch(keys[0])
        with patch.object(execution, "is_replaying", return_value=True):
            self.runtime.retire_forward_scratch(keys[-1])
        self.runtime.arena.release_regions.assert_not_called()

    def test_forward_retirement_rejects_stale_final_key(self):
        """A stale invocation cannot release a later phase's temporary regions."""
        key = self._forward_trace()[-1]
        self.runtime.arena = Mock()
        self.runtime.version += 1
        with self.assertRaisesRegex(RuntimeError, "current final execution"):
            self.runtime.retire_forward_scratch(key)
        self.assertTrue(self.runtime.poisoned)
        self.runtime.arena.release_regions.assert_not_called()

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

    def test_forward_rank_identity_mismatch_prevents_plan_creation(self):
        """A peer's different layer identity fails after the shared count gather."""
        def _mismatched(output, value, **kwargs):
            self._copy_counts(output, value, **kwargs)
            output.view(2, -1)[1, 4] = 1

        self.count_gather.side_effect = _mismatched
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
            expert.plan(self.ids, key=key)

        self.count_gather.assert_called_once()
        self.compute.assert_not_called()
        self.assertTrue(self.runtime.poisoned)

    def test_matching_but_wrong_registered_layer_is_rejected(self):
        """All ranks starting at the second registered layer is still invalid."""
        expert = self.executors[1]
        key = self.runtime.new_key(expert)

        with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
            expert.plan(self.ids, key=key)

        self.assertEqual(self.count_gather.call_args.args[1][-1].item(), 1)
        self.compute.assert_not_called()

    def test_local_forward_error_is_carried_to_peers(self):
        """A local contract error participates once with a failed header."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)

        with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
            expert.plan(self.ids, key=key, error=ValueError("invalid score layout"))

        self.count_gather.assert_called_once()
        self.assertEqual(self.count_gather.call_args.args[1][-1].item(), 1)
        self.compute.assert_not_called()

    def test_out_of_range_expert_id_converges_in_header(self):
        """Omitted histogram IDs set the shared error flag without changing shape."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)

        with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
            expert.plan(torch.tensor([[0, 4]], dtype=torch.int64), key=key)

        self.assertEqual(self.count_gather.call_args.args[1].shape, (10,))
        self.assertEqual(self.count_gather.call_args.args[1][-1].item(), 1)

    def test_negative_and_large_ids_set_the_same_collective_error(self):
        """FP32 conversion cannot turn an invalid integer into an admitted ID."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        for invalid in (-1, 2**40):
            with self.subTest(invalid=invalid):
                self.runtime.poisoned = False
                with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
                    expert.plan(torch.tensor([[0, invalid]], dtype=torch.int64), key=key)
                self.assertEqual(self.count_gather.call_args.args[1].shape, (10,))
                self.assertEqual(self.count_gather.call_args.args[1][-1].item(), 1)
        self.compute.assert_not_called()

    def test_fixed_histogram_does_not_query_bincount_shape(self):
        """Counts and the admitted header are exact with an all-hot route."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        with patch.object(torch, "bincount", side_effect=AssertionError("dynamic histogram")):
            expert.plan(torch.full((17, 2), 3, dtype=torch.int64), key=key)
        payload = self.count_gather.call_args.args[1]
        torch.testing.assert_close(payload[:4], torch.tensor([0, 0, 0, 34], dtype=torch.int64))
        self.assertEqual(payload[-1].item(), 0)

    def test_invalid_dtype_rejects_before_flatten_or_histogram(self):
        """Invalid IDs participate in admission without a hidden allocating cast."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        with patch.object(torch, "histc") as histogram:
            with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
                expert.plan(self.ids.float(), key=key)
        histogram.assert_not_called()
        self.assertEqual(self.count_gather.call_args.args[1][-1].item(), 1)

    def test_flatten_failure_converges_before_payload(self):
        """Noncontiguous ID preparation errors set the existing collective flag."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        original_reshape = torch.Tensor.reshape

        def fail_ids(value: torch.Tensor, *shape: Any) -> torch.Tensor:
            """Inject only the ID flatten failure, leaving collective buffers usable."""
            if value is self.ids:
                raise MemoryError("ID reshape allocation failed")
            return original_reshape(value, *shape)

        with patch.object(torch.Tensor, "reshape", fail_ids):
            with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
                expert.plan(self.ids, key=key)
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

    def test_backward_rejects_multiple_retained_forward_traces(self):
        """Accumulation must use sequential F/B pairs, not multiple retained graphs."""
        self._forward_trace(cycles=2)

        with self.assertRaisesRegex(RuntimeError, "one complete registered forward"):
            self.runtime.backward(torch.ones((), requires_grad=True))

        self.trace_gather.assert_called_once()
        self.assertTrue(self.runtime.poisoned)

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

    def test_backward_options_are_rejected_after_shared_admission(self):
        """Unsupported graph modes never enter autograd or leave before admission."""
        self._forward_trace()
        loss = torch.ones((), requires_grad=True)

        with patch.object(torch.Tensor, "backward") as backward:
            with self.assertRaisesRegex(RuntimeError, "admission failed"):
                self.runtime.backward(loss, retain_graph=True)

        backward.assert_not_called()
        self.trace_gather.assert_called_once()
        self.assertIn("retain_graph", self.trace_gather.call_args.args[1][1][0])
        self.assertTrue(self.runtime.poisoned)

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

    def test_create_graph_is_rejected_before_autograd(self):
        """Higher-order mode participates in admission and never executes locally."""
        self._forward_trace()
        with patch.object(torch.Tensor, "backward") as backward:
            with self.assertRaisesRegex(RuntimeError, "admission failed"):
                self.runtime.backward(torch.ones((), requires_grad=True), create_graph=True)

        backward.assert_not_called()
        self.trace_gather.assert_called_once()

    def test_backward_trace_mismatch_is_rejected_before_autograd(self):
        """Different invocation counters cannot share one admitted backward."""
        def _mismatch(output, value, **_kwargs):
            state, errors = value
            remote = state[0], state[1] + 1, *state[2:]
            output[:] = [value, (remote, errors)]

        self.trace_gather.side_effect = _mismatch
        self._forward_trace()
        with patch.object(torch.Tensor, "backward") as backward:
            with self.assertRaisesRegex(RuntimeError, "traces disagree=True"):
                self.runtime.backward(torch.ones((), requires_grad=True))

        backward.assert_not_called()
        self.assertTrue(self.runtime.poisoned)

    def test_incomplete_forward_trace_fails_entry(self):
        """A partial registered model execution cannot start controlled backward."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        self.runtime.executions[key] = expert, SimpleNamespace(route=self.route)

        with self.assertRaisesRegex(RuntimeError, "complete registered forward"):
            self.runtime.backward(torch.ones((), requires_grad=True))

        self.trace_gather.assert_called_once()

    def test_uncontrolled_node_poisons_without_inventing_a_collective(self):
        """The local guard fails closed; it cannot converge arbitrary rank divergence."""
        keys = self._forward_trace()

        with self.assertRaisesRegex(RuntimeError, "controlled execution token"):
            self.runtime.validate_backward(keys[-1])

        self.trace_gather.assert_not_called()
        self.assertTrue(self.runtime.poisoned)

    def test_wrong_node_during_admitted_backward_poisons(self):
        """A node differing from the admitted reverse order is rejected locally."""
        keys = self._forward_trace()
        loss = torch.ones((), requires_grad=True)

        def _wrong_node(gradient):
            self.runtime.validate_backward(keys[0])
            return gradient

        loss.register_hook(_wrong_node)
        with self.assertRaisesRegex(RuntimeError, "controlled execution token"):
            self.runtime.backward(loss)

        self.trace_gather.assert_called_once()
        self.assertFalse(self.runtime.backward_active)
        self.assertIsNone(self.runtime.backward_token)
        self.assertTrue(self.runtime.poisoned)

    def test_incomplete_autograd_consumption_is_rejected(self):
        """Returning from autograd without every registered node cannot finish a step."""
        self._forward_trace()

        with self.assertRaisesRegex(RuntimeError, "did not complete every admitted"):
            self.runtime.backward(torch.ones((), requires_grad=True))

        self.assertTrue(self.runtime.poisoned)

    def test_finish_step_cannot_overtake_active_dense_backward(self):
        """Even after every sparse node completes, active autograd blocks version advance."""
        self.runtime.backward_active = True

        with self.assertRaisesRegex(RuntimeError, "live or failed execution state"):
            self.runtime.finish_step()

        self.assertEqual(self.runtime.version, 0)

    def test_close_arena_after_workspace_quiescence(self):
        """Physical phase storage is freed only after the token window is drained."""
        order = []
        self.runtime.workspace.close = Mock(side_effect=lambda: order.append("workspace"))
        arena = SimpleNamespace(close=Mock(side_effect=lambda: order.append("arena")),
                                release_regions=Mock(side_effect=lambda _: order.append("retire")),
                                reserved_bytes=0)
        self.runtime.arena = arena
        self.runtime.weight_domain = SimpleNamespace(drain=Mock())
        self.runtime.gradient_domain = SimpleNamespace(drain=Mock())

        self.runtime.close()

        self.assertEqual(order, ["retire", "workspace", "arena"])
        arena.close.assert_called_once()
        self.assertIsNone(self.runtime.arena)
        self.assertTrue(self.runtime.closed)

    def test_completed_phase_drops_borrowed_accumulator_reference(self):
        """Runtime completion releases every borrowed view after backward joins G."""
        keys = self._forward_trace()
        self.runtime.gradient_accumulators[keys[-1]] = torch.empty(8)
        self.runtime.gradient_staging[keys[-1]] = (torch.empty(16), torch.empty(16))
        self.runtime.complete(keys[-1])
        self.assertEqual(self.runtime.gradient_accumulators, {})
        self.assertEqual(self.runtime.gradient_staging, {})

    def test_stale_borrowed_staging_blocks_backward_and_step(self):
        """A staging borrow cannot silently survive a supposedly completed phase."""
        keys = self._forward_trace()
        self.runtime.gradient_staging[keys[-1]] = (torch.empty(16), torch.empty(16))
        errors = self.runtime._backward_errors(torch.ones((), requires_grad=True), False, False)
        self.assertIn("a previous gradient phase is unfinished", errors)
        self.runtime.trace.clear()
        self.runtime.executions.clear()
        with self.assertRaisesRegex(RuntimeError, "live or failed execution state"):
            self.runtime.finish_step()

    def test_stale_borrowed_accumulator_blocks_backward_and_step(self):
        """A lingering scratch borrow is unfinished state even without other gradient objects."""
        keys = self._forward_trace()
        self.runtime.gradient_accumulators[keys[-1]] = torch.empty(8)
        errors = self.runtime._backward_errors(torch.ones((), requires_grad=True), False, False)
        self.assertIn("a previous gradient phase is unfinished", errors)
        self.runtime.trace.clear()
        self.runtime.executions.clear()
        with self.assertRaisesRegex(RuntimeError, "live or failed execution state"):
            self.runtime.finish_step()

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

    def test_missing_admitted_shapes_converge_in_existing_count_header(self):
        """Missing geometry does not proceed to local unpacking after peers start W."""
        expert = self.executors[0]
        expert.shapes = None
        key = self.runtime.new_key(expert)
        with patch.object(torch, "histc") as histogram:
            with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
                expert.plan(self.ids, key=key)
        histogram.assert_not_called()
        self.count_gather.assert_called_once()
        self.trace_gather.assert_not_called()
        self.assertEqual(self.count_gather.call_args.args[1][-1].item(), 1)
        self.compute.assert_not_called()
        self.assertTrue(self.runtime.poisoned)

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

    def test_later_transfer_compiler_failure_is_also_fatal(self):
        """Both dW role plans remain inside the local fatal preparation boundary."""
        expert = self.executors[0]
        key = self.runtime.new_key(expert)
        compile_rounds = execution.compile_transfer_rounds
        for failure_call in (2, 3):
            with self.subTest(call=failure_call):
                self.runtime.poisoned = False
                self.count_gather.reset_mock()
                calls = 0

                def compile_one(*args: Any, **kwargs: Any) -> tuple:
                    """Fail one selected role while executing all earlier compilers."""
                    nonlocal calls
                    calls += 1
                    if calls == failure_call:
                        raise ValueError("injected role compiler")
                    return compile_rounds(*args, **kwargs)

                with patch.object(execution, "compile_transfer_rounds", side_effect=compile_one):
                    with self.assertRaisesRegex(RuntimeError, "whole-job termination"):
                        expert.plan(self.ids, key=key)
                self.assertTrue(self.runtime.poisoned)
                self.assertEqual(calls, failure_call)
                self.count_gather.assert_called_once()
                self.trace_gather.assert_not_called()

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
