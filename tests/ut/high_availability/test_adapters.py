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
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.high_availability.adapters import ModelRecoveryAdapter
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.core.context_parallel.context_parallel import ContextParallel
from hyper_parallel.core.context_parallel.async_context_parallel import AsyncContextParallel
from hyper_parallel.core.context_parallel.async_dsa_context_parallel import _AsyncSequenceReplicateSlot


class TestModelRecoveryAdapter(unittest.TestCase):
    """Exercise shard cleanup without distributed setup or accelerator hardware."""

    def test_repeated_cleanup_deduplicates_schedulers_and_preserves_parameters(self):
        """Clear each shared scheduler once while preserving parameter identities."""
        module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        first = SimpleNamespace(
            hsdp_state=SimpleNamespace(offload_policy=None, hsdp_params=[], reshard_after_backward=True),
            abort_iter_state=Mock(),
        )
        second = SimpleNamespace(
            hsdp_state=SimpleNamespace(offload_policy=None, hsdp_params=[], reshard_after_backward=True),
            abort_iter_state=Mock(),
        )
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

    def test_custom_offload_is_rejected_before_cleanup(self):
        """Feature: Recovery rejects unrecognized parameter-offload lifecycles.

        Description: Attach a scheduler with a custom offload policy.
        Expectation: Adapter construction fails before registering cleanup state.
        """
        module = torch.nn.Linear(4, 4)
        module.hsdp_scheduler = SimpleNamespace(
            hsdp_state=SimpleNamespace(offload_policy=object(), hsdp_params=[], reshard_after_backward=True),
        )
        with self.assertRaisesRegex(ValueError, "offload"):
            ModelRecoveryAdapter(module)

    def test_retained_full_parameters_use_scheduler_cleanup(self) -> None:
        """Recovery delegates interrupted accumulation to the fully_shard scheduler."""
        module = torch.nn.Linear(4, 4)
        scheduler = SimpleNamespace(
            hsdp_state=SimpleNamespace(offload_policy=None, hsdp_params=[module.weight], reshard_after_backward=False),
            abort_iter_state=Mock(),
        )
        module.hsdp_scheduler = scheduler
        ModelRecoveryAdapter(module).abort_iteration()
        scheduler.abort_iter_state.assert_called_once_with()

    def test_activation_reset_is_separate_from_stopped_device_cleanup(self) -> None:
        """Feature: Reset activation swap only after device restart.

        Description: Bind a model with a registered swap group and run both cleanup phases.
        Expectation: Abort does not query swap events; the post-restart phase resets the model's swap modules.
        """
        module = torch.nn.Linear(4, 4)
        module._swap_group_name = "attention"
        adapter = ModelRecoveryAdapter(module)
        with patch("hyper_parallel.high_availability.adapters.SwapManager") as manager:
            adapter.abort_iteration()
            manager.assert_not_called()
            adapter.reset_after_device_restart(Mock())
        manager.return_value.reset_after_device_restart.assert_called_once_with([module])

    def test_model_without_activation_swap_does_not_initialize_swap_manager(self) -> None:
        """Feature: Ordinary recovery does not initialize activation-swap resources.

        Description: Run the post-restart phase on an unwrapped model.
        Expectation: No swap manager or device synchronization is introduced.
        """
        adapter = ModelRecoveryAdapter(torch.nn.Linear(4, 4, dtype=torch.float16))
        with patch("hyper_parallel.high_availability.adapters.SwapManager") as manager:
            adapter.reset_after_device_restart(Mock())
        manager.assert_not_called()

    def test_sync_cp_failed_forward_discards_layout_stack_and_preserves_hooks(self) -> None:
        """Feature: Synchronous CP reentry after an interrupted boundary.

        Description: Interrupt two CP forward calls and clean before retrying real CPU autograd.
        Expectation: Stale layout records disappear while hooks and parameter identities remain stable.
        """
        module = torch.nn.Linear(4, 4)
        mesh = SimpleNamespace(ndim=1, mesh=torch.arange(2))
        ContextParallel().apply(module, mesh)
        adapter = ModelRecoveryAdapter(module)
        hooks = tuple(module._forward_pre_hooks.values())
        parameters = tuple(module.parameters())
        reference = torch.nn.Linear(4, 4)
        reference.load_state_dict(module.state_dict())
        value = torch.ones(2, 4)
        reference(value).sum().backward()
        prefix = "hyper_parallel.core.context_parallel.context_parallel."
        with ExitStack() as stack:
            scatter = stack.enter_context(patch(prefix + "_scatter_seq_to_head", side_effect=lambda x, *args: x))
            stack.enter_context(patch(prefix + "_gather_head_to_seq", side_effect=lambda x, *args: x))
            stack.enter_context(patch(prefix + "_finalize_ata_output", side_effect=lambda x, *args: x))
            for _ in range(2):
                scatter.side_effect = RuntimeError("FORCE STOP")
                with self.assertRaisesRegex(RuntimeError, "FORCE STOP"):
                    module(value)
                self.assertEqual(len(module._context_parallel_output_layout_stack), 1)
                adapter.abort_iteration()
                adapter.reset_after_device_restart(Mock())
                self.assertNotIn("_context_parallel_output_layout_stack", vars(module))
                scatter.side_effect = lambda x, *args: x
                module.zero_grad()
                module(value).sum().backward()
                self.assertNotIn("_context_parallel_output_layout_stack", vars(module))
                for actual, expected in zip(module.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        self.assertEqual(tuple(module._forward_pre_hooks.values()), hooks)
        for actual, expected in zip(module.parameters(), parameters):
            self.assertIs(actual, expected)

    def test_async_cp_clears_slots_in_place_before_repeated_autograd_reentry(self) -> None:
        """Feature: Recovery preserves async CP hook containers across repeated failures.

        Description: Interrupt after projection launch and after reverse A2A launch using real CPU autograd.
        Expectation: Old Work objects are never waited, hooks retain lists, and recovered gradients match baseline.
        """
        mesh = SimpleNamespace(ndim=1, mesh=torch.arange(2), get_group=object)
        model = torch.nn.Module()
        model.attention = torch.nn.Identity()
        model.projections = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        style = AsyncContextParallel()
        style.apply(model.attention, mesh, *model.projections)
        state = model.attention._context_parallel_async_state
        adapter = ModelRecoveryAdapter(model)
        projection = model.projections[0]
        baseline = torch.nn.Linear(4, 4)
        baseline.load_state_dict(projection.state_dict())
        value = torch.ones(1, 2, 2, 4)
        baseline(value).sum().backward()
        forward_slots = state.forward_slots
        backward_slots = state.backward_slots
        backward_q = backward_slots["q"]
        works = []

        def _exchange(output, tensor, **kwargs):
            del kwargs
            output.copy_(tensor)
            work = Mock()
            works.append(work)
            return work

        def _consume(output):
            return style._wait_a2a(output, object(), 2, forward_slots, "q", backward_q)

        with patch("hyper_parallel.core.context_parallel.utils.dist.all_to_all_single", side_effect=_exchange):
            for phase in ("forward", "backward", "forward", "backward"):
                if phase == "backward":
                    def _interrupt(module, grad_output):
                        del module, grad_output
                        raise RuntimeError("FORCE STOP")
                    guard = projection.register_full_backward_pre_hook(_interrupt, prepend=True)
                    try:
                        output = projection(value)
                        with self.assertRaisesRegex(RuntimeError, "FORCE STOP"):
                            _consume(output).sum().backward()
                    finally:
                        guard.remove()
                    self.assertTrue(backward_q)
                    obsolete = backward_q[-1][0]
                else:
                    projection(value)
                    obsolete = forward_slots["q"]["work"]
                obsolete.wait.reset_mock()
                obsolete.wait.side_effect = AssertionError("obsolete Work must not be waited")
                fence = Mock()
                def _check_live_slot(context=(phase,)):
                    """Require the slot to remain alive until the device fence."""
                    self.assertIsNotNone(forward_slots["q"] if context[0] == "forward" else backward_q[-1])

                fence.side_effect = _check_live_slot
                adapter.reset_after_device_restart(fence)
                fence.assert_called_once_with()
                obsolete.wait.assert_not_called()
                self.assertIs(state.forward_slots, forward_slots)
                self.assertIs(state.backward_slots, backward_slots)
                self.assertIs(state.backward_slots["q"], backward_q)
                self.assertTrue(all(slot is None for slot in forward_slots.values()))
                self.assertFalse(backward_q)
                model.zero_grad()
                _consume(projection(value)).sum().backward()
                torch.testing.assert_close(projection.weight.grad, baseline.weight.grad, rtol=0, atol=0)
                torch.testing.assert_close(projection.bias.grad, baseline.bias.grad, rtol=0, atol=0)
                self.assertFalse(backward_q)
        self.assertTrue(works)

    def test_async_dsa_preserves_producer_queue_references_and_relaunches(self) -> None:
        """Feature: DSA async queues are reusable after recovery.

        Description: Retain producer hook lists, clear old forward/backward work, and launch a new projection.
        Expectation: The same containers receive fresh entries and no old Work is waited.
        """
        mesh = SimpleNamespace(ndim=1, mesh=torch.arange(2), get_group=object)
        state = _AsyncSequenceReplicateSlot(mesh, 1)
        module = torch.nn.Linear(4, 4)
        state.register_launch_hook(module, "key")
        module._context_parallel_async_state = state
        adapter = ModelRecoveryAdapter(module)
        old = Mock()
        old.wait.side_effect = AssertionError("obsolete Work must not be waited")
        forward_queue = state._slots.setdefault("key", [])
        backward_queue = state._bwd_slots["key"]
        for _ in range(2):
            forward_queue.append({"work": old, "local": torch.ones(1)})
            backward_queue.append((old, torch.ones(1), 1))
            adapter.reset_after_device_restart(Mock())
            self.assertIs(state._slots["key"], forward_queue)
            self.assertIs(state._bwd_slots["key"], backward_queue)
            self.assertFalse(forward_queue)
            self.assertFalse(backward_queue)
            with patch.object(state, "_local_tensor", side_effect=lambda value: value):
                with patch("hyper_parallel.core.context_parallel.async_dsa_context_parallel."
                           "_launch_async_allgather_seq", return_value=(Mock(), torch.ones(1))):
                    with patch.object(state, "_make_slot", return_value={"fresh": True}):
                        module(torch.ones(1, 4))
            self.assertEqual(forward_queue, [{"fresh": True}])
        old.wait.assert_not_called()

    def test_failed_async_cp_fence_preserves_slots_and_layout_stack(self) -> None:
        """Feature: Failed device quiescence prevents asynchronous buffer reuse.

        Description: Fail the post-restart fence with a pending DSA slot and stale CP layout.
        Expectation: Propagate the error while preserving both pending work and layout ownership.
        """
        module = torch.nn.Linear(4, 4)
        state = _AsyncSequenceReplicateSlot(SimpleNamespace(mesh=torch.arange(2), get_group=object), 1)
        pending = Mock()
        state._slots["key"] = [pending]
        module._context_parallel_async_state = state
        module._context_parallel_output_layout_stack = ["old"]
        adapter = ModelRecoveryAdapter(module)
        with self.assertRaisesRegex(RuntimeError, "fence failed"):
            adapter.reset_after_device_restart(Mock(side_effect=RuntimeError("fence failed")))
        self.assertEqual(state._slots["key"], [pending])
        self.assertEqual(module._context_parallel_output_layout_stack, ["old"])

    def test_unknown_async_cp_state_requires_cleanup_contract(self) -> None:
        """Feature: Reject unsupported asynchronous CP ownership.

        Description: Bind a model with a custom async state lacking the cleanup method.
        Expectation: Fail before training instead of silently retaining failed Work objects.
        """
        module = torch.nn.Linear(4, 4)
        module._context_parallel_async_state = object()
        with self.assertRaisesRegex(ValueError, "cleanup method"):
            ModelRecoveryAdapter(module)

    def test_async_dsa_native_autograd_uses_fresh_backward_queue_after_reset(self) -> None:
        """Feature: DSA producer hooks retain the correct reverse-communication queue.

        Description: Clean pending work, then run native gather autograd and the producer backward hook.
        Expectation: Only fresh handles are consumed and the producer receives the expected gradient.
        """
        mesh = SimpleNamespace(mesh=torch.arange(2), get_group=object)
        state = _AsyncSequenceReplicateSlot(mesh, 1)
        module = torch.nn.Linear(4, 4)
        state.register_launch_hook(module, "key")
        module._context_parallel_async_state = state
        adapter = ModelRecoveryAdapter(module)
        reference = torch.nn.Linear(4, 4)
        reference.load_state_dict(module.state_dict())
        value = torch.ones(1, 2, 4)
        (reference(value).sum() * 2).backward()
        forward_queue = state._slots.setdefault("key", [])
        backward_queue = state._bwd_slots["key"]
        old = Mock()
        old.wait.side_effect = AssertionError("obsolete Work must not be waited")

        def _gather(output, tensor, **kwargs):
            del kwargs
            output.copy_(torch.cat((tensor, tensor), dim=0))
            return Mock()

        def _reduce_scatter(output, tensor, **kwargs):
            del kwargs
            output.copy_(tensor.chunk(2, dim=0)[0] * 2)
            return Mock()

        with ExitStack() as stack:
            stack.enter_context(patch.object(state, "_local_tensor", side_effect=lambda item: item))
            stack.enter_context(patch.object(state, "_wrap_gathered", side_effect=lambda item, slot: item))
            stack.enter_context(patch("hyper_parallel.core.context_parallel.utils.dist.all_gather_into_tensor",
                                     side_effect=_gather))
            stack.enter_context(patch("hyper_parallel.core.context_parallel.utils.dist.reduce_scatter_tensor",
                                     side_effect=_reduce_scatter))
            for _ in range(2):
                forward_queue.append({"work": old})
                backward_queue.append((old, torch.ones(1), 1))
                adapter.reset_after_device_restart(Mock())
                module.zero_grad()
                state.wait("key", module(value)).sum().backward()
                self.assertIs(state._bwd_slots["key"], backward_queue)
                self.assertFalse(backward_queue)
                self.assertFalse(forward_queue)
                torch.testing.assert_close(module.weight.grad, reference.weight.grad, rtol=0, atol=0)
                torch.testing.assert_close(module.bias.grad, reference.bias.grad, rtol=0, atol=0)
        old.wait.assert_not_called()

    def test_native_cpu_offload_fences_recovery_without_replacing_host_parameters(self) -> None:
        """Feature: Native HSDP CPU offload participates in recovery cleanup.

        Description: Bind a CPU-offloaded scheduler, abort, and fence after device restart.
        Expectation: Parameter storage stays on CPU, the fence runs once, and swap resources are not initialized.
        """
        module = torch.nn.Linear(4, 4)
        module.hsdp_scheduler = SimpleNamespace(
            hsdp_state=SimpleNamespace(
                offload_policy=CPUOffloadPolicy(pin_memory=False),
                hsdp_params=[],
                reshard_after_backward=True,
            ),
            abort_iter_state=Mock(),
        )
        parameter = module.weight
        storage = parameter.data_ptr()
        adapter = ModelRecoveryAdapter(module)
        fence = Mock()
        with patch("hyper_parallel.high_availability.adapters.SwapManager") as manager:
            adapter.abort_iteration()
            fence.assert_not_called()
            adapter.reset_after_device_restart(fence)
        module.hsdp_scheduler.abort_iter_state.assert_called_once_with()
        fence.assert_called_once_with()
        manager.assert_not_called()
        self.assertIs(module.weight, parameter)
        self.assertEqual(module.weight.data_ptr(), storage)
        self.assertEqual(module.weight.device.type, "cpu")

    def test_cpu_offload_reuses_activation_swap_fence(self) -> None:
        """Feature: Offload and activation swap share one post-restart device fence.

        Description: Bind both features and let the swap reset fence the device.
        Expectation: The adapter does not issue a second device synchronization.
        """
        module = torch.nn.Linear(4, 4)
        module._swap_group_name = "attention"
        module.hsdp_scheduler = SimpleNamespace(
            hsdp_state=SimpleNamespace(
                offload_policy=CPUOffloadPolicy(),
                hsdp_params=[],
                reshard_after_backward=True,
            ),
            abort_iter_state=Mock(),
        )
        adapter = ModelRecoveryAdapter(module)
        fence = Mock()
        with patch("hyper_parallel.high_availability.adapters.SwapManager") as manager:
            manager.return_value.reset_after_device_restart.side_effect = lambda modules: fence()
            adapter.reset_after_device_restart(fence)
        fence.assert_called_once_with()
