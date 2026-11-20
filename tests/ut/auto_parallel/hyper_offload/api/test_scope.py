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
"""HyperOffload must interoperate with unchanged framework storage ownership."""

import unittest
from collections import namedtuple
from contextlib import ExitStack
from types import FunctionType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession
from hyper_parallel.auto_parallel.hyper_offload.api.scope import (
    dispatch_framework_op,
    dispatch_framework_read,
    has_managed_inputs,
)
from hyper_parallel.auto_parallel.hyper_offload.api.session import (
    ActivationDispatchMode,
)
from hyper_parallel.auto_parallel.hyper_offload.execution.replay import ReplayExecutor
from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
from hyper_parallel.auto_parallel.hyper_offload.execution.warmup import WarmupExecutor
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import AccessKind, ActivationTrace
from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencyActionType, ResidencySchedule
from hyper_parallel.auto_parallel.hyper_offload.planning import PeakResidencyPlanner
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import (
    PhysicalBuffer,
    ResidencyManager,
)
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from tests.common.mark_utils import arg_mark
from tests.ut.auto_parallel.hyper_offload.execution.test_selective_replay import BytePool, CPUActivationTracker
from tests.ut.core.fully_shard.test_param import _new_param


class _CountingMode(TorchDispatchMode):
    """Observe unrelated dispatch mode preservation."""

    def __init__(self) -> None:
        """Initialize the operation count."""
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Count operations while preserving execution."""
        del types
        self.calls += 1
        return func(*args, **(kwargs or {}))


class TestFrameworkBoundary(unittest.TestCase):
    """Check real FSDP methods and exported storage lifetimes."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_framework_allocation_records_only_unique_fresh_storage_bytes(self) -> None:
        """Feature: Framework workspace accounting.

        Description: Return fresh output aliases and a view of an existing input.
        Expectation: Count fresh storage once and exclude the input storage.
        """
        allocations = []
        executor = SimpleNamespace(
            dispatch=lambda function, args, kwargs: function(*args, **kwargs),
            record_framework_allocation=allocations.append,
        )
        source = torch.arange(8, dtype=torch.float16)

        def convert(value: torch.Tensor) -> tuple:
            """Create fresh aliases and an input view.

            Args:
                value: Existing input storage to exclude from accounting.
            """
            converted = value.float()
            return converted, converted.view(2, 4), value.view(-1)

        dispatch_framework_read(executor, convert, (source,), {}, native_inputs=True)
        self.assertEqual(allocations, [32])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_framework_workspace_metadata_survives_finish_and_resets(self) -> None:
        """Feature: Framework workspace trace lifecycle.

        Description: Record different allocation sizes before completing warmup.
        Expectation: The trace retains the maximum and the executor resets its metadata.
        """
        manager = ResidencyManager(1024)
        with patch("hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.DeviceTimer"):
            executor = WarmupExecutor(manager, 1024)
        executor.record_framework_allocation(64)
        executor.record_framework_allocation(32)
        with patch("hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
                   return_value=(16.0, 16.0)):
            trace, _ = executor.finish()
        self.assertEqual(trace.framework_allocation_bytes, 64)
        self.assertEqual(executor._framework_allocation_bytes, 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_opaque_native_ops_preserve_other_modes_without_stack_scan(self) -> None:
        """Feature: Opaque native dispatch.

        Description: Execute under an unrelated mode with live-stack checks forbidden.
        Expectation: Preserve values and observe the operation in the unrelated mode.
        """
        source = torch.arange(4.0)
        executor = SimpleNamespace(in_opaque_region=True, dispatch=Mock())
        observer = _CountingMode()
        with observer, ActivationDispatchMode(SimpleNamespace(executor=executor)), patch(
            "hyper_parallel.auto_parallel.hyper_offload.api.session.inside_fsdp_runtime",
            side_effect=RuntimeError("opaque native operation inspected the stack"),
        ):
            result = source.sin()
        torch.testing.assert_close(result, source.sin(), rtol=0, atol=0)
        executor.dispatch.assert_not_called()
        self.assertEqual(observer.calls, 1)

    def test_managed_input_query_does_not_reconstruct_containers(self) -> None:
        """A read-only query must not invoke user container constructors."""
        pair_type = namedtuple("Pair", "left right")
        source = torch.arange(4.0)
        buffer = PhysicalBuffer(device=source.device, device_buffer=source.view(torch.uint8))
        shadow = ShadowTensor(source, buffer, 1)
        native = pair_type(source, {"shape": [2, 2]})
        managed = pair_type(source, {"nested": [None, shadow]})
        with patch.object(pair_type, "__new__", side_effect=RuntimeError("query rebuilt input")):
            self.assertFalse(has_managed_inputs((native,), {}))
            self.assertTrue(has_managed_inputs((), {"input": managed}))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_native_gradient_assignment_preserves_alias_and_other_modes(self) -> None:
        """Feature: FSDP storage ownership.

        Description: Assign a native gradient under the offload and observer modes.
        Expectation: Preserve the gradient view alias and bypass replay dispatch.
        """
        param = _new_param()
        source = torch.arange(4.0)
        param.to_sharded_dtensor = lambda value: value.detach()
        executor = SimpleNamespace(in_opaque_region=False, dispatch=Mock(side_effect=RuntimeError("unexpected replay")))
        observer = _CountingMode()
        with observer, ActivationDispatchMode(SimpleNamespace(executor=executor)), patch(
            "hyper_parallel.auto_parallel.hyper_offload.api.session.has_managed_inputs",
            side_effect=RuntimeError("native gradients should not require a pytree scan"),
        ):
            self.assertFalse(param.apply_reduced_grad(source))
        gradient = param.sharded_param.grad
        torch.testing.assert_close(gradient, source[:2], rtol=0, atol=0)
        self.assertEqual(gradient.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
        source.add_(3)
        torch.testing.assert_close(gradient, source[:2], rtol=0, atol=0)
        executor.dispatch.assert_not_called()
        self.assertGreater(observer.calls, 0)

    def test_framework_read_is_recorded_and_replayed_before_terminal_release(self) -> None:
        """Framework consumption extends the plan lifetime without tracking its outputs."""
        manager = ResidencyManager(1024)
        with patch("hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.DeviceTimer"):
            warmup = WarmupExecutor(manager, 1024)
        source = torch.arange(4, dtype=torch.float32)
        warmup._tracker.register_op_activations([], [source])
        sid = warmup._tracker._ensure_id(source)
        shadow = warmup.make_shadow(sid, source)
        param = _new_param()
        param.shard_world_size = 1
        param._unsharded_param.grad = shadow
        with ActivationDispatchMode(SimpleNamespace(executor=warmup)):
            param.reduce_scatter_grad()
        self.assertEqual(len(warmup._ops), 1)
        self.assertEqual(warmup._ops[0].accesses[0].storage_id, sid)
        self.assertEqual(warmup._ops[0].accesses[0].kind, AccessKind.READ)
        self.assertEqual(warmup._guide[0].output_bindings, {})
        del shadow
        warmup._last_flat_inputs = None
        warmup._last_args = ()
        warmup._last_func = None
        param.reduce_scatter_comm_ctx.reduce_scatter_output = None
        with patch(
            "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
            return_value=(16.0, 16.0),
        ):
            trace, guide = warmup.finish()
        self.assertNotIn(sid, trace.retained_sids)
        replay = ReplayExecutor(manager, PeakResidencyPlanner().build(trace), guide)
        param._unsharded_param.grad = replay.make_shadow(sid, source)
        with ActivationDispatchMode(SimpleNamespace(executor=replay)):
            param.reduce_scatter_grad()
        self.assertIsNone(manager.device_resident_size(sid))
        torch.testing.assert_close(param.reduce_scatter_comm_ctx.reduce_scatter_output, source)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_parameter_allocation_bypasses_only_offload(self) -> None:
        """Feature: FSDP storage ownership.

        Description: Run FSDP allocation and model computation under an observer.
        Expectation: Keep framework buffers resizable and preserve model dispatch.
        """
        param = _new_param()
        param.unsharded_param_buffers = [torch.empty(4)]
        param.unsharded_param_buffers[0].untyped_storage().resize_(0)
        executor = SimpleNamespace(
            in_opaque_region=False, dispatch=Mock(side_effect=lambda op, args, kwargs: op(*args, **kwargs)),
        )
        observer = _CountingMode()
        with observer, ActivationDispatchMode(SimpleNamespace(executor=executor)):
            with patch.object(param, "reduce_comm_dtype", return_value=torch.float32):
                param.shard_world_size = 1
                param._unsharded_param.grad = torch.arange(4, dtype=torch.float32)
                executor.dispatch.reset_mock()
                param.reduce_scatter_grad()
                self.assertEqual(executor.dispatch.call_count, 0)
            HSDPParamV2.alloc_unsharded_param_buffers(param)
            HSDPParamV2.free_unsharded_param(param)
            torch.ones(4)
            self.assertEqual(executor.dispatch.call_count, 1)
        self.assertGreater(observer.calls, 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_shadow_gradient_export_owns_storage(self) -> None:
        """Feature: FSDP storage ownership.

        Description: Export a managed gradient through the FSDP reduction boundary.
        Expectation: Keep exported storage independent of the managed activation.
        """
        source = torch.arange(4, dtype=torch.float32)
        buffer = PhysicalBuffer(device=source.device, device_buffer=source.view(torch.uint8))
        param = _new_param()
        param.shard_world_size = 1
        param._unsharded_param.grad = ShadowTensor(source, buffer, 1)
        executor = SimpleNamespace(
            in_opaque_region=False, dispatch=Mock(side_effect=lambda op, args, kwargs: op(*args, **kwargs)),
        )
        with ActivationDispatchMode(SimpleNamespace(executor=executor)):
            param.reduce_scatter_grad()
        self.assertEqual(executor.dispatch.call_count, 1)
        self.assertTrue(executor.dispatch.call_args.args[0].__name__.startswith("framework_read:"))
        output = param.reduce_scatter_comm_ctx.reduce_scatter_output
        self.assertNotIsInstance(output, ShadowTensor)
        self.assertNotEqual(output.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
        output.untyped_storage().resize_(0)
        torch.testing.assert_close(source, torch.arange(4, dtype=torch.float32))

    def test_export_preserves_output_aliases_offsets_and_input_writes(self) -> None:
        """Framework writes reach the source; exported views share independent storage."""
        source = torch.arange(8, dtype=torch.float32)
        buffer = PhysicalBuffer(device=source.device, device_buffer=source.view(torch.uint8))
        shadow = ShadowTensor(source, buffer, 1)

        def operation(tensor: torch.Tensor) -> tuple:
            """Return two overlapping views after an in-place update."""
            tensor.add_(1)
            return tensor[2:6], tensor[4:]

        first, second = dispatch_framework_op(operation, (shadow,), {})
        torch.testing.assert_close(source, torch.arange(8, dtype=torch.float32) + 1)
        self.assertEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())
        self.assertEqual(first.storage_offset(), 2)
        source.untyped_storage().resize_(0)
        torch.testing.assert_close(first, torch.arange(3, 7, dtype=torch.float32))
        torch.testing.assert_close(second, torch.arange(5, 9, dtype=torch.float32))


class TestRecordedFrameworkBoundary(unittest.TestCase):
    """Replay framework ownership from the trace without inspecting live stacks."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_native_framework_output_keeps_alias_and_other_modes(self) -> None:
        """Feature: Native framework operations recorded during warmup.

        Description: Replay a framework view followed by an ordinary model op.
        Expectation: Preserve native alias ownership, other modes, and sequence validation.
        """
        source = torch.arange(4.0)
        replay = ReplayExecutor(ResidencyManager(1024), ResidencySchedule(), [
            OpGuide("framework_read:view.default", input_bindings={}),
            OpGuide("sin.default", 1, {0: 1}, {}),
        ])
        observer = _CountingMode()
        with observer, ActivationDispatchMode(SimpleNamespace(executor=replay), record_framework_ops=True), patch(
            "hyper_parallel.auto_parallel.hyper_offload.api.session.inside_fsdp_runtime",
            side_effect=RuntimeError("native replay inspected the stack"),
        ):
            view = source.view(-1)
            actual = view.sin()
        replay.finish()
        torch.testing.assert_close(actual, source.sin(), rtol=0, atol=0)
        self.assertEqual(view.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
        self.assertIs(type(view), torch.Tensor)
        self.assertEqual(observer.calls, 2)
        replay.reset()
        with self.assertRaisesRegex(RuntimeError, "operator differs"):
            replay.dispatch(torch.ops.aten.cos.default, (source,), {})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_replay_framework_read_preserves_copies_aliases_and_gradients(self) -> None:
        """Feature: Recorded framework reads with activation transfers.

        Description: Offload and reload an activation before FSDP exports a view.
        Expectation: Exact forward/gradient parity and independently owned view storage.
        """
        def view(value: torch.Tensor) -> torch.Tensor:
            """Run identical view code under the framework module identity."""
            return value.view(-1)

        framework_view = FunctionType(view.__code__, {"__name__": "hyper_parallel.core.fully_shard.test"})

        def build(trace: ActivationTrace) -> ResidencySchedule:
            """Move the producer storage and reload before the framework read."""
            producer = next(op for op in trace.ops if op.name == "sin.default")
            consumer = next(op for op in trace.ops if op.name == "framework_read:view.default")
            sid = next(access.storage_id for access in producer.accesses if access.kind == AccessKind.WRITE)
            schedule = ResidencySchedule()
            schedule.add_post(trace.ops.index(producer), ResidencyActionType.COPY_D2H, sid)
            schedule.add_post(trace.ops.index(producer), ResidencyActionType.RELEASE_DEVICE, sid)
            schedule.add_pre(trace.ops.index(consumer), ResidencyActionType.COPY_H2D, sid)
            schedule.add_post(trace.ops.index(consumer), ResidencyActionType.RELEASE_HOST, sid)
            return schedule

        def run(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Consume both framework-owned and ordinary activation views."""
            activation = value.sin()
            native = value.cos()
            exported = framework_view(activation)  # pylint: disable=not-callable
            loss = (exported * native.view(-1)).sum()
            loss.backward()
            return activation, exported, loss

        data = torch.linspace(-2, 2, 16).reshape(4, 4)
        reference = data.clone().requires_grad_()
        expected = run(reference)
        with ExitStack() as stack:
            for target in ("torch.accelerator.current_stream", "torch.Event", "torch.Tensor.record_stream"):
                stack.enter_context(patch(target))
            stack.enter_context(patch(
                "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
                return_value=(10.0, 10.0),
            ))
            session = OffloadSession(OffloadConfig(planner=SimpleNamespace(build=build)))
            manager = session.residency_manager
            manager._host_pool = BytePool()
            manager._copy_stream = MagicMock(device=torch.device("cpu"))
            manager.wait_for_transfers = lambda: None
            session.executor._timer = Mock()
            session.executor._timer.stop.return_value = 0.0
            session.executor._tracker = CPUActivationTracker()
            warmup = data.clone().requires_grad_()
            with session:
                run(warmup)
            for _ in range(2):
                value = data.clone().requires_grad_()
                with patch(
                    "hyper_parallel.auto_parallel.hyper_offload.api.session.inside_fsdp_runtime",
                    side_effect=RuntimeError("replay inspected the stack"),
                ), patch.object(manager, "copy_d2h", wraps=manager.copy_d2h) as d2h, patch.object(
                    manager, "copy_h2d", wraps=manager.copy_h2d,
                ) as h2d:
                    with session:
                        activation, exported, loss = run(value)
                    d2h.assert_called_once()
                    h2d.assert_called_once()
                torch.testing.assert_close(loss, expected[2], rtol=0, atol=0)
                torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
                self.assertIs(type(exported), torch.Tensor)
                self.assertNotEqual(
                    exported.untyped_storage().data_ptr(), activation.resolve().untyped_storage().data_ptr(),
                )
                manager.release_device(activation.storage_id)
                torch.testing.assert_close(exported, expected[1], rtol=0, atol=0)
