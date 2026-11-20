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
"""Host dispatch optimizations preserve lifetime and replay contracts."""

import gc
import unittest
import weakref
from types import FunctionType
from unittest.mock import patch

import torch

from hyper_parallel.auto_parallel.hyper_offload.api.scope import (
    _is_fsdp_module,
    inside_fsdp_runtime,
)
from hyper_parallel.auto_parallel.hyper_offload.execution.replay import ReplayExecutor
from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyActionType as Action,
)
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencySchedule
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import (
    ResidencyManager,
)


class TestHostDispatch(unittest.TestCase):
    """Use real CPU storage and inspect externally visible contracts."""

    def test_binding_alias_reuses_view_and_transfer_fences(self) -> None:
        """Alias binding preserves the existing byte view and pending events."""
        manager = ResidencyManager(1024)
        source = torch.arange(8.0)
        buffer = manager.bind(1, source)
        byte_view = buffer.device_buffer
        host_event, device_event = object(), object()
        buffer.host_event, buffer.device_event = host_event, device_event
        alias = source[2:6]
        with patch("hyper_parallel.auto_parallel.hyper_offload.runtime.residency.torch.empty") as allocate:
            self.assertIs(manager.bind(1, alias), buffer)
            allocate.assert_not_called()
        self.assertIs(buffer.device_buffer, byte_view)
        self.assertIs(buffer.host_event, host_event)
        self.assertIs(buffer.device_event, device_event)
        source.add_(1)
        torch.testing.assert_close(byte_view.view(source.dtype), source, rtol=0, atol=0)

    def test_binding_rebuilds_view_after_storage_growth(self) -> None:
        """An unchanged storage identity must not hide a changed byte extent."""
        manager = ResidencyManager(1024)
        source = torch.arange(8.0)
        buffer = manager.bind(1, source)
        old_view = buffer.device_buffer
        identity = source.untyped_storage()._cdata
        source.resize_(16)
        source.fill_(3)
        self.assertEqual(source.untyped_storage()._cdata, identity)
        self.assertIs(manager.bind(1, source), buffer)
        self.assertIsNot(buffer.device_buffer, old_view)
        self.assertEqual(manager.resident_bytes, source.untyped_storage().nbytes())
        torch.testing.assert_close(buffer.device_buffer.view(source.dtype), source, rtol=0, atol=0)

    def test_binding_rebuilds_after_replacement_and_release(self) -> None:
        """The same logical ID can acquire replacement or newly resident storage."""
        manager = ResidencyManager(1024)
        source = torch.arange(8.0)
        buffer = manager.bind(1, source)
        old_view = buffer.device_buffer
        replacement = source + 2
        manager.bind(1, replacement)
        self.assertIsNot(buffer.device_buffer, old_view)
        torch.testing.assert_close(buffer.device_buffer.view(source.dtype), replacement, rtol=0, atol=0)
        manager.release_device(1)
        self.assertIsNone(buffer.device_buffer)
        buffer = manager.bind(1, source)
        torch.testing.assert_close(buffer.device_buffer.view(source.dtype), source, rtol=0, atol=0)

    def test_duplicate_inputs_resolve_once_without_retaining_storage(self) -> None:
        """Reuse a view within an op, then observe replacement storage next op."""
        source = torch.arange(6.0)
        manager = ResidencyManager(1024)
        buffer = manager.bind(1, source)
        shadow = ShadowTensor(source, buffer, 1)
        with patch.object(ShadowTensor, "resolve", autospec=True, side_effect=ShadowTensor.resolve) as resolve:
            result = torch.add(shadow, shadow)
            self.assertEqual(resolve.call_count, 1)
        torch.testing.assert_close(result, source * 2, rtol=0, atol=0)
        replacement = torch.arange(6.0) + 10
        buffer.device_buffer = replacement.view(torch.uint8)
        torch.testing.assert_close(torch.add(shadow, shadow), replacement * 2, rtol=0, atol=0)
        buffer.device_buffer = None
        with self.assertRaisesRegex(RuntimeError, "No device buffer"):
            torch.add(shadow, shadow)

    def test_resolution_cache_dies_without_cyclic_collection(self) -> None:
        """An op-local cache must release views without waiting for cyclic GC."""
        manager = ResidencyManager(1024)
        source = torch.arange(8.0)
        shadow = ShadowTensor(source, manager.bind(1, source), 1)
        refs = []
        original = ShadowTensor.resolve

        def resolve(value: ShadowTensor) -> torch.Tensor:
            """Track the lifetime of the temporary resolved view."""
            result = original(value)
            refs.append(weakref.ref(result))
            return result

        enabled = gc.isenabled()
        gc.disable()
        try:
            with patch.object(ShadowTensor, "resolve", autospec=True, side_effect=resolve):
                result = torch.add(shadow, shadow)
            torch.testing.assert_close(result, source * 2, rtol=0, atol=0)
            self.assertEqual(len(refs), 1)
            self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if enabled:
                gc.enable()

    def test_native_output_avoids_shadow_traversal_and_survives_reset(self) -> None:
        """No-copy native storage is not owned or released by the residency table."""
        manager = ResidencyManager(1024)
        replay = ReplayExecutor(manager, ResidencySchedule(), [OpGuide(
            name="sin.default", output_leaf_count=1, output_bindings={0: 1}, input_bindings={},
        )])
        source = torch.arange(4.0)
        with patch("hyper_parallel.auto_parallel.hyper_offload.execution.base.tree_unflatten") as rebuild:
            output = replay.dispatch(torch.ops.aten.sin.default, (source,), {})
            rebuild.assert_not_called()
        self.assertIs(type(output), torch.Tensor)
        self.assertEqual(manager.resident_bytes, 0)
        replay.finish()
        replay.reset()
        torch.testing.assert_close(output, source.sin(), rtol=0, atol=0)

    def test_truncated_and_wrong_operator_replay_are_rejected(self) -> None:
        """Selective ownership cannot silently advance an incompatible schedule."""
        replay = ReplayExecutor(ResidencyManager(1024), ResidencySchedule(), [OpGuide(
            name="sin.default", output_leaf_count=1, input_bindings={},
        )])
        with self.assertRaisesRegex(RuntimeError, "op count differs"):
            replay.finish()
        with self.assertRaisesRegex(RuntimeError, "operator differs"):
            replay.dispatch(torch.ops.aten.cos.default, (torch.ones(4),), {})

    def test_active_plan_leaves_unrelated_ops_native(self) -> None:
        """One scheduled copy must not flatten or wrap unrelated native ops."""
        manager = ResidencyManager(1024)
        schedule = ResidencySchedule()
        schedule.add_post(2, Action.COPY_D2H, 2)
        replay = ReplayExecutor(manager, schedule, [
            OpGuide("sin.default", 1, {0: 1}, {}),
            OpGuide("detach.default", 1, {0: 1}, {0: 1}),
            OpGuide("cos.default", 1, {0: 2}, {}),
        ])
        source = torch.arange(4.0)
        with patch("hyper_parallel.auto_parallel.hyper_offload.execution.base.tree_flatten") as flatten:
            native = replay.dispatch(torch.ops.aten.sin.default, (source,), {})
            flatten.assert_not_called()
        self.assertIs(type(native), torch.Tensor)
        self.assertEqual(manager.resident_bytes, 0)
        with patch.object(manager, "copy_d2h") as copy:
            moved = replay.dispatch(torch.ops.aten.cos.default, (source,), {})
            copy.assert_called_once_with(2)
        self.assertIsInstance(moved, ShadowTensor)
        self.assertEqual(replay.op_idx, 2)
        replay.finish()
        torch.testing.assert_close(native, source.sin(), rtol=0, atol=0)
        torch.testing.assert_close(moved, source.cos(), rtol=0, atol=0)

    def test_transfer_on_optional_boundary_keeps_anchor(self) -> None:
        """A real transfer on detach or framework-read cannot disappear."""
        for name in ("detach.default", "framework_read:view.default"):
            with self.subTest(boundary=name):
                schedule = ResidencySchedule()
                schedule.add_pre(1, Action.COPY_H2D, 99)
                replay = ReplayExecutor(ResidencyManager(1024), schedule, [
                    OpGuide("sin.default", 1, {0: 1}, {}),
                    OpGuide(name, 1, {0: 1}, {0: 1}),
                    OpGuide("cos.default", 1, {0: 2}, {}),
                ])
                source = torch.arange(4.0)
                anchor = replay.dispatch(torch.ops.aten.sin.default, (source,), {})
                self.assertIsInstance(anchor, ShadowTensor)
                with self.assertRaisesRegex(RuntimeError, "operator differs"):
                    replay.dispatch(torch.ops.aten.cos.default, (source,), {})

    def test_sparse_managed_op_does_not_retain_native_inputs(self) -> None:
        """Input caches release native tensors before the next sparse boundary."""
        manager = ResidencyManager(1024)
        schedule = ResidencySchedule()
        schedule.add_post(0, Action.COPY_D2H, 1)
        replay = ReplayExecutor(manager, schedule, [OpGuide("sin.default", 1, {0: 1}, {})])
        source = torch.arange(4.0)
        ref = weakref.ref(source)
        with patch.object(manager, "copy_d2h"):
            output = replay.dispatch(torch.ops.aten.sin.default, (source,), {})
        del source
        self.assertIsNone(ref())
        self.assertIsInstance(output, ShadowTensor)
        replay.finish()

    def test_fsdp_classification_does_not_cache_calling_context(self) -> None:
        """The same code called within and outside FSDP gets distinct decisions."""
        def invoke() -> bool:
            """Inspect the current execution stack."""
            return inside_fsdp_runtime()

        framework = FunctionType(invoke.__code__, {
            "__name__": "hyper_parallel.core.fully_shard.host_test",
            "inside_fsdp_runtime": inside_fsdp_runtime,
        })
        _is_fsdp_module.cache_clear()
        self.addCleanup(_is_fsdp_module.cache_clear)
        self.assertFalse(invoke())
        self.assertTrue(framework())  # pylint: disable=not-callable  # FunctionType creates a callable.
        self.assertFalse(invoke())
        self.assertTrue(framework())  # pylint: disable=not-callable  # FunctionType creates a callable.
        self.assertGreater(_is_fsdp_module.cache_info().hits, 0)
