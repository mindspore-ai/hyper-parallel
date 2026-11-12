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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from hyper_parallel.auto_parallel.hyper_offload.api.scope import (
    dispatch_framework_op,
    has_managed_inputs,
)
from hyper_parallel.auto_parallel.hyper_offload.api.session import (
    ActivationDispatchMode,
)
from hyper_parallel.auto_parallel.hyper_offload.execution.replay import ReplayExecutor
from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
from hyper_parallel.auto_parallel.hyper_offload.execution.warmup import WarmupExecutor
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import AccessKind
from hyper_parallel.auto_parallel.hyper_offload.planning import PeakResidencyPlanner
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import (
    PhysicalBuffer,
    ResidencyManager,
)
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from tests.ut.core.fully_shard.test_param import _new_param


class _CountingMode(TorchDispatchMode):
    """Observe unrelated dispatch mode preservation."""

    def __init__(self) -> None:
        """Initialize the operation count."""
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Count operations while preserving execution."""
        self.calls += 1
        return func(*args, **(kwargs or {}))


class TestFrameworkBoundary(unittest.TestCase):
    """Check real FSDP methods and exported storage lifetimes."""

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

    def test_native_gradient_assignment_preserves_alias_and_other_modes(self) -> None:
        """Framework metadata bypass keeps the exact gradient view and observer."""
        param = _new_param()
        source = torch.arange(4.0)
        param.to_sharded_dtensor = lambda value: value.detach()
        executor = SimpleNamespace(dispatch=Mock(side_effect=RuntimeError("unexpected replay")))
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

    def test_parameter_allocation_bypasses_only_offload(self) -> None:
        """FSDP gets a resizable buffer, while model computation remains tracked."""
        param = _new_param()
        param.unsharded_param_buffers = [torch.empty(4)]
        param.unsharded_param_buffers[0].untyped_storage().resize_(0)
        executor = SimpleNamespace(dispatch=Mock(side_effect=lambda op, args, kwargs: op(*args, **kwargs)))
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

    def test_shadow_gradient_export_owns_storage(self) -> None:
        """Unmodified FSDP can retire a gradient without resizing activation storage."""
        source = torch.arange(4, dtype=torch.float32)
        buffer = PhysicalBuffer(device=source.device, device_buffer=source.view(torch.uint8))
        param = _new_param()
        param.shard_world_size = 1
        param._unsharded_param.grad = ShadowTensor(source, buffer, 1)
        executor = SimpleNamespace(dispatch=Mock(side_effect=lambda op, args, kwargs: op(*args, **kwargs)))
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
