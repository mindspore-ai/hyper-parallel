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
"""CPU regression tests for mHC contracts and per-invocation state ownership."""

from dataclasses import fields, replace
import importlib
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_mhc.plan import MegaMhcPlan
from hyper_parallel.core.multicore.modules.mega_mhc_common import (
    validate_mega_mhc_hidden_size,
    validate_mega_mhc_native_inputs,
    validate_mega_mhc_tensor,
    validate_mega_mhc_token_count,
)
from hyper_parallel.core.multicore.modules.mega_mhc_grad.common import (
    MegaMhcGradCache,
    validate_mega_mhc_grad_tensors,
)
from hyper_parallel.core.multicore.modules.mega_mhc_grad.plan import MegaMhcGradPlan


def _make_inputs() -> tuple[torch.Tensor, ...]:
    """Create CPU metadata fixtures without any accelerator initialization."""
    return (
        torch.zeros(40, 128, dtype=torch.bfloat16),
        torch.zeros(40, 4, 128, dtype=torch.bfloat16),
        torch.zeros(40, 4),
        torch.zeros(40, 4),
        torch.zeros(40, 4, 4),
        torch.zeros(24, 512),
        torch.ones(3),
        torch.zeros(24),
        torch.ones(128, dtype=torch.bfloat16),
    )


def _make_cache() -> MegaMhcGradCache:
    """Create native-layout CPU cache fixtures, not a numerical reference."""
    return MegaMhcGradCache(
        new_residual=torch.zeros(1, 40, 4, 128, dtype=torch.bfloat16),
        hc_before_norm=torch.zeros(1, 40, 24),
        inv_rms=torch.zeros(1, 40, 1),
        sum_out=torch.zeros(40, 1, 40, 4),
        norm_out=torch.zeros(40, 1, 40, 4, 4),
        mixed_input=torch.zeros(1, 40, 128, dtype=torch.bfloat16),
        rms_rstd=torch.zeros(1, 40, 1),
    )


def _make_gradients() -> tuple[torch.Tensor, ...]:
    """Create gradients in public forward-output order."""
    inputs = _make_inputs()
    return (inputs[1], inputs[2], inputs[3], inputs[4], inputs[0])


class TestMegaMhcContracts(unittest.TestCase):
    """Validate all native tensor metadata before launching a kernel."""

    def test_training_hidden_and_token_limits(self) -> None:
        """Feature: Native shape contracts.

        Description: Exercise alignment, backward hidden limits and physical AIV counts.
        Expectation: Unsupported training shapes fail with descriptive errors.
        """
        validate_mega_mhc_hidden_size(5760, need_backward=True)
        validate_mega_mhc_hidden_size(7168, need_backward=False)
        validate_mega_mhc_hidden_size(12288, need_backward=False)
        with self.assertRaisesRegex(ValueError, "forward requires hidden_size <= 12288"):
            validate_mega_mhc_hidden_size(12416, need_backward=False)
        for hidden in (0, -128, 127, True, 128.0, 7168):
            with self.subTest(hidden=hidden), self.assertRaises(ValueError):
                validate_mega_mhc_hidden_size(hidden, need_backward=True)
        for cores, minimum in ((32, 40), (40, 40), (48, 48)):
            with self.subTest(cores=cores):
                validate_mega_mhc_token_count(minimum, cores)
                with self.assertRaisesRegex(ValueError, f"at least {minimum} tokens"):
                    validate_mega_mhc_token_count(minimum - 1, cores)

    def test_native_inputs_validate_every_dtype_and_device(self) -> None:
        """Feature: Production input validation.

        Description: Supply NPU metadata mocks, then corrupt each input independently.
        Expectation: Valid metadata passes; every wrong dtype/device/shape is rejected.
        """
        device = SimpleNamespace(type="npu", index=0)
        inputs = []
        for tensor in _make_inputs():
            value = Mock(spec=torch.Tensor)
            value.shape, value.ndim = tuple(tensor.shape), tensor.ndim
            value.dtype, value.device = tensor.dtype, device
            value.numel.return_value = math.prod(value.shape)
            inputs.append(value)
        inputs = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                  inputs[5], inputs[6], inputs[7], inputs[8])
        validate_mega_mhc_native_inputs(*inputs, need_backward=True, token_tile=32)
        for index, value in enumerate(inputs):
            with self.subTest(index=index, invalid="dtype"):
                original = value.dtype
                value.dtype = torch.float64
                with self.assertRaisesRegex(TypeError, "must have dtype"):
                    validate_mega_mhc_native_inputs(*inputs, need_backward=True, token_tile=32)
                value.dtype = original
            with self.subTest(index=index, invalid="device"):
                value.device = SimpleNamespace(type="npu", index=1)
                with self.assertRaisesRegex(ValueError, "must be on"):
                    validate_mega_mhc_native_inputs(*inputs, need_backward=True, token_tile=32)
                value.device = device
        inputs[0].shape = (40, 127)
        with self.assertRaisesRegex(ValueError, "previous_output must have shape"):
            validate_mega_mhc_native_inputs(*inputs, need_backward=True, token_tile=32)

    def test_rejects_cpu_and_invalid_scalar_inputs(self) -> None:
        """Feature: Native launch preconditions.

        Description: Pass CPU tensors, non-tensors, empty tokens and invalid tile sizes.
        Expectation: Validation fails before device discovery or runtime allocation.
        """
        inputs = _make_inputs()
        with self.assertRaisesRegex(TypeError, "requires NPU tensors"):
            validate_mega_mhc_native_inputs(*inputs, need_backward=True, token_tile=32)
        for tile in (0, -1, True, 1.5):
            with self.subTest(tile=tile), self.assertRaisesRegex(ValueError, "token_tile"):
                validate_mega_mhc_native_inputs(*inputs, need_backward=True, token_tile=tile)
        with self.assertRaisesRegex(TypeError, "previous_output must be a Tensor"):
            validate_mega_mhc_native_inputs(None, *inputs[1:], need_backward=True, token_tile=32)
        empty = tuple(value[:0] if index < 5 else value for index, value in enumerate(inputs))
        with self.assertRaisesRegex(ValueError, "at least one token"):
            validate_mega_mhc_native_inputs(*empty, need_backward=True, token_tile=32)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            validate_mega_mhc_tensor("test", inputs[0], (128,), inputs[0].dtype, inputs[0].device)
        with self.assertRaisesRegex(ValueError, "must be on"):
            validate_mega_mhc_tensor("test", inputs[0], (40, 128), inputs[0].dtype, torch.device("meta"))

    def test_validates_every_gradient_and_cache_tensor(self) -> None:
        """Feature: Backward cache and gradient contracts.

        Description: Corrupt each gradient and cache tensor independently.
        Expectation: Exact shapes, dtypes and device ownership are enforced.
        """
        residual, gradients, cache = _make_inputs()[1], _make_gradients(), _make_cache()
        validate_mega_mhc_grad_tensors(gradients, residual, cache)
        for index, value in enumerate(gradients):
            for invalid in (value.to(torch.float64), value.flatten(), value.to("meta")):
                with self.subTest(gradient=index, shape=invalid.shape, dtype=invalid.dtype, device=invalid.device):
                    changed = list(gradients)
                    changed[index] = invalid
                    with self.assertRaises((TypeError, ValueError)):
                        validate_mega_mhc_grad_tensors(tuple(changed), residual, cache)
        for field in fields(cache):
            value = getattr(cache, field.name)
            for invalid in (value.to(torch.float64), value.flatten(), value.to("meta")):
                with self.subTest(cache=field.name, shape=invalid.shape, dtype=invalid.dtype, device=invalid.device):
                    with self.assertRaises((TypeError, ValueError)):
                        validate_mega_mhc_grad_tensors(gradients, residual, replace(cache, **{field.name: invalid}))
        with self.assertRaisesRegex(TypeError, "MegaMhcGradCache"):
            validate_mega_mhc_grad_tensors(gradients, residual, None)
        with self.assertRaisesRegex(ValueError, "five output gradients"):
            validate_mega_mhc_grad_tensors(gradients[:-1], residual, cache)


class TestMegaMhcInvocationState(unittest.TestCase):
    """Exercise actual launch wrappers while mocking only native/device boundaries."""

    def test_cached_plans_do_not_own_event_tensors(self) -> None:
        """Feature: Immutable plan caching.

        Description: Inspect forward and backward plan dataclasses.
        Expectation: Only event capacity, never mutable counters, is cached.
        """
        for plan_type in (MegaMhcPlan, MegaMhcGradPlan):
            names = {field.name for field in fields(plan_type)}
            self.assertIn("event_capacity", names)
            self.assertNotIn("event_counters", names)

    def _check_invocation_buffers(self, *, backward: bool) -> None:
        """Run two same-plan calls while retaining both live counter buffers."""
        package = "mega_mhc_grad" if backward else "mega_mhc"
        function = importlib.import_module(f"hyper_parallel.core.multicore.modules.{package}.function")
        inputs, cache, buffers = _make_inputs(), _make_cache(), []
        plan = SimpleNamespace(runtime=SimpleNamespace(runtime_config=object(), is_active=False), event_capacity=1024)

        def _launch(*native_args, **options):
            """Retain and mutate counters as an in-flight native invocation would."""
            events = native_args[22 if backward else 10]
            self.assertNotIn("diagnostics", options)
            self.assertEqual(torch.count_nonzero(events).item(), 0)
            buffers.append(events)
            # Simulate counters mutated by an in-flight native invocation.
            events.fill_(1)
            return native_outputs

        if backward:
            native_outputs = (inputs[1], inputs[5], inputs[6], inputs[7], inputs[0],
                              inputs[2], inputs[3], inputs[4], inputs[8])
            target = function.hyper_mega_mhc_grad
            args = (*_make_gradients(), *inputs, cache)
            launch_name = "multicore_ops.hyper_mega_mhc_grad"
        else:
            native_outputs = (cache.new_residual, inputs[2], inputs[3], inputs[4], inputs[0],
                              cache.hc_before_norm, cache.inv_rms, cache.sum_out, cache.norm_out,
                              cache.mixed_input, cache.rms_rstd)
            target = function._hyper_mega_mhc_forward
            args = inputs
            launch_name = "multicore_ops.hyper_mega_mhc"
        with (
            patch.object(function, "validate_mega_mhc_native_inputs"),
            patch.object(function, "_resolve_core_counts", return_value=(20, 40)),
            patch.object(function, "_get_cached_plan", return_value=plan),
            patch.object(torch, "npu", SimpleNamespace(current_device=lambda: 0), create=True),
            patch(f"{function.__name__}.{launch_name}", side_effect=_launch),
        ):
            target(*args)
            target(*args)
        self.assertEqual(len(buffers), 2)
        self.assertNotEqual(buffers[0].data_ptr(), buffers[1].data_ptr())
        for buffer in buffers:
            self.assertEqual(buffer.dtype, torch.uint8)
            self.assertEqual(buffer.numel(), 4096)
            self.assertEqual(torch.count_nonzero(buffer).item(), 4096)

    def test_forward_invocations_use_private_counters(self) -> None:
        """Feature: Forward invocation isolation.

        Description: Launch twice through the same cached plan.
        Expectation: Each launch receives independent zero-initialized counters.
        """
        self._check_invocation_buffers(backward=False)

    def test_backward_invocations_use_private_counters(self) -> None:
        """Feature: Backward invocation isolation.

        Description: Launch twice through the same cached plan.
        Expectation: No counter clearing can corrupt another in-flight invocation.
        """
        self._check_invocation_buffers(backward=True)
