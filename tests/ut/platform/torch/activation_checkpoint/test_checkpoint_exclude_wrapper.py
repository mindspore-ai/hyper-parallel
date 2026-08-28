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
"""Tests for the PyTorch checkpoint exclusion wrapper."""
import importlib
import unittest

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.platform.torch.activation_checkpoint import (
    ckpt_wrapper as checkpoint_wrapper,
    checkpoint_exclude_wrapper as exclude_factory,
    clear_recompute_session,
    recompute_handle,
    recompute_handle_collector_ctx,
    recompute_session_ctx,
)
from hyper_parallel.platform.torch.activation_checkpoint.checkpoint_exclude_wrapper import (
    CheckpointExcludeWrapper,
)
from tests.ut.platform.mindspore._ensure_mindspore_platform import restore_torch_platform_for_ut

checkpoint_exclude_wrapper = exclude_factory
exclude_impl = importlib.import_module(
    "hyper_parallel.platform.torch.activation_checkpoint.checkpoint_exclude_wrapper"
)


class _CountedSquare(torch.nn.Module):
    """Square its input and record real executions."""

    def __init__(self) -> None:
        """Initialize the call counter."""
        super().__init__()
        self.calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a value whose backward saves the exact input."""
        self.calls += 1
        return tensor.square()


class _SavedInputBlock(torch.nn.Module):
    """Checkpointed block whose SAVE tail uses a recomputed input."""

    def __init__(self) -> None:
        """Initialize the excluded region."""
        super().__init__()
        self.excluded = checkpoint_exclude_wrapper(_CountedSquare())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Recompute the head but execute the SAVE tail only once."""
        return self.excluded(tensor * 3).sum()


class _FirstSave(torch.nn.Module):
    """Return two differentiable outputs from the first SAVE region."""

    def __init__(self) -> None:
        """Initialize the call counter."""
        super().__init__()
        self.calls = 0

    def forward(self, tensor: torch.Tensor) -> tuple:
        """Return two outputs that both contribute to backward."""
        self.calls += 1
        return tensor.square(), tensor * tensor * tensor


class _SecondSave(torch.nn.Module):
    """Consume the complete first SAVE output as one argument."""

    def __init__(self) -> None:
        """Initialize the call counter."""
        super().__init__()
        self.calls = 0

    def forward(self, outputs: tuple) -> torch.Tensor:
        """Combine both outputs without exposing them to replay."""
        self.calls += 1
        return outputs[0] + outputs[1]


class _MultiOutputSaveBlock(torch.nn.Module):
    """Checkpointed block with adjacent multi-output SAVE regions."""

    def __init__(self, save_first_output: bool) -> None:
        """Initialize adjacent SAVE wrappers."""
        super().__init__()
        self.first = checkpoint_exclude_wrapper(_FirstSave(), save_output=save_first_output)
        self.second = checkpoint_exclude_wrapper(_SecondSave())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a scalar depending on every first-region output."""
        return self.second(self.first(tensor * 2)).sum()


class _PrecisionBlock(torch.nn.Module):
    """Deterministic network used to compare checkpoint precision."""

    def __init__(self, exclude_middle: bool) -> None:
        """Initialize deterministic layers and an optional SAVE middle."""
        super().__init__()
        self.head = torch.nn.Linear(4, 5)
        self.middle = torch.nn.Sequential(torch.nn.Linear(5, 5), torch.nn.Tanh())
        self.tail = torch.nn.Linear(5, 3)
        if exclude_middle:
            self.middle = checkpoint_exclude_wrapper(self.middle)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss."""
        output = self.tail(self.middle(self.head(tensor)))
        return output.square().mean()


class _SplitBackwardBlock(torch.nn.Module):
    """Checkpointed block used to validate prefired dx/dw sessions."""

    def __init__(self) -> None:
        """Initialize one excluded Linear region."""
        super().__init__()
        self.linear = checkpoint_exclude_wrapper(torch.nn.Linear(4, 4, bias=False))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a scalar whose backward needs input and weight gradients."""
        return self.linear(tensor * 2).square().sum()


def _initialize_precision_parameters(module: torch.nn.Module) -> None:
    """Fill every parameter with deterministic values."""
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.linspace(-0.3 + index * 0.01, 0.3 + index * 0.01, parameter.numel())
            parameter.copy_(values.reshape_as(parameter))


def _run_precision_step(module: torch.nn.Module, scale: float) -> tuple:
    """Run one backward step and clone value, input grad, and parameter grads."""
    tensor = torch.linspace(-1.5, 1.25, 24).reshape(6, 4).mul(scale).requires_grad_()
    value = module(tensor)
    value.backward()
    result = (
        value.detach().clone(),
        tensor.grad.detach().clone(),
        tuple(parameter.grad.detach().clone() for parameter in module.parameters()),
    )
    for parameter in module.parameters():
        parameter.grad = None
    return result


class TestCheckpointExcludeWrapper(unittest.TestCase):
    """Validate PyTorch function and Module checkpoint exclusion wrappers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Restore Torch core aliases after cross-backend test collection."""
        restore_torch_platform_for_ut()
        previous_device_type = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")
        cls.addClassCleanup(DefaultDeviceType.set_device_type, previous_device_type)

    def test_factory_validates_inputs(self) -> None:
        """Reject non-callables and non-boolean save_output values."""
        with self.assertRaisesRegex(ValueError, "module must"):
            exclude_factory(1)
        with self.assertRaisesRegex(ValueError, "save_output must be a bool"):
            exclude_factory(lambda value: value, save_output=1)

    def test_factory_wraps_module_and_callable(self) -> None:
        """Normalize both supported input forms to one wrapper type."""
        self.assertIsInstance(exclude_factory(_CountedSquare()), CheckpointExcludeWrapper)
        self.assertIsInstance(exclude_factory(lambda value: value.square()), CheckpointExcludeWrapper)

    def test_wrapper_is_transparent_outside_checkpoint(self) -> None:
        """Execute the wrapped region normally when no checkpoint state exists."""
        region = exclude_factory(_CountedSquare())
        tensor = torch.tensor([2.0], requires_grad=True)

        region(tensor).sum().backward()

        self.assertEqual(region.calls, 1)
        torch.testing.assert_close(tensor.grad, torch.tensor([4.0]))

    def test_saved_tensor_hook_uses_transient_tensor_attribute(self) -> None:
        """The pack hook should replace a marked input with its deferred handle."""
        tensor = torch.arange(4.0, requires_grad=True)
        recomputed = tensor.detach().mul(2)
        handle = exclude_impl._RecomputedInputHandle()
        setattr(tensor, exclude_impl._RECOMPUTE_INPUT_HANDLE_ATTR, handle)

        packed = exclude_impl._pack_saved_tensor(tensor)

        self.assertIs(packed, handle)
        self.assertTrue(handle.used)
        with self.assertRaisesRegex(RuntimeError, "before recomputation"):
            exclude_impl._unpack_saved_tensor(packed)
        handle.materialize(recomputed)
        self.assertIs(exclude_impl._unpack_saved_tensor(packed), recomputed)

    def test_input_marking_restores_existing_attribute(self) -> None:
        """Restore an input's previous private attribute after one SAVE call."""
        tensor = torch.arange(4.0, requires_grad=True)
        previous = object()
        setattr(tensor, exclude_impl._RECOMPUTE_INPUT_HANDLE_ATTR, previous)

        bindings, previous_handles = exclude_impl._mark_recompute_inputs(object(), (tensor,), {})
        self.assertEqual(len(bindings), 1)
        exclude_impl._restore_recompute_inputs(previous_handles)

        self.assertIs(getattr(tensor, exclude_impl._RECOMPUTE_INPUT_HANDLE_ATTR), previous)

    def test_parameter_input_is_not_marked(self) -> None:
        """Parameter inputs should retain their normal saved-tensor behavior."""
        parameter = torch.nn.Parameter(torch.arange(4.0))

        bindings, previous_handles = exclude_impl._mark_recompute_inputs(object(), (parameter,), {})

        self.assertEqual(bindings, [])
        self.assertEqual(previous_handles, [])

    def test_recompute_boundary_saves_only_zero_element_trigger(self) -> None:
        """The boundary should not retain storage from its tensor output."""
        packed_shapes = []
        unpacked_shapes = []

        def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
            """Record the tensor packed by the boundary."""
            packed_shapes.append(tuple(tensor.shape))
            return tensor

        def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
            """Record the tensor unpacked by the boundary."""
            unpacked_shapes.append(tuple(tensor.shape))
            return tensor

        tensor = torch.arange(4.0, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            boundary = exclude_impl._get_recompute_boundary()
            output = boundary.apply(tensor, exclude_impl._get_recompute_trigger())
            output.sum().backward()

        self.assertEqual(packed_shapes, [(0,)])
        self.assertEqual(unpacked_shapes, [(0,)])
        torch.testing.assert_close(tensor.grad, torch.ones(4))

    def test_recompute_to_save_uses_replayed_input(self) -> None:
        """SAVE backward should use a replayed upstream activation without rerunning SAVE."""
        block = checkpoint_wrapper(_SavedInputBlock())
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)

        value = block(tensor)
        value.backward()

        self.assertEqual(block.excluded.calls, 1)
        torch.testing.assert_close(value, torch.tensor(45.0))
        torch.testing.assert_close(tensor.grad, torch.tensor([18.0, 36.0]))

    def test_adjacent_save_multi_output_elides_first_output(self) -> None:
        """A multi-output middle SAVE should preserve gradients and execute once."""
        block = checkpoint_wrapper(_MultiOutputSaveBlock(save_first_output=False))
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)

        value = block(tensor)
        value.backward()

        self.assertEqual(block.first.calls, 1)
        self.assertEqual(block.second.calls, 1)
        torch.testing.assert_close(value, torch.tensor(92.0))
        torch.testing.assert_close(tensor.grad, torch.tensor([32.0, 112.0]))

    def test_default_save_output_preserves_adjacent_save_precision(self) -> None:
        """The default output-cache behavior should match output elision numerically."""
        optimized = checkpoint_wrapper(_MultiOutputSaveBlock(save_first_output=False))
        legacy = checkpoint_wrapper(_MultiOutputSaveBlock(save_first_output=True))
        optimized_input = torch.tensor([1.0, 2.0], requires_grad=True)
        legacy_input = optimized_input.detach().clone().requires_grad_()

        optimized_value = optimized(optimized_input)
        legacy_value = legacy(legacy_input)
        optimized_value.backward()
        legacy_value.backward()

        torch.testing.assert_close(optimized_value, legacy_value)
        torch.testing.assert_close(optimized_input.grad, legacy_input.grad)

    def test_repeated_steps_match_full_recompute_precision(self) -> None:
        """SAVE boundaries should match ordinary checkpoint values and gradients across steps."""
        baseline = checkpoint_wrapper(_PrecisionBlock(exclude_middle=False))
        optimized = checkpoint_wrapper(_PrecisionBlock(exclude_middle=True))
        _initialize_precision_parameters(baseline)
        optimized.load_state_dict(baseline.state_dict())

        for scale in (1.0, 1.7):
            baseline_result = _run_precision_step(baseline, scale)
            optimized_result = _run_precision_step(optimized, scale)
            torch.testing.assert_close(optimized_result[0], baseline_result[0])
            torch.testing.assert_close(optimized_result[1], baseline_result[1])
            for optimized_grad, baseline_grad in zip(optimized_result[2], baseline_result[2]):
                torch.testing.assert_close(optimized_grad, baseline_grad)

    def test_prefired_recompute_supports_split_input_and_weight_gradients(self) -> None:
        """One retained replay should serve separate input and weight gradient calls."""
        block = checkpoint_wrapper(_SplitBackwardBlock())
        tensor = torch.linspace(-1.0, 1.0, 12).reshape(3, 4).requires_grad_()
        reference_tensor = tensor.detach().clone().requires_grad_()
        reference_weight = block.linear.weight.detach().clone().requires_grad_()
        reference = ((reference_tensor * 2) @ reference_weight.t()).square().sum()
        expected_dx, expected_dw = torch.autograd.grad(reference, (reference_tensor, reference_weight))

        with recompute_handle_collector_ctx() as handles:
            output = block(tensor)
        self.assertEqual(len(handles), 1)
        session_id = ("checkpoint_exclude_split", id(output))
        try:
            with recompute_session_ctx(session_id=session_id, retain_on_unpack=True):
                recompute_handle(handles[0], session_id)
                actual_dx = torch.autograd.grad(output, tensor, retain_graph=True)[0]
            with recompute_session_ctx(session_id=session_id, retain_on_unpack=False):
                actual_dw = torch.autograd.grad(output, block.linear.weight)[0]
        finally:
            clear_recompute_session(session_id)

        torch.testing.assert_close(actual_dx, expected_dx)
        torch.testing.assert_close(actual_dw, expected_dw)


if __name__ == "__main__":
    unittest.main()
