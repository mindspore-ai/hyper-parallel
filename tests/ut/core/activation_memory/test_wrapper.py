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
"""Unit tests for ``hyper_parallel.core.activation_memory.wrapper``."""
# The Torch backend must be selected before importing HyperParallel modules.
# Tests intentionally inspect protected wrapper state and helpers.
# pylint: disable=wrong-import-position,protected-access,unused-argument
import contextlib
import gc
import importlib
import os
import unittest
import warnings
import weakref
from typing import Any, Callable, cast
from unittest.mock import MagicMock, call, patch, sentinel

import torch
from torch.utils.checkpoint import DefaultDeviceType


from hyper_parallel.core.activation_memory.api import (
    CheckpointPolicy,
    clear_recompute_session,
    recompute_handle,
    recompute_handle_collector_ctx,
    recompute_session_ctx,
)
from hyper_parallel.core.activation_memory import wrapper as wrapper_module
from hyper_parallel.core.activation_memory.wrapper import (
    ActivationWrapper,
    AsyncSaveOnCpu,
    CheckpointExcludeWrapper,
    CheckpointWrapper,
    FuncModule,
    SwapWrapper,
    base_check_fn,
    checkpoint_exclude_wrapper,
    ckpt_wrapper,
    swap_tensor_wrapper,
    swap_wrapper,
)

_api_module = importlib.import_module("hyper_parallel.core.activation_memory.api")
_checkpoint_exclude = importlib.import_module("hyper_parallel.core.activation_memory.checkpoint_exclude")
_swap_module = importlib.import_module("hyper_parallel.core.activation_memory.swap")


class _TinyModule(torch.nn.Module):
    """Small module used by wrapper tests."""

    def __init__(self) -> None:
        """Initialize a tiny module with one linear layer."""
        super().__init__()
        self.factor = 3
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer and apply the fixed scaling factor."""
        return self.linear(x) * self.factor


class _CallableObject:
    """Per-instance callable used by overlap-tracking tests."""

    def __call__(self, value: int) -> int:
        """Return the input unchanged."""
        return value


class _PassthroughWrapper(ActivationWrapper):
    """Concrete activation wrapper used to exercise base-class behavior."""

    def forward(self, *args, **kwargs):
        """Forward directly to the wrapped module."""
        wrapped_module = cast(Callable[..., Any], self._wrapped_module)
        return wrapped_module(*args, **kwargs)


class TestWrapperHelpers(unittest.TestCase):
    """Unit tests for wrapper validation and overlap-tracking helpers."""

    def test_raise_if_compiling_allows_eager_execution(self):
        """Eager execution should not reject activation wrappers."""
        with patch.object(torch.compiler, "is_compiling", return_value=False):
            self.assertIsNone(wrapper_module._raise_if_compiling("feature"))

    def test_raise_if_compiling_rejects_feature(self):
        """Compile capture should report the unsupported wrapper feature."""
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            with self.assertRaisesRegex(ValueError, "HyperParallel feature is not supported"):
                wrapper_module._raise_if_compiling("feature")

    def test_callable_exemption_recognizes_shared_function_types(self):
        """Functions, builtins, and bound methods should be exempt from object marking."""
        def _function(value):
            return value

        cases = [_function, len, self.test_callable_exemption_recognizes_shared_function_types]

        for callable_obj in cases:
            with self.subTest(callable_type=type(callable_obj)):
                self.assertTrue(wrapper_module._is_callable_exempt_from_overlap_check(callable_obj))
        self.assertFalse(wrapper_module._is_callable_exempt_from_overlap_check(_CallableObject()))

    def test_iter_wrappable_callable_attrs_only_yields_instance_callables(self):
        """Attribute discovery should ignore private, module, and shared function attributes."""
        module = torch.nn.Module()
        callable_obj = _CallableObject()
        module.public_callable = callable_obj
        module._private_callable = _CallableObject()
        module.child = torch.nn.Identity()
        module.shared_function = len

        attributes = list(wrapper_module._iter_wrappable_callable_attrs(module))

        self.assertEqual(attributes, [("public_callable", callable_obj)])

    def test_mark_wrapped_handles_mutable_and_immutable_objects(self):
        """Object marking should set supported attributes and ignore immutable objects."""
        callable_obj = _CallableObject()

        wrapper_module._mark_wrapped(callable_obj)
        self.assertTrue(callable_obj._is_wrapped)
        self.assertIsNone(wrapper_module._mark_wrapped(object()))

    def test_get_wrapped_callable_unwraps_function_modules(self):
        """Callable lookup should support direct and nested ``FuncModule`` objects."""
        function = lambda value: value  # pylint: disable=C3001
        function_module = FuncModule(function)
        wrapper = _PassthroughWrapper(function)

        self.assertIs(wrapper_module._get_wrapped_callable(function_module), function)
        self.assertIs(wrapper_module._get_wrapped_callable(wrapper), function)
        self.assertIsNone(wrapper_module._get_wrapped_callable(torch.nn.Identity()))

    def test_check_and_mark_callable_warns_on_second_wrap(self):
        """Per-instance callables should be marked and warn on overlapping wraps."""
        callable_obj = _CallableObject()

        wrapper_module._check_and_mark_callable(callable_obj)
        with self.assertWarnsRegex(UserWarning, "already wrapped"):
            wrapper_module._check_and_mark_callable(callable_obj)

        self.assertTrue(callable_obj._is_wrapped)

    def test_wrapping_parent_warns_for_wrapped_parameterized_child(self):
        """A wrapped parameterized child should be detected when wrapping its parent."""
        child = swap_wrapper(torch.nn.Linear(2, 2))
        parent = torch.nn.Sequential(child)

        with self.assertWarnsRegex(UserWarning, "Submodule.*already wrapped"):
            swap_wrapper(parent)

    def test_shared_parameterless_child_does_not_warn(self):
        """A marked parameterless child shared by another module should remain valid."""
        child = torch.nn.ReLU()
        swap_wrapper(torch.nn.Sequential(child))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            swap_wrapper(torch.nn.Sequential(child))

        self.assertEqual(caught, [])

    def test_shared_callable_attribute_warns_on_second_owner(self):
        """A per-instance callable reused by two modules should be detected as overlap."""
        callable_obj = _CallableObject()
        first = torch.nn.Module()
        second = torch.nn.Module()
        first.helper = callable_obj
        second.helper = callable_obj
        swap_wrapper(first)

        with self.assertWarnsRegex(UserWarning, "already wrapped"):
            swap_wrapper(second)


class TestActivationWrapper(unittest.TestCase):
    """Unit tests for the common activation-wrapper behavior."""

    def test_plain_callable_is_adapted_to_func_module(self):
        """A plain callable should be registered as a child ``FuncModule``."""
        function = lambda value, offset=0: value + offset  # pylint: disable=C3001

        wrapper = _PassthroughWrapper(function)

        self.assertIsInstance(wrapper._wrapped_module, FuncModule)
        self.assertEqual(wrapper(2, offset=3), 5)

    def test_getattr_and_getitem_delegate_to_wrapped_module(self):
        """Missing attributes and indexing should delegate to the wrapped module."""
        module = torch.nn.Sequential(torch.nn.Identity(), torch.nn.ReLU())
        module.label = "wrapped"
        wrapper = _PassthroughWrapper(module)

        self.assertEqual(wrapper.label, "wrapped")
        self.assertIs(wrapper[1], module[1])

    def test_named_modules_hides_internal_wrapper_prefix(self):
        """Named-module traversal should expose the wrapped module at the wrapper root."""
        module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        wrapper = _PassthroughWrapper(module)

        named_modules = list(wrapper.named_modules())

        self.assertEqual([name for name, _ in named_modules], ["", "", "0", "1"])
        self.assertIs(named_modules[0][1], wrapper)
        self.assertIs(named_modules[1][1], module)

    def test_state_dict_round_trip_uses_unwrapped_keys(self):
        """State dict save/load should interoperate with an unwrapped module."""
        source = torch.nn.Linear(2, 2)
        destination = torch.nn.Linear(2, 2)
        wrapper = _PassthroughWrapper(destination)

        load_result = wrapper.load_state_dict(source.state_dict())

        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        torch.testing.assert_close(wrapper.weight, source.weight)
        torch.testing.assert_close(wrapper.bias, source.bias)
        self.assertEqual(set(wrapper.state_dict()), {"weight", "bias"})

    def test_abstract_forward_raises_when_called_directly(self):
        """The base forward contract should fail clearly when bypassing abstraction."""
        with self.assertRaisesRegex(ValueError, "Subclasses should implement forward"):
            ActivationWrapper.forward(sentinel.wrapper)


class TestBaseCheckFn(unittest.TestCase):
    """Unit tests for base_check_fn()."""

    def test_returns_true_for_regular_tensor(self):
        """Regular tensors should satisfy the default swap predicate."""
        self.assertTrue(base_check_fn(torch.ones(2, 2)))

    def test_returns_false_for_parameter_and_parameter_view(self):
        """Parameters and their views should not be treated as activations."""
        param = torch.nn.Parameter(torch.ones(4))

        self.assertFalse(base_check_fn(param))
        self.assertFalse(base_check_fn(param[:2]))

    def test_returns_false_for_empty_storage_tensor(self):
        """Empty tensors should be filtered out by the swap predicate."""
        self.assertFalse(base_check_fn(torch.empty(0)))


class TestSwapWrapper(unittest.TestCase):
    """Unit tests for SwapWrapper and swap_wrapper()."""

    def test_swap_wrapper_returns_swap_wrapper(self):
        """swap_wrapper() should return a configured SwapWrapper instance."""
        mod = _TinyModule()

        result = swap_wrapper(mod, group_swap=True)

        self.assertIsInstance(result, SwapWrapper)
        self.assertTrue(result.group_swap)
        self.assertIs(result._wrapped_module, mod)

    def test_forward_runs_under_async_save_context(self):
        """Wrapper forward should run inside the async-save context manager."""
        mod = _TinyModule()
        wrapper = swap_wrapper(mod, policy_fn=lambda tensor: CheckpointPolicy.MUST_SAVE, group_swap=True)
        x = torch.randn(2, 2)

        with patch.object(wrapper_module, "AsyncSaveOnCpu", return_value=contextlib.nullcontext()) as mock_ctx:
            result = wrapper(x)

        self.assertEqual(result.shape, (2, 2))
        mock_ctx.assert_called_once_with(policy_fn=wrapper.policy_fn, group_swap=True, cpu_pool=None)

    def test_wraps_callable_in_func_module(self):
        """Plain callables should be adapted into FuncModule instances."""
        fn = lambda x: x + 1  # pylint: disable=C3001

        wrapper = swap_wrapper(fn)

        self.assertIsInstance(wrapper._wrapped_module, FuncModule)
        self.assertEqual(wrapper(torch.tensor(2)).item(), 3)

    def test_rejects_overlapping_wrap(self):
        """Wrapping the same module twice should warn."""
        mod = _TinyModule()
        swap_wrapper(mod)

        with self.assertWarnsRegex(UserWarning, "already wrapped"):
            swap_wrapper(mod)

    def test_forwards_attributes_and_strips_state_dict_prefix(self):
        """Wrapper metadata should mirror the wrapped module cleanly."""
        mod = _TinyModule()
        wrapper = swap_wrapper(mod)

        self.assertEqual(wrapper.factor, 3)
        self.assertTrue(all(not name.startswith("_swap_wrapped_module.") for name, _ in wrapper.named_parameters()))
        self.assertTrue(all(not key.startswith("_swap_wrapped_module.") for key in wrapper.state_dict()))

    def test_parent_named_parameters_strips_wrapped_module_prefix(self):
        """Parent module traversal should see the same parameter keys as state_dict."""
        parent = torch.nn.Module()
        parent.layer = swap_wrapper(torch.nn.Linear(2, 2))

        parameter_names = [name for name, _ in parent.named_parameters()]

        self.assertIn("layer.weight", parameter_names)
        self.assertIn("layer.bias", parameter_names)
        self.assertTrue(all("_swap_wrapped_module" not in name for name in parameter_names))


class TestAsyncSaveOnCpu(unittest.TestCase):
    """Unit tests for AsyncSaveOnCpu."""

    def test_packed_tensor_does_not_retain_original_after_storage_clear(self):
        """Clearing swap storage should release the original while keeping packed data valid."""
        expected = torch.tensor([1.0, 2.0])
        original = expected.clone()
        original_ref = weakref.ref(original)
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = "group0"
        fake_manager.is_last_group.return_value = False

        with patch.object(wrapper_module, "SwapManager", return_value=fake_manager):
            saved_tensors = AsyncSaveOnCpu(group_swap=True)
            packed = saved_tensors.pack_hook(original)

        self.assertIsNot(packed, original)
        self.assertIs(saved_tensors.storage[0][0].val, packed)
        del original

        unpacked = saved_tensors.unpack_hook(packed)
        gc.collect()

        self.assertIsNone(saved_tensors.storage)
        self.assertIsNone(original_ref())
        self.assertIs(unpacked, packed)
        self.assertTrue(torch.equal(unpacked, expected))

    def test_invalid_policy_raises_when_tensor_is_saved(self):
        """Saving tensors under an invalid policy should raise immediately."""
        x = torch.randn(2, requires_grad=True)

        with self.assertRaisesRegex(RuntimeError, "invalid policy"):
            with AsyncSaveOnCpu(policy_fn=lambda tensor: CheckpointPolicy.PREFER_SAVE):
                (x * x).sum()

    def test_adds_storage_once_for_registered_group(self):
        """MUST_SWAP policies should register swap storage once per group."""
        x = torch.randn(2, requires_grad=True)
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = "group0"
        fake_manager.is_last_group.return_value = False

        with patch.object(wrapper_module, "SwapManager", return_value=fake_manager):
            with AsyncSaveOnCpu(policy_fn=lambda tensor: CheckpointPolicy.MUST_SWAP, group_swap=True):
                (x * x).sum()

        fake_manager.add_storage.assert_called_once()

    def test_skips_storage_registration_for_last_group(self):
        """Last groups should keep saved tensors on device without registering swap storage."""
        tensor = torch.randn(2, requires_grad=True)
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = "group0"
        fake_manager.is_last_group.return_value = True

        with patch.object(wrapper_module, "SwapManager", return_value=fake_manager):
            saved_tensors = AsyncSaveOnCpu(group_swap=True)
            packed = saved_tensors.pack_hook(tensor)

        self.assertIsNot(packed, tensor)
        self.assertFalse(packed.requires_grad)
        self.assertTrue(torch.equal(packed, tensor))
        fake_manager.is_last_group.assert_called_once_with("group0")
        fake_manager.add_storage.assert_not_called()


class TestSwapTensorWrapper(unittest.TestCase):
    """Unit tests for swap_tensor_wrapper()."""

    def test_warns_and_returns_target_when_group_unregistered(self):
        """Missing swap groups should warn and leave tensors unchanged."""
        target = torch.ones(2)
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = None

        with patch.object(wrapper_module, "SwapManager", return_value=fake_manager):
            with self.assertWarnsRegex(UserWarning, "cannot be swapped"):
                result = swap_tensor_wrapper(target, tag="hidden")

        self.assertIs(result, target)
        fake_manager.add_storage.assert_not_called()

    def test_returns_target_when_current_group_is_last_group(self):
        """Last groups should bypass swap registration for their tensors."""
        target = torch.ones(2)
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = "group0"
        fake_manager.is_last_group.return_value = True

        with patch.object(wrapper_module, "SwapManager", return_value=fake_manager):
            result = swap_tensor_wrapper(target)

        self.assertIs(result, target)
        fake_manager.add_storage.assert_not_called()

    def test_registers_nested_tensors_into_storage(self):
        """Nested tensor structures should register one swap storage entry."""
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = "group0"
        fake_manager.is_last_group.return_value = False
        target = {"x": torch.ones(2), "meta": [1, torch.ones(1)]}

        with patch.object(wrapper_module, "SwapManager", return_value=fake_manager):
            result = swap_tensor_wrapper(target, tag="hidden", group_swap=True)

        self.assertIs(result["x"], target["x"])
        self.assertIs(result["meta"][1], target["meta"][1])
        fake_manager.add_storage.assert_called_once()


if __name__ == "__main__":
    unittest.main()
