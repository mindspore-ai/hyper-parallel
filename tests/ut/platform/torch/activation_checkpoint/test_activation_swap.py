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
"""Unit tests for PyTorch activation swap platform implementation."""
import contextlib
import gc
import os
import unittest
import weakref
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.activation_checkpoint.activation_checkpoint import CheckpointPolicy
from hyper_parallel.platform.torch.activation_checkpoint import activation_swap
from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import (
    AsyncSaveOnCpu,
    FuncModule,
    SwapWrapper,
    base_check_fn,
    swap_tensor_wrapper,
    swap_wrapper,
)


class _TinyModule(torch.nn.Module):
    """Small module used by wrapper tests."""

    def __init__(self):
        """Initialize a tiny module with one linear layer."""
        super().__init__()
        self.factor = 3
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        """Run the linear layer and apply the fixed scaling factor."""
        return self.linear(x) * self.factor


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

        with patch.object(activation_swap, "AsyncSaveOnCpu", return_value=contextlib.nullcontext()) as mock_ctx:
            result = wrapper(x)

        self.assertEqual(result.shape, (2, 2))
        mock_ctx.assert_called_once_with(policy_fn=wrapper.policy_fn, group_swap=True)

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

        with patch.object(activation_swap, "SwapManager", return_value=fake_manager):
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

        with patch.object(activation_swap, "SwapManager", return_value=fake_manager):
            with AsyncSaveOnCpu(policy_fn=lambda tensor: CheckpointPolicy.MUST_SWAP, group_swap=True):
                (x * x).sum()

        fake_manager.add_storage.assert_called_once()


class TestSwapTensorWrapper(unittest.TestCase):
    """Unit tests for swap_tensor_wrapper()."""

    def test_warns_and_returns_target_when_group_unregistered(self):
        """Missing swap groups should warn and leave tensors unchanged."""
        target = torch.ones(2)
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = None

        with patch.object(activation_swap, "SwapManager", return_value=fake_manager):
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

        with patch.object(activation_swap, "SwapManager", return_value=fake_manager):
            result = swap_tensor_wrapper(target)

        self.assertIs(result, target)
        fake_manager.add_storage.assert_not_called()

    def test_registers_nested_tensors_into_storage(self):
        """Nested tensor structures should register one swap storage entry."""
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = "group0"
        fake_manager.is_last_group.return_value = False
        target = {"x": torch.ones(2), "meta": [1, torch.ones(1)]}

        with patch.object(activation_swap, "SwapManager", return_value=fake_manager):
            result = swap_tensor_wrapper(target, tag="hidden", group_swap=True)

        self.assertIs(result["x"], target["x"])
        self.assertIs(result["meta"][1], target["meta"][1])
        fake_manager.add_storage.assert_called_once()


if __name__ == "__main__":
    unittest.main()
