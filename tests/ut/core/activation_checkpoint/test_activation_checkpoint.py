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
"""Unit tests for activation checkpoint module."""
import contextlib
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.activation_checkpoint.activation_checkpoint import (
    CheckpointPolicy,
    checkpoint,
    checkpoint_wrapper,
    swap,
)
from hyper_parallel.platform.torch.activation_checkpoint.checkpoint_wrapper import CheckpointWrapper


class TestCheckpointPolicy(unittest.TestCase):
    """Unit tests for CheckpointPolicy enum."""

    def test_enum_values(self):
        """Test all enum values are correct."""
        self.assertEqual(CheckpointPolicy.MUST_SAVE.value, 0)
        self.assertEqual(CheckpointPolicy.PREFER_SAVE.value, 1)
        self.assertEqual(CheckpointPolicy.MUST_RECOMPUTE.value, 2)
        self.assertEqual(CheckpointPolicy.PREFER_RECOMPUTE.value, 3)
        self.assertEqual(CheckpointPolicy.MUST_SWAP.value, 4)

    def test_enum_membership(self):
        """Test enum member type checking."""
        self.assertIsInstance(CheckpointPolicy.MUST_SAVE, CheckpointPolicy)
        self.assertIsInstance(CheckpointPolicy.MUST_SWAP, CheckpointPolicy)

    def test_enum_str(self):
        """Test string representation of enum members."""
        self.assertEqual(str(CheckpointPolicy.MUST_SWAP), "CheckpointPolicy.MUST_SWAP")


@patch("hyper_parallel.core.activation_checkpoint.activation_checkpoint.plat")
class TestCheckpointFunction(unittest.TestCase):
    """Unit tests for checkpoint() function."""

    def test_checkpoint_no_swap_no_policy(self, mock_plat):
        """Test checkpoint without swap_inputs and without policy_fn."""
        mock_plat.checkpoint.return_value = "checkpoint_result"
        mock_plat.noop_context_fn = "noop_ctx_fn"

        def dummy_fn(x):
            return x * 2

        result = checkpoint(dummy_fn, 3)

        self.assertEqual(result, "checkpoint_result")
        mock_plat.checkpoint.assert_called_once()

        call_args, call_kwargs = mock_plat.checkpoint.call_args
        self.assertIs(call_args[0], dummy_fn)
        self.assertEqual(call_args[1], 3)
        self.assertEqual(call_kwargs.get("use_reentrant"), False)
        self.assertEqual(call_kwargs.get("context_fn"), "noop_ctx_fn")

    def test_checkpoint_with_swap_inputs(self, mock_plat):
        """Test checkpoint with swap_inputs=True."""
        mock_plat.checkpoint.return_value = "result"
        mock_plat.noop_context_fn = "noop_ctx_fn"
        mock_plat.async_save_on_cpu = MagicMock()

        def dummy_fn(x):
            return x * 2

        result = checkpoint(dummy_fn, 3, swap_inputs=True)

        self.assertEqual(result, "result")
        mock_plat.async_save_on_cpu.assert_called_once()

    def test_checkpoint_with_policy_fn(self, mock_plat):
        """Test checkpoint with a policy function."""
        mock_plat.checkpoint.return_value = "result"
        mock_plat.noop_context_fn = "noop"

        policy = lambda op, idx: CheckpointPolicy.MUST_SAVE  # pylint: disable=C3001

        def dummy_fn(x):
            return x * 2

        result = checkpoint(dummy_fn, 3, policy_fn=policy)

        self.assertEqual(result, "result")
        call_kwargs = mock_plat.checkpoint.call_args[1]
        self.assertIsNotNone(call_kwargs.get("context_fn"))

    def test_checkpoint_with_group_swap(self, mock_plat):
        """Test checkpoint with group_swap=True."""
        mock_plat.checkpoint.return_value = "result"
        mock_plat.noop_context_fn = "noop"

        def dummy_fn(x):
            return x * 2

        result = checkpoint(dummy_fn, 3, group_swap=True)

        self.assertEqual(result, "result")
        call_kwargs = mock_plat.checkpoint.call_args[1]
        self.assertIsNotNone(call_kwargs.get("context_fn"))

    def test_checkpoint_with_policy_and_group_swap_no_duplicate(self, mock_plat):
        """Test checkpoint with both policy_fn and group_swap uses create_selective_checkpoint_contexts once."""
        mock_plat.checkpoint.return_value = "result"

        policy = lambda op, idx: CheckpointPolicy.MUST_SAVE  # pylint: disable=C3001

        def dummy_fn(x):
            return x * 2

        result = checkpoint(dummy_fn, 3, policy_fn=policy, group_swap=True)

        self.assertEqual(result, "result")
        call_kwargs = mock_plat.checkpoint.call_args[1]
        self.assertIsNotNone(call_kwargs.get("context_fn"))

    def test_checkpoint_with_kwargs(self, mock_plat):
        """Test checkpoint passes kwargs to underlying function."""
        mock_plat.checkpoint.return_value = "result"
        mock_plat.noop_context_fn = "noop"

        def dummy_fn(x, scale=1.0):
            return x * scale

        result = checkpoint(dummy_fn, 3, scale=2.0)

        self.assertEqual(result, "result")
        call_args = mock_plat.checkpoint.call_args[0]
        self.assertIn(3, call_args)


@patch("hyper_parallel.core.activation_checkpoint.activation_checkpoint.plat")
class TestSwapFunction(unittest.TestCase):
    """Unit tests for swap() function."""

    def test_swap_no_policy(self, mock_plat):
        """Test swap passes through function result without policy."""
        mock_plat.async_save_on_cpu.return_value = contextlib.nullcontext()

        def dummy_fn(x):
            return x * 2

        result = swap(dummy_fn, 3)

        self.assertEqual(result, 6)
        mock_plat.async_save_on_cpu.assert_called_once_with(policy_fn=None, group_swap=False)

    def test_swap_with_policy_fn(self, mock_plat):
        """Test swap passes policy_fn to async_save_on_cpu."""
        mock_plat.async_save_on_cpu.return_value = contextlib.nullcontext()

        policy = lambda t: CheckpointPolicy.MUST_SAVE  # pylint: disable=C3001

        def dummy_fn(x):
            return x * 2

        result = swap(dummy_fn, 3, policy_fn=policy)

        self.assertEqual(result, 6)
        mock_plat.async_save_on_cpu.assert_called_once_with(policy_fn=policy, group_swap=False)

    def test_swap_with_kwargs(self, mock_plat):
        """Test swap forwards kwargs to the function."""
        mock_plat.async_save_on_cpu.return_value = contextlib.nullcontext()

        def dummy_fn(x, scale=1.0):
            return x * scale

        result = swap(dummy_fn, 3, scale=2.0)

        self.assertEqual(result, 6.0)
        mock_plat.async_save_on_cpu.assert_called_once_with(policy_fn=None, group_swap=False)

    def test_swap_with_args_and_kwargs(self, mock_plat):
        """Test swap with args and kwargs."""
        mock_plat.async_save_on_cpu.return_value = contextlib.nullcontext()

        def dummy_fn(a, b, c=1):
            return a + b + c

        result = swap(dummy_fn, 1, 2, c=3)

        self.assertEqual(result, 6)
        mock_plat.async_save_on_cpu.assert_called_once_with(policy_fn=None, group_swap=False)


class _BaseWrapperModule(torch.nn.Module):
    """Minimal torch module used in checkpoint_wrapper alias tests."""

    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x) * self.factor


@patch("hyper_parallel.core.activation_checkpoint.activation_checkpoint.plat")
class TestCkptWrapper(unittest.TestCase):
    """Unit tests for checkpoint_wrapper() factory function."""

    def test_returns_checkpoint_wrapper(self, mock_plat):
        """Test checkpoint_wrapper returns a CheckpointWrapper instance."""
        mod = _BaseWrapperModule()

        result = checkpoint_wrapper(mod)

        self.assertIsInstance(result, CheckpointWrapper)

    def test_passes_group_swap(self, mock_plat):
        """Test checkpoint_wrapper passes group_swap to CheckpointWrapper."""
        mod = _BaseWrapperModule()

        result = checkpoint_wrapper(mod, group_swap=True)

        self.assertTrue(result.checkpoint_kwargs["group_swap"])

    def test_passes_checkpoint_kwargs(self, mock_plat):
        """Test checkpoint_wrapper passes kwargs to CheckpointWrapper."""
        mod = _BaseWrapperModule()
        policy = lambda x: x  # pylint: disable=C3001

        result = checkpoint_wrapper(mod, policy_fn=policy)

        self.assertIn("policy_fn", result.checkpoint_kwargs)
        self.assertEqual(result.checkpoint_kwargs["policy_fn"], policy)

    def test_with_callable(self, mock_plat):
        """Test checkpoint_wrapper works with callable (lambda)."""
        fn = lambda x: x * 2  # pylint: disable=C3001

        result = checkpoint_wrapper(fn)

        self.assertIsInstance(result, CheckpointWrapper)


@patch("hyper_parallel.core.activation_checkpoint.activation_checkpoint.plat")
class TestModuleLevelAliases(unittest.TestCase):
    """Unit tests for module-level aliases (swap_wrapper, swap_tensor_wrapper, checkpoint_wrapper)."""

    def test_swap_wrapper_is_callable(self, mock_plat):
        """Test swap_wrapper is a callable."""
        from hyper_parallel.core.activation_checkpoint import swap_wrapper

        self.assertTrue(callable(swap_wrapper))

    def test_swap_tensor_wrapper_is_callable(self, mock_plat):
        """Test swap_tensor_wrapper is a callable."""
        from hyper_parallel.core.activation_checkpoint import swap_tensor_wrapper

        self.assertTrue(callable(swap_tensor_wrapper))


if __name__ == "__main__":
    unittest.main()
