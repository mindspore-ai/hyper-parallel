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
"""Unit tests for MindSpore activation checkpoint wrapper."""
import os
import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (
    ensure_mindspore_platform_default,
)

ensure_mindspore_platform_default()

import mindspore as ms
from mindspore import nn

from hyper_parallel.core.activation_checkpoint.activation_checkpoint import CheckpointPolicy
from hyper_parallel.core.activation_checkpoint.swap import SwapManager
from hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
    ckpt_wrapper,
)


class _BaseWrapperCell(nn.Cell):
    """Minimal MindSpore cell used as wrapped module in CheckpointWrapper tests."""

    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def construct(self, x):
        return x * self.factor


@patch("hyper_parallel.core.activation_checkpoint.activation_checkpoint.plat")
class TestCheckpointWrapper(unittest.TestCase):
    """Unit tests for MindSpore CheckpointWrapper."""

    def test_init_stores_attributes(self, mock_plat):
        """Test __init__ stores wrapped module and checkpoint kwargs."""
        mod = _BaseWrapperCell()
        wrapper = CheckpointWrapper(mod, group_swap=True, policy_fn=lambda x: x)

        self.assertIsNotNone(wrapper._wrapped_module)
        self.assertTrue(wrapper.checkpoint_kwargs["group_swap"])
        self.assertIn("policy_fn", wrapper.checkpoint_kwargs)

    def test_init_default_checkpoint_kwargs_empty(self, mock_plat):
        """Test checkpoint kwargs default to empty."""
        mod = _BaseWrapperCell()
        wrapper = CheckpointWrapper(mod)

        self.assertEqual(wrapper.checkpoint_kwargs, {})

    def test_construct_returns_result(self, mock_plat):
        """Test construct() calls checkpoint and returns result."""
        mock_plat.checkpoint.return_value = "construct_result"
        mock_plat.noop_context_fn = "noop"

        mod = _BaseWrapperCell()
        wrapper = CheckpointWrapper(mod)

        result = wrapper.construct(ms.Tensor([1.0, 2.0], ms.float32))

        self.assertEqual(result, "construct_result")

    def test_do_checkpoint_with_policy_and_group_swap_uses_selective_context(self, mock_plat):
        """Test _do_checkpoint with policy_fn and group_swap=True uses selective checkpoint context."""
        mock_plat.checkpoint.return_value = "result"
        policy = lambda x: CheckpointPolicy.MUST_SWAP  # pylint: disable=C3001

        mod = _BaseWrapperCell()
        wrapper = CheckpointWrapper(mod, policy_fn=policy, group_swap=True)

        result = wrapper.construct(ms.Tensor([1.0, 2.0], ms.float32))

        self.assertEqual(result, "result")
        call_kwargs = mock_plat.checkpoint.call_args[1]
        ctx_fn = call_kwargs.get("context_fn")
        self.assertTrue(callable(ctx_fn))
        ctx_fn()
        mock_plat.create_selective_checkpoint_contexts.assert_called_once_with(
            policy, group_swap=True)

    def test_do_checkpoint_with_swap_group_name(self, mock_plat):
        """Test _do_checkpoint with _swap_group_name attribute sets current group."""
        mock_plat.checkpoint.return_value = "result"
        mock_plat.noop_context_fn = "noop"

        mod = _BaseWrapperCell()
        wrapper = CheckpointWrapper(mod)
        wrapper._swap_group_name = "test_group"

        with patch.object(SwapManager, "set_current_group_name") as mock_set_group:
            result = wrapper.construct(ms.Tensor([1.0, 2.0], ms.float32))

            self.assertEqual(result, "result")
            mock_set_group.assert_called_once_with("test_group")

    def test_do_checkpoint_without_swap_group_name(self, mock_plat):
        """Test _do_checkpoint without _swap_group_name does not call SwapManager."""
        mock_plat.checkpoint.return_value = "result"
        mock_plat.noop_context_fn = "noop"

        mod = _BaseWrapperCell()
        wrapper = CheckpointWrapper(mod)

        result = wrapper.construct(ms.Tensor([1.0, 2.0], ms.float32))

        self.assertEqual(result, "result")

    def test_do_checkpoint_passes_checkpoint_kwargs(self, mock_plat):
        """Test _do_checkpoint forwards stored checkpoint_kwargs to context_fn builder."""
        mock_plat.checkpoint.return_value = "result"

        mod = _BaseWrapperCell()
        policy = lambda x: CheckpointPolicy.MUST_SAVE  # pylint: disable=C3001
        wrapper = CheckpointWrapper(mod, policy_fn=policy)

        result = wrapper.construct(ms.Tensor([1.0, 2.0], ms.float32))

        self.assertEqual(result, "result")
        call_kwargs = mock_plat.checkpoint.call_args[1]
        ctx_fn = call_kwargs.get("context_fn")
        self.assertTrue(callable(ctx_fn))
        ctx_fn()
        mock_plat.create_selective_checkpoint_contexts.assert_called_once_with(
            policy, group_swap=False)


class TestCkptWrapper(unittest.TestCase):
    """Unit tests for MindSpore ckpt_wrapper() factory function."""

    def test_returns_checkpoint_wrapper(self):
        """Test ckpt_wrapper returns a CheckpointWrapper instance."""
        mod = _BaseWrapperCell()

        result = ckpt_wrapper(mod)

        self.assertIsInstance(result, CheckpointWrapper)

    def test_passes_group_swap(self):
        """Test ckpt_wrapper passes group_swap to CheckpointWrapper."""
        mod = _BaseWrapperCell()

        result = ckpt_wrapper(mod, group_swap=True)

        self.assertTrue(result.checkpoint_kwargs["group_swap"])

    def test_passes_checkpoint_kwargs(self):
        """Test ckpt_wrapper passes kwargs to CheckpointWrapper."""
        mod = _BaseWrapperCell()
        policy = lambda x: x  # pylint: disable=C3001

        result = ckpt_wrapper(mod, policy_fn=policy)

        self.assertIn("policy_fn", result.checkpoint_kwargs)
        self.assertEqual(result.checkpoint_kwargs["policy_fn"], policy)

    def test_with_callable(self):
        """Test ckpt_wrapper works with callable."""
        fn = lambda x: x * 2  # pylint: disable=C3001

        result = ckpt_wrapper(fn)

        self.assertIsInstance(result, CheckpointWrapper)


if __name__ == "__main__":
    unittest.main()
