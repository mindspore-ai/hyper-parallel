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
"""Unit tests for PyTorch selective activation checkpoint contexts."""
from collections import defaultdict
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.activation_checkpoint import CheckpointPolicy
from hyper_parallel.core.activation_checkpoint.swap import Storage
from hyper_parallel.platform.torch.activation_checkpoint import sac
from hyper_parallel.platform.torch.activation_checkpoint.sac import (
    SelectiveCheckpointContext,
    _CachedTorchDispatchMode,
    _CachingTorchDispatchMode,
    _SwapCacheEntry,
    _VersionWrapper,
    _maybe_detach,
    _policy_from_bool,
    create_selective_checkpoint_contexts,
)


class TestSacHelpers(unittest.TestCase):
    """Unit tests for small SAC helper objects."""

    def test_policy_from_bool(self):
        """Boolean policy helpers should map to the expected checkpoint policy."""
        self.assertEqual(_policy_from_bool(True), CheckpointPolicy.MUST_SAVE)
        self.assertEqual(_policy_from_bool(False), CheckpointPolicy.PREFER_RECOMPUTE)

    def test_context_stores_recompute_flag(self):
        """SelectiveCheckpointContext should preserve the recompute flag."""
        self.assertFalse(SelectiveCheckpointContext(is_recompute=False).is_recompute)
        self.assertTrue(SelectiveCheckpointContext(is_recompute=True).is_recompute)

    def test_version_wrapper_detects_mutation(self):
        """Version wrappers should detect cached tensor mutation on replay."""
        source = torch.ones(2)
        wrapper = _VersionWrapper(source)

        source.add_(1)

        with self.assertRaisesRegex(RuntimeError, "mutated"):
            wrapper.get_val(allow_cache_entry_mutation=False)
        self.assertIs(wrapper.get_val(allow_cache_entry_mutation=True), wrapper.val)

    def test_maybe_detach_detaches_float_tensor(self):
        """_maybe_detach() should return a detached tensor copy when needed."""
        source = torch.ones(2, requires_grad=True)

        result = _maybe_detach(source, any_ret_has_alias_info=False)

        self.assertFalse(result.requires_grad)
        self.assertIsNot(result, source)

    def test_swap_cache_entry_uses_cached_tensor_for_save_and_swap(self):
        """Swap cache entries should wrap the same cached value for save and swap."""
        source = torch.ones(2, requires_grad=True)
        cached = _maybe_detach(source, any_ret_has_alias_info=False)

        entry = _SwapCacheEntry(cached, "aten.add.Tensor", group_swap=True)

        self.assertIs(entry.save.val, cached)
        self.assertIs(entry.swap.val, cached)
        self.assertTrue(entry.swap.group_swap)


class TestCreateSelectiveCheckpointContexts(unittest.TestCase):
    """Unit tests for create_selective_checkpoint_contexts()."""

    def test_none_policy_defaults_to_prefer_recompute(self):
        """A missing policy should default caching mode to recomputation."""
        caching, cached = create_selective_checkpoint_contexts(None)

        self.assertIsInstance(caching, _CachingTorchDispatchMode)
        self.assertIsInstance(cached, _CachedTorchDispatchMode)
        self.assertEqual(caching.policy_fn(None, torch.ops.aten.add.Tensor), CheckpointPolicy.PREFER_RECOMPUTE)

    def test_op_list_policy_saves_listed_ops(self):
        """Op-list policies should save listed ops and recompute others."""
        caching, _ = create_selective_checkpoint_contexts([torch.ops.aten.add.Tensor])

        self.assertEqual(caching.policy_fn(None, torch.ops.aten.add.Tensor), CheckpointPolicy.MUST_SAVE)
        self.assertEqual(caching.policy_fn(None, torch.ops.aten.mul.Tensor), CheckpointPolicy.PREFER_RECOMPUTE)

    def test_op_list_rejects_op_overload_packet(self):
        """Op-list policies should require specific OpOverload instances."""
        with self.assertRaisesRegex(ValueError, "specific OpOverload"):
            create_selective_checkpoint_contexts([torch.ops.aten.add])

    def test_invalid_policy_type_raises(self):
        """Unsupported policy inputs should raise a TypeError."""
        with self.assertRaisesRegex(TypeError, "policy_fn_or_list"):
            create_selective_checkpoint_contexts(object())


class TestTorchDispatchModes(unittest.TestCase):
    """Unit tests for the caching and cached dispatch modes."""

    def test_must_save_cache_is_restored_during_recompute(self):
        """Saved activations should be replayed from cache during recompute."""
        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=unused-argument
            """Always save the intercepted operator output."""
            return CheckpointPolicy.MUST_SAVE

        caching, cached = create_selective_checkpoint_contexts(policy_fn)
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        zero_x = torch.zeros_like(x)
        zero_y = torch.zeros_like(y)

        with caching:
            expected = torch.add(x, y)
        with cached:
            restored = torch.add(zero_x, zero_y)

        self.assertTrue(torch.equal(restored, expected))

    def test_bool_policy_is_normalized(self):
        """Boolean-returning policies should be normalized before dispatch."""
        caching, cached = create_selective_checkpoint_contexts(lambda ctx, op, *args, **kwargs: True)
        x = torch.tensor([1.0])
        zero_x = torch.zeros_like(x)

        with caching:
            expected = torch.add(x, x)
        with cached:
            restored = torch.add(zero_x, zero_x)

        self.assertTrue(torch.equal(restored, expected))

    def test_invalid_policy_raises_in_caching_mode(self):
        """Invalid policies should raise when caching mode evaluates an op."""
        caching, _ = create_selective_checkpoint_contexts(lambda ctx, op, *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE)
        x = torch.tensor([1.0])

        with self.assertRaisesRegex(RuntimeError, "invalid policy"):
            with caching:
                torch.add(x, x)

    def test_cached_mode_missing_storage_raises(self):
        """Cached mode should fail when no saved storage exists for an op."""
        cached = _CachedTorchDispatchMode(
            lambda ctx, op, *args, **kwargs: CheckpointPolicy.MUST_SAVE,
            Storage(),
            {},
            allow_cache_entry_mutation=False,
        )

        with self.assertRaisesRegex(RuntimeError, "not found in storage"):
            with cached:
                torch.add(torch.tensor([1.0]), torch.tensor([1.0]))

    def test_cached_mode_empty_storage_raises(self):
        """Cached mode should fail when the saved storage list is exhausted."""
        storage = defaultdict(list)
        storage[torch.ops.aten.add.Tensor] = []
        cached = _CachedTorchDispatchMode(
            lambda ctx, op, *args, **kwargs: CheckpointPolicy.MUST_SAVE,
            Storage(),
            storage,
            allow_cache_entry_mutation=False,
        )
        x = torch.tensor([1.0])
        y = torch.tensor([1.0])

        with self.assertRaisesRegex(RuntimeError, "extra time"):
            with cached:
                torch.add(x, y)

    def test_must_swap_registers_swap_storage(self):
        """MUST_SWAP policies should register swap storage once per dispatch."""
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = "group0"

        with patch.object(sac, "SwapManager", return_value=fake_manager):
            caching, _ = create_selective_checkpoint_contexts(
                lambda ctx, op, *args, **kwargs: CheckpointPolicy.MUST_SWAP,
                group_swap=True,
            )
            with caching:
                torch.add(torch.tensor([1.0]), torch.tensor([2.0]))

        fake_manager.add_storage.assert_called_once()


if __name__ == "__main__":
    unittest.main()
