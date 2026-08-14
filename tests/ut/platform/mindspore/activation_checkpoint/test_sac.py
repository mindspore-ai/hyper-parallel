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
"""Unit tests for MindSpore selective activation checkpoint contexts."""
from collections import defaultdict
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from tests.ut.platform.mindspore._ensure_mindspore_platform import (
    ensure_mindspore_platform_default,
)

enable_mindspore_backward_compat()
ensure_mindspore_platform_default()

import mindspore as ms

from hyper_parallel.core.activation_checkpoint import CheckpointPolicy
from hyper_parallel.core.activation_checkpoint.swap import Storage
from hyper_parallel.platform.mindspore.activation_checkpoint import sac
from hyper_parallel.platform.mindspore.activation_checkpoint.sac import (
    SelectiveCheckpointContext,
    _CachedMindSporeDispatchMode,
    _CachingMindSporeDispatchMode,
    _SwapCacheEntry,
    _VersionWrapper,
    _maybe_detach,
    create_selective_checkpoint_contexts,
    ignore_sac_ops,
)


class _FakeOp:
    """Minimal callable op carrying the MindSpore dispatch ``name`` attribute."""

    def __init__(self, name, fn):
        """Store the fake op name and callable implementation."""
        self.name = name
        self._fn = fn

    def __call__(self, *args, **kwargs):
        """Invoke the wrapped fake operator implementation."""
        return self._fn(*args, **kwargs)


def _tensor(values):
    """Build a float32 MindSpore tensor from Python values."""
    return ms.Tensor(np.array(values, np.float32))


class TestSacHelpers(unittest.TestCase):
    """Unit tests for small SAC helper objects."""

    def test_context_stores_recompute_flag(self):
        """SelectiveCheckpointContext should preserve the recompute flag."""
        self.assertFalse(SelectiveCheckpointContext(is_recompute=False).is_recompute)
        self.assertTrue(SelectiveCheckpointContext(is_recompute=True).is_recompute)

    def test_version_wrapper_returns_value_for_non_tensor(self):
        """Non-tensor cached values should be returned directly."""
        wrapper = _VersionWrapper("cached")

        self.assertEqual(wrapper.get_val(allow_cache_entry_mutation=False), "cached")

    def test_maybe_detach_returns_tensor(self):
        """_maybe_detach() should still return a MindSpore tensor object."""
        source = _tensor([1.0, 2.0])

        result = _maybe_detach(source)

        self.assertIsInstance(result, ms.Tensor)

    def test_swap_cache_entry_uses_cached_tensor_for_save_and_swap(self):
        """Swap cache entries should wrap the same cached value for save and swap."""
        source = _tensor([1.0, 2.0])
        cached = _maybe_detach(source)

        entry = _SwapCacheEntry(cached, "Add", group_swap=True)

        self.assertIs(entry.save.val, cached)
        self.assertIs(entry.swap.val, cached)
        self.assertTrue(entry.swap.group_swap)

    def test_ignore_sac_ops_adds_available_operator_names(self):
        """Runtime operator names should be added while unavailable entries are omitted."""
        original_ignored_ops = set(sac.SAC_IGNORED_OPS)
        self.addCleanup(sac.SAC_IGNORED_OPS.update, original_ignored_ops)
        self.addCleanup(sac.SAC_IGNORED_OPS.intersection_update, original_ignored_ops)

        ignore_sac_ops(["AllGather", None])

        self.assertIn("AllGather", sac.SAC_IGNORED_OPS)
        self.assertNotIn(None, sac.SAC_IGNORED_OPS)


class TestCreateSelectiveCheckpointContexts(unittest.TestCase):
    """Unit tests for create_selective_checkpoint_contexts()."""

    def test_none_policy_defaults_to_prefer_recompute(self):
        """A missing policy should default caching mode to recomputation."""
        caching, cached = create_selective_checkpoint_contexts(None)

        self.assertIsInstance(caching, _CachingMindSporeDispatchMode)
        self.assertIsInstance(cached, _CachedMindSporeDispatchMode)
        self.assertEqual(caching.policy_fn(None, object()), CheckpointPolicy.PREFER_RECOMPUTE)

    def test_callable_policy_is_used(self):
        """Callable policies should be passed through to the caching mode."""
        policy_fn = lambda ctx, op, *args, **kwargs: CheckpointPolicy.MUST_SAVE  # pylint: disable=C3001

        caching, _ = create_selective_checkpoint_contexts(policy_fn, group_swap=True)

        self.assertIs(caching.policy_fn, policy_fn)
        self.assertTrue(caching.group_swap)

    def test_invalid_policy_type_raises(self):
        """Unsupported policy inputs should raise a TypeError."""
        with self.assertRaisesRegex(TypeError, "policy_fn_or_list"):
            create_selective_checkpoint_contexts([object()])


class TestMindSporeDispatchModes(unittest.TestCase):
    """Unit tests for the caching and cached dispatch modes."""

    def test_ignored_op_bypasses_policy(self):
        """Ignored ops should skip policy evaluation and execute directly."""
        policy_fn = MagicMock(return_value=CheckpointPolicy.MUST_SAVE)
        caching, _ = create_selective_checkpoint_contexts(policy_fn)
        op = _FakeOp("StopGradient", lambda x: x + 1)
        x = _tensor([1.0])

        result = caching.__ms_dispatch__(op, args=(x,))

        self.assertTrue(np.allclose(result.asnumpy(), np.array([2.0], np.float32)))
        policy_fn.assert_not_called()

    def test_must_save_cache_is_restored_during_recompute(self):
        """Saved activations should be replayed from cache during recompute."""
        policy_fn = lambda ctx, op, *args, **kwargs: CheckpointPolicy.MUST_SAVE  # pylint: disable=C3001
        caching, cached = create_selective_checkpoint_contexts(policy_fn)
        op = _FakeOp("Add", lambda x, y: x + y)
        x = _tensor([1.0, 2.0])
        y = _tensor([3.0, 4.0])

        expected = caching.__ms_dispatch__(op, args=(x, y))
        restored = cached.__ms_dispatch__(op, args=(_tensor([0.0, 0.0]), _tensor([0.0, 0.0])))

        self.assertTrue(np.allclose(restored.asnumpy(), expected.asnumpy()))

    def test_prefer_recompute_calls_op_during_recompute(self):
        """Recompute-preferred policies should rerun the op in cached mode."""
        policy_fn = lambda ctx, op, *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE  # pylint: disable=C3001
        _, cached = create_selective_checkpoint_contexts(policy_fn)
        op = _FakeOp("Add", lambda x, y: x + y)

        result = cached.__ms_dispatch__(op, args=(_tensor([1.0]), _tensor([2.0])))

        self.assertTrue(np.allclose(result.asnumpy(), np.array([3.0], np.float32)))

    def test_invalid_policy_raises_in_caching_mode(self):
        """Invalid policies should raise when caching mode evaluates an op."""
        caching, _ = create_selective_checkpoint_contexts(
            lambda ctx, op, *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE
        )
        op = _FakeOp("Add", lambda x, y: x + y)

        with self.assertRaisesRegex(RuntimeError, "invalid policy"):
            caching.__ms_dispatch__(op, args=(_tensor([1.0]), _tensor([2.0])))

    def test_cached_mode_missing_storage_raises(self):
        """Cached mode should fail when no saved storage exists for an op."""
        cached = _CachedMindSporeDispatchMode(
            lambda ctx, op, *args, **kwargs: CheckpointPolicy.MUST_SAVE,
            Storage(),
            {},
            allow_cache_entry_mutation=False,
        )
        op = _FakeOp("Add", lambda x, y: x + y)

        with self.assertRaisesRegex(RuntimeError, "not found in storage"):
            cached.__ms_dispatch__(op, args=(_tensor([1.0]), _tensor([2.0])))

    def test_cached_mode_empty_storage_raises(self):
        """Cached mode should fail when the saved storage list is exhausted."""
        storage = defaultdict(list)
        storage["Add"] = []
        cached = _CachedMindSporeDispatchMode(
            lambda ctx, op, *args, **kwargs: CheckpointPolicy.MUST_SAVE,
            Storage(),
            storage,
            allow_cache_entry_mutation=False,
        )
        op = _FakeOp("Add", lambda x, y: x + y)

        with self.assertRaisesRegex(RuntimeError, "extra time"):
            cached.__ms_dispatch__(op, args=(_tensor([1.0]), _tensor([2.0])))

    def test_must_swap_registers_swap_storage(self):
        """MUST_SWAP policies should register swap storage once per dispatch."""
        fake_manager = MagicMock()
        fake_manager.get_current_group_name.return_value = "group0"
        op = _FakeOp("Add", lambda x, y: x + y)

        with patch.object(sac, "SwapManager", return_value=fake_manager):
            caching, _ = create_selective_checkpoint_contexts(
                lambda ctx, op, *args, **kwargs: CheckpointPolicy.MUST_SWAP,
                group_swap=True,
            )
            caching.__ms_dispatch__(op, args=(_tensor([1.0]), _tensor([2.0])))

        fake_manager.add_storage.assert_called_once()


if __name__ == "__main__":
    unittest.main()
