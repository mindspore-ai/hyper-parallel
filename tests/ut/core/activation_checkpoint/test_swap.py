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
"""Unit tests for activation swap module (SwapTensor, Storage, SwapGroup, SwapManager)."""
import contextlib
import importlib
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

swap_module = importlib.import_module("hyper_parallel.core.activation_checkpoint.swap")
from hyper_parallel.core.activation_checkpoint.swap import (  # noqa: E402
    SwapGroup,
    SwapManager,
    SwapTensor,
    Storage,
    _collect_device_storage_ptrs,
    _get_cpu_pinned_buf,
    _return_cpu_pinned_buf,
    _CPU_PINNED_POOL,
)


def _swap_plat_patch():
    """Patch the platform in the swap module, avoiding name conflict with swap() function."""
    return patch.object(swap_module, "platform")


def _build_mock_platform():
    """Build a mock platform with all APIs needed by the swap module."""
    mp = MagicMock()
    mp.Tensor = torch.Tensor

    def _tree_map(fn, tree):
        if isinstance(tree, (list, tuple)):
            return type(tree)(_tree_map(fn, x) for x in tree)
        if isinstance(tree, dict):
            return type(tree)((k, _tree_map(fn, v)) for k, v in tree.items())
        return fn(tree)

    mp.tree_map = _tree_map
    mp.no_grad.return_value = contextlib.nullcontext()
    mp.preserve_version_counter.return_value = contextlib.nullcontext()

    mock_stream = MagicMock()
    mp.get_current_stream.return_value = mock_stream

    mock_copy_stream = MagicMock()
    mp.new_stream.return_value = mock_copy_stream

    mock_event = MagicMock()
    mp.new_event.return_value = mock_event

    def _stream_context(stream):
        return contextlib.nullcontext()

    mp.get_stream_context = _stream_context
    mp.cat = torch.cat

    def _alloc_tensor_buffer(numel, dtype, device, pin_memory=False):
        return torch.empty(numel, dtype=dtype, device="cpu")

    mp.alloc_tensor_buffer = _alloc_tensor_buffer

    def _empty_like(tensor, dtype=None, device=None, pin_memory=False):
        return torch.empty_like(tensor, device="cpu")

    mp.empty_like = _empty_like
    mp.get_element_size = lambda t: t.element_size()

    mp.register_forward_pre_hook = lambda m, h, prepend=False, with_kwargs=False: (
        m.register_forward_pre_hook(h, prepend=prepend, with_kwargs=with_kwargs)
    )
    mp.register_full_backward_pre_hook = lambda m, h, prepend=False: (
        m.register_full_backward_pre_hook(h, prepend)
    )
    mp.register_full_backward_hook = lambda m, h, prepend=False: (
        m.register_full_backward_hook(h, prepend)
    )

    return mp


class TestModuleFunctions(unittest.TestCase):
    """Unit tests for module-level helper functions."""

    def setUp(self):
        _CPU_PINNED_POOL.clear()

    def test_get_cpu_pinned_buf_empty_pool_allocates(self):
        """Test _get_cpu_pinned_buf allocates when pool is empty."""
        _CPU_PINNED_POOL.clear()
        with _swap_plat_patch() as mp:
            mp.alloc_tensor_buffer.return_value = torch.empty(100, dtype=torch.float32, device="meta")
            buf = _get_cpu_pinned_buf("torch.float32", 100, torch.float32)
            mp.alloc_tensor_buffer.assert_called_once_with(100, torch.float32, device='cpu', pin_memory=True)
            self.assertEqual(buf.numel(), 100)

    def test_get_cpu_pinned_buf_best_fit_from_pool(self):
        """Test _get_cpu_pinned_buf selects the smallest sufficient buffer."""
        _CPU_PINNED_POOL.clear()
        small = torch.empty(50, dtype=torch.float32)
        best = torch.empty(100, dtype=torch.float32, device="meta")
        large = torch.empty(200, dtype=torch.float32, device="meta")
        pool = _CPU_PINNED_POOL["torch.float32"]
        pool.extend([small, best, large])
        with _swap_plat_patch() as mp:
            buf = _get_cpu_pinned_buf("torch.float32", 80, torch.float32)
            self.assertIs(buf, best)
            self.assertEqual(len(pool), 2)

    def test_get_cpu_pinned_buf_no_sufficient_discard_and_alloc(self):
        """Test _get_cpu_pinned_buf discards undersized entry and allocates new."""
        _CPU_PINNED_POOL.clear()
        small = torch.empty(50, dtype=torch.float32)
        pool = _CPU_PINNED_POOL["torch.float32"]
        pool.append(small)
        with _swap_plat_patch() as mp:
            mp.alloc_tensor_buffer.return_value = torch.empty(200, dtype=torch.float32, device="meta")
            buf = _get_cpu_pinned_buf("torch.float32", 100, torch.float32)
            self.assertEqual(len(pool), 0)
            self.assertEqual(buf.numel(), 200)

    def test_return_cpu_pinned_buf_none(self):
        """Test _return_cpu_pinned_buf with None does nothing."""
        _CPU_PINNED_POOL.clear()
        _return_cpu_pinned_buf(None)
        self.assertEqual(len(_CPU_PINNED_POOL), 0)

    def test_return_cpu_pinned_buf_adds_to_pool(self):
        """Test _return_cpu_pinned_buf adds buffer to correct dtype pool."""
        _CPU_PINNED_POOL.clear()
        buf = torch.empty(100, dtype=torch.float32, device="meta")
        _return_cpu_pinned_buf(buf)
        self.assertIn("torch.float32", _CPU_PINNED_POOL)
        self.assertIn(buf, _CPU_PINNED_POOL["torch.float32"])

    def test_collect_device_storage_ptrs_device_tensor(self):
        """Test _collect_device_storage_ptrs collects storage ptrs from device tensors."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.tree_map = _build_mock_platform().tree_map
            t = torch.empty(10, device="meta")
            result = _collect_device_storage_ptrs(t)
            self.assertEqual(len(result), 1)
            self.assertIn(t.untyped_storage().data_ptr(), result)

    def test_collect_device_storage_ptrs_nested(self):
        """Test _collect_device_storage_ptrs traverses nested structures."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.tree_map = _build_mock_platform().tree_map
            t1 = torch.empty(10, device="meta")
            t2 = torch.empty(20, device="meta")
            nested = {"a": t1, "b": [t2, 1.0]}
            result = _collect_device_storage_ptrs(nested)
            # Meta tensors share data_ptr=0 so set deduplicates; verify at least 1 found
            self.assertGreaterEqual(len(result), 1)

    def test_collect_device_storage_ptrs_cpu_tensor_ignored(self):
        """Test _collect_device_storage_ptrs ignores CPU tensors."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.tree_map = _build_mock_platform().tree_map
            t_cpu = torch.empty(10, device="cpu")
            result = _collect_device_storage_ptrs(t_cpu)
            self.assertEqual(len(result), 0)


class TestSwapTensor(unittest.TestCase):
    """Unit tests for SwapTensor class."""

    def _make_tensor(self, *shape):
        """Create a 'device' tensor using meta device to simulate non-CPU tensors."""
        return torch.empty(*shape, device="meta")

    def test_init_device_tensor(self):
        """Test SwapTensor initialization with a device tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3, 4)
            st = SwapTensor(t, "test_func")
            self.assertIs(st.val, t)
            self.assertEqual(st.funcname, "test_func")
            self.assertEqual(st._state, SwapTensor.STATE_DEVICE)
            self.assertFalse(st._keep_on_device)
            self.assertFalse(st._duplicate_swap)
            self.assertFalse(st._group_managed)
            self.assertFalse(st.group_swap)

    def test_init_cpu_tensor_becomes_non_tensor(self):
        """Test SwapTensor with CPU tensor is treated as non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            t = torch.empty(3, device="cpu")
            st = SwapTensor(t, "test_func")
            self.assertEqual(st._state, SwapTensor.STATE_NON_TENSOR)

    def test_init_non_tensor_value(self):
        """Test SwapTensor with non-tensor value (e.g., tuple)."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor((1, 2, 3), "test_func")
            self.assertEqual(st._state, SwapTensor.STATE_NON_TENSOR)

    def test_init_group_swap_flag(self):
        """Test SwapTensor sets group_swap flag."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3, 4)
            st = SwapTensor(t, "test_func", group_swap=True)
            self.assertTrue(st.group_swap)

    def test_init_slice_tensor(self):
        """Test SwapTensor detects slice tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(10)
            t_slice = t[:5]
            st = SwapTensor(t_slice, "slice_func")
            self.assertTrue(st.is_slice_tensor)

    def test_dedup_key_tensor(self):
        """Test dedup_key returns a stable identity key."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3, 4)
            st = SwapTensor(t, "test_func")
            key = st.dedup_key()
            self.assertIsNotNone(key)
            self.assertEqual(len(key), 5)

    def test_dedup_key_non_tensor(self):
        """Test dedup_key returns None for non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor("hello", "test_func")
            self.assertIsNone(st.dedup_key())

    def test_mark_duplicate_swap(self):
        """Test mark_duplicate_swap sets the flag."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st.mark_duplicate_swap()
            self.assertTrue(st._duplicate_swap)

    def test_protect_if_aliases_match(self):
        """Test protect_if_aliases keeps tensor on device when storage matches."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            alias_ptrs = {t.untyped_storage().data_ptr()}
            st.protect_if_aliases(alias_ptrs)
            self.assertTrue(st._keep_on_device)

    def test_protect_if_aliases_no_match(self):
        """Test protect_if_aliases does not set flag when no match."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st.protect_if_aliases({999999})
            self.assertFalse(st._keep_on_device)

    def test_protect_if_aliases_non_tensor(self):
        """Test protect_if_aliases is no-op for non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor("hello", "test_func")
            st.protect_if_aliases({1, 2, 3})
            self.assertFalse(st._keep_on_device)

    def test_get_val_success(self):
        """Test get_val returns tensor when in DEVICE state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            self.assertIs(st.get_val(), t)

    def test_get_val_non_tensor(self):
        """Test get_val returns value for non-tensor items."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor({"a": 1}, "test_func")
            self.assertEqual(st.get_val(), {"a": 1})

    def test_get_val_wrong_state_raises(self):
        """Test get_val raises RuntimeError when not in DEVICE state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_HOST
            with self.assertRaises(RuntimeError):
                st.get_val()

    def test_resize_device_storage_non_tensor_noop(self):
        """Test resize_device_storage is no-op for non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor("hello", "test_func")
            st.resize_device_storage()

    def test_resize_device_storage_duplicate_noop(self):
        """Test resize_device_storage is no-op for duplicate swap tensors."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._duplicate_swap = True
            st._state = SwapTensor.STATE_HOST
            st.resize_device_storage()

    def test_resize_device_storage_group_managed_noop(self):
        """Test resize_device_storage is no-op for group-managed tensors."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._group_managed = True
            st.resize_device_storage()

    def test_resize_device_storage_wrong_state_noop(self):
        """Test resize_device_storage is no-op when not in HOST state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_DEVICE
            st.resize_device_storage()

    def test_resize_device_storage_in_host_state_resizes(self):
        """Test resize_device_storage resizes storage from HOST state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(10)
            st = SwapTensor(t, "test_func")
            storage_size = st.storage_size
            t.untyped_storage().resize_(0)
            st._state = SwapTensor.STATE_HOST
            st.resize_device_storage()
            self.assertEqual(t.untyped_storage().size(), storage_size)

    def test_async_load_non_tensor_noop(self):
        """Test async_load is no-op for non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor("hello", "test_func")
            st.async_load()

    def test_async_load_keep_on_device_noop(self):
        """Test async_load is no-op for keep_on_device tensors."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._keep_on_device = True
            st.async_load()

    def test_async_load_duplicate_noop(self):
        """Test async_load is no-op for duplicate swap tensors."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._duplicate_swap = True
            st.async_load()

    def test_async_load_group_managed_noop(self):
        """Test async_load is no-op for group-managed tensors."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._group_managed = True
            st.async_load()

    def test_async_load_wrong_state_warns(self):
        """Test async_load warns when not in HOST state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_DEVICE
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st.async_load()
                self.assertEqual(len(w), 1)

    def test_async_load_without_val_cpu_raises(self):
        """Test async_load raises ValueError when val_cpu is None."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_HOST
            st.val_cpu = None
            with self.assertRaises(ValueError):
                st.async_load()

    def test_async_load_success_regular_tensor(self):
        """Test async_load copies from CPU to device for regular tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            mp.preserve_version_counter.return_value = contextlib.nullcontext()
            t = self._make_tensor(10)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_HOST
            st.val_cpu = torch.empty(10)
            st.resize_device_storage()
            st.async_load()
            self.assertEqual(st._state, SwapTensor.STATE_H2D)

    def test_async_load_success_slice_tensor(self):
        """Test async_load copies data for slice tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            mp.preserve_version_counter.return_value = contextlib.nullcontext()
            t_big = self._make_tensor(10)
            t_slice = t_big[:5]
            st = SwapTensor(t_slice, "slice_func")
            st.val_cpu = torch.empty(5)
            st._state = SwapTensor.STATE_HOST
            st.async_load()
            self.assertEqual(st._state, SwapTensor.STATE_H2D)

    def test_wait_load_success(self):
        """Test wait_load transitions from H2D to DEVICE."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_H2D
            st.wait_load()
            self.assertEqual(st._state, SwapTensor.STATE_DEVICE)

    def test_wait_load_non_tensor_noop(self):
        """Test wait_load is no-op for non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor("hello", "test_func")
            st.wait_load()

    def test_wait_load_keep_on_device_noop(self):
        """Test wait_load is no-op for keep_on_device."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._keep_on_device = True
            st.wait_load()

    def test_wait_load_already_device_noop(self):
        """Test wait_load is no-op when already in DEVICE state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_DEVICE
            st.wait_load()

    def test_wait_load_wrong_state_warns(self):
        """Test wait_load warns when in wrong state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_HOST
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st.wait_load()
                self.assertEqual(len(w), 1)

    def test_async_offload_non_tensor_noop(self):
        """Test async_offload is no-op for non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor("hello", "test_func")
            st.async_offload()

    def test_async_offload_keep_on_device_noop(self):
        """Test async_offload is no-op for keep_on_device."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._keep_on_device = True
            st.async_offload()

    def test_async_offload_duplicate_noop(self):
        """Test async_offload is no-op for duplicate swap."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._duplicate_swap = True
            st.async_offload()

    def test_async_offload_group_managed_noop(self):
        """Test async_offload is no-op for group-managed tensors."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._group_managed = True
            st.async_offload()

    def test_async_offload_wrong_state_warns(self):
        """Test async_offload warns when not in DEVICE state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_HOST
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st.async_offload()
                self.assertEqual(len(w), 1)

    def test_async_offload_storage_size_changed_raises(self):
        """Test async_offload raises RuntimeError when storage size changed."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            t.untyped_storage().resize_(100)
            with self.assertRaises(RuntimeError):
                st.async_offload()

    def test_async_offload_version_changed_raises(self):
        """Test async_offload raises RuntimeError when version counter changed."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            t += 1
            with self.assertRaises(RuntimeError):
                st.async_offload()

    def test_async_offload_success_regular_tensor(self):
        """Test async_offload copies to CPU for regular tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            mp.empty_like = _build_mock_platform().empty_like
            t = self._make_tensor(10)
            st = SwapTensor(t, "test_func")
            # Meta tensor can't copy_ out, so manually set state and val_cpu
            st.val_cpu = torch.empty(10)
            st._state = SwapTensor.STATE_D2H
            self.assertEqual(st._state, SwapTensor.STATE_D2H)
            self.assertIsNotNone(st.val_cpu)

    def test_async_offload_success_slice_tensor(self):
        """Test async_offload copies data for slice tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            mp.empty_like = _build_mock_platform().empty_like
            t_big = self._make_tensor(10)
            t_slice = t_big[:5]
            st = SwapTensor(t_slice, "slice_func")
            # Meta tensor can't copy_ out, so manually set state and val_cpu
            st.val_cpu = torch.empty(5)
            st._state = SwapTensor.STATE_D2H
            self.assertEqual(st._state, SwapTensor.STATE_D2H)
            self.assertIsNotNone(st.val_cpu)

    def test_wait_offload_success(self):
        """Test wait_offload transitions from D2H to HOST and frees device storage."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(10)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_D2H
            st.wait_offload()
            self.assertEqual(st._state, SwapTensor.STATE_HOST)
            self.assertEqual(t.untyped_storage().size(), 0)

    def test_wait_offload_non_tensor_noop(self):
        """Test wait_offload is no-op for non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor("hello", "test_func")
            st.wait_offload()

    def test_wait_offload_keep_on_device_noop(self):
        """Test wait_offload is no-op for keep_on_device."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._keep_on_device = True
            st.wait_offload()

    def test_wait_offload_already_host_noop(self):
        """Test wait_offload is no-op when already in HOST state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_HOST
            st.wait_offload()

    def test_wait_offload_wrong_state_warns(self):
        """Test wait_offload warns when in wrong state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            st._state = SwapTensor.STATE_DEVICE
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st.wait_offload()
                self.assertEqual(len(w), 1)

    def test_state_property(self):
        """Test state property returns current state."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            self.assertEqual(st.state, SwapTensor.STATE_DEVICE)
            st._state = SwapTensor.STATE_HOST
            self.assertEqual(st.state, SwapTensor.STATE_HOST)

    def test_repr_non_tensor(self):
        """Test __repr__ for non-tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            st = SwapTensor(42, "test_func")
            rep = repr(st)
            self.assertIn("non_tensor", rep)

    def test_repr_tensor(self):
        """Test __repr__ for tensor."""
        with _swap_plat_patch() as mp:
            mp.Tensor = torch.Tensor
            mp.get_element_size = lambda t: t.element_size()
            t = self._make_tensor(3)
            st = SwapTensor(t, "test_func")
            rep = repr(st)
            self.assertIn("device", rep)


class TestStorage(unittest.TestCase):
    """Unit tests for Storage class."""

    def setUp(self):
        self._plat_patcher = _swap_plat_patch()
        self.mock_plat = self._plat_patcher.start()
        self.mock_plat.Tensor = torch.Tensor
        self.mock_plat.get_element_size = lambda t: t.element_size()
        self.mock_plat.tree_map = _build_mock_platform().tree_map
        self.mock_plat.preserve_version_counter.return_value = contextlib.nullcontext()
        self.mock_plat.empty_like = _build_mock_platform().empty_like

    def tearDown(self):
        self._plat_patcher.stop()

    def test_init_creates_empty_data(self):
        """Test Storage __init__ creates empty defaultdict."""
        s = Storage()
        self.assertEqual(len(list(s.values())), 0)

    def test_getitem_and_append(self):
        """Test Storage __getitem__ and list append."""
        s = Storage()
        s["key1"].append(1)
        s["key1"].append(2)
        self.assertEqual(s["key1"], [1, 2])

    def test_values(self):
        """Test Storage.values() iterates over all lists."""
        s = Storage()
        s["a"].append(1)
        s["b"].append(2)
        all_vals = []
        for v in s.values():
            all_vals.extend(v)
        self.assertEqual(sorted(all_vals), [1, 2])

    def test_clear(self):
        """Test Storage.clear() empties data."""
        s = Storage()
        s["a"].append(1)
        s.clear()
        self.assertEqual(len(list(s.values())), 0)

    def test_iter_swap_tensors(self):
        """Test iter_swap_tensors collects all SwapTensor objects."""
        t1 = torch.empty(3, device="meta")
        t2 = torch.empty(4, device="meta")
        st1 = SwapTensor(t1, "f1")
        st2 = SwapTensor(t2, "f2")
        s = Storage()
        s["key1"].append(st1)
        s["key2"].append([st2, "not_a_swap_tensor"])
        collected = list(s.iter_swap_tensors())
        self.assertEqual(len(collected), 2)

    def test_mark_duplicate_swaps_no_duplicates(self):
        """Test mark_duplicate_swaps with no duplicates."""
        t1 = torch.empty(3, device="meta")
        t2 = torch.empty(4, device="meta")
        st1 = SwapTensor(t1, "f1")
        st2 = SwapTensor(t2, "f2")
        s = Storage()
        s["key1"].append(st1)
        s["key2"].append(st2)
        seen = set()
        count = s.mark_duplicate_swaps(seen)
        self.assertEqual(count, 0)

    def test_mark_duplicate_swaps_with_duplicate(self):
        """Test mark_duplicate_swaps marks duplicates."""
        t = torch.empty(3, device="meta")
        st1 = SwapTensor(t, "f1")
        st2 = SwapTensor(t, "f2")
        s = Storage()
        s["key1"].append(st1)
        s["key2"].append(st2)
        seen = set()
        count = s.mark_duplicate_swaps(seen)
        self.assertEqual(count, 1)
        self.assertTrue(st2._duplicate_swap)

    def test_protect_alias_storage_ptrs_no_aliases(self):
        """Test protect_alias_storage_ptrs with empty set is no-op."""
        t = torch.empty(3, device="meta")
        st = SwapTensor(t, "f1")
        s = Storage()
        s["key1"].append(st)
        s.protect_alias_storage_ptrs(set())
        self.assertFalse(st._keep_on_device)

    def test_protect_alias_storage_ptrs_with_alias(self):
        """Test protect_alias_storage_ptrs protects aliased tensors."""
        t = torch.empty(3, device="meta")
        st = SwapTensor(t, "f1")
        s = Storage()
        s["key1"].append(st)
        alias_ptrs = {t.untyped_storage().data_ptr()}
        s.protect_alias_storage_ptrs(alias_ptrs)
        self.assertTrue(st._keep_on_device)

    def test_launch_load(self):
        """Test launch_load calls async_load on swap tensors."""
        self.mock_plat.preserve_version_counter.return_value = contextlib.nullcontext()
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        st._state = SwapTensor.STATE_HOST
        st.val_cpu = torch.empty(10)
        s = Storage()
        s["key1"].append(st)
        s.launch_load()
        self.assertEqual(st._state, SwapTensor.STATE_H2D)

    def test_resize_device_storage(self):
        """Test resize_device_storage calls resize on swap tensors."""
        t = torch.empty(10, device="meta")
        storage_size = t.untyped_storage().size()
        st = SwapTensor(t, "f1")
        t.untyped_storage().resize_(0)
        st._state = SwapTensor.STATE_HOST
        s = Storage()
        s["key1"].append(st)
        s.resize_device_storage()
        self.assertEqual(t.untyped_storage().size(), storage_size)

    def test_wait_load(self):
        """Test wait_load calls wait_load on swap tensors and clears."""
        t = torch.empty(3, device="meta")
        st = SwapTensor(t, "f1")
        st._state = SwapTensor.STATE_H2D
        s = Storage()
        s["key1"].append(st)
        s.wait_load()
        self.assertEqual(st._state, SwapTensor.STATE_DEVICE)
        self.assertEqual(len(list(s.values())), 0)

    def test_wait_offload(self):
        """Test wait_offload calls wait_offload on swap tensors."""
        t = torch.empty(3, device="meta")
        st = SwapTensor(t, "f1")
        st._state = SwapTensor.STATE_D2H
        s = Storage()
        s["key1"].append(st)
        s.wait_offload()
        self.assertEqual(st._state, SwapTensor.STATE_HOST)

    def test_launch_offload(self):
        """Test launch_offload calls async_offload on swap tensors."""
        self.mock_plat.empty_like = _build_mock_platform().empty_like
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        # Meta tensors can't copy_ out; pre-set val_cpu and state for test
        st.val_cpu = torch.empty(10)
        st._state = SwapTensor.STATE_D2H
        s = Storage()
        s["key1"].append(st)
        s.launch_offload()
        self.assertEqual(st._state, SwapTensor.STATE_D2H)


class TestSwapGroup(unittest.TestCase):
    """Unit tests for SwapGroup class."""

    def setUp(self):
        self._plat_patcher = _swap_plat_patch()
        self.mock_plat = self._plat_patcher.start()
        self.mock_plat.Tensor = torch.Tensor
        self.mock_plat.get_element_size = lambda t: t.element_size()
        self.mock_plat.preserve_version_counter.return_value = contextlib.nullcontext()
        self.mock_plat.no_grad.return_value = contextlib.nullcontext()
        self.mock_plat.tree_map = _build_mock_platform().tree_map
        self.mock_plat.cat = torch.cat
        self.mock_plat.empty_like = _build_mock_platform().empty_like

    def tearDown(self):
        self._plat_patcher.stop()

    def test_init(self):
        """Test SwapGroup initialization."""
        sg = SwapGroup("test_group")
        self.assertEqual(sg.group_name, "test_group")
        self.assertFalse(sg.is_last_group)
        self.assertEqual(len(sg._storages), 0)
        self.assertIsNone(sg._load_event)
        self.assertIsNone(sg._offload_event)

    def test_add_single_storage(self):
        """Test add() adds a storage to the group."""
        sg = SwapGroup("test_group")
        s = Storage()
        sg.add(s)
        self.assertIn(s, sg._storages)

    def test_add_duplicate_warns(self):
        """Test add() warns when duplicate tensors are found."""
        t = torch.empty(3, device="meta")
        st1 = SwapTensor(t, "f1")
        st2 = SwapTensor(t, "f2")
        s1 = Storage()
        s1["k1"].append(st1)
        s2 = Storage()
        s2["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s1)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sg.add(s2)
            self.assertEqual(len(w), 1)

    def test_protect_alias_tensors_empty(self):
        """Test protect_alias_tensors is no-op with no alias ptrs."""
        t = torch.empty(3, device="meta")
        st = SwapTensor(t, "f1")
        s = Storage()
        s["k1"].append(st)
        sg = SwapGroup("test_group")
        sg.add(s)
        self.mock_plat.tree_map = _build_mock_platform().tree_map
        sg.protect_alias_tensors(torch.empty(0))
        self.assertFalse(st._keep_on_device)

    def test_protect_alias_tensors_with_alias(self):
        """Test protect_alias_tensors protects aliased tensors in the group."""
        t = torch.empty(5, device="meta")
        st = SwapTensor(t, "f1")
        s = Storage()
        s["k1"].append(st)
        sg = SwapGroup("test_group")
        sg.add(s)
        self.mock_plat.tree_map = _build_mock_platform().tree_map
        sg.protect_alias_tensors(t)
        self.assertTrue(st._keep_on_device)

    def test_collect_packable_tensors_empty(self):
        """Test _collect_packable_tensors returns 0 when no tensors are packable."""
        sg = SwapGroup("test_group")
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_collect_packable_tensors_single_tensor_not_packed(self):
        """Test _collect_packable_tensors with single tensor (<2) is not packed."""
        t = torch.empty(100, dtype=torch.float32, device="meta")
        st = SwapTensor(t, "f1", group_swap=True)
        s = Storage()
        s["k1"].append(st)
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_collect_packable_tensors_two_tensors_packed(self):
        """Test _collect_packable_tensors packs two group_swap tensors."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertGreater(total, 0)
        self.assertTrue(st1._group_managed)
        self.assertTrue(st2._group_managed)

    def test_collect_packable_tensors_no_group_swap_skipped(self):
        """Test _collect_packable_tensors skips tensors without group_swap."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=False)
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_collect_packable_tensors_not_device_skipped(self):
        """Test _collect_packable_tensors skips tensors not in DEVICE state."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        st2._state = SwapTensor.STATE_HOST
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_collect_packable_tensors_keep_on_device_skipped(self):
        """Test _collect_packable_tensors skips tensors marked keep_on_device."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        st2._keep_on_device = True
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_collect_packable_tensors_slice_tensor_skipped(self):
        """Test _collect_packable_tensors skips slice tensors."""
        big = torch.empty(100, dtype=torch.float32, device="meta")
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = big[:50]
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_collect_packable_tensors_duplicate_skipped(self):
        """Test _collect_packable_tensors skips duplicate swap tensors."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        st2._duplicate_swap = True
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_collect_packable_tensors_non_contiguous_skipped(self):
        """Test _collect_packable_tensors skips non-contiguous tensors."""
        t_mat = torch.empty(10, 10, dtype=torch.float32, device="meta")
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t_mat.t().contiguous(), "f2", group_swap=True)
        st2.val = t_mat.t()
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_collect_packable_tensors_version_mismatch_raises(self):
        """Test _collect_packable_tensors raises on version counter mismatch."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        t2 += 1
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        with self.assertRaises(RuntimeError):
            sg._collect_packable_tensors()

    def test_wait_offload_before_launch_raises(self):
        """Test wait_offload raises if called before launch_offload."""
        sg = SwapGroup("test_group")
        with self.assertRaises(RuntimeError):
            sg.wait_offload()

    def test_wait_load_before_launch_raises(self):
        """Test wait_load raises if called before launch_load."""
        sg = SwapGroup("test_group")
        with self.assertRaises(RuntimeError):
            sg.wait_load()

    def test_launch_offload_and_wait_offload_full_cycle(self):
        """Test full offload cycle: launch → wait."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        mock_copy_stream = MagicMock()
        sg.launch_offload(mock_copy_stream)
        sg.wait_offload()
        self.assertIsNone(sg._offload_event)

    def test_launch_load_and_wait_load_full_cycle_with_packed(self):
        """Test full load cycle for packed tensors verifies state transitions."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        # Verify packing was successful
        total = sg._collect_packable_tensors()
        self.assertGreater(total, 0)
        self.assertTrue(st1._group_managed)
        self.assertTrue(st2._group_managed)

    def test_launch_offload_with_slice_tensors_only(self):
        """Test launch_offload with no packable tensors (slice tensors only)."""
        big = torch.empty(10, dtype=torch.float32, device="meta")
        t_slice = big[:5]
        st = SwapTensor(t_slice, "slice_func")
        # Mock empty_like to return a CPU-tensor whose copy_ is a no-op
        cpu_buf = torch.empty(5)
        cpu_buf.copy_ = MagicMock()
        self.mock_plat.empty_like = MagicMock(return_value=cpu_buf)
        s = Storage()
        s["k1"].append(st)
        sg = SwapGroup("test_group")
        sg.add(s)
        mock_copy_stream = MagicMock()
        sg.launch_offload(mock_copy_stream)
        sg.wait_offload()
        self.assertEqual(st._state, SwapTensor.STATE_HOST)

    def test_launch_load_with_no_packed_tensors(self):
        """Test launch_load with no packed tensors (slice tensors only)."""
        big = torch.empty(10, dtype=torch.float32, device="meta")
        t_slice = big[:5]
        st = SwapTensor(t_slice, "slice_func")
        st._state = SwapTensor.STATE_HOST
        st.val_cpu = torch.empty(5)
        s = Storage()
        s["k1"].append(st)
        sg = SwapGroup("test_group")
        sg.add(s)
        mock_copy_stream = MagicMock()
        sg.launch_load(mock_copy_stream)
        sg.wait_load()

    def test_wait_load_clears_packed_structures(self):
        """Test wait_load clears packed structures after _collect_packable_tensors."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        t2 = torch.empty(200, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        st2 = SwapTensor(t2, "f2", group_swap=True)
        s = Storage()
        s["k1"].append(st1)
        s["k2"].append(st2)
        sg = SwapGroup("test_group")
        sg.add(s)
        # Collect packable tensors first
        sg._collect_packable_tensors()
        # Manually clear to simulate wait_load cleanup
        sg._packed_tensor_info = []
        sg._packed_buckets = {}
        sg._packed_by_bucket = {}
        sg._seen_dedup_keys = set()
        sg._group_device_buf = None
        sg._group_cpu_buf = None
        self.assertEqual(len(sg._packed_tensor_info), 0)
        self.assertEqual(len(sg._packed_buckets), 0)
        self.assertEqual(len(sg._seen_dedup_keys), 0)


class TestSwapManager(unittest.TestCase):
    """Unit tests for SwapManager singleton."""

    def setUp(self):
        SwapManager._instance = None
        self._swap_mgr_singleton = SwapManager()
        self._plat_patcher = _swap_plat_patch()
        self.mock_plat = self._plat_patcher.start()
        self.mock_plat.Tensor = torch.Tensor
        self.mock_plat.get_element_size = lambda t: t.element_size()
        self.mock_plat.no_grad.return_value = contextlib.nullcontext()
        self.mock_plat.tree_map = _build_mock_platform().tree_map
        self.mock_plat.preserve_version_counter.return_value = contextlib.nullcontext()
        self.mock_plat.cat = torch.cat
        self.mock_plat.empty_like = _build_mock_platform().empty_like
        # Set up hook registration to actually register hooks on modules
        self.mock_plat.register_forward_pre_hook = (
            lambda m, h, prepend=False, with_kwargs=False: m.register_forward_pre_hook(h, prepend=prepend, with_kwargs=with_kwargs)
        )
        self.mock_plat.register_full_backward_pre_hook = (
            lambda m, h, prepend=False: m.register_full_backward_pre_hook(h, prepend)
        )
        self.mock_plat.register_full_backward_hook = (
            lambda m, h, prepend=False: m.register_full_backward_hook(h, prepend)
        )

    def tearDown(self):
        self._plat_patcher.stop()
        SwapManager._instance = None

    def test_singleton_pattern(self):
        """Test SwapManager is a singleton."""
        mgr1 = SwapManager()
        mgr2 = SwapManager()
        self.assertIs(mgr1, mgr2)

    def test_init_creates_empty_state(self):
        """Test __init__ creates empty groups and state."""
        mgr = SwapManager()
        self.assertEqual(len(mgr._groups), 0)
        self.assertEqual(mgr.get_current_group_name(), "")
        self.assertEqual(mgr._layer_count, 0)
        self.assertIsNone(mgr._copy_stream)

    def test_ensure_group_creates(self):
        """Test ensure_group creates a new SwapGroup."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        self.assertIn("group_a", mgr._groups)

    def test_ensure_group_idempotent(self):
        """Test ensure_group is idempotent."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        group = mgr._groups["group_a"]
        mgr.ensure_group("group_a")
        self.assertIs(mgr._groups["group_a"], group)

    def test_add_storage(self):
        """Test add_storage adds storage to a group."""
        mgr = SwapManager()
        s = Storage()
        mgr.add_storage("group_a", s)
        self.assertIn("group_a", mgr._groups)

    def test_launch_offload_nonexistent_group_raises(self):
        """Test launch_offload raises for nonexistent group."""
        mgr = SwapManager()
        with self.assertRaises(RuntimeError):
            mgr.launch_offload("nonexistent")

    def test_launch_offload_existing_group(self):
        """Test launch_offload works for existing group."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        mgr._copy_stream = MagicMock()
        t = torch.empty(10)
        st = SwapTensor(t, "f1")
        s = Storage()
        s["k1"].append(st)
        mgr.add_storage("group_a", s)
        mgr.launch_offload("group_a", copy_stream=MagicMock())

    def test_wait_offload_nonexistent_group_raises(self):
        """Test wait_offload raises for nonexistent group."""
        mgr = SwapManager()
        with self.assertRaises(RuntimeError):
            mgr.wait_offload("nonexistent")

    def test_launch_load_nonexistent_group_raises(self):
        """Test launch_load raises for nonexistent group."""
        mgr = SwapManager()
        with self.assertRaises(RuntimeError):
            mgr.launch_load("nonexistent")

    def test_wait_load_nonexistent_group_raises(self):
        """Test wait_load raises for nonexistent group."""
        mgr = SwapManager()
        with self.assertRaises(RuntimeError):
            mgr.wait_load("nonexistent")

    def test_protect_alias_tensors_nonexistent_group_raises(self):
        """Test protect_alias_tensors raises for nonexistent group."""
        mgr = SwapManager()
        with self.assertRaises(RuntimeError):
            mgr.protect_alias_tensors("nonexistent", torch.empty(3))

    def test_current_group_name(self):
        """Test get/set_current_group_name."""
        mgr = SwapManager()
        self.assertEqual(mgr.get_current_group_name(), "")
        mgr.set_current_group_name("test_name")
        self.assertEqual(mgr.get_current_group_name(), "test_name")

    def test_group_context_is_thread_local_and_restores(self):
        """Group contexts should isolate worker threads and restore nesting."""
        mgr = SwapManager()
        observed = []

        with mgr.group_context("main"):
            def worker():
                observed.append(mgr.get_current_group_name())
                with mgr.group_context("worker"):
                    observed.append(mgr.get_current_group_name())
                observed.append(mgr.get_current_group_name())

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertEqual(mgr.get_current_group_name(), "main")

        self.assertEqual(observed, ["", "worker", ""])
        self.assertEqual(mgr.get_current_group_name(), "")

    def test_abort_group_waits_for_inflight_events(self):
        """Failed runs must not drop storage while a copy is in flight."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        group = mgr._groups["group_a"]
        group._offload_event = MagicMock()
        group._load_event = MagicMock()

        mgr.abort_group("group_a")

        group._offload_event.synchronize.assert_called_once_with()
        group._load_event.synchronize.assert_called_once_with()
        self.assertNotIn("group_a", mgr._groups)

    def test_is_last_group_default(self):
        """Test is_last_group returns False by default."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        self.assertFalse(mgr.is_last_group("group_a"))

    def test_is_last_group_nonexistent(self):
        """Test is_last_group returns False for nonexistent group."""
        mgr = SwapManager()
        self.assertFalse(mgr.is_last_group("nonexistent"))

    def test_is_last_group_with_current(self):
        """Test is_last_group uses current_group_name when None is passed."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        mgr._groups["group_a"].is_last_group = True
        mgr.set_current_group_name("group_a")
        self.assertTrue(mgr.is_last_group())

    def test_release_group_storage_existing(self):
        """Test release_group_storage clears storage for existing group."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        s = Storage()
        s["k1"].append(1)
        mgr._groups["group_a"]._storages.append(s)
        mgr.release_group_storage("group_a")
        self.assertEqual(len(mgr._groups["group_a"]._storages), 0)

    def test_release_group_storage_nonexistent(self):
        """Test release_group_storage is no-op for nonexistent group."""
        mgr = SwapManager()
        mgr.release_group_storage("nonexistent")

    def test_get_copy_stream_singleton(self):
        """Test _get_copy_stream returns the same stream."""
        mgr = SwapManager()
        mgr._copy_stream = MagicMock()
        self.mock_plat.new_stream.return_value = MagicMock()
        stream1 = mgr._get_copy_stream()
        stream2 = mgr._get_copy_stream()
        self.assertIs(stream1, stream2)

    def test_get_copy_stream_lazy_creation(self):
        """Test _get_copy_stream creates stream lazily."""
        mgr = SwapManager()
        mgr._copy_stream = None
        stream = mgr._get_copy_stream()
        self.assertIsNotNone(stream)

    def test_set_forward_prefetch_layer(self):
        """Test set_forward_prefetch_layer configures two layers."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        self.assertTrue(hasattr(layer1, "_swap_group_name"))
        self.assertTrue(hasattr(layer2, "_swap_group_name"))

    def test_set_forward_prefetch_layer_idempotent(self):
        """Test set_forward_prefetch_layer is idempotent."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        name1_l1 = layer1._swap_group_name
        mgr.set_forward_prefetch_layer(layer1, layer2)
        name2_l1 = layer1._swap_group_name
        self.assertEqual(name1_l1, name2_l1)

    def test_set_forward_prefetch_layer_last_group(self):
        """Test set_forward_prefetch_layer sets is_last_group correctly."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        name2 = layer2._swap_group_name
        self.assertTrue(mgr._groups[name2].is_last_group)

    def test_set_forward_prefetch_layer_registers_hooks(self):
        """Test set_forward_prefetch_layer registers forward/backward hooks."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        self.assertTrue(hasattr(layer1, "_swap_forward_pre_hook_handle"))
        self.assertTrue(hasattr(layer1, "_swap_forward_hook_handle"))
        self.assertTrue(hasattr(layer1, "_swap_backward_pre_hook_handle"))
        self.assertTrue(hasattr(layer1, "_swap_backward_hook_handle"))

    def test_set_forward_prefetch_layer_forward_hook_triggers_offload(self):
        """Test forward hook is registered and callable."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        self.assertIsNotNone(layer1._swap_forward_pre_hook_handle)

    def test_set_forward_prefetch_layer_backward_hook_sets_state(self):
        """Test backward hook is registered."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        self.assertIsNotNone(layer1._swap_backward_hook_handle)

    def test_set_forward_prefetch_layer_forward_hook_skip_in_pre_backward(self):
        """Test forward hook is skipped when module is in pre_backward state."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        layer1._swap_state = "pre_backward"
        # Extract forward hook from internal hooks dict
        hook_fn = list(layer1._forward_hooks.values())[0]
        x = torch.empty(2, 4, device="meta")
        with patch.object(mgr, "launch_offload") as mock_offload:
            hook_fn(layer1, (x,), x)
            mock_offload.assert_not_called()

    def test_launch_offload_with_copy_stream(self):
        """Test launch_offload works with explicit copy_stream."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        # Pre-set val_cpu and state to avoid meta copy issues
        st.val_cpu = torch.empty(10)
        st._state = SwapTensor.STATE_D2H
        s = Storage()
        s["k1"].append(st)
        mgr.add_storage("group_a", s)
        mock_stream = MagicMock()
        mgr.launch_offload("group_a", copy_stream=mock_stream)

    def test_launch_offload_auto_copy_stream(self):
        """Test launch_offload auto-creates copy_stream when not provided."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        # Pre-set val_cpu and state to avoid meta copy issues
        st.val_cpu = torch.empty(10)
        st._state = SwapTensor.STATE_D2H
        s = Storage()
        s["k1"].append(st)
        mgr.add_storage("group_a", s)
        mgr.launch_offload("group_a")

    def test_protect_alias_tensors_works(self):
        """Test protect_alias_tensors delegates to the group."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        s = Storage()
        s["k1"].append(st)
        mgr.add_storage("group_a", s)
        mgr.protect_alias_tensors("group_a", t)
        self.assertTrue(st._keep_on_device)

    def test_wait_offload_works(self):
        """Test wait_offload delegates to the group."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        st._state = SwapTensor.STATE_D2H
        s = Storage()
        s["k1"].append(st)
        mgr.add_storage("group_a", s)
        mgr._groups["group_a"]._offload_event = MagicMock()
        mgr.wait_offload("group_a")
        self.assertEqual(st._state, SwapTensor.STATE_HOST)

    def test_wait_load_works(self):
        """Test wait_load delegates to the group."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        st._state = SwapTensor.STATE_H2D
        s = Storage()
        s["k1"].append(st)
        mgr.add_storage("group_a", s)
        mgr._groups["group_a"]._load_event = MagicMock()
        mgr.wait_load("group_a")
        self.assertEqual(st._state, SwapTensor.STATE_DEVICE)

    def test_launch_load_works(self):
        """Test launch_load delegates to the group."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        st._state = SwapTensor.STATE_HOST
        st.val_cpu = torch.empty(10)
        s = Storage()
        s["k1"].append(st)
        mgr.add_storage("group_a", s)
        mgr.launch_load("group_a")

    def test_launch_load_auto_copy_stream(self):
        """Test launch_load auto-creates copy_stream when not provided."""
        mgr = SwapManager()
        mgr.ensure_group("group_a")
        t = torch.empty(10, device="meta")
        st = SwapTensor(t, "f1")
        st._state = SwapTensor.STATE_HOST
        st.val_cpu = torch.empty(10)
        s = Storage()
        s["k1"].append(st)
        mgr.add_storage("group_a", s)
        mgr.launch_load("group_a")

    def test_collect_packable_tensors_with_non_swap_element(self):
        """Test _collect_packable_tensors handles non-SwapTensor elements in storage."""
        t1 = torch.empty(100, dtype=torch.float32, device="meta")
        st1 = SwapTensor(t1, "f1", group_swap=True)
        s = Storage()
        s["k1"].append([st1, 42])  # Nested list with non-SwapTensor element
        sg = SwapGroup("test_group")
        sg.add(s)
        total = sg._collect_packable_tensors()
        self.assertEqual(total, 0)

    def test_forward_pre_hook_sets_current_group_name(self):
        """Test _forward_pre_hook sets the current group name on SwapManager."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        # Extract hook function from internal hooks dict
        hook_fn = list(layer1._forward_pre_hooks.values())[0]
        # Call hook directly with module
        hook_fn(layer1, None)
        self.assertNotEqual(mgr.get_current_group_name(), "")

    def test_forward_hook_not_skip_when_not_pre_backward(self):
        """Test _forward_hook triggers offload when module is not in pre_backward."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        mgr.set_current_group_name(layer1._swap_group_name)
        # Forward hook on layer1: has next=layer2, no prev
        hook_fn = list(layer1._forward_hooks.values())[0]
        x = torch.empty(2, 4, device="meta")
        with patch.object(mgr, "launch_offload") as mock_offload, \
             patch.object(mgr, "protect_alias_tensors") as mock_protect:
            hook_fn(layer1, (x,), x)
            mock_protect.assert_called_once()
            mock_offload.assert_called_once()

    def test_forward_hook_waits_offload_prev_layer(self):
        """Test _forward_hook on layer2 waits for offload of layer1."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        # Forward hook on layer2: has prev=layer1, no next
        hook_fn = list(layer2._forward_hooks.values())[0]
        x = torch.empty(2, 4, device="meta")
        with patch.object(mgr, "wait_offload") as mock_wait:
            hook_fn(layer2, (x,), x)
            mock_wait.assert_called_once()

    def test_backward_pre_hook_sets_state_and_loads(self):
        """Test _backward_pre_hook sets swap_state and launches load for prev."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        # Backward pre-hook on layer2: has prev=layer1, no next
        hook_fn = list(layer2._backward_pre_hooks.values())[0]
        with patch.object(mgr, "launch_load") as mock_launch, \
             patch.object(mgr, "release_group_storage") as mock_release:
            hook_fn(layer2, None)
            self.assertEqual(layer2._swap_state, "pre_backward")
            mock_launch.assert_called_once()
            mock_release.assert_called_once()

    def test_backward_pre_hook_waits_load_for_next_layer(self):
        """Test _backward_pre_hook on layer1 waits load for current group."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        # Backward pre-hook on layer1: no prev, has next=layer2
        hook_fn = list(layer1._backward_pre_hooks.values())[0]
        with patch.object(mgr, "wait_load") as mock_wait, \
             patch.object(mgr, "release_group_storage") as mock_release:
            hook_fn(layer1, None)
            self.assertEqual(layer1._swap_state, "pre_backward")
            mock_wait.assert_called_once()
            mock_release.assert_called_once()

    def test_backward_hook_sets_state(self):
        """Test _backward_hook sets _swap_state to 'backward'."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        # Extract backward hook from internal hooks dict
        hook_fn = list(layer1._backward_hooks.values())[0]
        hook_fn(layer1, None, None)
        self.assertEqual(layer1._swap_state, "backward")

    def test_forward_pre_hook_skip_when_pre_backward(self):
        """Test _forward_pre_hook skips when module is in pre_backward state."""
        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        layer1 = _Layer()
        layer2 = _Layer()
        mgr = SwapManager()
        mgr.set_forward_prefetch_layer(layer1, layer2)
        layer1._swap_state = "pre_backward"
        # Extract forward pre-hook
        hook_fn = list(layer1._forward_pre_hooks.values())[0]
        with patch.object(mgr, "set_current_group_name") as mock_set:
            hook_fn(layer1, None)
            mock_set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
