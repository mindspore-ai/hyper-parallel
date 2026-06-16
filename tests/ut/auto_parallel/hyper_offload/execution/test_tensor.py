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
"""Unit tests for ShadowTensor and PhysicalBuffer device_storage."""

from __future__ import annotations

import unittest

import torch

from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import PhysicalBuffer




class TestShadowTensor(unittest.TestCase):
    """ShadowTensor lifecycle and resolve()."""

    def setUp(self) -> None:
        self.t = torch.randn(2, 3)
        self.buffer = PhysicalBuffer(device=self.t.device)
        self.wrapped = ShadowTensor(self.t, buffer=self.buffer, storage_id=1)

    def test_is_shadow_tensor(self) -> None:
        self.assertIsInstance(self.wrapped, ShadowTensor)
        self.assertNotIsInstance(torch.randn(2, 2), ShadowTensor)
        self.assertNotIsInstance(None, ShadowTensor)

    def test_storage_id(self) -> None:
        self.assertEqual(self.wrapped.storage_id, 1)

    def test_metadata_matches_original(self) -> None:
        self.assertEqual(self.wrapped.dtype, self.t.dtype)
        self.assertEqual(self.wrapped.shape, self.t.shape)
        self.assertEqual(self.wrapped.device, self.t.device)
        self.assertEqual(self.wrapped.requires_grad, self.t.requires_grad)

    def test_shape_stride_preserved(self) -> None:
        self.assertEqual(self.wrapped.shape, self.t.shape)
        self.assertEqual(self.wrapped.stride(), self.t.stride())
        self.assertEqual(self.wrapped.dtype, self.t.dtype)

    def test_resolve_reconstructs_from_device_storage(self) -> None:
        """Verify resolve creates a view from the PhysicalBuffer's storage."""
        storage = self.t.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=self.t.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        self.buffer.device_buffer = dev_view

        result = self.wrapped.resolve()
        self.assertEqual(result.shape, self.t.shape)
        self.assertEqual(result.dtype, self.t.dtype)
        self.assertEqual(result.stride(), self.t.stride())

    def test_resolve_raises_when_no_device_or_host(self) -> None:
        self.buffer.device_buffer = None
        self.buffer.host_buffer = None
        with self.assertRaisesRegex(RuntimeError, "No device buffer available"):
            self.wrapped.resolve()

    def test_resolve_demand_pages_from_host(self) -> None:
        """When device_buffer is None but host_buffer exists, resolve should demand-page."""
        # Create a 1D host buffer matching what PinnedMemoryPool produces
        host_data = self.t.cpu()
        byte_size = host_data.untyped_storage().size()
        host_buf = host_data.view(dtype=torch.uint8).clone().flatten()
        self.assertEqual(host_buf.numel(), byte_size)
        self.buffer.host_buffer = host_buf
        self.buffer.device_buffer = None

        result = self.wrapped.resolve()
        self.assertEqual(result.shape, self.t.shape)
        self.assertEqual(result.dtype, self.t.dtype)
        # After resolve, device_buffer should be populated
        self.assertIsNotNone(self.buffer.device_buffer)

    def test_torch_dispatch_resolves_and_forwards(self) -> None:
        """__torch_dispatch__ should resolve the shadow and call the function."""
        storage = self.t.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=self.t.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        self.buffer.device_buffer = dev_view

        # Perform an operation on the shadow tensor
        result = self.wrapped + 1.0
        expected = self.t + 1.0
        torch.testing.assert_close(result, expected)

    def test_torch_dispatch_multiple_args(self) -> None:
        """Dispatch with multiple shadow tensor arguments."""
        storage = self.t.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=self.t.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        self.buffer.device_buffer = dev_view

        other = torch.randn(2, 3)
        result = self.wrapped + other
        expected = self.t + other
        torch.testing.assert_close(result, expected)

    def test_shadow_tensor_storage_id_property(self) -> None:
        self.assertEqual(self.wrapped.storage_id, 1)

    def test_multiple_shadows_different_sids(self) -> None:
        t2 = torch.randn(4, 5)
        buf2 = PhysicalBuffer(device=t2.device)
        shadow2 = ShadowTensor(t2, buf2, storage_id=2)
        self.assertEqual(self.wrapped.storage_id, 1)
        self.assertEqual(shadow2.storage_id, 2)
        self.assertNotEqual(self.wrapped.storage_id, shadow2.storage_id)

    def test_torch_dispatch_unwraps_nested_structures(self) -> None:
        """__torch_dispatch__ should unwrap ShadowTensor inside lists/tuples/dicts."""
        t = torch.randn(2, 3)
        buf = PhysicalBuffer(device=t.device)
        # Set up device buffer
        storage = t.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=t.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        buf.device_buffer = dev_view

        shadow = ShadowTensor(t, buf, storage_id=1)

        # Test with tuple containing shadow
        result = torch.add(shadow, torch.tensor(1.0))
        expected = t + 1.0
        torch.testing.assert_close(result, expected)

    def test_shadow_tensor_with_zero_dim(self) -> None:
        """ShadowTensor with a 0-dim (scalar) tensor should work."""
        t = torch.tensor(3.14)
        buf = PhysicalBuffer(device=t.device)
        storage = t.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=t.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        buf.device_buffer = dev_view

        shadow = ShadowTensor(t, buf, storage_id=1)
        result = shadow.resolve()
        self.assertEqual(result.shape, t.shape)
        self.assertEqual(result.dtype, t.dtype)
        self.assertAlmostEqual(result.item(), 3.14, places=5)


class TestShadowTensorRequiresGrad(unittest.TestCase):
    """ShadowTensor with requires_grad."""


    def test_requires_grad_propagated(self) -> None:
        t = torch.randn(2, 3, requires_grad=True)
        buf = PhysicalBuffer(device=t.device)
        shadow = ShadowTensor(t, buf, storage_id=1)
        self.assertTrue(shadow.requires_grad)

    def test_no_requires_grad(self) -> None:
        t = torch.randn(2, 3, requires_grad=False)
        buf = PhysicalBuffer(device=t.device)
        shadow = ShadowTensor(t, buf, storage_id=1)
        self.assertFalse(shadow.requires_grad)

    def test_shadow_tensor_dispatch_with_requires_grad(self) -> None:
        """Dispatch through ShadowTensor should work without error."""
        t = torch.randn(2, 3, requires_grad=True)
        buf = PhysicalBuffer(device=t.device)
        storage = t.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=t.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        buf.device_buffer = dev_view

        shadow = ShadowTensor(t, buf, storage_id=1)
        result = shadow * 2
        # Verify result is a regular tensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertFalse(isinstance(result, ShadowTensor))
        # Verify mathematical correctness
        expected = t * 2
        torch.testing.assert_close(result, expected)

    def test_torch_dispatch_with_kwargs_none(self) -> None:
        """Dispatch with kwargs=None should not crash (kwargs = kwargs or {})."""
        t = torch.randn(2, 3)
        buf = PhysicalBuffer(device=t.device)
        storage = t.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=t.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        buf.device_buffer = dev_view
        shadow = ShadowTensor(t, buf, storage_id=1)

        # Call __torch_dispatch__ directly with kwargs=None
        result = ShadowTensor.__torch_dispatch__(
            torch.ops.aten.add.Tensor,
            (ShadowTensor,),
            (shadow, torch.tensor(1.0)),
            kwargs=None,
        )
        expected = t + 1.0
        torch.testing.assert_close(result, expected)

    def test_torch_dispatch_unwraps_list_containing_shadow(self) -> None:
        """Dispatch with args containing a list with a ShadowTensor should unwrap correctly."""
        t = torch.randn(2, 3)
        buf = PhysicalBuffer(device=t.device)
        storage = t.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=t.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        buf.device_buffer = dev_view
        shadow = ShadowTensor(t, buf, storage_id=1)

        # Call __torch_dispatch__ with args containing a list with a ShadowTensor
        result = ShadowTensor.__torch_dispatch__(
            torch.ops.aten.cat,
            (ShadowTensor,),
            ([shadow, shadow],),
            kwargs=None,
        )
        expected = torch.cat([t, t])
        torch.testing.assert_close(result, expected)
