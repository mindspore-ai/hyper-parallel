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
"""Unit tests for :mod:`hyper_parallel.core.dtensor.dtensor_base` — the ``data`` descriptor."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor_base import DTensorBase
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Shard

_MODULE = "hyper_parallel.core.dtensor.dtensor_base"


class _FakeDataDescriptor:
    """Records every ``Tensor.data.__set__`` call instead of touching storage."""

    def __init__(self):
        self.set_calls = []

    def __get__(self, obj, objtype=None):
        return "fake-data"

    def __set__(self, obj, value):
        self.set_calls.append((obj, value))


class TestDTensorBaseDataDescriptor(unittest.TestCase):
    """``DTensorBase.data`` must keep the wrapper and the local shard in sync."""

    def test_data_setter_updates_wrapper_and_local_tensor(self):
        """Assigning ``dtensor.data = x`` writes through to the local shard too."""
        fake_descriptor = _FakeDataDescriptor()
        fake_tensor_cls = SimpleNamespace(data=fake_descriptor)
        fake_dtensor = SimpleNamespace(_local_tensor=object())

        with patch(f"{_MODULE}.Tensor", fake_tensor_cls):
            DTensorBase.data.fset(fake_dtensor, "payload")

        self.assertEqual(
            fake_descriptor.set_calls,
            [(fake_dtensor, "payload"), (fake_dtensor._local_tensor, "payload")],
        )

    def test_data_setter_uses_local_tensor_for_dtensor_input(self):
        """Assigning another DTensor propagates that tensor's local shard payload."""
        class FakeInputDTensor:
            def __init__(self):
                self._local_tensor = "local-shard"

            def to_local(self):
                return self._local_tensor

        fake_descriptor = _FakeDataDescriptor()
        fake_tensor_cls = SimpleNamespace(data=fake_descriptor)
        fake_dtensor = SimpleNamespace(_local_tensor=object())
        input_dtensor = FakeInputDTensor()

        with patch(f"{_MODULE}.Tensor", fake_tensor_cls):
            with patch(f"{_MODULE}.DTensorBase", FakeInputDTensor):
                DTensorBase.data.fset(fake_dtensor, input_dtensor)

        self.assertEqual(
            fake_descriptor.set_calls,
            [(fake_dtensor, "local-shard"), (fake_dtensor._local_tensor, "local-shard")],
        )


class TestDTensorMetadata(unittest.TestCase):
    """All public numel entry points retain logical global tensor semantics."""

    def test_global_numel_matches_for_real_and_fake_parameters(self):
        """A shard's local capacity must not leak into global metadata queries."""
        layout = Layout((2,), ("dp",), init_backend=False)
        layout.set_placements((Shard(0),))
        layout.placement_to_tensor_map(2)
        for fake in (False, True):
            with self.subTest(fake=fake), FakeTensorMode() if fake else nullcontext():
                local = torch.empty(4, 3)
                parameter = torch.nn.Parameter(DTensor.from_local_with_layout(local, layout, shape=(8, 3)))
                for skip in (False, True):
                    with self.subTest(skip=skip), SkipDTensorDispatch() if skip else nullcontext():
                        self.assertEqual(parameter.shape, (8, 3))
                        self.assertEqual(parameter.size(), (8, 3))
                        self.assertEqual(parameter.numel(), 24)
                        self.assertEqual(torch.numel(parameter), 24)
                        self.assertEqual(torch.numel(input=parameter), 24)
                        self.assertEqual(torch.Tensor.numel(parameter), 24)
                        self.assertEqual(parameter.to_local().numel(), 12)


class TestDTensorFakeLocalSupport(unittest.TestCase):
    """DTensor wraps FakeTensor locals natively via a storage-less meta shell."""

    @staticmethod
    def _make_layout():
        layout = Layout((2,), ("dp",), init_backend=False)
        layout.set_placements((Shard(0),))
        layout.placement_to_tensor_map(2)
        return layout

    def test_meta_shell_construction_and_storage_forwarding(self):
        """A FakeTensor local yields a meta-shell wrapper answering for the local."""
        layout = self._make_layout()
        with FakeTensorMode():
            local = torch.empty(4, 3)
        dtensor = DTensor.from_local_with_layout(local, layout, shape=(8, 3))
        self.assertTrue(dtensor.is_meta_shell())
        self.assertIs(dtensor._local_tensor, local)
        self.assertIs(dtensor.untyped_storage(), local.untyped_storage())
        self.assertEqual(dtensor.device, local.device)
        detached = dtensor.detach()
        self.assertIsInstance(detached, DTensor)
        self.assertTrue(detached.is_meta_shell())
        parameter = torch.nn.Parameter(dtensor, requires_grad=True)
        self.assertIsInstance(parameter, DTensor)

    def test_real_local_is_not_meta_shell(self):
        """Real locals keep aliasing the wrapper storage; no shell is involved."""
        layout = self._make_layout()
        local = torch.empty(4, 3)
        dtensor = DTensor.from_local_with_layout(local, layout, shape=(8, 3))
        self.assertFalse(dtensor.is_meta_shell())
        self.assertIs(dtensor.untyped_storage(), local.untyped_storage())

    def test_backward_populates_local_grad_through_dispatcher(self):
        """Autograd over fake locals lands gradients on the local leaf."""
        layout = self._make_layout()
        mode = FakeTensorMode()
        with mode:
            local = torch.empty(4, 3, requires_grad=True)
        dtensor = DTensor.from_local_with_layout(local, layout, shape=(8, 3))
        with mode:
            (dtensor * 2).sum().backward()
        self.assertIsNotNone(local.grad)
        self.assertIsNotNone(dtensor.grad)

    def test_meta_shell_dtensor_grad_assignment_round_trips(self):
        """Assigning a meta-shell DTensor gradient stores the local and returns the wrapper."""
        layout = self._make_layout()
        with FakeTensorMode():
            local = torch.empty(4, 3, requires_grad=True)
            reduced = torch.empty(4, 3)
        parameter = DTensor.from_local_with_layout(local, layout, shape=(8, 3))
        grad_dtensor = DTensor.from_local_with_layout(reduced, layout, shape=(8, 3))
        parameter.grad = grad_dtensor
        self.assertIs(parameter.grad, grad_dtensor)
        self.assertIs(local.grad, reduced)
        parameter.grad = None
        self.assertIsNone(parameter.grad)
        self.assertIsNone(local.grad)

    def test_data_accessors_use_local_payload_for_meta_shell(self):
        """``.data`` reads and writes go to the local tensor on meta shells."""
        layout = self._make_layout()
        mode = FakeTensorMode()
        with mode:
            local = torch.empty(4, 3)
            replacement = torch.empty(4, 3)
        dtensor = DTensor.from_local_with_layout(local, layout, shape=(8, 3))
        self.assertIs(
            dtensor.data.untyped_storage(), dtensor._local_tensor.untyped_storage()
        )
        dtensor.data = replacement
        self.assertIs(
            dtensor._local_tensor.untyped_storage(), replacement.untyped_storage()
        )


if __name__ == "__main__":
    unittest.main()
