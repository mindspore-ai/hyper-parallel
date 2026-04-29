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
"""Unit tests for MindSpore autograd compatibility helpers."""
# pylint: disable=assignment-from-no-return,protected-access,unused-import

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

try:
    import mindspore  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest("mindspore is required for autograd compat tests") from exc

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_default,
)

ensure_mindspore_platform_default()

from hyper_parallel.platform.mindspore import autograd_compat


class FakeDType:
    """Minimal dtype stub for backward input validation."""

    def __init__(self, is_complex=False, is_float=True, name="Float32"):
        self.is_complex = is_complex
        self._is_float = is_float
        self.name = name

    def __repr__(self):
        return self.name


class FakeFloatMeta(type):
    """Treat fake float dtypes as instances of typing.Float/BFloat."""

    def __instancecheck__(cls, instance):
        return getattr(instance, "_is_float", False)


class FakeTensor:
    """Minimal tensor stub for testing autograd_compat helpers."""

    def __init__(
        self,
        shape=(),
        dtype=None,
        requires_grad=True,
        is_leaf=True,
        grad=None,
        dtensor_grad=None,
        grad_node=None,
        output_index=0,
        retains_grad=False,
    ):
        self.shape = shape
        self.dtype = dtype or FakeDType()
        self._requires_grad = requires_grad
        self._is_leaf = is_leaf
        self._grad = grad
        self._dtensor_grad = dtensor_grad
        self._grad_node = grad_node
        self._output_index = output_index
        self._retains_grad = retains_grad
        self.retain_grad_called = False

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def retains_grad(self):
        return self._retains_grad

    @property
    def grad_fn(self):
        return self._grad_node

    @property
    def output_nr(self):
        return self._output_index

    def numel(self):
        if not self.shape:
            return 1
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def _retain_grad(self):
        self.retain_grad_called = True
        return "retain_grad_called"


class FakeLeafNode:
    """Fake grad node with leaf marker."""

    def __init__(self, is_leaf_value):
        self._is_leaf_value = is_leaf_value

    def is_leaf(self):
        return self._is_leaf_value


class FakeDTensor:
    """Fake DTensor wrapper exposing a local tensor."""

    def __init__(self, local_tensor):
        self._local_tensor = local_tensor


class TestGradProperty(unittest.TestCase):
    """Tests for the `.grad` compatibility property."""

    def test_grad_prefers_dtensor_grad(self):
        """DTensor grad should take precedence over the local cached grad."""
        tensor = FakeTensor(grad="local_grad", dtensor_grad="dist_grad")

        result = autograd_compat.grad.fget(tensor)

        self.assertEqual(result, "dist_grad")

    def test_grad_warns_for_non_leaf_tensor(self):
        """Reading `.grad` on a non-leaf tensor should emit the compatibility warning."""
        tensor = FakeTensor(is_leaf=False, requires_grad=True, grad="leaf_grad")

        with self.assertWarnsRegex(UserWarning, "not a leaf Tensor"):
            result = autograd_compat.grad.fget(tensor)

        self.assertEqual(result, "leaf_grad")

    def test_grad_setter_syncs_dtensor_local_tensor(self):
        """Setting a DTensor grad should also cache its local tensor."""
        tensor = FakeTensor()
        local_grad = FakeTensor(shape=(2,))
        dist_grad = FakeDTensor(local_grad)
        fake_dtensor_module = SimpleNamespace(DTensor=FakeDTensor)

        with patch.dict(sys.modules, {"hyper_parallel.core.dtensor.dtensor": fake_dtensor_module}):
            autograd_compat.grad.fset(tensor, dist_grad)

        self.assertIs(tensor._dtensor_grad, dist_grad)
        self.assertIs(tensor._grad, local_grad)


class TestGraphMetadataProperties(unittest.TestCase):
    """Tests for `grad_fn` and related graph metadata accessors."""

    def test_grad_fn_returns_none_for_leaf_node(self):
        tensor = FakeTensor(grad_node=FakeLeafNode(True))

        self.assertIsNone(autograd_compat.grad_fn.fget(tensor))

    def test_grad_fn_returns_non_leaf_node(self):
        node = FakeLeafNode(False)
        tensor = FakeTensor(grad_node=node)

        self.assertIs(autograd_compat.grad_fn.fget(tensor), node)

    def test_retain_grad_forwards_to_tensor_impl(self):
        tensor = FakeTensor()

        result = autograd_compat.retain_grad(tensor)

        self.assertEqual(result, "retain_grad_called")
        self.assertTrue(tensor.retain_grad_called)


class TestDetach(unittest.TestCase):
    """Tests for `detach()` compatibility behavior."""

    def test_detach_clears_dtensor_grad(self):
        tensor = FakeTensor(dtensor_grad="old_dist_grad")
        detached = FakeTensor(dtensor_grad="should_be_cleared")

        with patch.object(autograd_compat, "pyboost_detach", return_value=detached):
            result = autograd_compat.detach(tensor)

        self.assertIs(result, detached)
        self.assertIsNone(detached._dtensor_grad)


class TestBackward(unittest.TestCase):
    """Tests for torch-style backward argument handling."""

    def test_backward_uses_implicit_scalar_grad(self):
        """Scalar outputs should receive an implicit ones-like gradient."""
        loss = FakeTensor(shape=(), dtype=FakeDType(is_float=True))
        auto_grad = FakeTensor(shape=())

        fake_float = FakeFloatMeta("FakeFloat", (), {})
        fake_bfloat = FakeFloatMeta("FakeBFloat", (), {})
        fake_typing = SimpleNamespace(Float=fake_float, BFloat=fake_bfloat)

        with patch.object(autograd_compat, "TensorPy", FakeTensor), patch.object(
            autograd_compat, "typing", fake_typing
        ), patch.object(autograd_compat.ops, "ones_like", return_value=auto_grad) as mock_ones, patch.object(
            autograd_compat, "run_backward", return_value="backward_called"
        ) as mock_backward:
            result = autograd_compat.backward(loss)

        self.assertEqual(result, "backward_called")
        mock_ones.assert_called_once_with(loss)
        mock_backward.assert_called_once_with(
            (loss,),
            (auto_grad,),
            False,
            False,
            (),
            allow_unreachable=True,
            accumulate_grad=True,
        )

    def test_backward_supports_explicit_sens_tensor(self):
        """An explicit sensitivity tensor should be forwarded to backward unchanged."""
        loss = FakeTensor(shape=(2,), dtype=FakeDType(is_float=True))
        sens = FakeTensor(shape=(2,), dtype=FakeDType(is_float=True))
        input_tensor = FakeTensor(shape=(2,), dtype=FakeDType(is_float=True))

        with patch.object(autograd_compat, "TensorPy", FakeTensor), patch.object(
            autograd_compat, "run_backward", return_value="backward_called"
        ) as mock_backward:
            result = autograd_compat.backward(loss, gradient=sens, inputs=input_tensor)

        self.assertEqual(result, "backward_called")
        mock_backward.assert_called_once_with(
            (loss,),
            (sens,),
            False,
            False,
            (input_tensor,),
            allow_unreachable=True,
            accumulate_grad=True,
        )

    def test_backward_rejects_missing_grad_for_non_scalar_output(self):
        """Non-scalar outputs must provide an explicit gradient tensor."""
        loss = FakeTensor(shape=(2,), dtype=FakeDType(is_float=True))
        fake_float = FakeFloatMeta("FakeFloat", (), {})
        fake_bfloat = FakeFloatMeta("FakeBFloat", (), {})
        fake_typing = SimpleNamespace(Float=fake_float, BFloat=fake_bfloat)

        with patch.object(autograd_compat, "TensorPy", FakeTensor), patch.object(
            autograd_compat, "typing", fake_typing
        ):
            with self.assertRaisesRegex(
                RuntimeError, "implicitly created only for scalar outputs"
            ):
                autograd_compat.backward(loss)

    def test_backward_rejects_empty_inputs_tuple(self):
        loss = FakeTensor(shape=(), dtype=FakeDType(is_float=True))

        with self.assertRaisesRegex(RuntimeError, "cannot be empty"):
            autograd_compat.backward(loss, inputs=())


if __name__ == "__main__":
    unittest.main()
