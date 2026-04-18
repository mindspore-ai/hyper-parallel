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
"""Unit tests for ``hyper_parallel.core.shard.dfunction``.

All tests run on CPU without any distributed setup.  The torch platform is
selected so that ``platform.Function = torch.autograd.Function``.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.core.shard.dfunction import DFunction, _LocalCallable


# ---------------------------------------------------------------------------
# Shared subclasses used across multiple test classes
# ---------------------------------------------------------------------------

class _AddFunc(DFunction):
    """Element-wise add: output = x + y."""

    @staticmethod
    def forward(ctx, x, y):  # pylint: disable=W0221
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        x, y = ctx.saved_tensors
        return torch.ones_like(x) * grad, torch.ones_like(y) * grad


class _ScaleFunc(DFunction):
    """Scale: output = x * scale (scale is a non-tensor positional arg)."""

    @staticmethod
    def forward(ctx, x, scale):  # pylint: disable=W0221
        ctx.save_for_backward(x)
        ctx.scale = scale
        return x * scale

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        # None for the non-tensor scale argument
        return grad * ctx.scale, None


class _MultiOutputFunc(DFunction):
    """Returns two outputs for testing tuple backward."""

    @staticmethod
    def forward(ctx, x):  # pylint: disable=W0221
        ctx.save_for_backward(x)
        return x * 2, x * 3

    @staticmethod
    def backward(ctx, grad_a, grad_b):  # pylint: disable=W0221
        return grad_a * 2 + grad_b * 3


class _NoOpNameFunc(DFunction):
    """DFunction subclass with no _op_name — for error-case testing."""

    @staticmethod
    def forward(ctx, x):  # pylint: disable=W0221
        return x

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=W0221
        return grad


# ---------------------------------------------------------------------------
# C1: Forward / backward correctness (single device, CPU)
# ---------------------------------------------------------------------------

class TestDFunctionBasics(unittest.TestCase):
    """Forward and backward correctness on plain CPU tensors."""

    def test_basic_forward(self):
        """apply returns the correct element-wise sum."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        result = _AddFunc.apply(x, y)
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result, expected), (
            f"Forward result mismatch: expected={expected}, got={result}"
        )

    def test_backward_gradient(self):
        """Gradients flow correctly through a differentiable apply call."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=True)
        result = _AddFunc.apply(x, y)
        result.sum().backward()
        assert x.grad is not None, "x.grad should not be None after backward"
        assert y.grad is not None, "y.grad should not be None after backward"
        assert torch.allclose(x.grad, torch.ones(2)), (
            f"x.grad mismatch: expected ones, got {x.grad}"
        )
        assert torch.allclose(y.grad, torch.ones(2)), (
            f"y.grad mismatch: expected ones, got {y.grad}"
        )

    def test_save_for_backward(self):
        """ctx.save_for_backward and ctx.saved_tensors round-trip correctly."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=True)
        result = _AddFunc.apply(x, y)
        # If save_for_backward fails the call above raises; passing is sufficient.
        result.sum().backward()

    def test_non_tensor_arg_forwarded(self):
        """Non-tensor positional arguments are forwarded to forward correctly."""
        x = torch.tensor([2.0, 4.0], requires_grad=True)
        result = _ScaleFunc.apply(x, 3.0)
        expected = torch.tensor([6.0, 12.0])
        assert torch.allclose(result, expected), (
            f"Scale forward mismatch: expected={expected}, got={result}"
        )

    def test_backward_with_non_tensor_arg(self):
        """Backward propagates through a function that used a non-tensor arg."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = _ScaleFunc.apply(x, 5.0)
        result.sum().backward()
        assert torch.allclose(x.grad, torch.full((2,), 5.0)), (
            f"Gradient mismatch: expected 5.0, got {x.grad}"
        )

    def test_multi_output_forward(self):
        """forward returning a tuple is handled correctly."""
        x = torch.tensor([1.0, 2.0])
        out_a, out_b = _MultiOutputFunc.apply(x)
        assert torch.allclose(out_a, x * 2), (
            f"out_a mismatch: expected {x * 2}, got {out_a}"
        )
        assert torch.allclose(out_b, x * 3), (
            f"out_b mismatch: expected {x * 3}, got {out_b}"
        )

    def test_multi_output_backward(self):
        """Backward sums contributions from both output gradients correctly."""
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out_a, out_b = _MultiOutputFunc.apply(x)
        (out_a.sum() + out_b.sum()).backward()
        # grad_a=1, grad_b=1, x.grad = 1*2 + 1*3 = 5
        assert torch.allclose(x.grad, torch.full((2,), 5.0)), (
            f"Multi-output gradient mismatch: expected 5.0, got {x.grad}"
        )


# ---------------------------------------------------------------------------
# C2: DFunction base-class structure
# ---------------------------------------------------------------------------

class TestDFunctionBaseClass(unittest.TestCase):
    """Verify the platform base class and class structure."""

    def test_platform_base_is_torch_function(self):
        """DFunction inherits from torch.autograd.Function."""
        assert issubclass(DFunction, torch.autograd.Function), (
            f"DFunction should subclass torch.autograd.Function, "
            f"got bases: {DFunction.__bases__}"
        )

    def test_subclass_inherits_dfunction(self):
        """User subclasses are subclasses of both DFunction and torch.autograd.Function."""
        assert issubclass(_AddFunc, DFunction), (
            "_AddFunc should subclass DFunction"
        )
        assert issubclass(_AddFunc, torch.autograd.Function), (
            "_AddFunc should subclass torch.autograd.Function"
        )

    def test_op_name_is_none_by_default(self):
        """DFunction subclasses without _op_name have _op_name=None."""
        assert _NoOpNameFunc._op_name is None, (
            f"Expected _op_name=None, got {_NoOpNameFunc._op_name}"
        )


# ---------------------------------------------------------------------------
# C3: _LocalCallable naming and caching
# ---------------------------------------------------------------------------

class TestLocalCallable(unittest.TestCase):
    """Tests for _LocalCallable naming and caching behaviour."""

    def test_local_callable_name_attributes(self):
        """_LocalCallable exposes both __name__ and .name for platform dispatchers."""
        def _identity_fn(*a):
            return a
        fn = _identity_fn
        lc = _LocalCallable(fn, "TestOp")
        assert lc.__name__ == "TestOp", (
            f"__name__ mismatch: expected 'TestOp', got {lc.__name__}"
        )
        assert lc.name == "TestOp", (
            f".name mismatch: expected 'TestOp', got {lc.name}"
        )

    def test_local_callable_invocation(self):
        """_LocalCallable forwards calls to the wrapped function."""
        called_with = []

        def _capture_fn(*a, **kw):
            called_with.append((a, kw))
            return 42

        fn = _capture_fn
        lc = _LocalCallable(fn, "Op")
        result = lc(1, 2, key="val")
        assert result == 42, f"Expected 42, got {result}"
        assert called_with == [((1, 2), {"key": "val"})], (
            f"Call args mismatch: {called_with}"
        )

    def test_get_local_callable_cached(self):
        """Repeated _get_local_callable() calls return the same object."""
        lc1 = _AddFunc._get_local_callable()
        lc2 = _AddFunc._get_local_callable()
        assert lc1 is lc2, (
            f"_get_local_callable() should return the same cached object, "
            f"got different objects: id(lc1)={id(lc1)}, id(lc2)={id(lc2)}"
        )

    def test_two_subclasses_have_independent_callables(self):
        """Each DFunction subclass has its own independent _local_callable."""

        class _FuncA(DFunction):
            _op_name = "OpA"

            @staticmethod
            def forward(ctx, x):  # pylint: disable=W0221
                return x

            @staticmethod
            def backward(ctx, grad):  # pylint: disable=W0221
                return grad

        class _FuncB(DFunction):
            _op_name = "OpB"

            @staticmethod
            def forward(ctx, x):  # pylint: disable=W0221
                return x * 2

            @staticmethod
            def backward(ctx, grad):  # pylint: disable=W0221
                return grad * 2

        lc_a = _FuncA._get_local_callable()
        lc_b = _FuncB._get_local_callable()
        assert lc_a is not lc_b, (
            "_FuncA and _FuncB should have independent _local_callable objects"
        )
        assert lc_a.__name__ == "OpA", (
            f"_FuncA callable name mismatch: expected 'OpA', got {lc_a.__name__}"
        )
        assert lc_b.__name__ == "OpB", (
            f"_FuncB callable name mismatch: expected 'OpB', got {lc_b.__name__}"
        )


# ---------------------------------------------------------------------------
# C4: DTensor dispatch routing
# ---------------------------------------------------------------------------

class TestDFunctionDTensorDispatch(unittest.TestCase):
    """Tests for the DTensor dispatch routing in apply()."""

    def test_dtensor_without_op_name_raises(self):
        """ValueError is raised when DTensor inputs are passed without _op_name."""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415

        mock_dtensor = MagicMock(spec=DTensor)

        with self.assertRaises(ValueError) as ctx:
            _NoOpNameFunc.apply(mock_dtensor)

        assert "_op_name" in str(ctx.exception), (
            f"ValueError message should mention '_op_name', got: {ctx.exception}"
        )

    def test_local_tensor_skips_dispatcher(self):
        """Plain tensor inputs bypass _OP_DISPATCHER entirely."""
        dispatch_called = []
        with patch("hyper_parallel.core.shard._op_dispatch._OP_DISPATCHER") as mock_disp:
            mock_disp.dispatch.side_effect = lambda *a, **kw: dispatch_called.append(True)
            x = torch.tensor([1.0, 2.0])
            y = torch.tensor([3.0, 4.0])
            _AddFunc.apply(x, y)

        assert len(dispatch_called) == 0, (
            f"Dispatcher should not be called for plain tensor inputs, "
            f"but was called {len(dispatch_called)} times"
        )


if __name__ == "__main__":
    unittest.main()
