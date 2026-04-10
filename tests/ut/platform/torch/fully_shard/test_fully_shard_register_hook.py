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
"""Unit tests for _register_post_backward_hook in TorchHSDPSchedulerV2.

Verifies that when forward inputs contain a mix of requires_grad=True and
requires_grad=False tensors, the requires_grad attribute is preserved correctly
after passing through PostBackwardFunction.apply.
"""
import os
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch

from hyper_parallel.platform.torch.fully_shard.scheduler import TorchHSDPSchedulerV2


def _make_scheduler_stub() -> TorchHSDPSchedulerV2:
    """Create a minimal TorchHSDPSchedulerV2 stub that can call _register_post_backward_hook."""
    scheduler = object.__new__(TorchHSDPSchedulerV2)
    return scheduler


def _call_register_post_backward_hook(scheduler, args, kwargs):
    """Call the tested protected hook helper in one place for lint cleanliness."""
    # pylint: disable=protected-access
    return scheduler._register_post_backward_hook(args, kwargs)


class TestRegisterPostBackwardHook(unittest.TestCase):
    """Unit tests for TorchHSDPSchedulerV2._register_post_backward_hook."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.scheduler = _make_scheduler_stub()

    def test_mixed_requires_grad_preserves_grad_flag(self):
        """Tensors with requires_grad=False stay False after the hook.

        description: Pass args containing both requires_grad=True and
            requires_grad=False floating-point tensors.
        expectation: The requires_grad flag of each tensor in the output
            matches the flag of the corresponding input tensor.
        feature: _register_post_backward_hook mixed requires_grad handling.
        """
        grad_tensor = torch.randn(3, 4, requires_grad=True)
        no_grad_tensor = torch.randn(3, 4, requires_grad=False)

        args = (grad_tensor, no_grad_tensor)
        kwargs = {}

        out_args, _ = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for first arg, "
            f"got {out_args[0].requires_grad}"
        )
        assert out_args[1].requires_grad is False, (
            f"Expected requires_grad=False for second arg, "
            f"got {out_args[1].requires_grad}"
        )
        assert torch.equal(out_args[0].data, grad_tensor.data), (
            f"Value mismatch for grad tensor: "
            f"expected {grad_tensor.data}, got {out_args[0].data}"
        )
        assert torch.equal(out_args[1].data, no_grad_tensor.data), (
            f"Value mismatch for no-grad tensor: "
            f"expected {no_grad_tensor.data}, got {out_args[1].data}"
        )

    def test_mixed_requires_grad_in_kwargs(self):
        """Tensors with requires_grad=False in kwargs stay False after the hook.

        description: Pass kwargs containing both requires_grad=True and
            requires_grad=False floating-point tensors.
        expectation: The requires_grad flag is preserved for kwargs tensors.
        feature: _register_post_backward_hook kwargs handling.
        """
        args = ()
        kwargs = {
            "x": torch.randn(2, 2, requires_grad=True),
            "y": torch.randn(2, 2, requires_grad=False),
        }

        _, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_kwargs["x"].requires_grad is True, (
            f"Expected requires_grad=True for kwarg 'x', "
            f"got {out_kwargs['x'].requires_grad}"
        )
        assert out_kwargs["y"].requires_grad is False, (
            f"Expected requires_grad=False for kwarg 'y', "
            f"got {out_kwargs['y'].requires_grad}"
        )
        assert torch.equal(out_kwargs["x"].data, kwargs["x"].data), (
            f"Value mismatch for kwarg 'x': "
            f"expected {kwargs['x'].data}, got {out_kwargs['x'].data}"
        )
        assert torch.equal(out_kwargs["y"].data, kwargs["y"].data), (
            f"Value mismatch for kwarg 'y': "
            f"expected {kwargs['y'].data}, got {out_kwargs['y'].data}"
        )

    def test_mixed_grad_with_non_tensor(self):
        """Non-tensor objects are passed through unchanged.

        description: Pass args containing a grad tensor, a no-grad tensor,
            an integer, and None.
        expectation: Tensors preserve requires_grad; non-tensors are unchanged.
        feature: _register_post_backward_hook non-tensor passthrough.
        """
        grad_tensor = torch.randn(2, requires_grad=True)
        no_grad_tensor = torch.randn(2, requires_grad=False)

        args = (grad_tensor, no_grad_tensor, 42, None)
        kwargs = {}

        out_args, _ = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for arg[0], "
            f"got {out_args[0].requires_grad}"
        )
        assert torch.equal(out_args[0].data, grad_tensor.data), (
            f"Value mismatch for grad tensor: "
            f"expected {grad_tensor.data}, got {out_args[0].data}"
        )
        assert out_args[1].requires_grad is False, (
            f"Expected requires_grad=False for arg[1], "
            f"got {out_args[1].requires_grad}"
        )
        assert torch.equal(out_args[1].data, no_grad_tensor.data), (
            f"Value mismatch for no-grad tensor: "
            f"expected {no_grad_tensor.data}, got {out_args[1].data}"
        )
        assert out_args[2] == 42, (
            f"Expected integer 42, got {out_args[2]}"
        )
        assert out_args[3] is None, (
            f"Expected None, got {out_args[3]}"
        )

    def test_all_no_grad_returns_early(self):
        """When no tensor requires grad, args/kwargs are returned unchanged.

        description: Pass only requires_grad=False tensors.
        expectation: Returns early with original args and kwargs unchanged.
        feature: _register_post_backward_hook early return.
        """
        t1 = torch.randn(2, requires_grad=False)
        t2 = torch.randn(3, requires_grad=False)

        args = (t1, t2)
        kwargs = {}

        out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args is args, (
            "Expected same args object (early return), "
            "got different object"
        )
        assert out_kwargs is kwargs, (
            "Expected same kwargs object (early return), "
            "got different object"
        )

    def test_grad_disabled_returns_early(self):
        """When grad is disabled, args/kwargs are returned unchanged.

        description: Disable grad globally and pass tensors.
        expectation: Returns early with original args and kwargs unchanged.
        feature: _register_post_backward_hook grad disabled path.
        """
        t1 = torch.randn(2, requires_grad=True)
        args = (t1,)
        kwargs = {}

        with torch.no_grad():
            out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args is args, (
            "Expected same args object under no_grad, "
            "got different object"
        )
        assert out_kwargs is kwargs, (
            "Expected same kwargs object under no_grad, "
            "got different object"
        )

    def test_multiple_grad_tensors_preserve_identity(self):
        """Multiple requires_grad=True tensors are passed through correctly.

        description: Pass multiple grad and no-grad tensors in both args and kwargs.
        expectation: All requires_grad flags are preserved; tensor data is unchanged.
        feature: _register_post_backward_hook multi-tensor correctness.
        """
        g1 = torch.randn(2, 3, requires_grad=True)
        g2 = torch.randn(4, requires_grad=True)
        ng1 = torch.randn(5, requires_grad=False)
        ng2 = torch.randn(1, 2, requires_grad=False)

        args = (g1, ng1)
        kwargs = {"a": g2, "b": ng2}

        out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for args[0], "
            f"got {out_args[0].requires_grad}"
        )
        assert torch.equal(out_args[0].data, g1.data), (
            f"Value mismatch for args[0]: "
            f"expected {g1.data}, got {out_args[0].data}"
        )
        assert out_args[1].requires_grad is False, (
            f"Expected requires_grad=False for args[1], "
            f"got {out_args[1].requires_grad}"
        )
        assert torch.equal(out_args[1].data, ng1.data), (
            f"Value mismatch for args[1]: "
            f"expected {ng1.data}, got {out_args[1].data}"
        )
        assert out_kwargs["a"].requires_grad is True, (
            f"Expected requires_grad=True for kwargs['a'], "
            f"got {out_kwargs['a'].requires_grad}"
        )
        assert torch.equal(out_kwargs["a"].data, g2.data), (
            f"Value mismatch for kwargs['a']: "
            f"expected {g2.data}, got {out_kwargs['a'].data}"
        )
        assert out_kwargs["b"].requires_grad is False, (
            f"Expected requires_grad=False for kwargs['b'], "
            f"got {out_kwargs['b'].requires_grad}"
        )
        assert torch.equal(out_kwargs["b"].data, ng2.data), (
            f"Value mismatch for kwargs['b']: "
            f"expected {ng2.data}, got {out_kwargs['b'].data}"
        )

    def test_integer_tensor_not_affected(self):
        """Integer tensors (non-floating) are not affected by requires_grad.

        description: Pass an integer tensor alongside floating-point tensors.
        expectation: Integer tensor is passed through unchanged; float tensors
            preserve their requires_grad flags.
        feature: _register_post_backward_hook integer tensor handling.
        """
        grad_float = torch.randn(3, requires_grad=True)
        int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        no_grad_float = torch.randn(3, requires_grad=False)

        args = (grad_float, int_tensor, no_grad_float)
        kwargs = {}

        out_args, _ = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for float grad tensor, "
            f"got {out_args[0].requires_grad}"
        )
        assert torch.equal(out_args[0].data, grad_float.data), (
            f"Value mismatch for grad float: "
            f"expected {grad_float.data}, got {out_args[0].data}"
        )
        assert out_args[1].requires_grad is False, (
            f"Expected requires_grad=False for int tensor, "
            f"got {out_args[1].requires_grad}"
        )
        assert torch.equal(out_args[1], int_tensor), (
            f"Value mismatch for int tensor: "
            f"expected {int_tensor}, got {out_args[1]}"
        )
        assert out_args[2].requires_grad is False, (
            f"Expected requires_grad=False for no-grad float tensor, "
            f"got {out_args[2].requires_grad}"
        )
        assert torch.equal(out_args[2].data, no_grad_float.data), (
            f"Value mismatch for no-grad float: "
            f"expected {no_grad_float.data}, got {out_args[2].data}"
        )


if __name__ == "__main__":
    unittest.main()
