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
"""Single-process tests for the MPipe Transpose execution math.

These validate the core numerical invariant of the schedule without any
distributed runtime: the transposed path (detached preprocess forward -> body
forward/backward -> recompute backward on stage 0) must produce exactly the
same preprocess gradient as an ordinary graph-connected forward/backward, and
the non-transposed path must stay graph-connected so its backward is automatic.
"""
import os
import unittest
from typing import Any, Optional

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # noqa: E402  pylint: disable=wrong-import-position
from tests.common.mark_utils import arg_mark  # noqa: E402  pylint: disable=wrong-import-position

from hyper_parallel.platform.torch.pipeline_parallel.mpipe_transpose import (  # noqa: E402  pylint: disable=wrong-import-position
    MPipeTransposeExecutor,
)


class _Ctx:
    """Minimal stand-in for PipelineContext exposing the fields the handlers use."""

    def __init__(self, arg_mbs: list, kwarg_mbs: Optional[list] = None) -> None:
        """Store the per-micro arg/kwarg slots and an empty send-handle list."""
        self.arg_mbs = arg_mbs
        self.kwarg_mbs = kwarg_mbs
        self.send_handles = []


def _step(micro_index):
    return type("MetaStepStub", (), {"micro_index": micro_index})()


def _make_executor(preprocess, num_transpose, has_trainable=True):
    """Build an executor with only the local-compute state populated (no distribution)."""
    executor = object.__new__(MPipeTransposeExecutor)
    # pylint: disable=protected-access
    executor._preprocess = preprocess
    executor._num_transpose = num_transpose
    executor._this_rank = 0
    executor._inputs = {}
    executor._outputs = {}
    executor._has_trainable = has_trainable
    return executor


class TestMPipeTransposeRecomputeEquivalence(unittest.TestCase):
    """Recompute backward on stage 0 must match a plain connected backward."""

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_transposed_micro_grad_matches_reference(self):
        """
        Feature: MPipe Transpose executor recompute backward.
        Description: Run a transposed micro through detached forward + recompute backward.
        Expectation: the preprocess weight/bias grads match a plain connected backward.
        """
        torch.manual_seed(0)
        dim = 8
        preprocess = torch.nn.Linear(dim, dim)
        body = torch.nn.Linear(dim, dim)
        x = torch.randn(4, dim)

        reference_loss = body(preprocess(x)).pow(2).sum()
        reference_loss.backward()
        ref_weight_grad = preprocess.weight.grad.clone()
        ref_bias_grad = preprocess.bias.grad.clone()
        preprocess.zero_grad(set_to_none=True)
        body.zero_grad(set_to_none=True)

        executor = _make_executor(preprocess, num_transpose=1)
        ctx = _Ctx(arg_mbs=[[x]])
        step = _step(0)

        executor.transpose_forward(step, ctx)
        body_input = ctx.arg_mbs[0][0]
        assert not body_input.requires_grad or body_input.grad_fn is None, \
            "transposed preprocess output must be a detached leaf, not graph-connected"
        body(body_input).pow(2).sum().backward()
        executor.transpose_backward(step, ctx)

        weight_diff = (preprocess.weight.grad - ref_weight_grad).abs().max().item()
        bias_diff = (preprocess.bias.grad - ref_bias_grad).abs().max().item()
        assert torch.allclose(preprocess.weight.grad, ref_weight_grad, atol=1e-6), \
            f"preprocess weight grad mismatch: max abs diff {weight_diff}"
        assert torch.allclose(preprocess.bias.grad, ref_bias_grad, atol=1e-6), \
            f"preprocess bias grad mismatch: max abs diff {bias_diff}"

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_nontransposed_micro_is_graph_connected(self):
        """
        Feature: MPipe Transpose executor forward (torch path).
        Description: Run a non-transposed micro (num_transpose=0) through the executor.
        Expectation: its preprocess output stays graph-connected and the body
            backward reaches the preprocess weights with correct grads.
        """
        torch.manual_seed(0)
        dim = 8
        preprocess = torch.nn.Linear(dim, dim)
        body = torch.nn.Linear(dim, dim)
        x = torch.randn(4, dim)

        body(preprocess(x)).pow(2).sum().backward()
        ref_weight_grad = preprocess.weight.grad.clone()
        preprocess.zero_grad(set_to_none=True)
        body.zero_grad(set_to_none=True)

        # num_transpose=0 -> micro 0 is non-transposed: inline, graph-connected.
        executor = _make_executor(preprocess, num_transpose=0)
        ctx = _Ctx(arg_mbs=[[x]])
        executor.transpose_forward(_step(0), ctx)

        body_input = ctx.arg_mbs[0][0]
        assert body_input.grad_fn is not None, \
            "non-transposed preprocess output must stay graph-connected for automatic backward"
        body(body_input).pow(2).sum().backward()

        weight_diff = (preprocess.weight.grad - ref_weight_grad).abs().max().item()
        assert torch.allclose(preprocess.weight.grad, ref_weight_grad, atol=1e-6), \
            f"non-transposed preprocess weight grad mismatch: max abs diff {weight_diff}"

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_kwargs_forwarded_to_preprocess(self):
        """
        Feature: MPipe Transpose executor kwargs forwarding.
        Description: Ship a preprocess that takes a per-micro ``scale`` kwarg.
        Expectation: the same kwargs reach both the forward and the recompute
            backward, so grads match the reference.
        """
        class _ScaledPreprocess(torch.nn.Module):
            def __init__(self) -> None:
                """A linear preprocess that also takes a scalar ``scale`` kwarg."""
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
                """Apply the linear layer and scale the output by ``scale``."""
                return self.linear(x) * scale

        torch.manual_seed(0)
        preprocess = _ScaledPreprocess()
        body = torch.nn.Linear(8, 8)
        x = torch.randn(4, 8)
        scale = 2.5

        body(preprocess(x, scale=scale)).pow(2).sum().backward()
        ref_grad = preprocess.linear.weight.grad.clone()
        preprocess.zero_grad(set_to_none=True)
        body.zero_grad(set_to_none=True)

        executor = _make_executor(preprocess, num_transpose=1)
        ctx = _Ctx(arg_mbs=[[x]], kwarg_mbs=[{"scale": scale}])
        step = _step(0)
        executor.transpose_forward(step, ctx)
        body(ctx.arg_mbs[0][0]).pow(2).sum().backward()
        executor.transpose_backward(step, ctx)

        diff = (preprocess.linear.weight.grad - ref_grad).abs().max().item()
        assert torch.allclose(preprocess.linear.weight.grad, ref_grad, atol=1e-6), \
            f"kwargs not applied consistently in forward + recompute: max abs diff {diff}"

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_dataload_only_passes_integer_input_ungraded(self):
        """
        Feature: MPipe Transpose executor T=0 dataload-only path.
        Description: Ship an integer input through a param-free identity preprocess.
        Expectation: the value is forwarded as-is and never marked grad-requiring,
            so integer input_ids survive (requires_grad would otherwise error).
        """
        class _Identity(torch.nn.Module):
            def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # pylint: disable=unused-argument
                """Return the input unchanged (param-free T=0 preprocess)."""
                return x

        executor = _make_executor(_Identity(), num_transpose=1, has_trainable=False)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        ctx = _Ctx(arg_mbs=[[input_ids]])
        executor.transpose_forward(_step(0), ctx)  # must not raise on the int tensor

        out = ctx.arg_mbs[0][0]
        assert out.dtype == torch.long and not out.requires_grad, \
            (f"T=0 identity output should be the unmodified, ungraded int input, "
             f"got dtype={out.dtype}, requires_grad={out.requires_grad}")

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_transpose_backward_noop_when_no_grad(self):
        """
        Feature: MPipe Transpose executor recompute backward.
        Description: Invoke transpose_backward when the body deposited no gradient.
        Expectation: it is a safe no-op and leaves the preprocess grads as None.
        """
        preprocess = torch.nn.Linear(4, 4)
        executor = _make_executor(preprocess, num_transpose=1)
        # pylint: disable=protected-access
        executor._inputs[0] = (torch.randn(2, 4),)
        leaf = torch.randn(2, 4, requires_grad=True)
        ctx = _Ctx(arg_mbs=[[leaf]])  # leaf.grad is None
        executor.transpose_backward(_step(0), ctx)
        assert preprocess.weight.grad is None, \
            f"expected no preprocess grad on no-op backward, got {preprocess.weight.grad}"


if __name__ == "__main__":
    unittest.main()
