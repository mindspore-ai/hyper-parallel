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
from unittest.mock import MagicMock, patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # noqa: E402  pylint: disable=wrong-import-position

from hyper_parallel.platform.torch.pipeline_parallel.mpipe_transpose import (  # noqa: E402  pylint: disable=wrong-import-position
    MPipeTransposeExecutor,
)
import hyper_parallel.core.pipeline_parallel.mpipe.executor_base as mpipe_base  # noqa: E402  pylint: disable=wrong-import-position


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


class _StubStage:
    """Minimal PipelineStage stand-in for the executor's __init__."""

    def __init__(self) -> None:
        """Expose the device / pp_group / stage_index the executor reads."""
        self.device = torch.device("cpu")
        self.pp_group = object()
        self.stage_index = 0


class _StubSchedule:
    """Minimal ScheduleMPipeTranspose stand-in for the executor's __init__ + send path."""

    def __init__(self, preprocess: torch.nn.Module, num_transpose: int = 2,
                 has_trainable: bool = True) -> None:
        """Expose the schedule attributes the executor reads, plus ``_send_handles``."""
        self.preprocess_module = preprocess
        self.stages = [_StubStage()]
        self.real_stage_num = 4
        self.num_transpose_micro_batches = num_transpose
        self.has_trainable_preprocess = has_trainable
        self._send_handles = []


class TestMPipeTransposeExecutorTransport(unittest.TestCase):
    """Cover ``__init__`` and the i->0 ``MPIPE_*`` transport handlers (mocked platform P2P)."""

    def test_init_binds_schedule_state(self):
        """
        Feature: MPipe Transpose executor construction.
        Description: Build the executor through its real ``__init__`` with a stub schedule.
        Expectation: it caches the per-rank transpose state (rank, NT, trainable flag, buffers).
        """
        sched = _StubSchedule(torch.nn.Linear(4, 4), num_transpose=2, has_trainable=True)
        executor = MPipeTransposeExecutor(sched)
        # pylint: disable=protected-access
        assert executor._this_rank == 0 and executor._num_transpose == 2
        assert executor._has_trainable and not executor._inputs and not executor._outputs
        executor.reset()
        assert not executor._inputs and not executor._outputs

    @patch.object(mpipe_base, "platform")
    def test_broadcast_params(self, mock_plat):
        """
        Feature: MPipe Transpose parameter broadcast.
        Description: Run ``broadcast_params`` for a trainable preprocess with mocked P2P.
        Expectation: every preprocess tensor is broadcast from stage 0.
        """
        mock_plat.get_global_rank.return_value = 0
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2)
        executor._pp_group = object()  # pylint: disable=protected-access
        executor.broadcast_params(_step(0), _Ctx(arg_mbs=[[torch.randn(2, 4)]]))
        assert mock_plat.broadcast.called

    @patch.object(mpipe_base, "platform")
    def test_fwd_send_and_graph_send(self, mock_plat):
        """
        Feature: MPipe Transpose i->0 send (output + recompute input).
        Description: Run ``fwd_send`` / ``graph_send`` with mocked P2P.
        Expectation: the (shape, dtype) meta is sent and each isend handle is deferred
            onto the schedule's ``_send_handles``.
        """
        mock_plat.get_global_rank.return_value = 1
        mock_plat.isend.return_value = "handle"
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2)
        # pylint: disable=protected-access
        executor._pp_group = object()
        executor._outputs[1] = torch.randn(2, 4)
        executor._inputs[1] = (torch.randn(2, 4),)
        ctx = _Ctx(arg_mbs=[[torch.randn(2, 4)], [torch.randn(2, 4)]])
        ctx.schedule = _StubSchedule(torch.nn.Linear(4, 4))
        executor.fwd_send(_step(1), ctx)
        executor.graph_send(_step(1), ctx)
        assert mock_plat.send_object_list.call_count >= 2
        assert len(ctx.schedule._send_handles) >= 2

    @patch.object(mpipe_base, "platform")
    def test_fwd_recv_and_graph_recv(self, mock_plat):
        """
        Feature: MPipe Transpose i->0 receive (output into stage-0 slot + recompute input).
        Description: Run ``fwd_recv`` / ``graph_recv`` with mocked P2P (a trainable preprocess
            marks the received buffer grad-requiring).
        Expectation: the received output lands in ``ctx.arg_mbs`` and the recompute input in
            the executor's per-micro input cache.
        """
        mock_plat.get_global_rank.return_value = 1

        def _recv_meta(meta, src):  # pylint: disable=unused-argument
            meta[0] = (2, 4)
            meta[1] = torch.float32
        mock_plat.recv_object_list.side_effect = _recv_meta
        mock_plat.empty.return_value = torch.zeros(2, 4)
        mock_plat.irecv.return_value = MagicMock()
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2, has_trainable=True)
        # pylint: disable=protected-access
        executor._pp_group = object()
        executor._device = torch.device("cpu")
        executor._inputs[0] = (torch.randn(2, 4),)   # arity source for graph_recv
        ctx = _Ctx(arg_mbs=[None, None])
        executor.fwd_recv(_step(1), ctx)
        assert ctx.arg_mbs[1] is not None and mock_plat.irecv.called
        executor.graph_recv(_step(1), ctx)
        assert 1 in executor._inputs


if __name__ == "__main__":
    unittest.main()
