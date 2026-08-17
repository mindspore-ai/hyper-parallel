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
distributed runtime: the (uniform, always-detached) preprocess forward -> body
forward/backward -> stage-0 backward path must produce exactly the same
preprocess gradient as an ordinary graph-connected forward/backward. Since the
round-robin change every micro currently takes the detached path (the old
graph-connected inline path for overflow micros was dropped); the stage-0
backward recomputes the grad instead. The in-flight owner-does-backward work
retains the forward graph on the owner and will reconnect it -- tracked by the
``test_preprocess_output_stays_graph_connected`` xfail below.
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
import hyper_parallel.platform.torch.pipeline_parallel.mpipe_transpose as mpipe_torch  # noqa: E402  pylint: disable=wrong-import-position
from hyper_parallel.core.pipeline_parallel.mpipe.schedule import (  # noqa: E402  pylint: disable=wrong-import-position
    ScheduleMPipeTranspose,
)
from hyper_parallel.core.pipeline_parallel.mpipe.step_types import MpipeStepType  # noqa: E402  pylint: disable=wrong-import-position
from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # noqa: E402  pylint: disable=wrong-import-position


class _Ctx:
    """Minimal stand-in for PipelineContext exposing the fields the handlers use."""

    def __init__(self, arg_mbs: list, kwarg_mbs: Optional[list] = None) -> None:
        """Store the per-micro arg/kwarg slots and an empty send-handle list."""
        self.arg_mbs = arg_mbs
        self.kwarg_mbs = kwarg_mbs
        self.send_handles = []


def _step(micro_index):
    return type("MetaStepStub", (), {"micro_index": micro_index})()


def _body_step(step_type, micro_index, stage_index):
    """A body-order step stub exposing the fields the rank-0 patcher reads."""
    return type("MetaStepStub", (), {
        "type": step_type, "micro_index": micro_index, "stage_index": stage_index,
    })()


def _make_executor(preprocess, num_transpose, has_trainable=True,  # pylint: disable=unused-argument
                   owner_backward=False, output_consumer=None, micro_batch_num=2):
    """Build an executor with only the local-compute state populated (no distribution).

    ``num_transpose`` is kept for call-site clarity but the executor no longer
    caches it (the schedule owns ``num_transpose_micro_batches``); ownership is
    resolved through the stub schedule's ``owner_of`` / ``owned_micros`` below.
    """
    executor = object.__new__(MPipeTransposeExecutor)
    # pylint: disable=protected-access
    executor._preprocess = preprocess
    executor._device = torch.device("meta")
    executor._output_arity_for_comm = None
    executor._input_arity_for_comm = None
    executor._this_rank = 0
    executor._inputs_for_explicit_forward = {}
    executor._outputs_for_stage0 = {}
    executor._outputs_for_bwd = {}     # owner-backward retained graphs; set in __init__
    executor._fwd_received = set()  # batched fwd_recv dedup; set in __init__
    executor._keep_grad = {}           # placed feature tensors (stage-0 / owner-backward grad read)
    executor._owner_backward = owner_backward
    executor._mpipe_group = None
    executor._mpipe_group_info = None
    executor._grad_snapshot = None  # tower-grad snapshot for the accumulation-safe reduce
    executor._has_trainable = has_trainable
    # Must be an INSTANCE attribute: as a class attribute a function consumer
    # would bind as a method, unlike the real schedule's property.
    sched_stub = type("_SchedStub", (), {})()
    sched_stub.output_consumer = output_consumer
    # Ownership the batched recv path reads: identity layout (micro m owned by
    # rank m), so rank 0 owns {0} and receives every other micro from rank m.
    sched_stub.micro_batch_num = micro_batch_num
    sched_stub.owner_of = lambda m: m
    sched_stub.owned_micros = lambda rank: frozenset(
        m for m in range(micro_batch_num) if m == rank)
    executor._schedule = sched_stub
    return executor


class _FakeDTensorGrad:
    """Stand-in for an FSDP sharded-DTensor grad: ``to_local()`` returns the plain
    local shard the pp reduce must operate on (the DTensor is never handed to a
    raw collective)."""

    def __init__(self, local: torch.Tensor) -> None:
        """Wrap the rank-local shard."""
        self._local = local

    def to_local(self) -> torch.Tensor:
        """Return the rank-local shard."""
        return self._local


class _FakeShardedParam:
    """A trainable tower param whose ``.grad`` is an FSDP DTensor stand-in."""

    def __init__(self, grad: object) -> None:
        """Hold a DTensor-like ``.grad``."""
        self.requires_grad = True
        self.grad = grad


class _FakeShardedPreprocess:
    """A preprocess module exposing only ``parameters()`` -- what the tower-grad
    reduce iterates."""

    def __init__(self, params: list) -> None:
        """Expose ``params`` to the tower-grad reduce."""
        self._params = params

    def parameters(self) -> iter:
        """Iterate the stand-in parameters."""
        return iter(self._params)


class TestMPipeTransposeStage0BackwardEquivalence(unittest.TestCase):
    """The stage-0 backward must match a plain connected backward."""

    def test_transposed_micro_grad_matches_reference(self):
        """
        Feature: MPipe Transpose executor stage-0 backward.
        Description: Run a transposed micro through detached forward + stage-0 backward.
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

    # The graph-connected inline forward was removed in the round-robin change, so
    # this fails today. It is kept as an executable SPEC (unittest's strict xfail)
    # for the in-flight "backward of a non-stage-0 / owner-does-backward" work,
    # which -- if it retains the forward graph rather than recomputing -- must
    # reconnect the output. When that lands this becomes an unexpected pass; remove
    # the decorator then. (Current detached+recompute correctness is already
    # covered by test_transposed_micro_grad_matches_reference.)
    @unittest.expectedFailure
    def test_preprocess_output_stays_graph_connected(self):
        """
        Feature: MPipe Transpose executor forward — graph-connected contract (TARGET).
        Description: Run a micro through the executor forward.
        Expectation (not yet met): the preprocess output stays graph-connected so a
            non-stage-0 / owner backward can flow into the preprocess without a
            recompute. The executor currently always detaches, so this is an
            expected failure until owner-does-backward reconnects the graph.
        """
        torch.manual_seed(0)
        dim = 8
        preprocess = torch.nn.Linear(dim, dim)
        x = torch.randn(4, dim)

        executor = _make_executor(preprocess, num_transpose=1)
        ctx = _Ctx(arg_mbs=[[x]])
        executor.transpose_forward(_step(0), ctx)

        body_input = ctx.arg_mbs[0][0]
        assert body_input.grad_fn is not None, \
            "preprocess output should stay graph-connected (owner-does-backward target)"

    def test_kwargs_forwarded_to_preprocess(self):
        """
        Feature: MPipe Transpose executor kwargs forwarding.
        Description: Ship a preprocess that takes a per-micro ``scale`` kwarg.
        Expectation: the same kwargs reach both the forward and the stage-0
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
            f"kwargs not applied consistently in forward + stage-0 backward: max abs diff {diff}"

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
        Feature: MPipe Transpose executor stage-0 backward.
        Description: Invoke transpose_backward when the body deposited no gradient.
        Expectation: it is a safe no-op and leaves the preprocess grads as None.
        """
        preprocess = torch.nn.Linear(4, 4)
        executor = _make_executor(preprocess, num_transpose=1)
        # pylint: disable=protected-access
        executor._save_grad_for_bwd(0, torch.randn(2, 4))
        leaf = torch.randn(2, 4, requires_grad=True)
        ctx = _Ctx(arg_mbs=[[leaf]])  # leaf.grad is None
        executor.transpose_backward(_step(0), ctx)
        assert preprocess.weight.grad is None, \
            f"expected no preprocess grad on no-op backward, got {preprocess.weight.grad}"
        executor.reset()  # the no-grad path must still have cleaned up its entries


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
        # Read by ``_place_output`` when the executor runs the preprocess forward
        # on rank 0 (``None`` = default placement into ``arg_mbs[micro]``).
        self.output_consumer = None
        self._send_handles = []


class TestMPipeTransposeExecutorTransport(unittest.TestCase):
    """Cover ``__init__`` and the i->0 ``MPIPE_*`` transport handlers (mocked platform P2P)."""

    def test_init_binds_schedule_state(self):
        """
        Feature: MPipe Transpose executor construction.
        Description: Build the executor through its real ``__init__`` with a stub schedule.
        Expectation: it caches the per-rank transpose state (rank, preprocess module,
            trainable flag, empty buffers). NT is no longer cached on the executor --
            the schedule owns ``num_transpose_micro_batches`` / ``owner_of``.
        """
        sched = _StubSchedule(torch.nn.Linear(4, 4), num_transpose=2, has_trainable=True)
        executor = MPipeTransposeExecutor(sched)
        # pylint: disable=protected-access
        assert executor._this_rank == 0
        assert executor._preprocess is sched.preprocess_module
        assert (executor._has_trainable and not executor._inputs_for_explicit_forward
                and not executor._outputs_for_stage0)
        executor.reset()
        assert not executor._inputs_for_explicit_forward and not executor._outputs_for_stage0

    @patch.object(mpipe_base, "platform")
    def test_broadcast_params(self, mock_plat):
        """
        Feature: MPipe Transpose parameter broadcast.
        Description: Run ``broadcast_params`` for a trainable preprocess with mocked P2P.
        Expectation: every preprocess tensor is broadcast from stage 0.
        """
        mock_plat.get_global_rank.return_value = 0
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2)
        executor.broadcast_params(_step(0), _Ctx(arg_mbs=[[torch.randn(2, 4)]]))
        assert mock_plat.broadcast.called

    @patch.object(mpipe_base, "platform")
    def test_fwd_send_and_graph_send(self, mock_plat):
        """
        Feature: MPipe Transpose i->0 send (output + stage-0-backward input).
        Description: Run ``fwd_send`` / ``fwd_bwd_graph_input_send`` with mocked P2P.
        Expectation: the (shape, dtype) meta is sent and each isend handle is deferred
            onto the schedule's ``_send_handles``.
        """
        mock_plat.get_global_rank.return_value = 1
        mock_plat.isend.return_value = "handle"
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2)
        # pylint: disable=protected-access
        executor._outputs_for_stage0[1] = torch.randn(2, 4)
        executor._inputs_for_explicit_forward[1] = (torch.randn(2, 4),)
        ctx = _Ctx(arg_mbs=[[torch.randn(2, 4)], [torch.randn(2, 4)]])
        ctx.schedule = _StubSchedule(torch.nn.Linear(4, 4))
        executor.fwd_send(_step(1), ctx)
        executor.fwd_bwd_graph_input_send(_step(1), ctx)
        assert mock_plat.send_object_list.call_count >= 2
        assert len(ctx.schedule._send_handles) >= 2

    @patch.object(mpipe_base, "platform")
    def test_fwd_recv_and_graph_recv(self, mock_plat):
        """
        Feature: MPipe Transpose i->0 receive (output into stage-0 slot + stage-0-backward input).
        Description: Run ``fwd_recv`` / ``fwd_bwd_graph_input_recv`` with mocked P2P (a trainable preprocess
            marks the received buffer grad-requiring).
        Expectation: the received output lands in ``ctx.arg_mbs`` and the stage-0-backward input in
            the executor's per-micro input cache.
        """
        mock_plat.get_global_rank.return_value = 1

        def _recv_meta(meta, src, group=None):  # pylint: disable=unused-argument
            meta[0] = (2, 4)
            meta[1] = torch.float32
        mock_plat.recv_object_list.side_effect = _recv_meta
        mock_plat.empty.return_value = torch.zeros(2, 4)
        mock_plat.irecv.return_value = MagicMock()
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2, has_trainable=True)
        # pylint: disable=protected-access
        executor._device = torch.device("cpu")
        executor._inputs_for_explicit_forward[0] = (torch.randn(2, 4),)   # arity source for fwd_bwd_graph_input_recv
        # stage-0's own micro-0 output: the arity source for the batched fwd_recv
        executor._outputs_for_stage0[0] = (torch.zeros(2, 4),)
        ctx = _Ctx(arg_mbs=[None, None])
        executor.fwd_recv(_step(1), ctx)
        assert ctx.arg_mbs[1] is not None and mock_plat.irecv.called
        executor.fwd_bwd_graph_input_recv(_step(1), ctx)
        assert 1 in executor._inputs_for_explicit_forward

    def test_recv_device_resolves_meta_to_materialized(self):
        """
        Feature: MPipe Transpose recv-buffer device under deferred / meta init.
        Description: When first_stage.device (cached as _device) is ``meta`` (FSDP2
            builds the stage on meta, materializes later), _recv_device() must
            resolve to a materialized tensor's real device so recv buffers are not
            Meta tensors -- c10d::recv_ has no Meta kernel and raises.
        Expectation: meta is replaced by the materialized output/input device and
            cached; a non-meta _device is returned unchanged.
        """
        # pylint: disable=protected-access
        # meta _device + materialized own output -> resolves to the output device.
        ex = _make_executor(torch.nn.Linear(4, 4), num_transpose=2)
        ex._device = torch.device("meta")
        ex._outputs_for_stage0[0] = (torch.zeros(2, 4),)
        assert ex._recv_device() == torch.device("cpu")
        assert ex._device == torch.device("cpu")  # cached after resolution
        # falls back to the retained input when only that is present.
        ex2 = _make_executor(torch.nn.Linear(4, 4), num_transpose=2)
        ex2._device = torch.device("meta")
        ex2._inputs_for_explicit_forward[0] = (torch.zeros(2, 4),)
        assert ex2._recv_device() == torch.device("cpu")
        # a real device is returned unchanged (fast path, no resolution).
        ex3 = _make_executor(torch.nn.Linear(4, 4), num_transpose=2)
        ex3._device = torch.device("cpu")
        assert ex3._recv_device() == torch.device("cpu")


class TestMPipeTransposeOwnerBackward(unittest.TestCase):
    """Owner-does-backward: retained-graph forward, ship-back, owner backward, reduce."""

    def test_owner_transpose_backward_matches_reference(self):
        """
        Feature: MPipe owner-does-backward tower backward.
        Description: Backprop dL/dfeatures through the owner's retained connected
            tower graph and compare to a plain ``preprocess(x).backward(grad)``.
        Expectation: identical tower param-grads (owner backward == centralized).
        """
        torch.manual_seed(0)
        preprocess = torch.nn.Linear(4, 4)
        ref = torch.nn.Linear(4, 4)
        ref.load_state_dict(preprocess.state_dict())
        x = torch.randn(2, 4)
        grad = torch.randn(2, 4)
        ref(x).backward(grad.clone())
        executor = _make_executor(preprocess, num_transpose=2, has_trainable=True, owner_backward=True)
        # pylint: disable=protected-access
        retained = executor._connected_forward((x,), {})
        executor._owner_transpose_backward((retained,), [grad.clone()])
        for param, ref_param in zip(preprocess.parameters(), ref.parameters()):
            diff = (param.grad - ref_param.grad).abs().max().item()
            assert torch.allclose(param.grad, ref_param.grad, atol=1e-6), \
                f"owner backward grad mismatch: max abs diff {diff}"

    def test_transpose_forward_retains_graph_for_owner(self):
        """
        Feature: MPipe owner-backward forward.
        Description: Under owner_backward, ``transpose_forward`` runs the connected
            (non-detached) forward and stores the graph-bearing output in
            ``_outputs_for_bwd`` (the owner backward's read site), aliasing the
            same tuple into ``_outputs_for_stage0`` for ``fwd_send`` to ship.
        Expectation: the stored output requires grad and has a grad_fn (the graph
            is kept on the owner), unlike the detached stage-0 backward; both
            dicts hold the same objects (no copy).
        """
        preprocess = torch.nn.Linear(4, 4)
        executor = _make_executor(preprocess, num_transpose=2, has_trainable=True, owner_backward=True)
        # pylint: disable=protected-access
        executor._this_rank = 1  # an owner rank (not stage 0) -> retain, don't place
        ctx = _Ctx(arg_mbs=[(torch.randn(2, 4),)], kwarg_mbs=[{}])
        executor.transpose_forward(_step(0), ctx)
        out = executor._outputs_for_bwd[0]
        assert out[0].requires_grad and out[0].grad_fn is not None
        assert executor._outputs_for_stage0[0] is out, \
            "fwd_send's ship source must alias the retained tuple, not copy it"

    def test_stage0_connected_micro_trains_tower_via_body_backward(self):
        """
        Feature: MPipe owner-backward stage-0 tower gradient.
        Description: On stage 0 (owner of micro 0), ``transpose_forward`` places a
            graph-CONNECTED body input (the owner-branch mirror of the stage-0
            backward's detached-leaf test) and keeps no ``_outputs_for_bwd`` entry;
            the body backward alone must reach the tower params, with no
            ``transpose_backward`` or other explicit tower-backward call.
        Expectation: the ctx body input has a grad_fn; after the body
            forward+backward the tower param-grads equal a plain connected
            reference's.
        """
        torch.manual_seed(0)
        preprocess = torch.nn.Linear(4, 4)
        ref = torch.nn.Linear(4, 4)
        ref.load_state_dict(preprocess.state_dict())
        body = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)
        executor = _make_executor(preprocess, num_transpose=2, has_trainable=True, owner_backward=True)
        # pylint: disable=protected-access
        ctx = _Ctx(arg_mbs=[(x,)], kwarg_mbs=[{}])
        executor.transpose_forward(_step(0), ctx)  # _this_rank == 0, micro 0
        body_input = ctx.arg_mbs[0][0]
        assert body_input.grad_fn is not None, \
            "owner-backward stage-0 body input must stay graph-connected"
        assert 0 not in executor._outputs_for_bwd, \
            "stage 0 must keep no owner-backward entry (its tower backprops via the body graph)"
        body(body_input).pow(2).sum().backward()
        body(ref(x)).pow(2).sum().backward()
        for param, ref_param in zip(preprocess.parameters(), ref.parameters()):
            diff = (param.grad - ref_param.grad).abs().max().item()
            assert torch.allclose(param.grad, ref_param.grad, atol=1e-6), \
                f"stage-0 tower grad via body backward mismatch: max abs diff {diff}"

    @patch.object(mpipe_base, "platform")
    def test_grad_send_ships_feature_grad_and_skips_self(self, mock_plat):
        """
        Feature: MPipe owner-backward feature-grad ship-back.
        Description: ``grad_send`` ships dL/dfeatures for a transposed micro owned
            by another rank, and is a no-op for stage 0's own micro 0 (whose
            connected tower already backpropagated).
        Expectation: meta + deferred isend for micro 1; nothing sent for micro 0.
        """
        mock_plat.get_global_rank.side_effect = lambda group, rank: rank  # identity map
        mock_plat.isend.return_value = "handle"
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2,
                                  has_trainable=True, owner_backward=True)
        # pylint: disable=protected-access
        feat = torch.randn(2, 4, requires_grad=True)
        feat.grad = torch.randn(2, 4)
        executor._keep_grad[1] = (feat,)
        ctx = _Ctx(arg_mbs=[None, None])
        ctx.schedule = _StubSchedule(torch.nn.Linear(4, 4))
        executor.grad_send(_step(1), ctx)
        assert mock_plat.send_object_list.called and len(ctx.schedule._send_handles) == 1
        before = mock_plat.send_object_list.call_count
        executor._keep_grad[0] = (feat,)
        executor.grad_send(_step(0), ctx)  # stage 0 owns micro 0 -> local no-op
        assert mock_plat.send_object_list.call_count == before

    @patch.object(mpipe_base, "platform")
    def test_grad_recv_with_backward_runs_owner_backward(self, mock_plat):
        """
        Feature: MPipe owner-backward feature-grad receive.
        Description: ``grad_recv_with_backward`` receives dL/dfeatures and backprops the retained
            tower graph (``_outputs_for_bwd``), depositing grads on the owner's tower replica.
        Expectation: after grad_recv_with_backward the preprocess params have grads.
        """
        mock_plat.get_global_rank.side_effect = lambda group, rank: rank

        def _recv_meta(meta, src, group=None):  # pylint: disable=unused-argument
            meta[0] = (2, 4)
            meta[1] = torch.float32
        mock_plat.recv_object_list.side_effect = _recv_meta
        mock_plat.empty.return_value = torch.ones(2, 4)
        mock_plat.irecv.return_value = MagicMock()
        preprocess = torch.nn.Linear(4, 4)
        executor = _make_executor(preprocess, num_transpose=2, has_trainable=True, owner_backward=True)
        # pylint: disable=protected-access
        retained = executor._connected_forward((torch.randn(2, 4),), {})
        executor._outputs_for_bwd[1] = (retained,)
        executor.grad_recv_with_backward(_step(1), _Ctx(arg_mbs=[None, None]))
        assert all(p.grad is not None for p in preprocess.parameters())

    @patch.object(mpipe_torch, "platform")
    def test_reduce_tower_grads_allreduces_and_zero_inits(self, mock_plat):
        """
        Feature: MPipe owner-backward tower-grad reduction.
        Description: ``reduce_tower_grads`` SUM-all-reduces each trainable tower
            param-grad over the pp group, zero-initialising params that produced
            none (so the collective is symmetric on every rank).
        Expectation: every trainable param ends with a grad; all_reduce runs once
            per trainable param.
        """
        mock_plat.all_reduce.side_effect = lambda data, group_info: (data, None)
        preprocess = torch.nn.Linear(4, 4)  # weight + bias = 2 trainable params
        executor = _make_executor(preprocess, num_transpose=2, has_trainable=True, owner_backward=True)
        params = list(preprocess.parameters())
        params[0].grad = torch.randn_like(params[0])  # weight has a grad; bias is None
        executor.reduce_tower_grads(_step(0), _Ctx(arg_mbs=[]))
        assert all(p.grad is not None for p in preprocess.parameters())  # bias zero-inited
        assert mock_plat.all_reduce.call_count == 2

    @patch.object(mpipe_torch, "platform")
    def test_reduce_tower_grads_reduces_only_delta_under_accum(self, mock_plat):
        """
        Feature: MPipe owner-backward tower-grad reduction under accumulation.
        Description: With a non-None grad snapshot (the already-reduced grad from a
            prior accumulation pass), reduce_tower_grads must SUM-reduce only this
            run's contribution (grad - snapshot) and add it back -- not re-reduce
            the accumulated total.
        Expectation: with an all_reduce that doubles its input (2-rank SUM of equal
            contributions), the result is snapshot + 2*delta, NOT 2*(snapshot+delta).
        """
        mock_plat.all_reduce.side_effect = lambda data, group_info: (data * 2, None)
        preprocess = torch.nn.Linear(4, 4)
        executor = _make_executor(preprocess, num_transpose=2, has_trainable=True, owner_backward=True)
        # pylint: disable=protected-access
        params = list(preprocess.parameters())
        snapshot = [torch.randn_like(p) for p in params]       # prior (reduced) accumulated grad
        deltas = [torch.randn_like(p) for p in params]         # this run's local contribution
        executor._grad_snapshot = [s.clone() for s in snapshot]
        for param, snap, delta in zip(params, snapshot, deltas):
            param.grad = snap + delta                          # accumulated grad before reduce
        executor.reduce_tower_grads(_step(0), _Ctx(arg_mbs=[]))
        for param, snap, delta in zip(params, snapshot, deltas):
            expected = snap + 2 * delta
            diff = (param.grad - expected).abs().max().item()
            assert torch.allclose(param.grad, expected, atol=1e-5), \
                f"delta-reduce wrong (re-reduced the accumulated total?): max abs diff {diff}"

    @patch.object(mpipe_torch, "platform")
    def test_reduce_grads_dtensor_reduces_local_shard_in_place(self, mock_plat):
        """
        Feature: MPipe owner-backward tower-grad reduction with an FSDP-sharded tower.
        Description: When the tower is FSDP-wrapped ``param.grad`` is a sharded
            DTensor; ``_reduce_grads`` must pp-SUM-reduce the LOCAL shard (never hand
            the DTensor to a raw collective) and write it back in place so the
            DTensor grad wrapper the optimizer reads is preserved.
        Expectation: the pp all_reduce receives a plain (no ``to_local``) tensor;
            ``param.grad`` is still the same DTensor object; its local shard holds
            the reduced value.
        """
        mock_plat.all_reduce.side_effect = lambda data, group_info: (data * 2, None)
        local = torch.randn(4, 4)
        dgrad = _FakeDTensorGrad(local.clone())
        param = _FakeShardedParam(dgrad)
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2,
                                  has_trainable=True, owner_backward=True)
        # pylint: disable=protected-access
        executor._preprocess = _FakeShardedPreprocess([param])
        executor._grad_snapshot = None  # first run
        executor.reduce_tower_grads(_step(0), _Ctx(arg_mbs=[]))
        reduced_arg = mock_plat.all_reduce.call_args[0][0]
        assert not hasattr(reduced_arg, "to_local"), \
            "pp reduce ran on the DTensor, not its local shard"
        assert param.grad is dgrad, "param.grad reassigned -> DTensor wrapper dropped"
        assert torch.allclose(dgrad.to_local(), local * 2, atol=1e-5)

    @patch.object(mpipe_torch, "platform")
    def test_reduce_grads_dtensor_reduces_only_delta_under_accum(self, mock_plat):
        """
        Feature: MPipe owner-backward FSDP tower reduction under grad accumulation.
        Description: With a local-shard snapshot from a prior (already-reduced) pass,
            only this run's contribution (local - snapshot) is pp-reduced and added
            back on the local shard -- the accumulated total is not re-reduced, and
            the DTensor wrapper is preserved.
        Expectation: an all_reduce that doubles its input yields local shard =
            snapshot + 2*delta (not 2*(snapshot+delta)); ``param.grad`` stays the DTensor.
        """
        mock_plat.all_reduce.side_effect = lambda data, group_info: (data * 2, None)
        snap = torch.randn(4, 4)
        delta = torch.randn(4, 4)
        dgrad = _FakeDTensorGrad((snap + delta).clone())  # accumulated local grad
        param = _FakeShardedParam(dgrad)
        executor = _make_executor(torch.nn.Linear(4, 4), num_transpose=2,
                                  has_trainable=True, owner_backward=True)
        # pylint: disable=protected-access
        executor._preprocess = _FakeShardedPreprocess([param])
        executor._grad_snapshot = [snap.clone()]  # prior reduced accumulation (local shard)
        executor.reduce_tower_grads(_step(0), _Ctx(arg_mbs=[]))
        assert param.grad is dgrad
        assert torch.allclose(dgrad.to_local(), snap + 2 * delta, atol=1e-5)


class TestMPipeTransposeOwnerBackwardSchedule(unittest.TestCase):
    """Owner-backward rewrites the MPIPE_* step ordering (static builders)."""

    def test_prefix_drops_graph_send(self):
        """
        Feature: MPipe owner-backward warmup prefix.
        Description: Under owner_backward the owner ships only its features
            (MPIPE_FWD_SEND) and keeps its graph -- no MPIPE_GRAPH_SEND.
        Expectation: an owner rank's prefix has FWD_SEND but not GRAPH_SEND; the
            stage-0 backward still emits GRAPH_SEND.
        """
        owner_types = [s.type for s in
                       ScheduleMPipeTranspose._build_transpose_prefix(
                           1, 4, 4, True, "min", owner_backward=True)]
        assert MpipeStepType.MPIPE_FWD_SEND in owner_types
        assert MpipeStepType.MPIPE_GRAPH_SEND not in owner_types
        stage0_backward_types = [s.type for s in
                           ScheduleMPipeTranspose._build_transpose_prefix(
                               1, 4, 4, True, "min", owner_backward=False)]
        assert MpipeStepType.MPIPE_GRAPH_SEND in stage0_backward_types

    def test_owner_backward_suffix_min(self):
        """
        Feature: MPipe owner-backward cooldown suffix, "min" overflow.
        Description: Under "min" each owner rank owns exactly one micro (== rank),
            so this is the original one-gradient-per-rank exchange. Stage 0 ships
            dL/dfeatures for micros 1..NT-1 (GRAD_SEND); owner ranks 1..NT-1 each
            receive theirs (GRAD_RECV_WITH_BACKWARD); ranks >= NT get none.
            Emitted as a suffix (after all body P2P) so the blocking grad
            transport never deadlocks against the body's send_bwd/recv.
        Expectation: rank 0 -> [GRAD_SEND]*(NT-1) for micros 1..NT-1; rank 1 ->
            [GRAD_RECV_WITH_BACKWARD] for micro 1; rank >= NT -> [].
        """
        stage0 = ScheduleMPipeTranspose._build_owner_backward_suffix(0, 4, 4, "min")
        assert [s.type for s in stage0] == [MpipeStepType.MPIPE_GRAD_SEND] * 3
        assert [s.micro_index for s in stage0] == [1, 2, 3]
        rank1 = ScheduleMPipeTranspose._build_owner_backward_suffix(1, 4, 4, "min")
        assert [s.type for s in rank1] == [MpipeStepType.MPIPE_GRAD_RECV_WITH_BACKWARD]
        assert [s.micro_index for s in rank1] == [1]
        assert ScheduleMPipeTranspose._build_owner_backward_suffix(5, 4, 4, "min") == []

    def test_owner_backward_suffix_full_multi_micro_owner(self):
        """
        Feature: MPipe owner-backward cooldown suffix, "full" (round-robin) overflow.
        Description: At PP=4, M=8, NT=4, rank i owns {m : m % NT == i} -- an owner
            rank can own more than one micro. The suffix must ship/receive a
            gradient for EVERY owned micro, not just the first (the bug this
            fixes: the old single-owner-per-rank suffix silently dropped the
            rest and leaked ``_outputs_for_bwd`` entries).
        Expectation: rank 0 -> GRAD_SEND for every non-rank-0-owned micro
            (1,2,3,5,6,7 -- ascending, so the per-destination order matches each
            owner's receive order); rank 1 -> GRAD_RECV_WITH_BACKWARD for BOTH
            micro 1 and micro 5 (its full owned set, not just micro 1).
        """
        stage0 = ScheduleMPipeTranspose._build_owner_backward_suffix(0, 4, 8, "full")
        assert [s.type for s in stage0] == [MpipeStepType.MPIPE_GRAD_SEND] * 6
        assert [s.micro_index for s in stage0] == [1, 2, 3, 5, 6, 7]
        rank1 = ScheduleMPipeTranspose._build_owner_backward_suffix(1, 4, 8, "full")
        assert [s.type for s in rank1] == [MpipeStepType.MPIPE_GRAD_RECV_WITH_BACKWARD] * 2
        assert [s.micro_index for s in rank1] == [1, 5]
        assert ScheduleMPipeTranspose._build_owner_backward_suffix(5, 4, 8, "full") == []

    def test_insert_no_inline_backward_step_under_owner_backward(self):
        """
        Feature: MPipe owner-backward stage-0 patch.
        Description: Under owner_backward, _insert_rank0_preprocess_steps inserts
            NO per-BWD step (no inline GRAD_SEND, no TRANSPOSE_BWD) -- the grad
            ship-back is a cooldown suffix instead. The stage-0 backward still
            inserts TRANSPOSE_BWD.
        Expectation: owner-backward -> neither GRAD_SEND nor TRANSPOSE_BWD;
            stage-0 backward -> TRANSPOSE_BWD present, GRAD_SEND absent.
        """
        order = [_body_step(MetaStepType.BWD, 0, 0), _body_step(MetaStepType.BWD, 1, 0)]
        owner_types = [s.type for s in ScheduleMPipeTranspose._insert_rank0_preprocess_steps(
            order, 2, True, "min", owner_backward=True)]
        assert MpipeStepType.MPIPE_GRAD_SEND not in owner_types
        assert MpipeStepType.MPIPE_TRANSPOSE_BWD not in owner_types
        stage0_backward_types = [s.type for s in ScheduleMPipeTranspose._insert_rank0_preprocess_steps(
            order, 2, True, "min", owner_backward=False)]
        assert MpipeStepType.MPIPE_TRANSPOSE_BWD in stage0_backward_types
        assert MpipeStepType.MPIPE_GRAD_SEND not in stage0_backward_types


class TestMPipeTransposeStage0BackwardKwargGrad(unittest.TestCase):
    """The stage-0 backward reads dL/dfeatures from the placed feature tensors, so a
    trainable tower trains even when the features are kwarg-routed (VL)."""

    def test_stage0_backward_reads_placed_grad_not_argmbs(self):
        """
        Feature: MPipe stage-0 backward grad source (VL kwarg routing).
        Description: transpose_backward must read dL/dfeatures from self._keep_grad
            (the actual placed feature tensors), which carries the grad whether the
            output was routed into arg_mbs (text) or a kwarg (VL mpipe_visual) --
            NOT from arg_mbs[micro][0], whose grad is None for VL.
        Expectation: the tower receives a gradient even when arg_mbs[micro][0]
            carries no grad (the VL case); the old arg_mbs reader trained nothing.
        """
        preprocess = torch.nn.Linear(4, 4)
        executor = _make_executor(preprocess, num_transpose=2, has_trainable=True)
        # pylint: disable=protected-access
        x = torch.randn(2, 4)
        executor._inputs_for_explicit_forward[0] = (x,)
        # the placed (kwarg-routed) feature: a detached grad-requiring leaf holding
        # the body-deposited dL/dfeatures, exactly as for a VL mpipe_visual tensor.
        feat = preprocess(x).detach().requires_grad_(True)
        feat.grad = torch.randn(2, 4)
        executor._keep_grad[0] = (feat,)
        # arg_mbs[0][0] is a plain non-grad tensor (the VL body text input) -> the
        # old arg_mbs reader saw grad=None and backprop'd nothing.
        executor.transpose_backward(_step(0), _Ctx(arg_mbs=[(torch.randn(2, 4),)]))
        assert all(p.grad is not None for p in preprocess.parameters()), \
            "stage-0 backward must train the tower from the placed feature grad"

    def test_output_consumer_receives_output(self):
        """
        Feature: MPipe output-consumer routing (VL kwarg injection).
        Description: When an output_consumer is set (VL routes the tower output into
            an injection kwarg, not the body input), the executor forward must hand
            the preprocess output to the consumer and leave arg_mbs[micro] as the
            body's own input -- the bug only bites once micro_batch_num > pp.
        Expectation: the consumer receives the output; arg_mbs[micro] is untouched.
            (Asserts routing only -- agnostic to whether the output is detached or
            graph-connected, so it holds across the owner-does-backward change.)
        """
        preprocess = torch.nn.Linear(4, 4)
        captured = {}

        def consumer(ctx: object, micro: int, out: object) -> None:  # pylint: disable=unused-argument
            """VL-style kwarg routing: record what stage 0 was handed."""
            captured[micro] = out

        executor = _make_executor(preprocess, num_transpose=2, has_trainable=True,
                                  output_consumer=consumer, micro_batch_num=4)
        # pylint: disable=protected-access
        body_input = torch.randn(2, 4)
        ctx = _Ctx(arg_mbs=[None, None, (body_input,), None], kwarg_mbs=[{}, {}, {}, {}])
        executor.transpose_forward(_step(2), ctx)
        assert 2 in captured and len(captured[2]) == 1, "consumer must receive the preprocess output"
        assert ctx.arg_mbs[2][0] is body_input, "arg_mbs must keep the body input, not the tower output"

    def test_explicit_backward_tolerates_partial_none_grads(self):
        """
        Feature: MPipe stage-0 backward with a partial-grad multi-output preprocess.
        Description: _explicit_forward_before_backward must backprop only the
            outputs that received a gradient (a None dL/dfeature = zero
            contribution) when some but not all outputs are consumed. Guards the
            torch path; the MindSpore path mirrors it by zero-filling the sens.
        Expectation: the grad-bearing output trains its params; the None-grad
            output contributes nothing (no crash).
        """
        class _TwoOut(torch.nn.Module):
            def __init__(self) -> None:
                """A preprocess returning two independent outputs."""
                super().__init__()
                self.a = torch.nn.Linear(4, 4)
                self.b = torch.nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> tuple:
                """Return (a(x), b(x)) so the two outputs have disjoint params."""
                return self.a(x), self.b(x)

        preprocess = _TwoOut()
        executor = _make_executor(preprocess, num_transpose=1, has_trainable=True)
        # pylint: disable=protected-access
        executor._explicit_forward_before_backward((torch.randn(2, 4),), {},
                                                   [torch.randn(2, 4), None])  # 2nd output: no grad
        assert preprocess.a.weight.grad is not None, "grad-bearing output must train its params"
        assert preprocess.b.weight.grad is None, "the None-grad output must contribute nothing"


class TestBroadcastReshardsTower(unittest.TestCase):
    """``_broadcast_tensors`` reshards the tower to its canonical sharded state
    before yielding, so every pp rank broadcasts the same-sized local shard.

    Regression for the p3f hang: under owner-backward (reshard_after_forward=
    False) stage 0 held the tower unsharded (full) while the replicas were
    sharded, so the pp-group broadcast of ``_local(param.data)`` got mismatched
    sizes and deadlocked (surfacing on NPU as HcclBroadcast ERR00100).
    """

    def test_reshard_called_on_hsdp_submodules_before_yield(self):
        """The broadcast must reshard HSDP submodules before yielding shards."""
        class _FSDPLeaf(torch.nn.Module):
            """A trainable leaf exposing a reshard() hook like an HSDPModule."""

            def __init__(self) -> None:
                """Build the leaf with a reshard counter."""
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                self.reshard_calls = 0

            def reshard(self) -> None:
                """Count a reshard call."""
                self.reshard_calls += 1

        class _Tower(torch.nn.Module):
            def __init__(self) -> None:
                """Build a tower of two resharding leaves."""
                super().__init__()
                self.blocks = torch.nn.ModuleList([_FSDPLeaf(), _FSDPLeaf()])

        tower = _Tower()
        executor = _make_executor(tower, num_transpose=2, has_trainable=True)
        # pylint: disable=protected-access
        tensors = list(executor._broadcast_tensors())
        assert [b.reshard_calls for b in tower.blocks] == [1, 1], \
            "every reshard-bearing submodule must be resharded before broadcast"
        # Still yields exactly the trainable params (weight + bias per leaf).
        assert len(tensors) == 4


class TestMPipeBackwardRetainFlag(unittest.TestCase):
    """A trainable transposed tower flags its stages to retain the body-backward
    graph, so the shared FSDP all-gather node survives a later backward (the
    ``vpp >= 2`` "backward through the graph a second time" fix)."""

    @staticmethod
    def _schedule(has_trainable, nstages=4):
        sched = object.__new__(ScheduleMPipeTranspose)
        # pylint: disable=protected-access
        sched._has_trainable_preprocess = has_trainable
        sched.stages = [type("S", (), {})() for _ in range(nstages)]
        return sched

    def test_trainable_tower_flags_all_stages(self):
        """A trainable tower marks every stage to retain its backward graph."""
        sched = self._schedule(has_trainable=True)
        sched._apply_backward_retain_flag()  # pylint: disable=protected-access
        assert all(getattr(s, "retain_backward_graph", False) for s in sched.stages), \
            "a trainable transposed tower must flag every stage to retain its body graph"

    def test_frozen_tower_leaves_stages_unflagged(self):
        """A frozen tower needs no retained graph, so no stage is flagged."""
        sched = self._schedule(has_trainable=False)
        sched._apply_backward_retain_flag()  # pylint: disable=protected-access
        assert all(not getattr(s, "retain_backward_graph", False) for s in sched.stages), \
            "a frozen tower must not force body-graph retention (normal PP behavior)"


if __name__ == "__main__":
    unittest.main()
