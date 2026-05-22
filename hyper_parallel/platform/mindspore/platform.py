# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""MindSpore platform api"""
from datetime import timedelta
from typing import Any, Optional, Union
import dataclasses
from collections import OrderedDict

import contextlib
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.mint.distributed import TCPStore

from mindspore.nn import Cell
from mindspore import mint
from mindspore.common.api import _no_grad
from mindspore.common._grad_function import _Function
from mindspore.common.dtype import type_size_in_bytes
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.recompute import null_context_fn
from mindspore.communication import GlobalComm
from mindspore.communication import get_group_size
from mindspore.communication import create_group as new_group
from mindspore.communication import get_rank as get_rank_id
from mindspore.ops import communication as ops_comm
from mindspore.ops.function import comm_func
# Private MindSpore symbols used by ``_MSAsyncA2ALazyBwd._issue_async_a2a`` to
# bypass the trailing reshape that ``comm_func.all_to_all_single`` performs on
# the default compute stream before the async ``CommHandle.wait()`` fires —
# see that helper's docstring for the full rationale.  If a future MindSpore
# release moves or renames either symbol, this module will fail to import
# loudly (intended — silently falling back to ``comm_func.all_to_all_single``
# would re-introduce the race).
from mindspore.ops.function.comm_func import _deal_comm_outputs
from mindspore.ops.auto_generate.gen_ops_prim import inner_comm_all_to_all_v_op
from mindspore._c_expression import TensorTransform
import mindspore.mint.distributed as dist

from hyper_parallel.platform.platform import Platform, PlatformType, EXISTING_COMM_GROUPS
from hyper_parallel.platform.mindspore.dtensor import DTensorBase
from hyper_parallel.platform.mindspore.pipeline_parallel.stage import PipelineStageBase
from hyper_parallel.platform.mindspore.parameter_init import init_parameters as _init_parameters
from hyper_parallel.platform.mindspore.init_weights import (
    init_on_device as _init_on_device,
    _install_cell_to_empty_patch,
)

comm_func.set_comm_ops_inplace(False)
_tensor_transform = TensorTransform.get_instance()


# pylint: disable=C0103


def _a2a_reconstruct_ms(out_perm: Tensor, concat_dim: int) -> Tensor:
    """Reconstruct A2A result from raw out_perm buffer."""
    new_ndim = out_perm.dim()
    chunk_in_perm = concat_dim + 1
    recon_perm = list(range(1, chunk_in_perm)) + [0] + list(range(chunk_in_perm, new_ndim))
    x_recon = out_perm.permute(recon_perm).contiguous()
    shape = list(x_recon.shape)
    merged = shape[concat_dim] * shape[concat_dim + 1]
    return x_recon.reshape(shape[:concat_dim] + [merged] + shape[concat_dim + 2:])


def _normalize_all_to_all_single_result(result, output: Tensor) -> tuple[Tensor, object]:
    """Normalize MindSpore all_to_all_single return values to ``(output, handle)``."""
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                "mindspore all_to_all_single returned an unexpected tuple "
                f"with length {len(result)}"
            )
        return result
    return output, result


def _mindspore_all_to_all_single(input_tensor: Tensor, output_shape, group, async_op=False) -> tuple[Tensor, object]:
    """Launch MindSpore all_to_all_single and normalize return values."""
    output = mint.empty(tuple(output_shape), dtype=input_tensor.dtype)
    result = ops_comm.all_to_all_single(output, input_tensor, group=group, async_op=async_op)
    normalized_output, handle = _normalize_all_to_all_single_result(result, output)
    if not async_op:
        return normalized_output, None
    return normalized_output, handle


class AsyncCollectiveTensor(Tensor):
    """MindSpore Tensor subclass that defers ``CommHandle.wait()`` to
    the first op that reads it.

    Mimics PyTorch's ``AsyncCollectiveTensor`` using MindSpore's
    per-tensor ``__ms_dispatch__`` mechanism.  Constructed by calling
    ``AsyncCollectiveTensor(inner_tensor, work)`` — :meth:`__new__`
    invokes ``Tensor._make_subclass`` which (per MindSpore C++ side)
    sets ``has_ms_dispatch=true`` on the new tensor because this class
    defines ``__ms_dispatch__``.  All subsequent ops involving this
    tensor are routed through that callback.

    Stream-side ``CommHandle.wait()`` (host non-blocking) means the
    overlap window between the async a2a issue and the first consumer
    op is preserved: the wait is only inserted on the consumer stream
    at the consumer dispatch site, not at the a2a issue site.

    Note:
        Currently every op (including view ops like reshape /
        transpose / permute) triggers ``work.wait()`` + unwrap.
        Once MindSpore exposes schema alias annotations on
        :class:`OpFunc` (planned per discussion with the MS team),
        this class can mirror PyTorch's ``_is_view_op`` to keep
        view chains lazy and stretch the overlap window further.

    Attributes:
        elem:           The underlying regular Tensor (PyTorch's
                        ``AsyncCollectiveTensor.elem``).  Returned by
                        :meth:`_wait_and_unwrap` after the wait fires
                        so downstream ops see a plain Tensor type.
        completed:      Whether ``work.wait()`` has already been
                        triggered (idempotency guard).
        _pending_work:  The async ``CommHandle`` returned by MindSpore.
                        PyTorch's equivalent class doesn't carry this
                        because PyTorch tracks tensor→work via the
                        global ``wait_tensor()`` aten op + c10d
                        registry.  MindSpore has no such infra, so we
                        have to stash the handle on the wrapper itself.
    """

    __slots__ = ("elem", "completed", "_pending_work")

    @staticmethod
    def __new__(cls, inner: Tensor, work):  # pylint: disable=W0613
        """Construct a wrapper tensor sharing storage with ``inner``.

        ``Tensor._make_subclass`` returns a tensor of class ``cls``
        that shares storage with ``inner``.  MindSpore C++ side then
        sets ``has_ms_dispatch=true`` because ``cls`` defines
        ``__ms_dispatch__``.  Per-instance state is set in
        :meth:`__init__`.
        """
        return Tensor._make_subclass(cls, inner)  # pylint: disable=W0212

    def __init__(self, inner: Tensor, work):  # pylint: disable=W0231
        """Initialize wrapper state (does NOT call ``super().__init__``).

        Skipping ``Tensor.__init__`` is intentional: the parent
        constructor would re-interpret ``inner`` as raw input data
        and ``work`` as a dtype, corrupting the tensor that
        :meth:`__new__` already built via ``Tensor._make_subclass``.
        """
        self.elem = inner
        self.completed = work is None
        self._pending_work = work

    def _wait_and_unwrap(self) -> Tensor:
        """Trigger ``work.wait()`` (idempotent) and return ``elem``.

        Mirrors PyTorch's ``trigger_wait``: returns the underlying
        regular Tensor so downstream ops see a plain ``Tensor``
        instance, not an ``AsyncCollectiveTensor`` (avoids re-entering
        ``__ms_dispatch__`` on every subsequent op).
        """
        if not self.completed:
            work = self._pending_work
            if work is not None:
                work.wait()  # stream-side: inserts streamWaitEvent on current stream
            self.completed = True
        return self.elem

    @classmethod
    def __ms_dispatch__(cls, func, args, kwargs=None):
        """Per-tensor dispatch callback invoked for every op touching a
        :class:`AsyncCollectiveTensor` instance.

        Must be a ``@classmethod`` so MindSpore's C++-side invocation
        (``tensor_py_reg.cc`` retrieves the attribute from the class
        and calls it as ``handler(op_func, packed_args, kwargs)`` —
        three positional args, no ``self`` binding) lines up with the
        signature ``(cls, func, args, kwargs)``.  Mirrors PyTorch's
        ``__torch_dispatch__`` decoration on ``AsyncCollectiveTensor``.

        Currently every op triggers wait + unwrap on any
        ``AsyncCollectiveTensor`` arg, then runs the op on the
        underlying inner tensors.  This is the conservative
        correctness-first behavior: it always defers the wait at
        least until the first op consumes the tensor (which is later
        than calling ``work.wait()`` immediately at a2a issue site,
        so the overlap window is preserved across the
        ``sync_hook("B")`` window).

        TODO: when MindSpore exposes schema alias annotations on
        ``func`` (the ``OpFunc`` parameter), add a fast path that
        keeps view ops (reshape / transpose / permute / etc.) lazy
        and only triggers wait on real data-touching ops, mirroring
        PyTorch's ``_is_view_op`` in
        ``torch/distributed/_functional_collectives.py``.  Until that
        annotation is available, treating views as real ops just
        shortens the overlap window for view-heavy paths — it does
        not affect correctness.
        """
        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}
        unwrapped_args = tuple(
            a._wait_and_unwrap() if isinstance(a, cls) else a  # pylint: disable=W0212
            for a in args
        )
        unwrapped_kwargs = {
            k: (v._wait_and_unwrap() if isinstance(v, cls) else v)  # pylint: disable=W0212
            for k, v in kwargs.items()
        }
        return func(*unwrapped_args, **unwrapped_kwargs)

    # ------------------------------------------------------------------
    # Data-export overrides
    # ------------------------------------------------------------------
    # The methods below all read raw tensor data (or print it) and
    # bypass ``__ms_dispatch__`` because they are Python-level methods
    # on ``Tensor``, not MindSpore ops.  Without these overrides they
    # would access ``self``'s data buffer before the pending async a2a
    # has finished, returning stale / uninitialized values.  Each
    # override forces a stream-side wait via ``_wait_and_unwrap`` and
    # delegates to the same method on the underlying inner tensor.
    #
    # Methods deliberately NOT overridden:
    #   ``__len__``       — metadata only (returns shape[0]); no data read.
    #   ``__hash__``      — id-based on MindSpore Tensor; no data read.
    #   ``__contains__``  — uses ``(elem == self).any().item()`` which
    #                       dispatches through ``==`` so wait fires
    #                       transitively before the chain reaches data.
    #   ``__getitem__``   — slicing dispatches through ``__ms_dispatch__``.
    #   ``__format__``    — calls ``__repr__`` which we override.

    def asnumpy(self):
        """Convert to numpy ndarray; waits the pending a2a first."""
        return self._wait_and_unwrap().asnumpy()

    def numpy(self):
        """Alias of :meth:`asnumpy` — same wait + unwrap path."""
        return self._wait_and_unwrap().numpy()

    def __array__(self, dtype=None):
        """``np.array(t)`` protocol; waits + delegates to inner tensor."""
        return self._wait_and_unwrap().__array__(dtype)

    def get_bytes(self):
        """Raw byte serialization; must wait before reading the buffer."""
        return self._wait_and_unwrap().get_bytes()

    def tolist(self):
        """Convert to nested Python list; waits first."""
        return self._wait_and_unwrap().tolist()

    def item(self):
        """Extract scalar value (0-d tensor); waits first."""
        return self._wait_and_unwrap().item()

    def __bool__(self):
        """``bool(t)`` / ``if t:``; reads scalar value, must wait."""
        return bool(self._wait_and_unwrap())

    def __int__(self):
        """``int(t)``; reads scalar value, must wait."""
        return int(self._wait_and_unwrap())

    def __float__(self):
        """``float(t)``; reads scalar value, must wait."""
        return float(self._wait_and_unwrap())

    def __index__(self):
        """Python index protocol; uses scalar value, must wait."""
        return self._wait_and_unwrap().__index__()

    def __repr__(self):
        """Eager debug print; force wait so the printout reflects real data.

        Mirrors PyTorch's ``AsyncCollectiveTensor.__repr__`` style by
        labelling the wrapper so a stray ``print(t)`` doesn't silently
        hide the lazy nature of the value.
        """
        return f"AsyncCollectiveTensor({self._wait_and_unwrap()})"

    def __str__(self):
        """``str(t)`` / format printing; falls through to :meth:`__repr__`."""
        return self.__repr__()

    def __iter__(self):
        """Iterate over dim-0 slices; one wait, then iterate inner."""
        return iter(self._wait_and_unwrap())


class _MSAsyncA2ALazyBwd(_Function):
    """Async all-to-all whose forward and backward both return
    :class:`AsyncCollectiveTensor`, deferring ``CommHandle.wait()``
    to the first consumer op via ``__ms_dispatch__``.

    Mirrors the Torch ``_AsyncA2ALazyBwd`` semantics: the kernel is
    queued on the HCCL group's stream, host returns immediately, and
    the wait fires lazily on the consumer's stream — giving the
    paired thread a window to dispatch its compute concurrently.
    """

    @staticmethod
    def _issue_async_a2a(flat_input, send_splits, recv_splits, group):
        """Issue an async all-to-all-v on a 1-D flat tensor.

        Bypasses ``comm_func.all_to_all_single``: that wrapper appends an
        unconditional ``result.reshape((-1,) + recv_shape_without_first_dim)``
        on the default compute stream *before* the async ``CommHandle.wait()``
        fires (the wait is deferred to the first consumer op via
        :class:`AsyncCollectiveTensor`).  MindSpore's mem_pool race_checker
        (``MS_ALLOC_CONF=memory_tracker:True``) flags that trailing reshape
        as a cross-stream race on the HCCL output, even though for 1-D
        inputs it is a metadata-only no-op.  Calling the inner primitive
        directly skips the tracker-visible read on stream 0.

        Args:
            flat_input:  1-D tensor — must already be flattened by the caller.
            send_splits: ``list[int]`` — element counts sent to each rank.
            recv_splits: ``list[int]`` — element counts received from each rank.
            group:       Process group.

        Returns:
            ``(output_tensor, CommHandle)`` — the 1-D output and the async handle.
        """
        rank_size = get_group_size(group)
        # Positional args follow the MS auto-generated primitive signature:
        # ``(input, group, send_splits, recv_splits, rank_size, block)``.
        # ``block=False`` selects the async path; the handle is returned in
        # the raw tuple and unpacked by ``_deal_comm_outputs`` below.
        raw = inner_comm_all_to_all_v_op(
            flat_input, group, list(send_splits), list(recv_splits), rank_size,
            False,
        )
        # ``_deal_comm_outputs(raw, is_async=True)`` mirrors the async branch
        # inside ``comm_func.all_to_all_single`` — unpacks the primitive's raw
        # output into ``(tensor, handle)`` without the trailing reshape.
        return _deal_comm_outputs(raw, True)

    @staticmethod
    def forward(ctx, input_tensor, output_splits, input_splits, group):  # pylint: disable=arguments-differ
        """Launch async a2a; return :class:`AsyncCollectiveTensor`.

        ``input_tensor`` must already be 1-D and the splits must be element
        counts (not row counts).  The caller is expected to flatten and
        translate splits beforehand — see
        :meth:`MindSporePlatform.differentiable_all_to_all_single_async`.
        """
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        flat_input = input_tensor.reshape(-1)
        actual_output, work = _MSAsyncA2ALazyBwd._issue_async_a2a(
            flat_input, input_splits, output_splits, group,
        )
        return AsyncCollectiveTensor(actual_output, work)

    @staticmethod
    def backward(ctx, grad_output):
        """Symmetric reverse a2a; returns :class:`AsyncCollectiveTensor`."""
        # If grad_output is still lazy, force unwrap before issuing the
        # reverse a2a (which is itself a "real" op on the data).
        if isinstance(grad_output, AsyncCollectiveTensor):
            grad_output = grad_output._wait_and_unwrap()  # pylint: disable=W0212
        flat_grad = grad_output.reshape(-1)
        actual_grad, work = _MSAsyncA2ALazyBwd._issue_async_a2a(
            flat_grad, ctx.output_splits, ctx.input_splits, ctx.group,
        )
        lazy_grad = AsyncCollectiveTensor(actual_grad, work)
        return lazy_grad, None, None, None


class _MSSyncHookFunction(_Function):
    """Identity autograd op that fires HookCoordinator rendezvous on
    forward and backward, mirroring the Torch ``_TorchSyncHookFunction``.

    The role tables are intentionally identical to the Torch backend so
    the dual-thread protocol (COMM-first dispatch ordering) is the same
    on MindSpore.

    Hook-name semantics:

    - ``"A"`` / ``"B"`` / ``"C"`` / ``"D"`` — full rendezvous on both
      forward and backward, using ``_FWD_ROLES`` / ``_BWD_ROLES``.
    - ``"CHUNK_START"`` — pair-0 entry hook.
      **Forward**: full rendezvous(COMPUTE) — pairs with
      ``D_LAST.bwd`` so the BWD thread's combine.bwd of the last
      layer is bracketed by a barrier-synced window.
      **Backward**: paired with ``CHUNK_END.fwd`` as the BWD-side of
      the exit barrier (roles ``(COMPUTE, COMPUTE)``).
    - ``"D_LAST"`` — closing D hook of the last MoE layer in a chunk.
      **Forward**: **pure skip** — neither notify nor rendezvous.
      The C_last → combine COMM event is left un-notified so BWD's
      COMPUTE waiter at ``A_0.bwd`` stays parked.  This keeps FWD's
      post-combine forward work serialised against BWD's Attn.bwd_0;
      required because MS PyNative does not support concurrent
      FWD-record + BWD-replay on its autograd executor.  (The Torch
      backend takes the looser ``notify(COMM) + skip`` path here for
      more overlap — Torch autograd is thread-safe.)
      **Backward**: full rendezvous using ``_BWD_ROLES["D"]``; this
      is the very first BWD rendezvous and pairs with
      ``CHUNK_START.fwd`` to bracket combine.bwd_last.
    - ``"CHUNK_END"`` — pair-N exit hook (FWD side).
      **Forward**: roles ``(COMM, COMPUTE)``.  ``notify_dispatched``
      sets the C_last event (waking BWD's A_0.bwd waiter), then
      ``rendezvous(COMPUTE)`` parks FWD on the exit barrier so BWD's
      Attn.bwd_0 runs with FWD already blocked — no concurrent
      FWD-record + BWD-replay.
      **Backward**: skipped (this would be the first node visited
      in BWD replay; its partner ``D_LAST.bwd`` already pairs with
      ``CHUNK_START.fwd`` on pair 0).
    """

    # Index encoding: 1 = COMM, 2 = COMPUTE.
    _FWD_ROLES = {
        # ``CHUNK_START``: chunk entry on FWD.  No "previous" op on
        # this thread within this overlap.run() — ``notify(COMPUTE)``
        # is a no-op anyway.  Next role is COMPUTE so FWD parks on
        # ``_comm_dispatched.wait`` for BWD's ``D_LAST.bwd`` COMM.
        "CHUNK_START": (2, 2),
        "A": (2, 1),   # prev=Attention COMPUTE | next=dispatch COMM
        "B": (1, 2),   # prev=dispatch COMM     | next=module COMPUTE
        "C": (2, 1),   # prev=module COMPUTE    | next=combine COMM
        "D": (1, 2),   # prev=combine COMM      | next=Attention COMPUTE
        # ``CHUNK_END``: chunk-exit hook on FWD.  Does two things in
        # one place — both critical for MS PyNative correctness:
        #   1. ``notify_dispatched(COMM)`` sets the C_last event from
        #      C_last's rendezvous(COMM).  ``D_LAST.fwd`` deliberately
        #      does NOT notify (it is a pure skip) so BWD's COMPUTE
        #      waiter at ``A_0.bwd`` stays parked until FWD has
        #      finished all chunk-local forward work (post-combine
        #      sort/index_select/multiply).
        #   2. ``rendezvous(COMPUTE)`` parks FWD on the exit barrier.
        #      By the time BWD wakes from step 1 and starts
        #      Attn.bwd_0, FWD is already blocked at this barrier —
        #      no concurrent FWD-record + BWD-replay window.
        "CHUNK_END": (1, 2),
    }
    _BWD_ROLES = {
        # ``CHUNK_START.bwd`` is intentionally NOT engaged here.
        # MS PyNative's autograd may skip the backward node if the
        # chunk input lacks ``requires_grad`` (the value of
        # ``x.grad`` is unused downstream), which would leave the
        # pair-8 BWD partner unmatched and deadlock FWD's
        # ``CHUNK_END`` barrier.  pair-8 BWD is instead taken out of
        # band: the OVERLAP_B_F callback's ``bwd_fn`` makes one
        # explicit ``coordinator.rendezvous(COMPUTE)`` after
        # ``backward_one_chunk`` returns, paired with FWD's
        # ``CHUNK_END.fwd`` rendezvous.
        # ``D_LAST`` on backward routes through D's BWD role (COMM
        # next: the upcoming combine.bwd) — see the docstring above
        # for why we no longer skip.
        "D": (2, 1),   # prev=Attn.bwd COMPUTE      | next=combine.bwd COMM
        "C": (1, 2),   # prev=combine.bwd COMM      | next=module.bwd COMPUTE
        "B": (2, 1),   # prev=module.bwd COMPUTE    | next=dispatch.bwd COMM
        "A": (1, 2),   # prev=dispatch.bwd COMM     | next=Attn.bwd COMPUTE
    }
    _ROLE_CACHE = None

    @staticmethod
    def _role_enum(idx: int):
        """Lazy import of HookRole to avoid a circular import at module load."""
        if _MSSyncHookFunction._ROLE_CACHE is None:
            # pylint: disable=C0415
            from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookRole
            _MSSyncHookFunction._ROLE_CACHE = (None, HookRole.COMM, HookRole.COMPUTE)
        return _MSSyncHookFunction._ROLE_CACHE[idx]

    @staticmethod
    def _passthrough(x):
        """Identity passthrough that defeats MS autograd's identity-output handling.

        When :meth:`forward` returns its input unchanged, MS PyNative's
        ``FunctionBase.apply`` sees ``is_same_as_input=True`` on the output
        and inserts a ``ViewAsSelfWithNoGrad`` (a ``view(self, self.shape)``
        kernel) on the current compute stream.  If the input is an
        :class:`AsyncCollectiveTensor` whose lazy ``CommHandle.wait()`` has
        not yet fired, that view runs on the default stream while the HCCL
        kernel is still writing the same memory on the comm stream — flagged
        by MS's mem_pool ``race_checker`` (``MS_ALLOC_CONF=memory_tracker:True``).

        Returning a freshly wrapped :class:`AsyncCollectiveTensor` keeps the
        same underlying buffer and pending work, but yields a new
        ``shared_ptr<Tensor>`` so ``is_same_as_input`` is ``False`` and no
        autograd view is emitted.  For regular tensors the original
        passthrough is safe (the view sits on the same stream as the data).

        Note:
            The clone shares ``_pending_work`` with the original but keeps
            an independent ``completed`` flag.  Two assumptions:

            * ``CommHandle.wait()`` is idempotent — relied on whenever both
              wrappers end up being consumed (matches the existing
              :meth:`AsyncCollectiveTensor._wait_and_unwrap` pattern, which
              also does not null out ``_pending_work`` after waiting).
            * Per-wrapper ``completed`` is intentional: a ``wait()`` on
              stream A does not synchronize stream B, so each consumer
              stream must be free to re-issue its own wait.
        """
        if isinstance(x, AsyncCollectiveTensor):
            new_wrapper = AsyncCollectiveTensor(x.elem, x._pending_work)  # pylint: disable=W0212
            new_wrapper.completed = x.completed
            return new_wrapper
        return x

    @staticmethod
    def forward(ctx, x, hook_name, coordinator):  # pylint: disable=arguments-differ
        """Fire forward-direction rendezvous and return ``x`` unchanged."""
        ctx.hook_name = hook_name
        ctx.coordinator = coordinator
        if not coordinator.is_enabled():
            return _MSSyncHookFunction._passthrough(x)
        if hook_name == "D_LAST":
            # Pure skip — neither notify nor rendezvous.  The
            # C_last → combine COMM event is left un-notified on
            # purpose so BWD's COMPUTE waiter at A_0.bwd stays parked
            # until FWD reaches CHUNK_END.fwd.  This keeps FWD's
            # post-combine forward work (sort / index_select / probs
            # mul / strided_slice) strictly serialised against BWD's
            # Attn.bwd_0 — required because MS PyNative does not
            # support concurrent FWD-record + BWD-replay on the
            # autograd executor.
            return _MSSyncHookFunction._passthrough(x)
        prev_idx, next_idx = _MSSyncHookFunction._FWD_ROLES[hook_name]
        role_of = _MSSyncHookFunction._role_enum
        coordinator.notify_dispatched(role_of(prev_idx))
        coordinator.rendezvous(role_of(next_idx))
        return _MSSyncHookFunction._passthrough(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Mirror of :meth:`forward` using ``_BWD_ROLES``."""
        hook_name = ctx.hook_name
        coordinator = ctx.coordinator
        if not coordinator.is_enabled():
            return _MSSyncHookFunction._passthrough(grad_output), None, None
        if hook_name in ("CHUNK_END", "CHUNK_START"):
            # Both boundary hooks skip in backward:
            # * ``CHUNK_END.bwd`` would fire FIRST in BWD replay (it
            #   wraps the chunk's last forward op).  We do not want
            #   a rendezvous here — pair 0 is handled by
            #   ``D_LAST.bwd`` ↔ ``CHUNK_START.fwd``.
            # * ``CHUNK_START.bwd`` would fire LAST.  We do not
            #   rendezvous here either, because MS autograd may skip
            #   the node entirely when the chunk input lacks
            #   ``requires_grad`` (unused ``x.grad``).  pair-8 BWD
            #   is taken out of band — see the role-table comment.
            return _MSSyncHookFunction._passthrough(grad_output), None, None
        # ``D_LAST.bwd`` reuses D's BWD role: it is the *first non-skip*
        # BWD rendezvous and pairs with FWD's ``CHUNK_START`` to lock
        # the combine.bwd_last launch inside a barrier-synced window.
        role_name = "D" if hook_name == "D_LAST" else hook_name
        prev_idx, next_idx = _MSSyncHookFunction._BWD_ROLES[role_name]
        role_of = _MSSyncHookFunction._role_enum
        coordinator.notify_dispatched(role_of(prev_idx))
        coordinator.rendezvous(role_of(next_idx))
        return _MSSyncHookFunction._passthrough(grad_output), None, None


class _MSAsyncA2AFunction(_Function):
    """Differentiable wrapper for pre-launched async all-to-all."""

    @staticmethod
    def forward(ctx, x, work, out_perm, group, world_size, concat_dim, split_dim, handle_box):  # pylint: disable=arguments-differ
        """Wait for pre-launched async A2A and return reconstructed output."""
        ctx.group = group
        ctx.world_size = world_size
        ctx.concat_dim = concat_dim
        ctx.split_dim = split_dim
        ctx.handle_box = handle_box
        ctx.x_shape = tuple(x.shape)
        work.wait()
        return _a2a_reconstruct_ms(out_perm, concat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Launch async head->seq A2A for backward overlap, or return zero grad."""
        if ctx.handle_box is not None:
            g = grad_output.contiguous()
            shape = list(g.shape)
            seq_dim = ctx.concat_dim
            s_full = shape[seq_dim]
            ndim = len(shape) + 1
            x_perm = g.reshape(
                shape[:seq_dim] + [ctx.world_size, s_full // ctx.world_size] + shape[seq_dim + 1:]
            ).permute(
                [seq_dim] + list(range(seq_dim)) + list(range(seq_dim + 1, ndim))
            ).contiguous()
            out_perm, work = _mindspore_all_to_all_single(
                x_perm,
                list(x_perm.shape),
                ctx.group,
                async_op=True,
            )
            ctx.handle_box.append((work, out_perm))
        return mint.zeros(ctx.x_shape, dtype=grad_output.dtype), None, None, None, None, None, None, None


class MindSporePlatform(Platform):
    """MindSpore platform api"""
    Tensor = Tensor
    tensor = Tensor
    Parameter = Parameter
    Module = Cell
    DTensorBase = DTensorBase
    PipelineStageBase = PipelineStageBase
    platform_type = PlatformType.MINDSPORE
    tensor_dtype = mstype
    dtype = ms.Type
    Function = _Function

    _custom_ops_cls = None

    @property
    def custom_ops(self):
        """Return the MindSpore platform custom ops instance.

        .. warning::
            This is an experimental API that subject to change or deletion.

        Returns:
            MindSporeCustomOps: Custom ops class that delegates to DFunction
            implementations wrapping Ascend NPU custom C++ kernels.
        """
        if self._custom_ops_cls is None:
            from hyper_parallel.platform.mindspore.custom_ops.custom_ops import (  # pylint: disable=import-outside-toplevel
                MindSporeCustomOps,
            )
            self._custom_ops_cls = MindSporeCustomOps
        return self._custom_ops_cls

    def __init__(self):
        # Ensure MindSpore ``nn.Cell.to_empty`` is patched as soon as the
        # MindSpore platform instance is created.
        _install_cell_to_empty_patch()

    @staticmethod
    def is_linear_module(module) -> bool:
        """Check whether *module* is a MindSpore ``Dense`` (linear) or ``mint.nn.Linear`` layer."""
        return isinstance(module, (ms.nn.Dense, mint.nn.Linear))

    @staticmethod
    def is_embedding_module(module) -> bool:
        """Check whether *module* is a MindSpore ``Embedding`` or ``mint.nn.Embedding`` layer."""
        return isinstance(module, (ms.nn.Embedding, mint.nn.Embedding))

    def device_count(self, device_handle):
        """
        Get the number of available devices.

        Args:
            device_handle: The device handle (e.g., ms.device_context).

        Returns:
            int: The number of available devices.
        """
        device_type = self.device_type()
        if device_type == "cpu":
            return device_handle.device_context.cpu.device_count()
        if device_type == "gpu":
            return device_handle.device_context.gpu.device_count()
        return device_handle.device_context.ascend.device_count()

    @staticmethod
    def get_rng_state(device=None, device_handle=None):
        """
        Get the random number generator state.

        Args:
            device (Optional): The device to get RNG state from (not used in MindSpore).
            device_handle (Optional): The device handle (not used in MindSpore).

        Returns:
            Tensor: The RNG state as a tensor.
        """
        _ = device, device_handle
        return ms.get_rng_state()

    @staticmethod
    def set_rng_state(state, device=None, device_handle=None):
        """
        Set the random number generator state.

        Args:
            state (Tensor): The RNG state to set.
            device (Optional): The device to set RNG state for (not used in MindSpore).
            device_handle (Optional): The device handle (not used in MindSpore).
        """
        _ = device, device_handle
        return ms.set_rng_state(state)

    def device_type(self):
        """
        Get the current device type.

        Returns:
            str: The device type string ("npu" for Ascend, "gpu" for GPU, "cpu" for CPU).
        """
        device_type = ms.get_context("device_target")
        if device_type == "Ascend":
            return "npu"
        return device_type.lower()

    def device(self, device_idx=None):
        """
        Get the device type string.

        Args:
            device_idx (Optional[int]): The device index (not used in MindSpore).

        Returns:
            str: The device type string.
        """
        _ = device_idx
        device_type = self.device_type()
        return device_type

    @staticmethod
    def get_device_handle():
        """
        Get the MindSpore module as the device handle.

        Returns:
            module: The mindspore module.
        """
        return ms

    @staticmethod
    def manual_seed(seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The random seed value.

        Returns:
            None
        """
        return ms.manual_seed(seed)

    @staticmethod
    def ones(size, dtype=None):
        """
        Create a tensor filled with ones.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[ms.Type]): The desired data type.

        Returns:
            Tensor: A tensor filled with ones.
        """
        return mint.ones(size, dtype=dtype)

    @staticmethod
    def zeros(size, dtype=None, device=None):
        """
        Create a tensor filled with zeros.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[ms.Type]): The desired data type.
            device (Optional[ms.device]): The device to create the tensor on.

        Returns:
            Tensor: A tensor filled with zeros.
        """
        tensor = mint.zeros(size, dtype=dtype)
        if device in ("GPU", "Ascend"):
            return tensor.to(device)
        return tensor

    @staticmethod
    def full(size, fill_value, dtype=None):
        """
        Create a tensor filled with a scalar value.

        Args:
            size (tuple): The shape of the output tensor.
            fill_value (scalar): The value to fill the tensor with.
            dtype (Optional[ms.Type]): The desired data type.

        Returns:
            Tensor: A tensor filled with the specified value.
        """
        return mint.full(size, fill_value, dtype=dtype)

    @staticmethod
    def empty(size, dtype=None):
        """
        Create an uninitialized tensor.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[ms.Type]): The desired data type.

        Returns:
            Tensor: An uninitialized tensor.
        """
        return mint.empty(size, dtype=dtype)

    @staticmethod
    def get_rank():
        """
        Get the rank of the current process in the distributed group.

        Returns:
            int: The rank of the current process.
        """
        return get_rank_id()

    @staticmethod
    def get_global_rank(group, group_rank):
        """
        Get the global rank from a group rank.

        Args:
            group (str): The process group name.
            group_rank (int): The rank within the group.

        Returns:
            int: The global rank.
        """
        return dist.get_global_rank(group, group_rank)

    @staticmethod
    def get_world_size():
        """
        Get the total number of processes in the distributed group.

        Returns:
            int: The world size.
        """
        return get_group_size()

    @staticmethod
    def get_op_name(func):
        """
        Extract the operation name from a function.

        Args:
            func: The function to extract the name from.

        Returns:
            str: The operation name.
        """
        return func.name

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim):
        output, _ = comm_func.all_gather_into_tensor(None, data, group=group)
        if concat_dim == 0:
            return output
        output_tensors = ms.ops.Split(output_num=concat_size)(output)
        return ms.mint.concat(output_tensors, concat_dim)

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        return ms.ops.Split(axis=split_dim, output_num=split_size)(data)[index]

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        output_tensor, _ = comm_func.all_to_all_single(
            output_shape,
            input_data,
            group=group,
            async_op=False
        )
        return output_tensor

    @staticmethod
    def tensor_type_cast(input_data, cast_type):
        """Cast tensor to specified data type."""
        type_mapping = {
            'float32': ms.float32,
            'float16': ms.float16,
            'int64': ms.int64,
            'int32': ms.int32
        }
        if cast_type not in type_mapping:
            raise ValueError(f"Unknown cast type: {cast_type}. Supported types: {list(type_mapping.keys())}")
        return input_data.to(type_mapping[cast_type])

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        output, _ = comm_func.all_reduce(data, op, group)
        return output

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        if axis > 0:
            data = ms.mint.concat(ms.ops.Split(axis=axis, output_num=dev_num)(data), dim=0)
        output_tensor, _ = comm_func.reduce_scatter_tensor(None, data, 'sum', group)
        if op == 'avg':
            output_tensor = output_tensor / dev_num
        return output_tensor

    @staticmethod
    def init_parameters(module, stage_index):
        return _init_parameters(module, stage_index)

    # pylint: disable=W0212
    @staticmethod
    def update_param_data(param, data):
        """update param data"""
        if isinstance(param, DTensorBase):
            param.set_data(data)
        else:
            param._update_data(data)

    @staticmethod
    def load_into_param(param, data):
        copy_tensor = MindSporePlatform.empty_like(data)
        copy_tensor.copy_(data)
        if isinstance(param, DTensorBase):
            param.set_data(copy_tensor)
        else:
            param._update(copy_tensor)

    @staticmethod
    def get_cell_construct(cell):
        return cell.construct

    @staticmethod
    def get_cells_and_names(cell):
        return cell.cells_and_names()

    @staticmethod
    def get_modules(module):
        return module.cells()

    @staticmethod
    def search_parameter_by_name(cell, param_name: str):
        """
        Find the parent Module of the parameter, the parameter's name in the parent Module, and the parameter.
        Return value: (parent Module instance, parameter's name in parent Module, parameter object).
        Returns None if not found.
        """
        # Remove the "self." prefix from param_name (to maintain compatibility with original logic)
        param_name = param_name.replace("self.", "")
        # Case 1: The parameter is a direct parameter of the current Module (not in any sub-Module)
        if param_name in cell._params:
            return (cell, param_name, cell._params[param_name])

        # Case 2: The parameter is in a sub-Module (supports multi-level nesting, e.g., "net_b.dense1.weight")
        if "." in param_name:
            # Split into: sub-Module path + parameter name (e.g., "net_b.dense1" + "weight")
            cell_path, param_key = param_name.rsplit(".", 1)
            try:
                # Locate the sub-Module where the parameter resides (supports multi-level paths)
                target_cell = cell.get_sub_cell(cell_path)
                # Check if the sub-Module directly contains this parameter
                if param_key in target_cell._params:
                    return target_cell, param_key, target_cell._params[param_key]
            except AttributeError:
                # Sub-Module path does not exist or the parameter is not in that sub-Module
                pass

        # Traverse all sub-Modules (recursively) to search for the parameter
        for _, child_cell in cell._cells.items():
            if isinstance(child_cell, Cell):
                # Recursively search within the sub-Module
                result = MindSporePlatform.search_parameter_by_name(child_cell, param_name)
                if result is not None:
                    return result

        return None

    @staticmethod
    def update_parameter_by_name(cell, result: tuple, new_param) -> bool:
        """
        Modify the original parameter in a Module or sub-Module using the search result
        Args:
            cell: The cell which parameter is to update
            result: A tuple contains parent Module, parameter key and old parameter.
            new_param: New Parameter object (used to replace the original parameter)
        """
        parent_cell, param_key, _ = result
        # Key operation: directly modify the _params dictionary of the parent Module (original storage location)
        parent_cell._params[param_key] = new_param

        if param_key in parent_cell.__dict__:
            parent_cell.__dict__[param_key] = new_param
        parent_cell._params_list[param_key] = new_param
        return True

    @staticmethod
    def set_layout_into_parameter(param, layout):
        """Set layout in to parameter"""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=import-outside-toplevel
        from hyper_parallel.core.dtensor.layout import _infer_slice_shape_by_layout, \
            _get_slice_tensor_by_layout  # pylint: disable=import-outside-toplevel
        if isinstance(param, DTensor):
            raise ValueError(f"Parameter {param.name} has been configured layout, cannot be set repeatedly.")
        param_info = param.param_info
        requires_grad = param.requires_grad
        name = param.name
        slice_shape = _infer_slice_shape_by_layout(param.shape, layout)

        if not param.has_init:
            # has been init, get slice data
            param_dtensor = DTensor.from_local(
                _get_slice_tensor_by_layout(param, layout).value(), layout.mesh, layout.alias_placements
            )
            param = Parameter(param_dtensor, name=name, requires_grad=requires_grad)
            param.param_info = param_info
        else:
            # has not been init, need to modify init shape
            param.init_mode.shape = slice_shape
            param_dtensor = DTensor.from_local(param.init_mode, layout.mesh, layout.alias_placements)
            param = Parameter(param_dtensor, name=name, requires_grad=requires_grad)
            param.param_info = param_info
        return param

    @staticmethod
    def get_param_local_shape(param):
        """get param local shape"""
        if isinstance(param, DTensorBase):
            return param.local_shape
        return param.shape

    @staticmethod
    def get_param_local_data(param):
        """get param local shape"""
        if isinstance(param, DTensorBase):
            return param.to_local()
        return param

    @staticmethod
    def get_param_type_size(param):
        return type_size_in_bytes(param.dtype)

    @staticmethod
    def is_tensor(obj: Any) -> bool:
        """Return True if ``obj`` is a ``mindspore.Tensor``."""
        return isinstance(obj, Tensor)

    @staticmethod
    def get_tensor_storage_size(tensor: Any) -> int:
        """Return serialized byte size (numel * itemsize) for a MindSpore tensor."""
        if not MindSporePlatform.is_tensor(tensor):
            raise TypeError(
                f"MindSporePlatform.get_tensor_storage_size expects mindspore.Tensor, got {type(tensor)!r}"
            )
        return int(tensor.numel()) * int(tensor.itemsize)

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad, device):
        param = Parameter(initializer("zeros", param_shape, param_type), requires_grad=requires_grad)
        if device in ("GPU", "Ascend"):
            return param.to(device)
        return param

    @staticmethod
    def new_tensor(tensor_shape, tensor_type, device):
        tensor = Tensor(shape=tensor_shape, dtype=tensor_type)
        if device in ("GPU", "Ascend"):
            return tensor.to(device)
        return tensor

    @staticmethod
    def full_like(tensor, fill_value, dtype=None):
        return mint.full_like(tensor, fill_value, dtype=dtype)

    @staticmethod
    def isend(tensor, dst=None, group=None, tag=0):
        return dist.isend(tensor, dst, group, tag)

    @staticmethod
    def irecv(tensor, src=None, group=None, tag=0):
        return dist.irecv(tensor, src, group, tag)

    @staticmethod
    def p2p_exchange(tensor, peer_rank: int, group=None):  # pylint: disable=unused-argument
        raise NotImplementedError(
            "p2p_exchange is not yet supported on the MindSpore platform."
        )

    @staticmethod
    def send_object_list(obj_list, dst=None, group=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.pipeline_parallel._utils import send_object_list
        send_object_list(obj_list, dst, group)

    @staticmethod
    def recv_object_list(obj_list, src=None, group=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.pipeline_parallel._utils import recv_object_list
        recv_object_list(obj_list, src, group)

    @staticmethod
    def set_tensor_requires_grad(input_tensor):
        """
        set requires grad flag for input tensor
        """
        input_tensor.requires_grad_()

    def _create_group(self, rank_list):
        world_group = self._maybe_reuse_world_group(rank_list)
        if world_group is not None:
            return world_group

        group_name = str(tuple(sorted(rank_list)))
        new_group(rank_ids=rank_list, group=group_name)
        EXISTING_COMM_GROUPS[group_name] = group_name
        return group_name

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        return comm_func.all_gather_into_tensor(None, data, group=group_info.group_name, async_op=async_op)

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        if isinstance(group_info, str):
            handle = dist.all_reduce(data, group=group_info, async_op=async_op)
        else:
            handle = dist.all_reduce(data, group=group_info.group_name, async_op=async_op)
        return data, handle

    @staticmethod
    def broadcast(data, src, group=None, async_op=False):
        handle = dist.broadcast(data, src, group, async_op)
        if async_op:
            handle.wait()
        return data

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        return comm_func.reduce_scatter_tensor(None, data, group=group_info.group_name, async_op=async_op)

    @staticmethod
    def all_to_all_single(input_tensor, output_shape, group, async_op=False):
        return _mindspore_all_to_all_single(input_tensor, output_shape, group, async_op=async_op)

    @staticmethod
    def differentiable_async_a2a_wait(x, work, out_perm, group, world_size, concat_dim, split_dim,  # pylint: disable=unused-argument
                                      handle_box=None):
        return _MSAsyncA2AFunction.apply(
            x, work, out_perm, group, world_size, concat_dim, split_dim, handle_box
        )

    @staticmethod
    def differentiable_all_to_all_single_async(input_tensor, input_splits, output_splits, group):
        """Launch an asynchronous, differentiable all-to-all-single.

        Token a2a entry point used by ``CommComputeOverlap``-driven MoE
        wrappers.  The kernel is queued on the HCCL group's stream and
        the host returns immediately, so the calling thread can proceed
        to the next sync hook (notify + rendezvous) before the
        collective finishes — this is what enables the comm/compute
        overlap window on the paired thread.

        Args:
            input_tensor:  **1-D** tensor — the caller is responsible for
                           flattening multi-dim inputs beforehand.
            input_splits:  ``list[int]`` — **element** counts sent to each
                           rank (not row counts).  For an originally
                           ``(N, D)`` tensor, each entry is ``rows_i * D``.
            output_splits: ``list[int]`` — element counts received from each rank.
            group:         Process group.

        Returns:
            ``AsyncCollectiveTensor`` of shape ``(sum(output_splits),)`` that
            defers ``CommHandle.wait()`` to the first consumer op via
            ``__ms_dispatch__``.

        Raises:
            ValueError: if ``input_tensor`` is not 1-D.

        Note:
            The 1-D + element-count contract diverges from the Torch
            implementation (which accepts N-D input + row-count splits).
            The divergence is intentional for now: it lets the MS path
            call the inner primitive directly and avoid the cross-stream
            race that ``comm_func.all_to_all_single``'s trailing reshape
            triggers under ``MS_ALLOC_CONF=memory_tracker:True`` —
            see :meth:`_MSAsyncA2ALazyBwd._issue_async_a2a`.
        """
        if input_tensor.ndim != 1:
            raise ValueError(
                "MindSporePlatform.differentiable_all_to_all_single_async requires a 1-D "
                f"input_tensor (got ndim={input_tensor.ndim}, shape={tuple(input_tensor.shape)}). "
                "Flatten the tensor and convert row-count splits to element counts before calling."
            )
        return _MSAsyncA2ALazyBwd.apply(input_tensor, output_splits, input_splits, group)

    @staticmethod
    def differentiable_sync_hook(x, hook_name: str, coordinator):
        """Fire a HookCoordinator rendezvous on forward and backward.

        Args:
            x:           Input tensor — returned unchanged.
            hook_name:   One of:
                         * ``"A"`` / ``"B"`` / ``"C"`` / ``"D"`` —
                           full rendezvous on both directions.
                         * ``"CHUNK_START"`` — chunk-entry hook on
                           forward; pairs with ``D_LAST.bwd`` so the
                           BWD thread's combine.bwd of the last layer
                           is bracketed by a barrier-synced sync point.
                           Skipped on backward.
                         * ``"D_LAST"`` — closing D of the last MoE
                           layer in a chunk.  Forward: ``notify_dispatched``
                           only (no Attention follows so rendezvous is
                           skipped).  Backward: full rendezvous via D's
                           BWD role; paired with ``CHUNK_START`` on FWD.
            coordinator: The :class:`HookCoordinator` driving the
                         rendezvous protocol.

        Returns:
            ``x`` unchanged.

        Note:
            Two-thread compatibility on MindSpore PyNative is not yet
            fully verified.  The HookCoordinator + ``_Function``
            primitives are individually thread-safe, but the
            interaction with MindSpore's autograd execution model
            under ``threading.Thread`` should be PoC-tested before
            production use.
        """
        return _MSSyncHookFunction.apply(x, hook_name, coordinator)

    @staticmethod
    def parameters_dict(cell: Cell):
        return cell.parameters_and_names()

    @staticmethod
    def get_tensor_transform():
        return _tensor_transform

    @staticmethod
    def construct_strided_slice(x, begin, end, stride):
        return ms.ops.strided_slice(x, begin, end, stride)

    @staticmethod
    def micro_batch(micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.pipeline_parallel._utils import _MicroBatch
        return _MicroBatch(micro_batch_num, args_batch_dim, kwargs_batch_dim)

    @staticmethod
    def get_model_state_dict(model, *, options=None):
        raise NotImplementedError(
            "get_model_state_dict is not yet supported on MindSpore"
        )

    @staticmethod
    def save_checkpoint(cell: Union[Cell, dict], file_path: str, ckpt_format: str = "safetensors") -> None:
        if isinstance(cell, dict):
            save_dict = {}
            for k, v in cell.items():
                if isinstance(v, Parameter):
                    save_dict[k] = v
                elif isinstance(v, Tensor):
                    save_dict[k] = Parameter(v, name=k)
                else:
                    save_dict[k] = v
        else:
            save_dict = cell._params
        ms.save_checkpoint(save_obj=save_dict, ckpt_file_name=file_path, format=ckpt_format)

    @staticmethod
    def load_checkpoint(file_path: str, ckpt_format: str = "safetensors") -> dict:
        return ms.load_checkpoint(ckpt_file_name=file_path, format=ckpt_format)

    @staticmethod
    def get_symmetric_memory_handler():
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.symmetric_memory import MSSymmetricMemoryHandler
        symmetric_memory = MSSymmetricMemoryHandler()
        return symmetric_memory

    @staticmethod
    def get_multicore_handler():
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.multicore import MSMulticoreHandler
        return MSMulticoreHandler()

    def new_stream(self):
        return ms.runtime.Stream()

    def get_stream_context(self):
        return ms.runtime.StreamCtx

    @staticmethod
    def all_gather_object(object_list, obj, group=None) -> None:
        """
        Gathers objects from the given group into object list.

        Args:
            object_list (list[Any]): Define the output list, which size equal to the size of group.
            obj (Any): The object on current rank and in given process group.
            group (ProcessGroup, optional): The process group to gather obj. Default is ``None``, and ``None`` means
                global group.

        Returns:
            None. Objs are gathered into ``object_list``.
        """
        dist.all_gather_object(object_list, obj, group)

    @staticmethod
    def barrier(group=None, async_op: bool = False, device_ids=None) -> Any:
        """
        Synchronize all processes in the given communication group.

        Args:
            group (str, optional): The communication group to work on. Default is ``None``,
                meaning the default world group.
            async_op (bool, optional): Whether this op should be asynchronous. Default: ``False``.
            device_ids (list[int], optional): Reserved parameter on Ascend. Default: ``None``.

        Returns:
            CommHandle if ``async_op`` is True; otherwise ``None``.
        """
        return dist.barrier(group, async_op, device_ids)

    @staticmethod
    def init_process_group(
            backend: str = None,
            *,
            init_method: Optional[str] = None,
            timeout: Optional[timedelta] = None,
            world_size: int = -1,
            rank: int = -1,
            store: TCPStore = None,
            pg_options=None,
            device_id=None
    ) -> None:
        """
        Initialize global process group.

        Args:
            backend (str): The backend used to init process group. Default is ``"hccl"`` and now only support hccl.
            init_method (str, optional): URL specifying how to initialize the process group. Default is ``None``.
            timeout (timedelta, optional): Timeout for API executed. Default is ``None``.
            world_size (int): Number of processes. Default is ``-1``.
            rank (int, optional): Rank of the current process. Default is ``-1``.
            store (Store, optional): An object that stores key/value data, facilitating the exchange of inter-process
                communication addresses and connection information. Default is ``None``. Currently, only the
                ``TCPStore`` type is supported.
            pg_options (ProcessGroupOptions, optional): Reserved parameter. Current not take effect.
            device_id (int, optional): Reserved parameter. Current not take effect.
        """
        if backend is None:
            backend = "hccl"
        try:
            if dist.is_initialized():
                return
        except AttributeError:
            pass
        dist.init_process_group(backend=backend, init_method=init_method, timeout=timeout, world_size=world_size,
                                rank=rank, store=store, pg_options=pg_options, device_id=device_id)

    @staticmethod
    def destroy_process_group(group: Optional[str] = None) -> None:
        """
        Destroy given process group.

        Args:
            group (str, optional): Specify the group to destroy. Default: ``None`` means ``hccl_world_group``. If group
                is None or "hccl_world_group", destroy global process group and all process groups relative to global
                process group.
        """
        if group in EXISTING_COMM_GROUPS.values():
            keys_to_destroy = [k for k, v in EXISTING_COMM_GROUPS.items() if v == group]
            for k in keys_to_destroy:
                del EXISTING_COMM_GROUPS[k]
        dist.destroy_process_group(group)

    @staticmethod
    def get_process_group_ranks(group: Optional[str] = None) -> list[int]:
        """
        Get all ranks in given process group.

        Args:
            group (str, optional): Specify the process group to work on. Default: ``None`` means ``hccl_world_group``.

        Returns:
            List[int]: List of ranks in given process group.
        """
        return dist.get_process_group_ranks(group)

    @staticmethod
    def get_backend(group: Optional[str] = None) -> str:
        """
        Get the backend of given process group.

        Args:
            group (str, optional): Specify the process group to work on. Default: ``None`` means ``hccl_world_group``.

        Returns:
            str: The backend of the group.
        """
        return dist.get_backend(group)

    @staticmethod
    def split_group(parent_pg: Optional[str] = None,
                    split_ranks: Optional[list] = None,
                    timeout: Optional[timedelta] = None,
                    pg_options: Optional[str] = None,
                    group_desc: Optional[str] = None,
                    ) -> str:
        """
        Create split group for a specific group rank in split_ranks, which group contains current rank id.

        Args:
            parent_pg (str, Optional): A process group which the goal group split from.
            split_ranks (Optional[list]): A list like ``list[list[int]]``.
            timeout (Optional[timedelta]): Timeout for API executed. Default is ``None``.
            pg_options (Optional[str]): Reserved parameter. Current not take effect.
            group_desc (Optional[str]): Description of process group.

        Returns:
            str: The split group name.
        """
        if split_ranks is None or len(split_ranks) == 0:
            raise ValueError("split_ranks cannot be None or empty")

        rank_id = MindSporePlatform.get_rank()
        for split_rank in split_ranks:
            if rank_id in split_rank:
                world_group = MindSporePlatform._maybe_reuse_world_group(split_rank)
                if world_group is not None:
                    return world_group
                split_group = MindSporePlatform.get_created_group(split_rank)
                if split_group:
                    return split_group
                group_name = str(tuple(sorted(split_rank)))
                new_group(rank_ids=split_rank, group=group_name)
                EXISTING_COMM_GROUPS[group_name] = group_name
                return group_name
        raise ValueError(f"Split group invalid rank, the Split_ranks {split_ranks} does not contain current rank"
                         f" {rank_id}")

    @staticmethod
    def get_group_local_rank(group=None) -> int:
        """get group local rank id."""
        return dist.get_group_rank(group, MindSporePlatform.get_rank())

    @staticmethod
    def no_grad():
        return _no_grad()

    @staticmethod
    def relu(tensor):
        return mint.nn.functional.relu(tensor)

    @staticmethod
    def cat(tensors, dim=0):
        return mint.cat(tensors, dim=dim)

    @staticmethod
    def empty_like(tensor, *, dtype=None, device=None, pin_memory=False):
        return mint.empty_like(tensor, dtype=dtype, device=device, pin_memory=pin_memory)

    def get_current_stream(self):
        return ms.runtime.current_stream()

    def new_event(self):
        return ms.runtime.Event()

    def tree_map(self, fn, tree):
        """
        Apply fn to each leaf in a nested structure (list / tuple / dict),
        preserving the original structure.
        """
        if isinstance(tree, dict):
            return type(tree)(
                (k, self.tree_map(fn, v)) for k, v in tree.items()
            )

        if isinstance(tree, tuple):
            return tuple(self.tree_map(fn, v) for v in tree)

        if isinstance(tree, list):
            return [self.tree_map(fn, v) for v in tree]

        # leaf
        return fn(tree)

    @staticmethod
    def register_forward_pre_hook(module, hook, prepend=False, with_kwargs=False):
        return module.register_forward_pre_hook(hook, with_kwargs=with_kwargs)

    @staticmethod
    def register_full_backward_hook(module, hook, prepend=False):
        return module.register_backward_hook(hook)

    @staticmethod
    def register_full_backward_pre_hook(module, hook, prepend=False):
        return module.register_backward_pre_hook(hook)

    @property
    def checkpoint(self):
        return ms.recompute

    @staticmethod
    def ckpt_wrapper(module, checkpoint_fn=None, **checkpoint_fn_kwargs):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_wrapper import checkpoint_wrapper
        return checkpoint_wrapper(module, checkpoint_fn=checkpoint_fn, **checkpoint_fn_kwargs)

    @staticmethod
    def swap_wrapper(module, policy_fn=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import swap_wrapper
        return swap_wrapper(module, policy_fn=policy_fn)

    @staticmethod
    def swap_tensor_wrapper(target, tag=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import swap_tensor_wrapper
        return swap_tensor_wrapper(target, tag=tag)

    @property
    def noop_context_fn(self):
        return null_context_fn

    @staticmethod
    def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.sac import create_selective_checkpoint_contexts
        return create_selective_checkpoint_contexts(policy_fn_or_list,
                                                    allow_cache_entry_mutation=allow_cache_entry_mutation)

    @staticmethod
    def async_save_on_cpu(policy_fn=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import AsyncSaveOnCpu
        return AsyncSaveOnCpu(policy_fn=policy_fn)

    @staticmethod
    def get_element_size(tensor):
        """Get Tensor Element Size"""
        return tensor.itemsize

    @staticmethod
    def tensor_to_numpy(tensor) -> np.ndarray:
        """Convert MindSpore tensor to numpy array."""
        return tensor.asnumpy()

    @staticmethod

    def clip_grad_norm_(
        parameters, max_norm, norm_type=2.0,
        error_if_nonfinite=False, foreach=None,
    ):
        raise NotImplementedError(
            "clip_grad_norm_ is not yet supported on MindSpore"
        )

    @property
    def meta_device(self):
        return "meta"

    def init_on_device(self, device, include_buffers=False):
        return _init_on_device(device, include_buffers=include_buffers)

    def cast_fp_tensor(self, dtype, x):
        """
        Cast floating-point tensor to target dtype if applicable.
        """
        if (
            not isinstance(x, ms.Tensor)
            or not ms.ops.is_floating_point(x)
            or x.dtype == dtype
        ):
            return x
        return x.to(dtype)

    def apply_to_tensors(self, fn, container):
        """Recursively apply to all tensor in different kinds of container types."""

        def apply(x):
            if isinstance(x, ms.Tensor):
                return fn(x)
            if hasattr(x, "__dataclass_fields__"):
                dc = dataclasses.replace(x)
                changes = {
                    f.name: apply(getattr(dc, f.name)) for f in dataclasses.fields(dc)
                }
                return dataclasses.replace(dc, **changes)
            if isinstance(x, OrderedDict):
                od = x.__class__()
                for key, value in x.items():
                    od[key] = apply(value)
                return od
            if isinstance(x, dict):
                return {key: apply(value) for key, value in x.items()}
            if isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields"):
                res = (apply(el) for el in x)
                return type(x)(*res)
            if isinstance(x, (list, tuple, set)):
                return type(x)(apply(el) for el in x)
            return x

        return apply(container)

    @staticmethod
    def profiler_record(name):
        """Profiler context manager for recording operations using mindspore.profiler."""
        return contextlib.nullcontext()

    def str_to_dtype(self, dtype_str: str) -> Any:
        """Resolve checkpoint dtype strings (``mindspore.*`` or short ``str(Tensor.dtype)`` e.g. ``Float32``)."""
        if "." in dtype_str:
            prefix, name = dtype_str.split(".", 1)
            if prefix == "mindspore":
                return getattr(ms, name)
        dtype = getattr(ms, dtype_str.lower(), None)
        if dtype is not None:
            return dtype
        raise ValueError(
            f"Expected dtype string like 'mindspore.float32' or 'Float32', got {dtype_str!r}."
        )

    def list_to_size(self, size_list: list[int]) -> tuple[int, ...]:
        return tuple(size_list)

    @staticmethod
    def _maybe_reuse_world_group(rank_list):
        """Reuse the default world group for full-world rank lists."""
        normalized = tuple(sorted(rank_list))
        world_ranks = tuple(range(MindSporePlatform.get_world_size()))
        if normalized != world_ranks:
            return None

        EXISTING_COMM_GROUPS[str(normalized)] = GlobalComm.WORLD_COMM_GROUP
        return GlobalComm.WORLD_COMM_GROUP
