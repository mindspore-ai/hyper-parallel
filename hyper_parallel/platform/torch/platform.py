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
"""Torch platform api"""
from datetime import timedelta
from typing import Any, Optional, Sequence, Union
import dataclasses
from collections import OrderedDict

import numpy as np
from safetensors.torch import save_file, load_file
import torch
from torch import nn
from torch import Tensor
from torch._C._distributed_c10d import Store, ProcessGroup
from torch.distributed import Backend
from torch.distributed.distributed_c10d import _get_default_group
from torch.nn import Parameter, Module
from torch.nn.utils.rnn import PackedSequence
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils.checkpoint import noop_context_fn

import torch.distributed.nn.functional as dist_func
import torch.distributed as dist
from hyper_parallel.platform.torch.dtensor import DTensorBase
from hyper_parallel.platform.torch.pipeline_parallel.stage import PipelineStageBase
from hyper_parallel.platform.torch.group_utils import create_sub_groups
from hyper_parallel.platform.platform import Platform, PlatformType, EXISTING_COMM_GROUPS
from hyper_parallel.platform.torch.function_override import override_functions
from hyper_parallel.platform.torch.init_weights import init_on_device as _init_on_device

override_functions()


# ---------------------------------------------------------------------------
# Module-level A2A reshape helpers
# ---------------------------------------------------------------------------

def _a2a_reconstruct(out_perm: torch.Tensor, concat_dim: int) -> torch.Tensor:
    """Reconstruct A2A result from raw out_perm buffer.

    ``out_perm`` has shape ``[ws, *rest_dims]``, chunk at ``concat_dim + 1``.
    Returns tensor with merged chunk dimension.
    """
    new_ndim = out_perm.dim()
    chunk_in_perm = concat_dim + 1
    recon_perm = list(range(1, chunk_in_perm)) + [0] + list(range(chunk_in_perm, new_ndim))
    x_recon = out_perm.permute(recon_perm).contiguous()
    shape = list(x_recon.shape)
    merged = shape[concat_dim] * shape[concat_dim + 1]
    return x_recon.reshape(shape[:concat_dim] + [merged] + shape[concat_dim + 2:])


def _normalize_dim(dim: int, ndim: int) -> int:
    """Normalize a possibly negative dimension index."""
    return dim + ndim if dim < 0 else dim


def _move_dim_to_front(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Move ``dim`` to the front while keeping the other dimensions ordered."""
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous()
    perm = [dim] + [i for i in range(tensor.dim()) if i != dim]
    return tensor.permute(perm).contiguous()


def _move_dim_from_front(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of :func:`_move_dim_to_front`."""
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous()
    perm = [dim] + [i for i in range(tensor.dim()) if i != dim]
    inverse = [0] * len(perm)
    for idx, value in enumerate(perm):
        inverse[value] = idx
    return tensor.permute(inverse).contiguous()


class _TorchAsyncA2AFunction(torch.autograd.Function):
    """Differentiable wrapper for pre-launched async all-to-all.

    Forward: wait async handle, reconstruct A2A result.
    Backward: launch async head→seq A2A and store handle in ``handle_box``
    for the projection pre-hook to wait, achieving GEMM–A2A overlap.
    """

    @staticmethod
    def forward(ctx, x, work, out_perm, group, world_size, concat_dim, split_dim,  # pylint: disable=arguments-differ
                handle_box):
        """Wait for pre-launched async A2A and return reconstructed output."""
        ctx.group = group
        ctx.world_size = world_size
        ctx.concat_dim = concat_dim
        ctx.split_dim = split_dim
        ctx.handle_box = handle_box
        ctx.x_shape = x.shape
        work.wait()
        return _a2a_reconstruct(out_perm, concat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Launch async head→seq A2A for backward overlap, or return zero grad."""
        if ctx.handle_box is not None:
            # Launch async head→seq A2A (reverse of forward seq→head)
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
            out_perm = torch.empty_like(x_perm)
            work = dist.all_to_all_single(out_perm, x_perm, group=ctx.group, async_op=True)
            ctx.handle_box.append((work, out_perm))
        return grad_output.new_zeros(ctx.x_shape), None, None, None, None, None, None, None


class _TorchAsyncAllGatherFunction(torch.autograd.Function):
    """Differentiable wrapper for pre-launched async all-gather."""

    @staticmethod
    def forward(ctx, x, work, out_perm, group, world_size, gather_dim, handle_box):  # pylint: disable=arguments-differ
        """Wait for pre-launched all-gather and reconstruct the gathered tensor."""
        ctx.group = group
        ctx.world_size = world_size
        ctx.gather_dim = gather_dim
        ctx.handle_box = handle_box
        ctx.x_shape = x.shape
        work.wait()
        return _move_dim_from_front(out_perm, gather_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Launch reverse reduce-scatter for the all-gather."""
        grad_perm = _move_dim_to_front(grad_output.contiguous(), ctx.gather_dim)
        output_shape = list(grad_perm.shape)
        if output_shape[0] % ctx.world_size != 0:
            raise ValueError(
                "all_gather backward expected gathered dimension to be divisible by world_size, "
                f"got {output_shape[0]} and {ctx.world_size}."
            )
        output_shape[0] //= ctx.world_size
        output = torch.empty(output_shape, dtype=grad_perm.dtype, device=grad_perm.device)
        work = dist.reduce_scatter_tensor(output, grad_perm, group=ctx.group, async_op=True)
        if ctx.handle_box is not None:
            ctx.handle_box.append((work, output, ctx.gather_dim))
            return grad_output.new_zeros(ctx.x_shape), None, None, None, None, None, None
        work.wait()
        return _move_dim_from_front(output, ctx.gather_dim), None, None, None, None, None, None


class _AsyncA2ALazyBwd(torch.autograd.Function):
    """All-to-all whose forward AND backward return ``AsyncCollectiveTensor``.

    PyTorch's stock ``all_to_all_single_autograd`` calls ``wait_tensor`` in
    its backward eagerly, and the autograd engine binds backward stream
    context to the forward stream — so even if the BWD thread is wrapped
    in a side-stream context, that wait still lands on the FWD main
    stream and blocks Attention launches.

    This Function bypasses the engine's binding by calling the
    non-autograd functional op in both directions and returning ACT.
    The wait is deferred to the next consumer's first non-view access
    (e.g. the indexing backward of ``_unpermute``), giving the FWD
    thread a small Python window to enqueue its Attention kernels onto
    the main stream **before** the wait lands there.
    """

    @staticmethod
    def forward(ctx, input_tensor, output_splits, input_splits, group):  # pylint: disable=arguments-differ
        """Perform the forward all-to-all single collective, saving splits and group for backward."""
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        # pylint: disable=C0415
        from torch.distributed._functional_collectives import all_to_all_single
        return all_to_all_single(
            input_tensor, output_splits, input_splits, group,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the backward pass by performing the inverse all-to-all with swapped splits."""
        # pylint: disable=C0415
        from torch.distributed._functional_collectives import all_to_all_single
        grad_input = all_to_all_single(
            grad_output, ctx.input_splits, ctx.output_splits, ctx.group,
        )
        return grad_input, None, None, None


class _TorchSyncHookFunction(torch.autograd.Function):
    """Autograd identity that fires HookCoordinator rendezvous on fwd/bwd.

    Uses a **4-hook** design (``A``, ``B``, ``C``, ``D``) with pure
    COMM / COMPUTE roles — no NONE role.  Every rendezvous is a strict
    COMM + COMPUTE pair, guaranteeing NCCL-first dispatch ordering at
    **all** points including layer boundaries.

    Hook placement per MoE layer::

        [A] → dispatch → [B] → module → [C] → combine → [D] → (Attention) → [A_next]

    At layer boundaries (D / A hooks), the Attention that runs between
    layers is treated as COMPUTE, and the combine / combine.bwd is treated
    as COMM, so the coordinator enforces comm-first ordering even across
    layer transitions.
    """

    # 4-hook role tables: (prev_role_idx, next_role_idx).
    # Index encoding: 1 = COMM, 2 = COMPUTE.
    #
    # Torch only uses the four core hooks A/B/C/D + D_LAST sentinel.
    # The MS backend adds ``CHUNK_START`` / ``CHUNK_END`` because of
    # MS-specific issues (stream binding follows the calling thread;
    # autograd cannot have FWD-record + BWD-replay concurrently).
    # Torch has neither problem — CUDA streams are process-wide and
    # Torch autograd is thread-safe — so we keep the original
    # 4-hook design here.  Do not add CHUNK_START / CHUNK_END to
    # the Torch tables; if a future test does need them, copy the
    # MS implementation and add the matching skip rules in
    # ``forward`` / ``backward``.
    _FWD_ROLES = {
        #         (prev, next)      prev op          next op
        "A": (2, 1),   # COMPUTE, COMM     Attention   | dispatch
        "B": (1, 2),   # COMM, COMPUTE     dispatch    | module
        "C": (2, 1),   # COMPUTE, COMM     module      | combine
        "D": (1, 2),   # COMM, COMPUTE     combine     | Attention
    }
    _BWD_ROLES = {
        "D": (2, 1),   # COMPUTE, COMM     Attn.bwd    | combine.bwd
        "C": (1, 2),   # COMM, COMPUTE     combine.bwd | module.bwd
        "B": (2, 1),   # COMPUTE, COMM     module.bwd  | dispatch.bwd
        "A": (1, 2),   # COMM, COMPUTE     dispatch.bwd| Attn.bwd
    }

    _ROLE_CACHE = None

    @staticmethod
    def _role_enum(idx: int):
        if _TorchSyncHookFunction._ROLE_CACHE is None:
            from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookRole  # pylint: disable=C0415
            _TorchSyncHookFunction._ROLE_CACHE = (None, HookRole.COMM, HookRole.COMPUTE)
        return _TorchSyncHookFunction._ROLE_CACHE[idx]

    @staticmethod
    def forward(ctx, x, hook_name, coordinator):  # pylint: disable=arguments-differ
        """Identity forward that fires a HookCoordinator rendezvous.

        Notifies the previous op's role and rendezvouses for the next op's
        role per the ``_FWD_ROLES`` table.  ``"D_LAST"`` is a sentinel
        meaning "skip this rendezvous" (last layer's closing D — no
        Attention follows).

        Args:
            ctx:         Autograd context, stores ``hook_name`` and
                         ``coordinator`` for the backward pass.
            x:           Input tensor, returned unchanged.
            hook_name:   One of ``"A"``, ``"B"``, ``"C"``, ``"D"``,
                         ``"D_LAST"``.
            coordinator: The :class:`HookCoordinator` driving the rendezvous.

        Returns:
            ``x`` unchanged.
        """
        ctx.hook_name = hook_name
        ctx.coordinator = coordinator

        if not coordinator.is_enabled():
            return x

        if hook_name == "D_LAST":
            # ``D_LAST`` marks the last layer's closing D hook — no
            # Attention follows in this chunk, so the rendezvous is
            # meaningless and is skipped.  We still
            # ``notify_dispatched(COMM)`` so the COMPUTE side of the
            # preceding ``C`` rendezvous unblocks early, letting
            # BWD's Attn.bwd_last overlap with FWD's post-combine
            # work — Torch autograd is thread-safe so this concurrent
            # FWD-record + BWD-replay is fine.
            prev_idx, _ = _TorchSyncHookFunction._FWD_ROLES["D"]
            role_of = _TorchSyncHookFunction._role_enum
            coordinator.notify_dispatched(role_of(prev_idx))
            return x

        prev_idx, next_idx = _TorchSyncHookFunction._FWD_ROLES[hook_name]
        role_of = _TorchSyncHookFunction._role_enum
        coordinator.notify_dispatched(role_of(prev_idx))
        coordinator.rendezvous(role_of(next_idx))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """Identity backward that fires a HookCoordinator rendezvous.

        Mirror of :meth:`forward` using the ``_BWD_ROLES`` table.
        ``"D_LAST"`` skips the rendezvous because this is the first BWD
        hook to fire and ``combine.bwd`` has already dispatched freely
        before any rendezvous can happen.

        Args:
            ctx:         Autograd context with ``hook_name`` and
                         ``coordinator`` saved during forward.
            grad_output: Gradient w.r.t. the forward output, returned
                         unchanged.

        Returns:
            ``(grad_output, None, None)`` — gradients only flow back to
            the tensor input, ``hook_name`` and ``coordinator`` are
            non-tensor inputs.
        """
        hook_name = ctx.hook_name
        coordinator = ctx.coordinator

        if not coordinator.is_enabled():
            return grad_output, None, None

        if hook_name == "D_LAST":
            # First BWD hook to fire; combine.bwd has already
            # dispatched freely before any rendezvous can happen.
            # Skipping here is safe on Torch because CUDA streams
            # are process-wide and the NCCL FIFO order is consistent
            # across ranks regardless of which thread launched
            # combine.bwd.
            return grad_output, None, None

        prev_idx, next_idx = _TorchSyncHookFunction._BWD_ROLES[hook_name]
        role_of = _TorchSyncHookFunction._role_enum
        coordinator.notify_dispatched(role_of(prev_idx))
        coordinator.rendezvous(role_of(next_idx))
        return grad_output, None, None


class _TorchP2PExchangeFunction(torch.autograd.Function):
    """Symmetric bidirectional P2P: send local tensor to peer, receive peer's tensor."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, peer_rank: int, group) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Perform symmetric bidirectional P2P exchange with peer_rank."""
        ctx.peer_rank = peer_rank
        ctx.group = group
        send_buf = tensor.contiguous()
        recv_buf = torch.empty_like(send_buf)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, peer_rank, group),
            dist.P2POp(dist.irecv, recv_buf, peer_rank, group),
        ])
        for req in reqs:
            req.wait()
        return recv_buf

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Perform symmetric P2P exchange for the backward gradient pass."""
        send_buf = grad_output.contiguous()
        recv_buf = torch.empty_like(send_buf)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, ctx.peer_rank, ctx.group),
            dist.P2POp(dist.irecv, recv_buf, ctx.peer_rank, ctx.group),
        ])
        for req in reqs:
            req.wait()
        return recv_buf, None, None


class _TorchDifferentiableVariableAllGather(torch.autograd.Function):
    """Variable dim-zero all-gather with an uneven reduce-scatter backward."""

    @staticmethod
    def forward(ctx, input_tensor, output_splits, group):  # pylint: disable=arguments-differ
        """Gather each rank's true row count without replicating inputs for A2A."""
        if input_tensor.ndim == 0:
            raise ValueError("variable all-gather input must have at least one dimension")
        splits = tuple(output_splits)
        if not splits:
            raise ValueError("output_splits must contain at least one group rank")
        if any(not isinstance(rows, int) or isinstance(rows, bool) or rows < 0 for rows in splits):
            raise ValueError(f"output_splits must contain non-negative integers, got {splits!r}")

        group_rank = dist.get_rank(group=group)
        if group_rank < 0 or group_rank >= len(splits):
            raise ValueError(f"group rank must be in [0, {len(splits)}), got {group_rank}")
        if input_tensor.shape[0] != splits[group_rank]:
            raise ValueError(
                "variable all-gather local rows must match output_splits at the group rank, "
                f"got local_rows={input_tensor.shape[0]}, group_rank={group_rank}, "
                f"output_splits={splits!r}"
            )

        input_tensor = input_tensor.contiguous()
        feature_shape = tuple(input_tensor.shape[1:])
        if input_tensor.device.type == "npu":
            gathered = [input_tensor.new_empty((rows, *feature_shape)) for rows in splits]
            dist.all_gather(gathered, input_tensor, group=group)
        else:
            max_rows = max(splits)
            if max_rows == 0:
                gathered = [input_tensor.new_empty((0, *feature_shape)) for _ in splits]
            else:
                padded = input_tensor.new_zeros((max_rows, *feature_shape))
                if input_tensor.shape[0] > 0:
                    padded[:input_tensor.shape[0]].copy_(input_tensor)
                padded_outputs = [torch.empty_like(padded) for _ in splits]
                dist.all_gather(padded_outputs, padded, group=group)
                gathered = [
                    output[:rows].contiguous()
                    for output, rows in zip(padded_outputs, splits)
                ]

        ctx.output_splits = splits
        ctx.group = group
        ctx.group_rank = group_rank
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Sum replicated output gradients and return this rank's uneven shard."""
        output_rows = ctx.output_splits[ctx.group_rank]
        output = grad_output.new_empty((output_rows, *grad_output.shape[1:]))
        if sum(ctx.output_splits) == 0:
            return output, None, None

        grad_output = grad_output.contiguous()
        if grad_output.device.type == "npu":
            from torch_npu.distributed import reduce_scatter_tensor_uneven  # pylint: disable=C0415
            reduce_scatter_tensor_uneven(
                output,
                grad_output,
                input_split_sizes=list(ctx.output_splits),
                op=dist.ReduceOp.SUM,
                group=ctx.group,
            )
        else:
            reduced = grad_output.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=ctx.group)
            start = sum(ctx.output_splits[:ctx.group_rank])
            output.copy_(reduced.narrow(0, start, output_rows))
        return output, None, None


# Mapping from string op names to torch.distributed.ReduceOp
_OP_MAP = {
    'sum': dist.ReduceOp.SUM,
    'prod': dist.ReduceOp.PRODUCT,
    'max': dist.ReduceOp.MAX,
    'min': dist.ReduceOp.MIN,
    # convert tensor elements to int32 and use MIN
    'all': dist.ReduceOp.MIN,
    # 'avg' is typically handled by SUM followed by division in current implementation logic
    'avg': dist.ReduceOp.SUM,
}

# Try to add AVG for 'mean' if supported by current torch version
if hasattr(dist.ReduceOp, "AVG"):
    _OP_MAP['mean'] = dist.ReduceOp.AVG
else:
    # Fallback for older torch versions if necessary, though this might require manual division upstream
    # Assuming standard behavior where 'mean' implies native AVG support or upstream handling
    _OP_MAP['mean'] = dist.ReduceOp.SUM


def _ensure_contiguous(x):
    """Return a contiguous copy of *x* if not already contiguous."""
    if not x.is_contiguous() or x.storage_offset() != 0:
        x = x.contiguous()
    return x


class _TorchBatchP2PWork:
    """Single ``.wait()`` handle wrapping the per-op works returned by
    ``torch.distributed.batch_isend_irecv``.

    Torch returns one ``Work`` per op in the batch (the ops are coalesced
    onto one comm stream), whereas the platform contract — and the scheduler
    that consumes it — expects a single handle covering the whole batch so
    the wait can be deferred to one consumption point (mirroring MindSpore's
    single packaging ``CommHandle``).  Waiting this handle waits every
    underlying op.
    """

    __slots__ = ("_works",)

    def __init__(self, works):
        self._works = works

    def wait(self):
        for work in self._works:
            if work is not None:
                work.wait()


# pylint: disable=C0103
class TorchPlatform(Platform):
    """Torch platform api"""
    Tensor = Tensor
    tensor = torch.tensor
    Parameter = Parameter
    Module = Module
    DTensorBase = DTensorBase
    PipelineStageBase = PipelineStageBase
    platform_type = PlatformType.PYTORCH
    tensor_dtype = torch
    dtype = torch.dtype
    Function = torch.autograd.Function

    _custom_ops_cls = None

    @property
    def custom_ops(self):
        """Return the Torch platform custom ops instance.

        .. warning::
            This is an experimental API that subject to change or deletion.

        Returns:
            TorchCustomOps: Custom ops class that raises NotImplementedError
            for all operators (MindSpore-only at this time).
        """
        if self._custom_ops_cls is None:
            from hyper_parallel.platform.torch.custom_ops import TorchCustomOps  # pylint: disable=import-outside-toplevel
            self._custom_ops_cls = TorchCustomOps
        return self._custom_ops_cls

    @staticmethod
    def get_swap_optimizer():
        """Return the Torch optimizer-state swap wrapper class."""
        from hyper_parallel.platform.torch.swap_optimizer.swap_optimizer import (  # pylint: disable=import-outside-toplevel
            get_swap_optimizer,
        )
        return get_swap_optimizer()

    @staticmethod
    def is_linear_module(module) -> bool:
        """Check whether *module* is a ``torch.nn.Linear`` instance."""
        return isinstance(module, nn.Linear)

    @staticmethod
    def is_embedding_module(module) -> bool:
        """Check whether *module* is a ``torch.nn.Embedding`` instance."""
        return isinstance(module, nn.Embedding)

    @staticmethod
    def device_count(device_handle):
        """
        Get the number of available devices.

        Args:
            device_handle: The device handle (e.g., torch.cuda, torch.npu).

        Returns:
            int: The number of available devices.
        """
        return device_handle.device_count()

    def device_type(self):
        """
        Get the current device type.

        Returns:
            str: The device type string ("npu" for NPU, "cuda" for GPU).
        """
        device_handle = self.get_device_handle()
        if device_handle == torch.npu:
            return "npu"
        return "cuda"

    def device(self, device_idx=None):
        """
        Get a torch.device object for the specified device index.

        Args:
            device_idx (Optional[int]): The device index. If None, returns device without index.

        Returns:
            torch.device: A torch device object.
        """
        device_type = self.device_type()
        if device_idx is None:
            return torch.device(device_type)
        return torch.device(f"{device_type}:{device_idx:d}")

    @staticmethod
    def get_rng_state(device=None, device_handle=None):
        """
        Get the random number generator state.

        Args:
            device (Optional): The device to get RNG state from.
            device_handle (Optional): The device handle (torch.cuda, torch.npu, etc.).

        Returns:
            Tensor: The RNG state as a byte tensor.
        """
        if device_handle is None:
            return torch.get_rng_state()
        if device is None:
            return device_handle.get_rng_state()
        return device_handle.get_rng_state(device)

    @staticmethod
    def set_rng_state(state, device=None, device_handle=None):
        """
        Set the random number generator state.

        Args:
            state (Tensor): The RNG state to set.
            device (Optional): The device to set RNG state for.
            device_handle (Optional): The device handle (torch.cuda, torch.npu, etc.).
        """
        if device_handle is None:
            return torch.set_rng_state(state)
        if device is None:
            return device_handle.set_rng_state(state)
        return device_handle.set_rng_state(state, device)

    @staticmethod
    def manual_seed(seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The random seed value.

        Returns:
            torch.Generator: The random number generator.
        """
        return torch.manual_seed(seed)

    @staticmethod
    def ones(size, dtype=None):
        """
        Create a tensor filled with ones.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[torch.dtype]): The desired data type.

        Returns:
            Tensor: A tensor filled with ones.
        """
        return torch.ones(size, dtype=dtype)

    @staticmethod
    def zeros(size, dtype=None, device=None):
        """
        Create a tensor filled with zeros.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[torch.dtype]): The desired data type.
            device (Optional[torch.device]): The device to create the tensor on.

        Returns:
            Tensor: A tensor filled with zeros.
        """
        return torch.zeros(size, dtype=dtype, device=device)

    @staticmethod
    def full(size, fill_value, dtype=None):
        """
        Create a tensor filled with a scalar value.

        Args:
            size (tuple): The shape of the output tensor.
            fill_value (scalar): The value to fill the tensor with.
            dtype (Optional[torch.dtype]): The desired data type.

        Returns:
            Tensor: A tensor filled with the specified value.
        """
        return torch.full(size, fill_value, dtype=dtype)

    @staticmethod
    def empty(size, dtype=None, device=None):
        """
        Create an uninitialized tensor.

        Args:
            size (tuple): The shape of the output tensor.
            dtype (Optional[torch.dtype]): The desired data type.
            device (Optional[torch.device or str]): Target device.  When
                ``None`` the tensor is allocated on the default device
                (CPU under PyTorch defaults), matching the original
                back-compat behavior.

        Returns:
            Tensor: An uninitialized tensor.
        """
        return torch.empty(size, dtype=dtype, device=device)

    @staticmethod
    def rand(size, dtype=None, device=None):
        """Create a tensor filled with uniform random values in ``[0, 1)``."""
        return torch.rand(size, dtype=dtype, device=device)

    @staticmethod
    def randn(size, dtype=None, device=None):
        """Create a tensor filled with standard-normal random values."""
        return torch.randn(size, dtype=dtype, device=device)

    @staticmethod
    def get_rank():
        """
        Get the rank of the current process in the distributed group.

        Returns:
            int: The rank of the current process.
        """
        return dist.get_rank()

    @staticmethod
    def get_global_rank(group, group_rank):
        """
        Get the global rank from a group rank.

        Args:
            group (ProcessGroup): The process group.
            group_rank (int): The rank within the group.

        Returns:
            int: The global rank.
        """
        return dist.get_global_rank(group, group_rank)

    @staticmethod
    def get_group_rank(group):
        """Return this process's rank within *group*."""
        return dist.get_group_rank(group, dist.get_rank())

    @staticmethod
    def get_world_size():
        """
        Get the total number of processes in the distributed group.

        Returns:
            int: The world size.
        """
        return dist.get_world_size()

    @staticmethod
    def get_param_local_shape(param):
        """
        Get the local shape of a parameter, handling both regular and distributed tensors.

        Args:
            param (Union[Tensor, DTensorBase]): The parameter tensor.

        Returns:
            torch.Size: The local shape of the parameter.
        """
        if isinstance(param, DTensorBase):
            return param.local_shape
        return param.shape

    @staticmethod
    def get_param_local_data(param):
        """
        Get the local data of a parameter, handling both regular and distributed tensors.

        Args:
            param (Union[Tensor, DTensorBase]): The parameter tensor.

        Returns:
            Tensor: The local tensor data.
        """
        if isinstance(param, DTensorBase):
            return param.to_local()
        return param

    @staticmethod
    def update_param_data(param, data):
        """
        Update the data of a parameter.

        Args:
            param (Parameter): The parameter to update.
            data (Tensor): The new data tensor.
        """
        param.data = data

    @staticmethod
    def load_into_param(param, data):
        """Load tensor *data* into *param* (plain tensor or DTensor)."""
        if isinstance(param, DTensorBase):
            local = param._local_tensor # pylint: disable=W0212
            if local.is_meta:
                # Meta tensor materialisation: replace the placeholder.
                orig_requires_grad = param.requires_grad
                param._local_tensor = data # pylint: disable=W0212
                if data.requires_grad != orig_requires_grad:
                    param.requires_grad_(orig_requires_grad)
            else:
                local.copy_(data)
        else:
            param.copy_(data)

    @staticmethod
    def get_op_name(func):
        """
        Extract the operation name from various function types.

        Args:
            func: The function or operation to extract the name from.

        Returns:
            str: The operation name.
        """
        if hasattr(func, "__name__"):
            return func.__name__
        if isinstance(func, OpOverload):
            full_name = func.name
            core_name = full_name.split("::")[-1].split(".")[0]
            return core_name
        if isinstance(func, OpOverloadPacket):
            return func.name.split("::")[-1]
        func_str = str(func)
        if "built-in function" in func_str:
            return func_str.split()[-1].strip(">")
        if "function" in func_str:
            return func_str.split()[1]
        return "unknown_op"

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim, rank_list=None):
        data = _ensure_contiguous(data)
        output = list(dist_func.all_gather(data, group=group))
        if rank_list is not None:
            group_ranks = dist.get_process_group_ranks(group)
            if tuple(rank_list) != tuple(group_ranks):
                rank_to_idx = {int(rank): idx for idx, rank in enumerate(group_ranks)}
                output = [output[rank_to_idx[int(rank)]] for rank in rank_list]
        return torch.cat(output, dim=concat_dim)

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        return torch.chunk(data, split_size, dim=split_dim)[index]

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        input_data = _ensure_contiguous(input_data)
        output_tensor = torch.empty(output_shape, device=input_data.device, dtype=input_data.dtype)
        output_tensor = dist_func.all_to_all_single(
            output_tensor,
            input_data,
            group=group
        )
        return output_tensor

    @staticmethod
    def tensor_type_cast(input_data, cast_type):
        """Cast tensor to specified data type."""
        type_mapping = {
            'float32': torch.float32,
            'float16': torch.float16,
            'int64': torch.int64,
            'int32': torch.int32
        }
        if cast_type not in type_mapping:
            raise ValueError(f"Unknown cast type: {cast_type}. Supported types: {list(type_mapping.keys())}")
        return input_data.to(type_mapping[cast_type])

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        data = _ensure_contiguous(data)
        # Resolve the op from string to ReduceOp enum if necessary
        reduce_op = _OP_MAP.get(op, dist.ReduceOp.SUM) if isinstance(op, str) else op
        return dist_func.all_reduce(data, op=reduce_op, group=group)

    @staticmethod
    def get_cell_construct(cell):
        return cell.forward

    @staticmethod
    def get_cells_and_names(cell):
        return cell.named_modules()

    @staticmethod
    def get_modules(module):
        return module.modules()

    @staticmethod
    def search_parameter_by_name(cell, param_name: str):
        """
        Find the parent Module of the parameter, the parameter's name in the parent Module, and the parameter.
        Return value: (parent Module instance, parameter's name in parent Module, parameter object).
        Returns None if not found.
        """
        # Remove the "self." prefix from param_name
        param_name = param_name.replace("self.", "")
        # Case 1: The parameter is a direct parameter of the current Module
        if param_name in cell._parameters:  # pylint: disable=protected-access
            return (cell, param_name, cell._parameters[param_name])  # pylint: disable=protected-access

        # Case 2: The parameter is in a sub-Module
        if "." in param_name:
            cell_path, param_key = param_name.rsplit(".", 1)
            try:
                # Locate the sub-Module where the parameter resides (supports multi-level paths)
                target_cell = cell.get_submodule(cell_path)
                # Check if the sub-Module directly contains this parameter
                if param_key in target_cell._parameters:  # pylint: disable=protected-access
                    return target_cell, param_key, target_cell._parameters[param_key]  # pylint: disable=protected-access
            except AttributeError:
                pass

        # Traverse all sub-Modules (recursively) to search for the parameter
        for _, child_cell in cell.named_children():
            if isinstance(child_cell, Module):
                result = TorchPlatform.search_parameter_by_name(child_cell, param_name)
                if result is not None:
                    return result

        return None

    @staticmethod
    def update_parameter_by_name(cell, result: tuple, new_param) -> bool:
        """
        Modify the original parameter in a Module or sub-Module using the search result
        """
        parent_cell, param_key, _ = result
        # Key operation: directly modify the _parameters dictionary.
        if param_key in parent_cell._parameters:  # pylint: disable=protected-access
            parent_cell._parameters[param_key] = new_param  # pylint: disable=protected-access
        else:
            parent_cell.register_parameter(param_key, new_param)
        return True

    @staticmethod
    def set_layout_into_parameter(param, layout):
        """Set layout into parameter"""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=import-outside-toplevel
        from hyper_parallel.core.dtensor.layout import _get_slice_tensor_by_layout  # pylint: disable=import-outside-toplevel
        if isinstance(param, DTensor):
            raise ValueError(f"Parameter {param} has been configured layout, cannot be set repeatedly.")
        requires_grad = param.requires_grad
        param_dtensor = DTensor.from_local(
            _get_slice_tensor_by_layout(param, layout),
            layout.mesh, layout.alias_placements)
        new_param = Parameter(param_dtensor, requires_grad=requires_grad)
        return new_param

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        data = _ensure_contiguous(data)
        input_tuple = torch.chunk(data, dev_num, dim=axis)
        output_tensor = torch.empty(input_tuple[0].shape, device=data.device, dtype=data.dtype)

        # Resolve the op from string to ReduceOp enum
        reduce_op = _OP_MAP.get(op, dist.ReduceOp.SUM) if isinstance(op, str) else op

        output_tensor = dist_func.reduce_scatter(output_tensor, input_tuple, op=reduce_op, group=group)

        # Keep manual handling for 'avg' string as it maps to SUM in _OP_MAP
        if op == 'avg':
            output_tensor = output_tensor / dev_num
        return output_tensor

    @staticmethod
    def get_device_handle(device_type: str = "npu"):
        """Return the torch device module (e.g. ``torch.npu`` or ``torch.cuda``) for the given device type."""
        try:
            handle =  getattr(torch, device_type)
        except AttributeError as e:
            raise RuntimeError(f"TorchPlatform expect got device handle: 'torch.{device_type}' failed.") from e
        return handle

    @staticmethod
    def get_param_type_size(param):
        # pylint: disable=W0212
        return torch._utils._element_size(param.dtype)

    @staticmethod
    def is_tensor(obj: Any) -> bool:
        """Return True if ``obj`` is a ``torch.Tensor``."""
        return isinstance(obj, Tensor)

    @staticmethod
    def get_tensor_storage_size(tensor: Any) -> int:
        """Return serialized byte size (numel * element size) for a PyTorch tensor."""
        if not TorchPlatform.is_tensor(tensor):
            raise TypeError(
                f"TorchPlatform.get_tensor_storage_size expects torch.Tensor, got {type(tensor)!r}"
            )
        return int(tensor.numel()) * int(tensor.element_size())

    @staticmethod
    def parameters_dict(cell: Module):
        return cell.named_parameters()

    @staticmethod
    def buffers_dict(cell: Module) -> Any:
        """Return all named buffers registered by the module tree."""
        return cell.named_buffers()

    @staticmethod
    def get_model_state_dict(model: Any, *, options: Any = None) -> dict[str, Any]:
        """Get the state dictionary of a model.

        Delegates to torch-specific implementation that handles DTensor
        gathering, CPU offloading and frozen-parameter filtering.
        """
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.fully_shard.state_dict_utils import (
            get_model_state_dict as _get_model_state_dict,
        )
        return _get_model_state_dict(model, options=options)

    @staticmethod
    def set_model_state_dict(model: Any, model_state_dict: dict[str, Any], *, options: Any = None) -> None:
        """Set the state dictionary of a model.

        Delegates to torch-specific implementation that scatters full tensors
        into DTensor shards and performs an in-place load.
        """
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.fully_shard.state_dict_utils import (
            set_model_state_dict as _set_model_state_dict,
        )
        return _set_model_state_dict(model, model_state_dict, options=options)

    @staticmethod
    def save_checkpoint(cell: Module, file_path: str, ckpt_format: str = "safetensors") -> None:
        if ckpt_format == "safetensors":
            save_file(tensors=cell, filename=file_path)
        else:
            torch.save(obj=cell, f=file_path)

    @staticmethod
    def load_checkpoint(file_path: str, ckpt_format: str = "safetensors") -> dict:
        if ckpt_format == "safetensors":
            return load_file(filename=file_path)
        return torch.load(f=file_path)

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad, device):
        return nn.Parameter(torch.zeros(param_shape, dtype=param_type, device=device), requires_grad=requires_grad)

    @staticmethod
    def new_tensor(tensor_shape, tensor_type, device):
        return torch.empty(size=tensor_shape, dtype=tensor_type, device=device)

    @staticmethod
    def full_like(tensor, fill_value, dtype=None):
        return torch.full_like(tensor, fill_value, dtype=dtype)

    @staticmethod
    def set_tensor_requires_grad(input_tensor):
        """
        set requires grad flag for input tensor, only effective for leaf node
        """
        if input_tensor.is_leaf:
            input_tensor.requires_grad = True

    def _create_group(self, rank_list):
        normalized_rank_list = tuple(sorted(rank_list))
        world_rank_list = tuple(range(self.get_world_size()))
        if normalized_rank_list == world_rank_list:
            group = _get_default_group()
            EXISTING_COMM_GROUPS[str(normalized_rank_list)] = group
            return group
        group_dict = create_sub_groups(rank_list)
        return group_dict[normalized_rank_list]

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        output_shape = list(data.shape)
        output_shape[0] = output_shape[0] * group_info.rank_size
        output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
        handle = dist.all_gather_into_tensor(output, data, group=group_info.group, async_op=async_op)
        return output, handle

    @staticmethod
    def all_gather_single(input_tensor, output_shape, group, async_op=False):
        output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        handle = dist.all_gather_into_tensor(output, input_tensor, group=group, async_op=async_op)
        return output, handle

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        if not data.is_contiguous():
            data = data.contiguous()
        handle = dist.all_reduce(data, group=group_info.group, async_op=async_op)
        return data, handle

    @staticmethod
    def broadcast(data, src=None, group=None, async_op=False, group_src=None):
        if group_src is not None:
            src = dist.get_global_rank(group, group_src)
        handle = dist.broadcast(data, src, group, async_op)
        if async_op and handle is not None:
            handle.wait()

    @staticmethod
    def scatter(output, scatter_list, src=None, group=None, async_op=False, group_src=None):
        if group_src is not None:
            src = dist.get_global_rank(group, group_src)
        handle = dist.scatter(output, scatter_list, src=src, group=group, async_op=async_op)
        if async_op and handle is not None:
            handle.wait()
        return output

    @staticmethod
    def isend(tensor, dst=None, group=None, tag=0):
        return dist.isend(tensor, dst, group, tag)

    @staticmethod
    def irecv(tensor, src=None, group=None, tag=0):
        return dist.irecv(tensor, src, group, tag)

    @staticmethod
    def p2p_op(op_type, tensor, peer, group=None):
        # torch's P2POp takes the op callable (dist.isend / dist.irecv), not
        # the "isend"/"irecv" string the stage specs builders emit.
        if op_type == "isend":
            op = dist.isend
        elif op_type == "irecv":
            op = dist.irecv
        else:
            raise ValueError(
                f"p2p_op op_type must be 'isend' or 'irecv', but got {op_type!r}."
            )
        return dist.P2POp(op, tensor, peer, group)

    @staticmethod
    def batch_isend_irecv(p2p_ops):
        """Launch a peer-batched P2P group as one coalesced op.

        ``torch.distributed.batch_isend_irecv`` coalesces the ops onto one
        comm stream and returns one ``Work`` per op; we wrap them in a single
        ``.wait()`` handle so a send and a recv to the same peer overlap on
        the duplex link and the caller can defer the whole batch's wait to one
        consumption point.
        """
        if not p2p_ops:
            return None
        works = dist.batch_isend_irecv(p2p_ops)
        return _TorchBatchP2PWork(works) if works else None

    @staticmethod
    def prepare_batch_p2p_group(group: Any = None) -> None:
        """Synchronize a group before its first subset batched P2P call.

        PyTorch requires every rank in a process group to participate when
        ``batch_isend_irecv`` is the first collective on that group. A barrier
        at the common pipeline run boundary initializes the communicator
        before ranks reach peer operations at different times.

        Args:
            group: The process group used by the batched P2P operations.
                ``None`` uses the default group.
        """
        dist.barrier(group=group)

    @staticmethod
    def p2p_exchange(tensor, peer_rank: int, group=None):
        if peer_rank == dist.get_rank(group):
            return tensor
        return _TorchP2PExchangeFunction.apply(tensor, peer_rank, group)

    @staticmethod
    def send_object_list(obj_list, dst=None, group=None):
        dist.send_object_list(obj_list, dst, group)

    @staticmethod
    def recv_object_list(obj_list, src=None, group=None):
        dist.recv_object_list(obj_list, src, group)

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        output_shape = list(data.shape)
        output_shape[0] = output_shape[0] // group_info.rank_size
        output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
        handle = dist.reduce_scatter_tensor(output, data, group=group_info.group, async_op=async_op)
        return output, handle

    @staticmethod
    def reduce_scatter_single(input_tensor, output_shape, group, async_op=False):
        output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        handle = dist.reduce_scatter_tensor(output, input_tensor, group=group, async_op=async_op)
        return output, handle

    @staticmethod
    def all_to_all_single(input_tensor, output_shape, group, async_op=False):
        output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
        work = dist.all_to_all_single(output, input_tensor, group=group, async_op=async_op)
        return output, work

    @staticmethod
    def differentiable_all_to_all_single(input_tensor, input_splits, output_splits, group):
        """Variable-split all-to-all with autograd support for EP token dispatch/combine."""
        out_total = sum(output_splits)
        output = torch.empty(
            out_total, *input_tensor.shape[1:],
            dtype=input_tensor.dtype, device=input_tensor.device,
        )
        output = dist_func.all_to_all_single(
            output, input_tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return output

    @staticmethod
    def differentiable_all_to_all_single_async(input_tensor, input_splits, output_splits, group):
        """Truly-async variant of :meth:`differentiable_all_to_all_single`.

        Both forward AND backward return :class:`AsyncCollectiveTensor`,
        so the ``wait_tensor`` op is queued lazily — only when a downstream
        kernel actually reads the result.

        Why both directions need lazy wait:

        * FWD: ACT lazy wait lets host return immediately and the paired
          BWD thread's compute kernel slip into the queue before the wait.
        * BWD: PyTorch's stock backward issues ``wait_tensor`` eagerly,
          and the autograd engine binds backward stream to the forward
          stream — so even running BWD inside a ``with torch.npu.stream
          (side_stream)`` context does not move that wait off the main
          stream.  Returning ACT from backward defers the wait to the
          next backward op's first consumption, opening a small window
          during which FWD's Attention kernels can be queued onto the
          main stream **before** the wait lands.

        Args:
            input_tensor:  Input tensor, split along dim 0 by ``input_splits``.
            input_splits:  ``list[int]`` — rows sent to each rank.
            output_splits: ``list[int]`` — rows received from each rank.
            group:         Process group.

        Returns:
            ``AsyncCollectiveTensor`` of shape
            ``[sum(output_splits), *input_tensor.shape[1:]]``.
        """
        return _AsyncA2ALazyBwd.apply(input_tensor, output_splits, input_splits, group)

    @staticmethod
    def differentiable_variable_all_gather(
            input_tensor: Tensor, output_splits: Sequence[int], group: Any) -> Tensor:
        """Gather variable dim-zero shards on HCCL or Gloo with autograd support."""
        return _TorchDifferentiableVariableAllGather.apply(
            input_tensor, tuple(output_splits), group
        )

    @staticmethod
    def wait_async_tensor(tensor):
        """Wait for an async collective tensor to become materialised.

        Idempotent — calling on an already-waited tensor is a no-op.

        Args:
            tensor: ``AsyncCollectiveTensor`` whose device-side values may
                not yet be ready.

        Returns:
            The same *tensor*, now fully materialised.
        """
        from torch.distributed._functional_collectives import wait_tensor  # pylint: disable=C0415
        wait_tensor(tensor)
        return tensor

    @staticmethod
    def differentiable_async_allgather_wait(x, work, out_perm, group, world_size, gather_dim,
                                            handle_box=None):
        """Wait async all-gather handle and reconstruct result (differentiable)."""
        return _TorchAsyncAllGatherFunction.apply(
            x, work, out_perm, group, world_size, gather_dim, handle_box
        )

    @staticmethod
    def arange(start, end=None, step=1, dtype=None, device=None):
        """Create a 1-D tensor with evenly spaced values."""
        if end is None:
            return torch.arange(start, dtype=dtype, device=device)
        return torch.arange(start, end, step, dtype=dtype, device=device)

    @staticmethod
    def differentiable_async_a2a_wait(x, work, out_perm, group, world_size, concat_dim, split_dim,
                                      handle_box=None):
        """Wait async A2A handle and reconstruct result (differentiable).

        Args:
            x: Input tensor.
            work: Async work handle from all_to_all.
            out_perm: Output buffer from all_to_all.
            group: Process group.
            world_size: World size.
            concat_dim: Dimension for concatenation.
            split_dim: Dimension for split.
            handle_box: Optional mutable list; backward appends (work, out_perm) here.
        """
        return _TorchAsyncA2AFunction.apply(
            x, work, out_perm, group, world_size, concat_dim, split_dim, handle_box
        )

    @staticmethod
    def differentiable_sync_hook(x, hook_name: str, coordinator):
        """Identity op that fires coordinator rendezvous on forward and backward.

        Always goes through ``_TorchSyncHookFunction.apply`` so that the
        autograd graph **records a SyncHook node regardless of whether the
        coordinator is currently enabled**.  Skipping ``apply`` when
        disabled would leave warmup-forwarded graphs without the hook
        nodes, and a later ``overlap.run`` — whose BWD thread back-props
        such a graph — would then traverse zero hooks while the paired FWD
        thread (whose current forward DOES record hooks) waits at a
        barrier for a partner that never arrives.

        Args:
            x:           Input tensor.
            hook_name:   One of:
                         * ``"A"`` / ``"B"`` / ``"C"`` / ``"D"`` —
                           full rendezvous on both directions.
                         * ``"D_LAST"`` — closing D of the last MoE
                           layer in a chunk.  Forward: ``notify_dispatched``
                           only (no Attention follows so rendezvous is
                           skipped).  Backward: pure skip (first BWD
                           hook to fire; combine.bwd has already
                           dispatched freely).
            coordinator: A :class:`HookCoordinator` instance.
        """
        return _TorchSyncHookFunction.apply(x, hook_name, coordinator)

    @staticmethod
    def get_tensor_transform():
        raise NotImplementedError("Unsupported get_tensor_transform for torch platform")

    @staticmethod
    def construct_strided_slice(x, begin, end, stride):
        raise NotImplementedError("Unsupported construct_strided_slice for torch platform")

    @staticmethod
    def micro_batch(micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.pipeline_parallel._utils import _MicroBatch
        return _MicroBatch(micro_batch_num, args_batch_dim, kwargs_batch_dim)

    @staticmethod
    def get_symmetric_memory_handler():
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.symmetric_memory import TorchSymmetricMemoryHandler
        symmetric_memory = TorchSymmetricMemoryHandler()
        return symmetric_memory

    @staticmethod
    def get_multicore_handler():
        """Return a TorchMulticoreHandler instance for multi-core device management."""
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.multicore import TorchMulticoreHandler
        return TorchMulticoreHandler()

    def new_stream(self):
        device = self.get_device_handle()
        return device.Stream()

    def get_stream_context(self):
        device = self.get_device_handle()
        return device.stream

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
        Synchronize all processes in the given process group.

        Args:
            group (ProcessGroup, optional): The process group to work on. Default is ``None``,
                meaning the default process group.
            async_op (bool, optional): Whether this op should be asynchronous. Default: ``False``.
            device_ids (list[int], optional): Device ids for backends that require a device for
                barrier (e.g. NCCL). Default: ``None``.

        Returns:
            Async work handle if ``async_op`` is True; otherwise ``None``.
        """
        return dist.barrier(group, async_op, device_ids)

    @staticmethod
    def init_process_group(
            backend: Optional[str] = None,
            *,
            init_method: Optional[str] = None,
            timeout: Optional[timedelta] = None,
            world_size: int = -1,
            rank: int = -1,
            store: Optional[Store] = None,
            pg_options: Optional[Any] = None,
            device_id: Optional[Union[torch.device, int]] = None,
    ) -> None:
        """
        Initialize global process group.

        Args:
            backend (str or Backend, optional): The backend to use for distributed communication.
            init_method (str, optional): URL specifying how to initialize the process group. Default is "env://",
                can not be specified at the same time with ``store``.
            timeout (timedelta, optional): Timeout for process group. Default 10 minutes for NCCL and for other
                backends 30 minutes.
            world_size (int, optional): Number of processes. If ``store`` is specified, world_size is required.
            rank (int, optional): Rank of the current process, which value must between 0 and ``world_size``-1. If
                ``store`` is specified, rank is required.
            store (Store, optional): Key/value store accessible to all workers, used to exchange connection/address
                information. Can not be specified at the same time with ``init_method``.
            pg_options (ProcessGroupOptions, optional): Extra options to pass during constructing process groups.
            device_id (torch.device | int, optional): Specific device this process will work on.
        """
        try:
            _get_default_group()
        # except multi version error
        except (ValueError, RuntimeError):
            if backend is None:
                backend = "hccl"
            dist.init_process_group(backend=backend, init_method=init_method, timeout=timeout, world_size=world_size,
                                    rank=rank, store=store, pg_options=pg_options, device_id=device_id)

    @staticmethod
    def destroy_process_group(group: Optional[ProcessGroup] = None) -> None:
        """
        Destroy given process group.

        Args:
            group (ProcessGroup, optional): Given process group will be destroyed, if not given, all process groups
                will be destroyed.
        """
        group = group or _get_default_group()
        if group in EXISTING_COMM_GROUPS.values():
            keys_to_destroy = [k for k, v in EXISTING_COMM_GROUPS.items() if v == group]
            for k in keys_to_destroy:
                del EXISTING_COMM_GROUPS[k]
        dist.destroy_process_group(group)

    @staticmethod
    def get_process_group_ranks(group: Optional[ProcessGroup] = None) -> list[int]:
        """
        Get all ranks relative to given process group.

        Args:
            group (Optional[ProcessGroup]): Process group worked on. Default is ``None``, and ``None`` means global
                group.

        Returns:
            Rank list.
        """
        group = group or _get_default_group()
        return dist.get_process_group_ranks(group)

    @staticmethod
    def get_backend(group: Optional[ProcessGroup] = None) -> Backend:
        """
        Get the backend of the given process group.

        Args:
            group (ProcessGroup, optional): Process group worked on. Default is ``None``, and ``None`` means global
                group.

        Returns:
            The backend object of the given process group.
        """
        group = group or _get_default_group()
        return dist.get_backend(group)

    @staticmethod
    def split_group(parent_pg: Optional[ProcessGroup] = None,
                    split_ranks: Optional[list] = None,
                    timeout: Optional[timedelta] = None,
                    pg_options: Optional[Any] = None,
                    group_desc: Optional[str] = None,
                    ) -> Optional[ProcessGroup]:
        """
        Create split groups for every group rank in split_ranks, and return the split process group which relative to
            current rank id.

        Args:
            parent_pg (Optional[ProcessGroup]): A process group which the goal group split from.
            split_ranks (Optional[list]): A list like ``list[list[int]]``.
            timeout (Optional[timedelta]): Timeout for process group. Default 10 minutes for NCCL and for other
                backend 30 minutes.
            pg_options (Optional[Any]): Extra options to pass during constructing process groups.
            group_desc (Optional[str]): Description of process group.

        Return:
            Optional[ProcessGroup]: One of split process group which relative to current rank id
        """
        if split_ranks is None or len(split_ranks) == 0:
            raise ValueError("split_ranks cannot be None or empty")

        split_group = None
        for split_rank in split_ranks:
            dist_group = TorchPlatform.get_created_group(split_rank)
            if dist_group is None:
                dist_group = dist.new_group(ranks=split_rank)
                EXISTING_COMM_GROUPS[str(tuple(sorted(split_rank)))] = dist_group
            if TorchPlatform.get_rank() in split_rank:
                split_group = dist_group

        return split_group

    @staticmethod
    def get_group_local_rank(group: ProcessGroup = None) -> int:
        """get group local rank id."""
        group = group or _get_default_group()
        return group.rank()

    @staticmethod
    def no_grad():
        return torch.no_grad()

    @staticmethod
    def preserve_version_counter(tensor):
        return torch.autograd._unsafe_preserve_version_counter(tensor)  # pylint: disable=W0212

    @staticmethod
    def relu(tensor):
        return torch.relu(tensor)

    @staticmethod
    def cat(tensors, dim=0):
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def empty_like(tensor, *, dtype=None, device=None, pin_memory=False):
        return torch.empty_like(tensor, dtype=dtype, device=device, pin_memory=pin_memory)

    def get_current_stream(self):
        device = self.get_device_handle()
        return device.current_stream()

    def new_event(self):
        device = self.get_device_handle()
        return device.Event()

    def tree_map(self, fn, tree):
        return torch.utils._pytree.tree_map(fn, tree)  # pylint: disable=protected-access

    @property
    def checkpoint(self):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.checkpoint import checkpoint
        return checkpoint

    @staticmethod
    def recompute_handle_collector_ctx():
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.checkpoint import recompute_handle_collector_ctx
        return recompute_handle_collector_ctx()

    @staticmethod
    def recompute_handle(handle, session_id):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.checkpoint import recompute_handle
        return recompute_handle(handle, session_id)

    @staticmethod
    def recompute_session_ctx(session_id, retain_on_unpack=False):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.checkpoint import recompute_session_ctx
        return recompute_session_ctx(session_id=session_id, retain_on_unpack=retain_on_unpack)

    @staticmethod
    def clear_recompute_session(session_id):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.checkpoint import clear_recompute_session
        return clear_recompute_session(session_id)

    @staticmethod
    def checkpoint_wrapper(module, **checkpoint_kwargs):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.checkpoint_wrapper import ckpt_wrapper
        return ckpt_wrapper(module, **checkpoint_kwargs)

    @staticmethod
    def swap_wrapper(module, policy_fn=None, group_swap=False):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import swap_wrapper
        return swap_wrapper(module, policy_fn=policy_fn, group_swap=group_swap)

    @staticmethod
    def swap_tensor_wrapper(target, tag=None, group_swap=False):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import swap_tensor_wrapper
        return swap_tensor_wrapper(target, tag=tag, group_swap=group_swap)

    @staticmethod
    def get_class_activation_wrapper():
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import ActivationWrapper
        return ActivationWrapper

    @property
    def noop_context_fn(self):
        return noop_context_fn

    @staticmethod
    def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False, group_swap=False):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.sac import create_selective_checkpoint_contexts
        return create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation, group_swap)

    @staticmethod
    def async_save_on_cpu(policy_fn=None, group_swap: bool = False):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.activation_checkpoint.activation_swap import AsyncSaveOnCpu
        return AsyncSaveOnCpu(policy_fn, group_swap=group_swap)

    @staticmethod
    def get_element_size(tensor):
        """Get Tensor Element Size"""
        return tensor.element_size()

    @staticmethod
    def alloc_tensor_buffer(numel: int, dtype, device, pin_memory: bool = False):
        """Allocate an uninitialized 1-D tensor buffer."""
        if pin_memory:
            return torch.empty(numel, dtype=dtype, device='cpu', pin_memory=True)
        return torch.empty(numel, dtype=dtype, device=device)

    @staticmethod
    def tensor_to_numpy(tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return tensor.cpu().numpy()

    @staticmethod
    def from_numpy(np_array):
        """Create a host (CPU) PyTorch tensor from a numpy array."""
        return torch.from_numpy(np_array)

    @staticmethod
    def clip_grad_norm_(
        parameters, max_norm, norm_type=2.0,
        error_if_nonfinite=False, foreach=None,
    ):
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.clip_grad import (
            clip_grad_norm_ as _clip_grad_norm,
        )
        return _clip_grad_norm(
            parameters, max_norm, norm_type,
            error_if_nonfinite=error_if_nonfinite, foreach=foreach,
        )

    @staticmethod
    def profiler_record(name):
        """Profiler context manager for recording operations using torch.profiler."""
        return torch.profiler.record_function(name)

    def cast_fp_tensor(self, dtype, x):
        """
        Cast floating-point tensor to target dtype if applicable.
        """
        if (
            not isinstance(x, torch.Tensor)
            or not torch.is_floating_point(x)
            or x.dtype == dtype
        ):
            return x
        return x.to(dtype)

    def apply_to_tensors(self, fn, container):
        """Recursively apply to all tensor in different kinds of container types."""

        def apply(x):

            if isinstance(x, torch.Tensor):
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
            if isinstance(x, PackedSequence):
                apply(x.data)
                return x
            if isinstance(x, dict):
                return {key: apply(value) for key, value in x.items()}
            if isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields"):
                res = (apply(el) for el in x)
                return type(x)(*res)
            if isinstance(x, (list, tuple, set)):
                return type(x)(apply(el) for el in x)
            return x

        return apply(container)


    @property
    def meta_device(self):
        return torch.device("meta")

    def init_on_device(self, device, include_buffers=False):
        return _init_on_device(device, include_buffers=include_buffers)

    def str_to_dtype(self, dtype_str: str) -> torch.dtype:
        """Map ``torch.<type>`` strings from checkpoint metadata to ``torch.dtype``."""
        parts = dtype_str.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Expected dtype string like 'torch.float32', got {dtype_str!r}."
            )
        prefix, name = parts
        if prefix != "torch":
            raise ValueError(
                f"Expected PyTorch dtype string with prefix 'torch', got {dtype_str!r}."
            )
        dtype = getattr(torch, name)
        if isinstance(dtype, torch.dtype):
            return dtype
        raise ValueError(f"{dtype_str!r} does not resolve to a torch.dtype.")

    def list_to_size(self, size_list: list[int]) -> torch.Size:
        return torch.Size(size_list)
