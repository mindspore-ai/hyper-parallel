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
"""Communication helpers shared by the ``core`` modules.

Everything here is collective-related and therefore Torch-only:
differentiable collective wrappers (plus the autograd ``Function``s backing
them), the process-group cache, and the P2P exchange.  The synchronous
collectives each caller needs are thin enough to stay local; what is shared
here is the part with nontrivial autograd/stream-lifetime semantics, which
must stay identical across dtensor, expert_parallel and context_parallel.
"""
# pylint: disable=C9006,C9007
from typing import Any, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_func
from torch import Tensor
from torch._C._distributed_c10d import ProcessGroup


def get_group_local_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the current rank's index within ``group``, or 0 before init.

    Args:
        group: The group to index into. ``None`` means the default group.

    Returns:
        int: The rank of the current process inside ``group``.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    if group is None:
        return torch.distributed.get_rank()
    if hasattr(group, "rank"):
        return group.rank()
    return torch.distributed.get_group_rank(group, torch.distributed.get_rank())


def get_device_handle(device_type: str = "npu"):
    """Return the ``torch`` device module for ``device_type``.

    Args:
        device_type: Device backend name, e.g. ``"npu"`` or ``"cuda"``.

    Returns:
        The matching ``torch.<device_type>`` module.

    Raises:
        RuntimeError: If torch exposes no module for ``device_type``.
    """
    try:
        handle = getattr(torch, device_type)
    except AttributeError as e:
        raise RuntimeError(f"Failed to resolve device handle: 'torch.{device_type}'.") from e
    return handle


def get_world_size() -> int:
    """Return the Torch distributed world size.

    Kept as a thin wrapper so single-process call sites and white-box unit
    tests share one patchable seam; ``torch.distributed.get_world_size`` itself
    raises before ``init_process_group``.
    """
    return dist.get_world_size()


# Process-group cache keyed by the string form of the sorted rank tuple, e.g.
# ``"(0, 1, 2, 3)"``. Lives here rather than in a module-specific ``utils`` so
# that every sub-group creator in ``core`` shares one map. Values are
# framework-native groups, hence ``Any``.
EXISTING_COMM_GROUPS: dict[str, Any] = {}


def ensure_contiguous(x):
    """Return a contiguous copy of *x* if not already contiguous.

    Under ``torch.compile`` the contiguity of a graph input is not knowable at
    trace time, so the copy is emitted unconditionally.
    """
    if torch.compiler.is_compiling():
        return x.contiguous()
    if not x.is_contiguous() or x.storage_offset() != 0:
        return x.contiguous()
    return x


# ---------------------------------------------------------------------------
# Process group helpers
# ---------------------------------------------------------------------------

def get_created_group(rank_list: Union[list[int], tuple[int, ...]]):
    """Return an existing process group by rank list, or ``None``."""
    group_key = str(tuple(sorted(rank_list)))
    if group_key in EXISTING_COMM_GROUPS:
        return EXISTING_COMM_GROUPS[group_key]
    return None


def split_group(parent_pg=None,
                split_ranks: Optional[list] = None,
                timeout=None,
                pg_options=None,
                group_desc=None,
                ):
    """Create split groups for every rank list in *split_ranks*.

    Returns the split process group relative to the current rank id.
    """
    del parent_pg, timeout, group_desc
    if split_ranks is None or len(split_ranks) == 0:
        raise ValueError("split_ranks cannot be None or empty")

    split_group_pg = None
    for split_rank in split_ranks:
        dist_group = get_created_group(split_rank)
        if dist_group is None:
            dist_group = dist.new_group(ranks=split_rank, pg_options=pg_options)
            EXISTING_COMM_GROUPS[str(tuple(sorted(split_rank)))] = dist_group
        if dist.get_rank() in split_rank:
            split_group_pg = dist_group

    return split_group_pg


def init_process_group(*args, **kwargs) -> None:
    """Initialize the default torch distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(*args, **kwargs)


# ---------------------------------------------------------------------------
# Differentiable collectives
# ---------------------------------------------------------------------------

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


def resolve_reduce_op(op: Union[str, Any]) -> Any:
    """Resolve a string op name (or pass through an already-resolved ``ReduceOp``)."""
    return _OP_MAP.get(op, dist.ReduceOp.SUM) if isinstance(op, str) else op


class _TorchContiguousGrad(torch.autograd.Function):  # pylint: disable=abstract-method
    """Autograd identity that materializes gradients before upstream collectives."""

    @staticmethod
    def forward(ctx: Any, tensor: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """Return the input unchanged in the forward pass."""
        del ctx
        return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """Return a contiguous gradient to the preceding autograd node."""
        del ctx
        return grad_output.contiguous()


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


class _TorchP2PExchangeFunction(torch.autograd.Function):
    """Symmetric bidirectional P2P: send local tensor to peer, receive peer's tensor."""

    @staticmethod
    def forward(ctx, tensor: Tensor, peer_rank: int, group) -> Tensor:  # pylint: disable=arguments-differ
        """Perform symmetric bidirectional P2P exchange with ``peer_rank``."""
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
    def backward(ctx, grad_output: Tensor):
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


def differentiable_all_gather_concat(data: Tensor, group, concat_size: int, concat_dim: int,
                                     rank_list=None) -> Tensor:
    """Autograd-aware all-gather whose results are concatenated along ``concat_dim``.

    Args:
        data: Local shard to gather.
        group: Process group to gather over.
        concat_size: Unused; kept for call-site compatibility.
        concat_dim: Dimension the gathered shards are concatenated along.
        rank_list: Optional global-rank order for the gathered shards.  When it
            differs from the group's own rank order the shards are permuted, so
            callers can request mesh order without changing the group.

    Returns:
        The gathered shards concatenated along ``concat_dim``.
    """
    del concat_size
    data = ensure_contiguous(data)
    output = [
        _TorchContiguousGrad.apply(tensor)
        for tensor in dist_func.all_gather(data, group=group)
    ]
    if rank_list is not None:
        group_ranks = dist.get_process_group_ranks(group)
        if tuple(rank_list) != tuple(group_ranks):
            rank_to_idx = {int(rank): idx for idx, rank in enumerate(group_ranks)}
            output = [output[rank_to_idx[int(rank)]] for rank in rank_list]
    return torch.cat(output, dim=concat_dim)


def differentiable_all_to_all_single(input_tensor: Tensor, input_splits: Sequence[int],
                                     output_splits: Sequence[int], group) -> Tensor:
    """Variable-split all-to-all with autograd support for EP token dispatch/combine."""
    out_total = sum(output_splits)
    output = torch.empty(
        out_total, *input_tensor.shape[1:],
        dtype=input_tensor.dtype, device=input_tensor.device,
    )
    return dist_func.all_to_all_single(
        output, input_tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )


def differentiable_all_to_all_single_async(input_tensor: Tensor, input_splits: Sequence[int],
                                           output_splits: Sequence[int], group) -> Tensor:
    """Truly-async variant of :func:`differentiable_all_to_all_single`.

    Both forward AND backward return ``AsyncCollectiveTensor``, so the
    ``wait_tensor`` op is queued lazily — only when a downstream kernel
    actually reads the result.  See :class:`_AsyncA2ALazyBwd` for why the
    backward needs the same treatment.

    Args:
        input_tensor: Input tensor, split along dim 0 by ``input_splits``.
        input_splits: ``list[int]`` — rows sent to each rank.
        output_splits: ``list[int]`` — rows received from each rank.
        group: Process group.

    Returns:
        ``AsyncCollectiveTensor`` of shape
        ``[sum(output_splits), *input_tensor.shape[1:]]``.
    """
    return _AsyncA2ALazyBwd.apply(input_tensor, output_splits, input_splits, group)


def differentiable_all_to_all(input_data: Tensor, output_shape: Sequence[int], group) -> Tensor:
    """Autograd-aware all-to-all producing a tensor of ``output_shape``."""
    input_data = ensure_contiguous(input_data)
    output_tensor = torch.empty(output_shape, device=input_data.device, dtype=input_data.dtype)
    return dist_func.all_to_all_single(output_tensor, input_data, group=group)


def differentiable_all_reduce(data: Tensor, op: Union[str, Any], group) -> Tensor:
    """Autograd-aware all-reduce with string or ``ReduceOp`` *op*."""
    data = ensure_contiguous(data)
    return dist_func.all_reduce(data, op=resolve_reduce_op(op), group=group)


def differentiable_reduce_scatter(data: Tensor, dev_num: int, axis: int,
                                  op: Union[str, Any], group) -> Tensor:
    """Autograd-aware reduce-scatter splitting ``axis`` into ``dev_num`` parts."""
    data = ensure_contiguous(data)
    input_tuple = torch.chunk(data, dev_num, dim=axis)
    output_tensor = torch.empty(input_tuple[0].shape, device=data.device, dtype=data.dtype)

    output_tensor = dist_func.reduce_scatter(
        output_tensor, input_tuple, op=resolve_reduce_op(op), group=group
    )

    # 'avg' maps to SUM in _OP_MAP, so the division stays manual.
    if op == 'avg':
        output_tensor = output_tensor / dev_num
    return output_tensor


def differentiable_variable_all_gather(
        input_tensor: Tensor, output_splits: Sequence[int], group) -> Tensor:
    """Gather variable dim-zero shards on HCCL or Gloo with autograd support."""
    return _TorchDifferentiableVariableAllGather.apply(
        input_tensor, tuple(output_splits), group
    )


def wait_async_tensor(tensor: Tensor) -> Tensor:
    """Wait for an async collective tensor to become materialised.

    Idempotent — calling on an already-waited tensor is a no-op.
    """
    from torch.distributed._functional_collectives import wait_tensor  # pylint: disable=C0415
    wait_tensor(tensor)
    return tensor


def p2p_exchange(tensor: Tensor, peer_rank: int, group=None) -> Tensor:
    """Symmetric bidirectional P2P exchange with ``peer_rank``."""
    if peer_rank == dist.get_rank(group):
        return tensor
    return _TorchP2PExchangeFunction.apply(tensor, peer_rank, group)


def exchange_splits_via_all_to_all(input_tensor: Tensor, group) -> Tensor:
    """All-to-all a per-rank split vector, returning the received counts.

    Used by the token dispatchers to learn how many tokens every peer sends
    them; the exchange carries no gradient, so plain ``dist`` collectives are
    used.  Runs asynchronously with an explicit ``handle.wait()`` rather than
    ``async_op=False``: the implicit cross-stream sync is NCCL-only, and on
    HCCL the compute stream may read the output before the collective write is
    visible, producing garbage values that blow up downstream allocations.
    """
    output = torch.empty(
        [input_tensor.shape[0]],
        dtype=input_tensor.dtype, device=input_tensor.device,
    )
    handle = dist.all_to_all_single(
        output, input_tensor, group=group, async_op=True,
    )
    if handle is not None:
        handle.wait()
    return output


def gather_counts_via_all_gather(input_tensor: Tensor, group, world_size: int) -> Tensor:
    """All-gather a per-rank count vector into an ``[world_size, *shape]`` view.

    Same async-with-explicit-wait rationale as
    :func:`exchange_splits_via_all_to_all`.
    """
    output = torch.empty(
        [world_size * input_tensor.shape[0]],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    handle = dist.all_gather_into_tensor(
        output, input_tensor, group=group, async_op=True,
    )
    if handle is not None:
        handle.wait()
    return output.view(world_size, input_tensor.shape[0])
