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
"""Torch implementations backing ``hyper_parallel.core.dtensor``.

The DTensor core used to reach these routines through the platform abstraction
layer.  Only the torch backend is supported now, so the nontrivial torch
implementations live here and every caller calls them directly.
"""
# pylint: disable=C9006,C9007
from contextlib import contextmanager
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch._C._distributed_c10d import ProcessGroup
from torch._ops import OpOverload, OpOverloadPacket
from torch.distributed.distributed_c10d import _get_default_group

import torch.distributed as dist
import torch.distributed.nn.functional as dist_func
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


# ---------------------------------------------------------------------------
# Small shared helpers
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


def _ensure_contiguous(x):
    """Return a contiguous copy of *x* if not already contiguous."""
    if torch.compiler.is_compiling():
        return x.contiguous()
    return x if x.is_contiguous() else x.contiguous()


def get_op_name(func):
    """Extract the canonical operation name from a callable or torch op overload."""
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


# ---------------------------------------------------------------------------
# Device / process group helpers
# ---------------------------------------------------------------------------

def get_device_handle(device_type: str = "npu"):  # pylint: disable=W0621
    """Return the torch device module (e.g. ``torch.npu`` or ``torch.cuda``) for the given device type."""
    try:
        handle = getattr(torch, device_type)
    except AttributeError as e:
        raise RuntimeError(f"expect got device handle: 'torch.{device_type}' failed.") from e
    return handle


def device_count(device_handle):
    """Return the number of available devices for *device_handle*."""
    return device_handle.device_count()


def device_type():
    """Return the current device type string ("npu" for NPU, "cuda" for GPU)."""
    device_handle = get_device_handle()
    if device_handle == getattr(torch, "npu", None):
        return "npu"
    return "cuda"


def device(device_idx=None):
    """Return a :class:`torch.device` for the current device type."""
    current_device_type = device_type()
    if device_idx is None:
        return torch.device(current_device_type)
    return torch.device(f"{current_device_type}:{device_idx:d}")


def manual_seed(seed):
    """Set the random seed for reproducibility."""
    return torch.manual_seed(seed)


def get_rng_state(device=None, device_handle=None):  # pylint: disable=W0621
    """Get the random number generator state."""
    if device_handle is None:
        return torch.get_rng_state()
    if device is None:
        return device_handle.get_rng_state()
    return device_handle.get_rng_state(device)


def set_rng_state(state, device=None, device_handle=None):  # pylint: disable=W0621
    """Set the random number generator state."""
    if device_handle is None:
        return torch.set_rng_state(state)
    if device is None:
        return device_handle.set_rng_state(state)
    return device_handle.set_rng_state(state, device)


def get_created_group(rank_list: Union[list[int], tuple[int]]):
    """Return an existing process group by rank list, or ``None``."""
    group_key = str(tuple(sorted(rank_list)))
    if group_key in EXISTING_COMM_GROUPS:
        return EXISTING_COMM_GROUPS[group_key]
    return None


def create_group(rank_list):
    """Create or retrieve a communication group with the specified ranks.

    If a group with the same rank list already exists, returns the existing
    group instead of creating a new one.
    """
    group_key = str(tuple(sorted(rank_list)))
    if group_key in EXISTING_COMM_GROUPS:
        return EXISTING_COMM_GROUPS[group_key]

    normalized_rank_list = tuple(sorted(rank_list))
    world_rank_list = tuple(range(dist.get_world_size()))
    if normalized_rank_list == world_rank_list:
        group = _get_default_group()
    else:
        group = create_sub_groups(rank_list)[normalized_rank_list]

    EXISTING_COMM_GROUPS[group_key] = group
    return group


def split_group(parent_pg: Optional[ProcessGroup] = None,
                split_ranks: Optional[list] = None,
                timeout: Optional[Any] = None,
                pg_options: Optional[Any] = None,
                group_desc: Optional[str] = None,
                ) -> Optional[ProcessGroup]:
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


def init_process_group(*args, **kwargs):
    """Initialize the default torch distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(*args, **kwargs)


# ---------------------------------------------------------------------------
# Sub-group construction
# ---------------------------------------------------------------------------

def _validate_intra_step(normalized_template: list[int], template_len: int) -> int:
    """Verify consistent intra-group step and return intra_step."""
    intra_step = normalized_template[1] - normalized_template[0]
    for i in range(1, template_len - 1):
        diff = normalized_template[i + 1] - normalized_template[i]
        if diff != intra_step:
            msg = (
                f"Template must have consistent intra-group step. "
                f"Found {normalized_template[i+1]} - {normalized_template[i]} = {diff}, "
                f"expected {intra_step}"
            )
            raise ValueError(msg)
    return intra_step


def _compute_group_starts(world_size: int, block_size: int, inter_step: int) -> list[int]:
    """Compute all valid block start positions."""
    return [s for s in range(0, world_size, inter_step) if s + block_size <= world_size]


def _build_groups_for_blocks(
    group_starts: list[int],
    block_size: int,
    template_span_int: int,
    normalized_template: list[int],
    template_len: int,
    world_size: int,
) -> list[list[int]]:
    """Build all groups from block starts."""
    all_groups = []
    for start_block in group_starts:
        max_offset = block_size - template_span_int
        for offset in range(0, max_offset):
            group = [start_block + offset + normalized_template[i] for i in range(template_len)]
            if all(0 <= r < world_size for r in group):
                all_groups.append(group)
    return all_groups


def generate_groups_from_template(
    template: Union[list[int], tuple[int, ...]],
    world_size: int,
    my_rank: int,
    verbose: bool = False
) -> list[list[int]]:
    """
    Auto-generate all communication groups from a template (supports any valid starting template).

    Args:
        template: Template group, e.g. [0,1], [0,2,4,6] or [1,3,5,7]
        world_size: Total number of processes
        my_rank: Current process rank (for debug output)
        verbose: Whether to print debug info

    Returns:
        Full rank list, e.g.:
        - template [0,1] + world_size=8 -> [[0,1], [2,3], [4,5], [6,7]]
        - template [0,2,4,6] + world_size=8 -> [[0,2,4,6], [1,3,5,7]]
        - template [1,3,5,7] + world_size=8 -> [[0,2,4,6], [1,3,5,7]]

    Algorithm:
        1. Template normalization: convert any starting template to 0-based
        2. Analyze pattern (intra-step, template span)
        3. Iterate by blocks, generate valid sub-groups per block
        4. Ensure each rank appears in exactly one group
    """
    # convert template to int list and sort (rank_list may come from numpy/tensor as float)
    template = sorted([int(x) for x in list(template)])
    world_size = int(world_size)
    my_rank = int(my_rank)
    template_len = len(template)

    if verbose:
        print(f"Rank {my_rank}: Original Template = {template}, World size = {world_size}")

    if template_len == 1:
        return [[i] for i in range(world_size)]

    if template_len < 2:
        raise ValueError(f"Template must have at least 2 ranks, got {template}")

    # 1. Template normalization: convert to 0-based template
    template_base = template[0]  # original template start value
    normalized_template = [x - template_base for x in template]  # normalize to 0-based
    if verbose:
        print(f"Rank {my_rank}: Normalized Template = {normalized_template}")

    # 2. Analyze normalized template core params
    # intra-step: spacing between elements in template
    intra_step = _validate_intra_step(normalized_template, template_len)
    # template span: last - first element of normalized template
    template_span = normalized_template[-1] - normalized_template[0]
    # block size: ranks per block (determines inter-step)
    block_size = int(intra_step * template_len)
    # inter-step: spacing between adjacent blocks (equals block_size)
    inter_step = block_size

    if verbose:
        print(
            f"Rank {my_rank}: Template analysis - "
            f"intra_step={intra_step}, template_span={template_span}, "
            f"block_size={block_size}, inter_step={inter_step}"
        )

    # 3. Compute all valid block start positions
    group_starts = _compute_group_starts(world_size, block_size, inter_step)
    if verbose:
        print(f"Rank {my_rank}: Possible block starts: {group_starts}")

    # 4. Generate all valid sub-groups for each block
    template_span_int = int(template_span)
    all_groups = _build_groups_for_blocks(
        group_starts, block_size, template_span_int,
        normalized_template, template_len, world_size
    )

    # 5. Validate: ensure each rank appears exactly once
    all_ranks = [rank for group in all_groups for rank in group]
    unique_ranks = set(all_ranks)
    if len(all_ranks) != len(unique_ranks):
        raise ValueError("Duplicate ranks found! Some ranks appear in multiple groups.")

    # 6. Sort: ensure all processes generate groups in same order
    all_groups.sort(key=lambda x: (x[0], x[1] if len(x) > 1 else 0))

    if verbose:
        print(
            f"Rank {my_rank}: Generated {len(all_groups)} groups, "
            f"covering {len(unique_ranks)} unique ranks\n"
            f"Final group list: {all_groups}"
        )

    return all_groups


def create_sub_groups(
    rank_list: Union[list[int], tuple[int, ...]],
    verbose: bool = False
) -> dict[tuple, dist.ProcessGroup]:
    """
    Create sub-communication groups, supports template auto-expansion.

    Args:
        rank_list: One of:
                  1. Full group list, e.g. [[0,1], [2,3], [4,5], [6,7]]
                  2. Template group, e.g. [0,1] or [0,2], will auto-expand
        verbose: Whether to print debug info

    Returns:
        Dict, key is tuple of group ranks, value is ProcessGroup
    """
    my_rank = dist.get_rank()
    world_size = dist.get_world_size()
    template = list(rank_list)
    full_rank_list = generate_groups_from_template(template, world_size, my_rank, verbose=verbose)

    if verbose:
        print(f"Rank {my_rank}: Full rank list to create: {full_rank_list}")

    # validate full group list format
    for i, group in enumerate(full_rank_list):
        if not isinstance(group, (list, tuple)):
            raise ValueError(f"Group {i} must be a list or tuple, got {type(group)}")
        if len(group) == 0:
            raise ValueError(f"Group {i} is empty")
        if len(group) != len(set(group)):
            raise ValueError(f"Group {i} contains duplicate ranks")
        for rank in group:
            if not isinstance(rank, int):
                raise ValueError(f"Rank must be integer, got {type(rank)} in group {i}")

    # sort by first element to ensure all processes create groups in same order
    sorted_groups = sorted(full_rank_list, key=lambda x: x[0])

    if verbose:
        print(f"Rank {my_rank}: Sorted groups for creation: {sorted_groups}")

    # create all groups and collect groups current process belongs to
    group_dict = {}
    for group_ranks in sorted_groups:
        # ensure ranks are ordered so each process passes same order
        sorted_ranks = sorted(group_ranks)

        if verbose:
            print(f"Rank {my_rank}: Creating group with ranks {sorted_ranks}")

        # key: all processes participate in each group creation
        group = dist.new_group(ranks=sorted_ranks)
        EXISTING_COMM_GROUPS[str(tuple(sorted_ranks))] = group

        # only save when current process is in the group
        if my_rank in sorted_ranks:
            group_dict[tuple(sorted_ranks)] = group

    if verbose:
        print(f"Rank {my_rank}: Created {len(group_dict)} groups I belong to")

    return group_dict


# ---------------------------------------------------------------------------
# Differentiable collectives
# ---------------------------------------------------------------------------

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
    # Only the four core hooks A/B/C/D + D_LAST sentinel are used.  CUDA
    # streams are process-wide and Torch autograd is thread-safe, so no
    # chunk-boundary hooks are needed.  Do not add CHUNK_START / CHUNK_END
    # to these tables; if a future test does need them, copy the MS
    # implementation and add the matching skip rules in ``forward`` /
    # ``backward``.
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
            from hyper_parallel.core.pipeline_parallel.hook_coordinator import (  # pylint: disable=C0415
                HookRole,
            )
            _TorchSyncHookFunction._ROLE_CACHE = (None, HookRole.COMM, HookRole.COMPUTE)
        return _TorchSyncHookFunction._ROLE_CACHE[idx]  # pylint: disable=E1136

    @staticmethod
    def forward(ctx, x, hook_name, coordinator):  # pylint: disable=arguments-differ
        """Identity forward that fires a HookCoordinator rendezvous.

        Notifies the previous op's role and rendezvouses for the next op's
        role per the ``_FWD_ROLES`` table.  ``"D_LAST"`` is a sentinel
        meaning "skip this rendezvous" (last layer's closing D — no
        Attention follows).
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


def differentiable_all_gather_concat(data, group, concat_size, concat_dim, rank_list=None):  # pylint: disable=W0613
    """Autograd-aware all-gather whose results are concatenated along ``concat_dim``."""
    data = _ensure_contiguous(data)
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


def chunk(data, split_dim, split_size, index):
    """Return chunk *index* of ``data`` split into ``split_size`` pieces."""
    return torch.chunk(data, split_size, dim=split_dim)[index]


def differentiable_all_to_all(input_data, output_shape, group):
    """Autograd-aware all-to-all producing a tensor of ``output_shape``."""
    input_data = _ensure_contiguous(input_data)
    output_tensor = torch.empty(output_shape, device=input_data.device, dtype=input_data.dtype)
    return dist_func.all_to_all_single(output_tensor, input_data, group=group)


def differentiable_all_reduce(data, op, group):
    """Autograd-aware all-reduce with string or ``ReduceOp`` *op*."""
    data = _ensure_contiguous(data)
    # Resolve the op from string to ReduceOp enum if necessary
    reduce_op = _OP_MAP.get(op, dist.ReduceOp.SUM) if isinstance(op, str) else op
    return dist_func.all_reduce(data, op=reduce_op, group=group)


def differentiable_reduce_scatter(data, dev_num, axis, op, group):
    """Autograd-aware reduce-scatter splitting ``axis`` into ``dev_num`` parts."""
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


def differentiable_all_to_all_single(input_tensor, input_splits, output_splits, group):
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


def differentiable_all_to_all_single_async(input_tensor, input_splits, output_splits, group):
    """Truly-async variant of :func:`differentiable_all_to_all_single`.

    Both forward AND backward return ``AsyncCollectiveTensor``, so the
    ``wait_tensor`` op is queued lazily — only when a downstream kernel
    actually reads the result.  See :class:`_AsyncA2ALazyBwd`.
    """
    return _AsyncA2ALazyBwd.apply(input_tensor, output_splits, input_splits, group)


def differentiable_variable_all_gather(
        input_tensor: Tensor, output_splits: Sequence[int], group: Any) -> Tensor:
    """Gather variable dim-zero shards on HCCL or Gloo with autograd support."""
    return _TorchDifferentiableVariableAllGather.apply(
        input_tensor, tuple(output_splits), group
    )


def wait_async_tensor(tensor):
    """Wait for an async collective tensor to become materialised.

    Idempotent — calling on an already-waited tensor is a no-op.
    """
    from torch.distributed._functional_collectives import wait_tensor  # pylint: disable=C0415
    wait_tensor(tensor)
    return tensor


def differentiable_async_allgather_wait(x, work, out_perm, group, world_size, gather_dim,
                                        handle_box=None):
    """Wait async all-gather handle and reconstruct result (differentiable)."""
    return _TorchAsyncAllGatherFunction.apply(
        x, work, out_perm, group, world_size, gather_dim, handle_box
    )


def differentiable_async_a2a_wait(x, work, out_perm, group, world_size, concat_dim, split_dim,
                                  handle_box=None):
    """Wait async A2A handle and reconstruct result (differentiable)."""
    return _TorchAsyncA2AFunction.apply(
        x, work, out_perm, group, world_size, concat_dim, split_dim, handle_box
    )


def differentiable_sync_hook(x, hook_name: str, coordinator):
    """Insert a HookCoordinator rendezvous into the autograd graph."""
    return _TorchSyncHookFunction.apply(x, hook_name, coordinator)


def p2p_exchange(tensor, peer_rank: int, group=None):
    """Symmetric bidirectional P2P exchange with *peer_rank*."""
    if peer_rank == dist.get_rank(group):
        return tensor
    return _TorchP2PExchangeFunction.apply(tensor, peer_rank, group)


# ---------------------------------------------------------------------------
# Unsupported legacy redistribution hooks
# ---------------------------------------------------------------------------

def get_tensor_transform():
    """Legacy MindSpore-side tensor transform hook — not available on torch."""
    raise NotImplementedError("Unsupported get_tensor_transform for torch platform")


def construct_strided_slice(x, begin, end, stride):
    """Legacy MindSpore-side strided-slice hook — not available on torch."""
    raise NotImplementedError("Unsupported construct_strided_slice for torch platform")


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

@contextmanager
def init_on_device(device, include_buffers=False):  # pylint: disable=W0621
    """Monkey-patch ``nn.Module`` so that every parameter (and optionally every
    buffer) is placed on *device* at registration time.

    Args:
        device (torch.device): Target device.
        include_buffers (bool): Also redirect buffers to *device*.
    """
    orig_register_parameter = nn.Module.register_parameter
    orig_register_buffer = nn.Module.register_buffer

    # pylint: disable=W0212
    def _register_parameter(module, name, param):
        orig_register_parameter(module, name, param)
        if param is None or param.device == device:
            return
        # Rebuild with data only, then restore instance attributes via __dict__:
        # forwarding them to __new__ crashes subclasses with a narrow signature.
        new_param = type(param)(param.to(device))
        new_param.__dict__.update(param.__dict__)
        new_param.requires_grad = param.requires_grad
        module._parameters[name] = new_param

    # pylint: disable=W0212
    def _register_buffer(module, name, buffer, persistent=True):
        orig_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    try:
        nn.Module.register_parameter = _register_parameter
        if include_buffers:
            nn.Module.register_buffer = _register_buffer
        yield
    finally:
        nn.Module.register_parameter = orig_register_parameter
        if include_buffers:
            nn.Module.register_buffer = orig_register_buffer
