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

Collectives and the small rank/device/module helpers that other ``core``
modules also needed were lifted into :mod:`hyper_parallel.core.utils`.  The
collectives stay re-exported here because the debug tracer patches these
attributes on this module, so ``_utils.<name>`` keeps working for that path;
callers elsewhere should import from :mod:`hyper_parallel.core.utils`.
"""
# pylint: disable=C9006,C9007
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
from torch import nn
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group

import torch.distributed as dist

from hyper_parallel.core.utils.communication import (
    EXISTING_COMM_GROUPS,
    differentiable_all_gather_concat,
    differentiable_all_to_all,
    differentiable_all_to_all_single,
    differentiable_all_to_all_single_async,
    differentiable_all_reduce,
    differentiable_reduce_scatter,
    differentiable_variable_all_gather,
)
from hyper_parallel.core.utils.communication import get_device_handle
from hyper_parallel.core.shard.utils import get_op_name


_HCCL_OP_EXPANSION_MODE_AIV = 3


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Device / process group helpers
# ---------------------------------------------------------------------------

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


def get_cp_hccl_process_group_options() -> Any:
    """Build AIV HCCL process-group options for the CP communication group."""
    # ``torch_npu`` is optional, so import it only for the NPU-specific CP group.
    try:
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL  # pylint: disable=C0415
    except ImportError as exc:
        raise RuntimeError("Creating the 'cp' group with AIV requires torch_npu.") from exc

    options = ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_op_expansion_mode": _HCCL_OP_EXPANSION_MODE_AIV}
    return options


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
    """Verify a normalized template has a uniform step between its members.

    The step between adjacent members defines the stride of the groups built
    from this template, so a template such as ``[0, 1, 3]`` (steps 1 then 2)
    has no single well-defined stride and is rejected.

    Args:
        normalized_template: Template rebased so its first member is ``0``.
        template_len: Number of members; ``normalized_template`` must hold at
            least two for a step to exist.

    Returns:
        int: The step shared by every adjacent pair of members.

    Raises:
        ValueError: If any adjacent pair differs in step from the first pair.
    """
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
    """Compute every block start position that fits inside ``world_size``.

    Start positions are spaced ``inter_step`` apart, and a start is kept only
    when the whole block fits, i.e. the last rank it can reach stays within
    ``world_size``. Ranks past the final full block are left uncovered.

    Args:
        world_size: Total number of processes.
        block_size: Ranks spanned by one block; used for the fit check.
        inter_step: Distance between adjacent candidate starts.

    Returns:
        list[int]: Ascending start positions of the blocks that fit.
    """
    return [s for s in range(0, world_size, inter_step) if s + block_size <= world_size]


def _build_groups_for_blocks(
    group_starts: list[int],
    block_size: int,
    template_span_int: int,
    normalized_template: list[int],
    template_len: int,
    world_size: int,
) -> list[list[int]]:
    """Expand each block start into the groups the template can place there.

    Within a block the template may be slid to offsets ``0 .. block_size -
    template_span_int``; an offset's group is kept only when every rank it
    produces lands inside ``world_size``. Groups are appended in start order,
    then offset order, so every process building the same template sees the
    same sequence before the caller sorts.

    Args:
        group_starts: Block start positions, as returned by
            :func:`_compute_group_starts`.
        block_size: Ranks spanned by one block.
        template_span_int: Distance from the template's first member to its
            last; bounds how far the template can slide within a block.
        normalized_template: Template rebased so its first member is ``0``.
        template_len: Number of members in the template.
        world_size: Total number of processes.

    Returns:
        list[list[int]]: One rank list per (block, offset) pair that fits.
    """
    all_groups = []
    for start_block in group_starts:
        max_offset = block_size - template_span_int
        for offset in range(0, max_offset):
            group = [start_block + offset + normalized_template[i] for i in range(template_len)]
            if all(0 <= r < world_size for r in group):
                all_groups.append(group)
    return all_groups


def _report_template_stage(verbose: bool, my_rank: int, message: str) -> None:
    """Print one ``generate_groups_from_template`` stage when *verbose* is set.

    Each generated group must be identical across processes, so a template bug
    shows up as a mismatch in the printed rank lists rather than as a crash.
    This helper exists so those stage prints stay a single ``verbose`` check
    each, keeping the caller's local-variable count down.

    Args:
        verbose: Whether the caller asked for debug output.
        my_rank: Current process rank, prefixed to every line.
        message: Stage-specific text; the caller formats it.
    """
    if verbose:
        print(f"Rank {my_rank}: {message}")


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

    _report_template_stage(verbose, my_rank, f"Original Template = {template}, World size = {world_size}")

    if template_len == 1:
        return [[i] for i in range(world_size)]

    if template_len < 2:
        raise ValueError(f"Template must have at least 2 ranks, got {template}")

    # 1. Template normalization: convert to 0-based template
    normalized_template = [x - template[0] for x in template]
    _report_template_stage(verbose, my_rank, f"Normalized Template = {normalized_template}")

    # 2. Analyze normalized template core params
    # intra-step: spacing between elements in template
    intra_step = _validate_intra_step(normalized_template, template_len)
    # block size: ranks per block; it is also the spacing between adjacent blocks
    block_size = intra_step * template_len
    template_span = normalized_template[-1] - normalized_template[0]
    _report_template_stage(
        verbose, my_rank,
        f"Template analysis - intra_step={intra_step}, template_span={template_span}, block_size={block_size}"
    )

    # 3. Compute all valid block start positions
    group_starts = _compute_group_starts(world_size, block_size, block_size)
    _report_template_stage(verbose, my_rank, f"Possible block starts: {group_starts}")

    # 4. Generate all valid sub-groups for each block
    all_groups = _build_groups_for_blocks(
        group_starts, block_size, template_span,
        normalized_template, template_len, world_size
    )

    # 5. Validate: ensure each rank appears exactly once
    all_ranks = [rank for group in all_groups for rank in group]
    unique_ranks = set(all_ranks)
    if len(all_ranks) != len(unique_ranks):
        raise ValueError("Duplicate ranks found! Some ranks appear in multiple groups.")

    # 6. Sort: ensure all processes generate groups in same order
    all_groups.sort(key=lambda x: (x[0], x[1] if len(x) > 1 else 0))

    _report_template_stage(
        verbose, my_rank,
        f"Generated {len(all_groups)} groups, covering {len(unique_ranks)} unique ranks\n"
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
# Unsupported legacy redistribution hooks
# ---------------------------------------------------------------------------

def get_tensor_transform():
    """Legacy tensor transform hook — not available on torch."""
    raise NotImplementedError("Unsupported get_tensor_transform for torch platform")


def construct_strided_slice(x, begin, end, stride):
    """Legacy strided-slice hook — not available on torch."""
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


__all__ = [
    # Tensor/device/rng helpers.
    "tensor_type_cast",
    "device_count",
    "device_type",
    "device",
    "get_rng_state",
    "set_rng_state",
    "get_tensor_transform",
    "construct_strided_slice",
    "init_on_device",
    # Process-group construction.
    "get_created_group",
    "create_group",
    "split_group",
    "init_process_group",
    "generate_groups_from_template",
    "create_sub_groups",
    # Re-exports kept for the debug tracer, which patches these attributes on
    # this module; the canonical home is
    # :mod:`hyper_parallel.core.utils.communication`.
    "differentiable_all_gather_concat",
    "differentiable_all_to_all",
    "differentiable_all_to_all_single",
    "differentiable_all_to_all_single_async",
    "differentiable_all_reduce",
    "differentiable_reduce_scatter",
    "differentiable_variable_all_gather",
    # Small shared helpers re-exported for callers that reach them here.
    "get_device_handle",
    "get_op_name",
]
