# Copyright 2025 Huawei Technologies Co., Ltd
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
"""_group_manager"""

from typing import Dict, List, Tuple, Union

import torch.distributed as dist

from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _validate_intra_step(normalized_template: List[int], template_len: int) -> int:
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


def _compute_group_starts(world_size: int, block_size: int, inter_step: int) -> List[int]:
    """Compute all valid block start positions."""
    return [s for s in range(0, world_size, inter_step) if s + block_size <= world_size]


def _build_groups_for_blocks(
    group_starts: List[int],
    block_size: int,
    template_span_int: int,
    normalized_template: List[int],
    template_len: int,
    world_size: int,
) -> List[List[int]]:
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
    template: Union[List[int], Tuple[int, ...]],
    world_size: int,
    my_rank: int,
    verbose: bool = False
) -> List[List[int]]:
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
    rank_list: Union[List[int], Tuple[int, ...]],
    verbose: bool = False
) -> Dict[tuple, dist.ProcessGroup]:
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
