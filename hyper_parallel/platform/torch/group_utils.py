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
    根据模板组自动生成所有通信组（支持任意合法起始的模板）。

    参数:
        template: 模板组，例如 [0,1]、[0,2,4,6] 或 [1,3,5,7]
        world_size: 总进程数
        my_rank: 当前进程rank（用于打印调试信息）
        verbose: 是否打印调试信息

    返回:
        完整的rank列表，例如：
        - 模板[0,1] + world_size=8 → [[0,1], [2,3], [4,5], [6,7]]
        - 模板[0,2,4,6] + world_size=8 → [[0,2,4,6], [1,3,5,7]]
        - 模板[1,3,5,7] + world_size=8 → [[0,2,4,6], [1,3,5,7]]

    原理:
        1. 模板归一化：将任意起始的模板转换为以0为起点的基准模板
        2. 分析基准模板的模式（组内步长、模板跨度）
        3. 按块遍历，在每个块内生成所有合法子组
        4. 确保每个rank恰好出现在一个组中
    """
    # 将模板转换为整数列表并排序（rank_list 可能来自 numpy/tensor 等为 float）
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

    # 1. 模板归一化：转换为以0为起点的基准模板（消除起始值影响）
    template_base = template[0] # 原始模板的起始值
    normalized_template = [x - template_base for x in template] # 归一化到0起点
    if verbose:
        print(f"Rank {my_rank}: Normalized Template = {normalized_template}")

    # 2. 分析归一化后的模板核心参数
    # 组内步长（模板内元素的间隔）
    intra_step = _validate_intra_step(normalized_template, template_len)
    # 模板跨度（归一化模板最后一个元素 - 第一个元素）
    template_span = normalized_template[-1] - normalized_template[0]
    # 块大小：每个块可容纳的rank数（决定组间步长）
    block_size = int(intra_step * template_len)
    # 组间步长：相邻块的起始间隔（等于块大小）
    inter_step = block_size

    if verbose:
        print(
            f"Rank {my_rank}: Template analysis - "
            f"intra_step={intra_step}, template_span={template_span}, "
            f"block_size={block_size}, inter_step={inter_step}"
        )

    # 3. 计算所有合法的块起始位置
    group_starts = _compute_group_starts(world_size, block_size, inter_step)
    if verbose:
        print(f"Rank {my_rank}: Possible block starts: {group_starts}")

    # 4. 为每个块生成所有合法子组
    template_span_int = int(template_span)
    all_groups = _build_groups_for_blocks(
        group_starts, block_size, template_span_int,
        normalized_template, template_len, world_size
    )

    # 5. 验证：确保每个rank只出现一次
    all_ranks = [rank for group in all_groups for rank in group]
    unique_ranks = set(all_ranks)
    if len(all_ranks) != len(unique_ranks):
        raise ValueError("Duplicate ranks found! Some ranks appear in multiple groups.")

    # 6. 排序：确保所有进程生成的组顺序一致
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
    创建子通信组，支持模板组自动扩展。

    参数:
        rank_list: 可以是以下两种格式之一：
                  1. 完整的组列表，例如 [[0,1], [2,3], [4,5], [6,7]]
                  2. 模板组，例如 [0,1] 或 [0,2]，会自动扩展
        verbose: 是否打印调试信息

    返回:
        字典，键为组ranks的元组，值为该组的ProcessGroup对象
    """
    my_rank = dist.get_rank()
    world_size = dist.get_world_size()
    template = list(rank_list)
    full_rank_list = generate_groups_from_template(template, world_size, my_rank, verbose=verbose)

    if verbose:
        print(f"Rank {my_rank}: Full rank list to create: {full_rank_list}")

    # 验证完整组列表格式
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

    # 按照第一个元素的顺序排序，确保所有进程以相同顺序创建组
    sorted_groups = sorted(full_rank_list, key=lambda x: x[0])

    if verbose:
        print(f"Rank {my_rank}: Sorted groups for creation: {sorted_groups}")

    # 创建所有组并收集当前进程所在的组
    group_dict = {}
    for group_ranks in sorted_groups:
        # 确保ranks有序，这样每个进程传入相同的顺序
        sorted_ranks = sorted(group_ranks)

        if verbose:
            print(f"Rank {my_rank}: Creating group with ranks {sorted_ranks}")

        # 关键：所有进程都参与每个组的创建
        group = dist.new_group(ranks=sorted_ranks)

        # 只在当前进程在组内时保存
        if my_rank in sorted_ranks:
            group_dict[tuple(sorted_ranks)] = group

    if verbose:
        print(f"Rank {my_rank}: Created {len(group_dict)} groups I belong to")

    return group_dict
