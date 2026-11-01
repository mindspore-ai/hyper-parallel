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
"""MoonEP 容量规划及一次性通信元数据编译；大索引始终留在设备。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from numbers import Integral

import numpy as np
import torch


@dataclass(frozen=True)
class ExecutionPlan:
    """不可变分发与权重归属元数据；例如active=[0,2,3]对应三个GMM组。"""

    allocation: torch.Tensor
    dispatch_order: torch.Tensor
    sort_order: torch.Tensor
    boundaries: torch.Tensor
    group_counts: tuple[int, ...]
    token_send: list[int]
    token_recv: list[int]
    weight_send: list[int]
    weight_recv: list[int]
    send_ids: torch.Tensor
    weight_spans: tuple[tuple[bool, int, int, torch.Tensor], ...]
    home_positions: torch.Tensor
    home_rows: torch.Tensor
    remote_positions: torch.Tensor
    remote_rows: torch.Tensor
    home_count: int
    any_migration: bool
    metrics: tuple[int, ...]
    used_fallback: bool
    weight_ranges: tuple[tuple[int, int] | None, ...]
    home_dense: bool


def _stable_order(ids: torch.Tensor) -> torch.Tensor:
    """在设备上稳定排序整数标签；例如[2,0,2]返回[1,0,2]。

    Returns:
        与输入同设备的排列索引；小复合键用昇腾AICore FP32排序。
    """
    count = ids.numel()
    if count == 0:
        return torch.empty(0, device=ids.device, dtype=torch.long)
    # 不读取设备最大值；调用边界保证标签的上界，长序列退回设备稳定排序。
    return torch.argsort(ids, stable=True)


def prepare_entry_order(ids: torch.Tensor, experts: int) -> torch.Tensor:
    """按专家和原选择位置排序，供后续目标卡排序复用。

    Args:
        ids: 本卡展开后的专家号，例如 [2,0,2]。
        experts: 全局专家数；该例至少为三。

    Returns:
        设备上的专家主序索引；该例为 [1,0,2]。

    Example:
        [2,0,2] 返回 [1,0,2]，目标卡排序再复用这个顺序。
    """
    size = ids.numel()
    positions = torch.arange(size, device=ids.device)
    if experts * size < 2**24:
        return torch.argsort((ids * size + positions).float())
    return _stable_order(ids)


def _entry_order(counts: torch.Tensor, allocation: torch.Tensor,
                 rank: int, ids: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
    """按计算卡、专家、原选择编号排列发送项；例如四项得到[0,2,1,3]。

    Args:
        counts: CPU小计数表[R,E]，只用于编译固定尺寸前缀。
        allocation: CPU小分配表[E,R]。
        rank: EP组内源编号。
        ids: 设备上的本地展开专家号，绝不搬到CPU排序。

    Returns:
        设备上的发送排列；每个索引指向一个token的一个top-k选择。
    """
    size = ids.numel()
    experts, ranks = allocation.shape
    positions = torch.arange(size, device=ids.device)
    expert_order = prepare_entry_order(ids, experts)
    sorted_ids = ids[expert_order]
    if metadata is None:
        offset = counts[:rank].sum(0) - (counts[rank].cumsum(0) - counts[rank])
        metadata = torch.cat((offset, allocation.cumsum(1).reshape(-1))).to(ids.device)
    offset, ends = metadata[:experts], metadata[experts:].reshape(experts, ranks)
    ordinal = positions + offset[sorted_ids]
    destinations = (ordinal[:, None] >= ends[sorted_ids]).sum(1)
    if ranks * size < 2**24:
        target_order = torch.argsort((destinations * size + positions).float())
    else:
        target_order = _stable_order(destinations)
    return expert_order[target_order]


def _checked_count_rows(counts_cpu: torch.Tensor) -> list[list[int]]:
    """检查已完成D2H的小计数表并返回Python行，不再读取设备。

    输入为CPU上的int64[R,E]，输出是同一计数的整数列表；例如
    [[2,0],[1,3]]原样返回，而[[-1,2],[1,3]]即使列和非负也拒绝。
    """
    rows = counts_cpu.tolist()
    if any(value < 0 for row in rows for value in row):
        raise ValueError("MoonEP源计数必须非负")
    return rows


def _cpu_plan_metrics(allocation: list[list[int]], totals: list[int],
                      capacities: list[int]) -> tuple[int, ...]:
    """从已有CPU整数列表计算与设备版完全相同的六项指标。

    输入分别是[E,R]分配、[E]专家总量、[R]副本容量；输出依次为
    合法标志、迁移量M、收发副本峰值z、总副本、迁移下界M0、M最优证书。
    例如allocation=[[2,2],[0,0]]、totals=[4,0]、capacities=[1,1]
    返回(1,2,1,1,2,1)：两条e0迁到rank1，达到最小迁移量。
    """
    experts, ranks = len(totals), len(capacities)
    if (ranks < 1 or experts < 1 or experts % ranks or len(allocation) != experts
            or any(len(row) != ranks for row in allocation)):
        raise ValueError("CPU规划指标要求合法的[E,R]、[E]、[R]形状")
    home_count = experts // ranks
    loads, incoming, outgoing = [0] * ranks, [0] * ranks, [0] * ranks
    valid = all(value >= 0 for value in totals) and all(value >= 0 for value in capacities)
    moved = 0
    for expert, row in enumerate(allocation):
        home = expert // home_count
        valid = valid and all(value >= 0 for value in row) and sum(row) == totals[expert]
        for target, value in enumerate(row):
            loads[target] += value
            if target != home:
                # 非法候选也保留设备版的精确指标语义：负配额计入M，但不算启用副本。
                moved += value
                if value > 0:
                    incoming[target] += 1
                    outgoing[home] += 1
    total = sum(totals)
    targets = [total // ranks + int(rank < total % ranks) for rank in range(ranks)]
    valid = valid and loads == targets and all(count <= capacity for count, capacity in zip(incoming, capacities))
    lower_bound = sum(max(0, sum(totals[rank * home_count:(rank + 1) * home_count]) - targets[rank])
                      for rank in range(ranks))
    return (int(valid), moved, max(incoming + outgoing), sum(incoming),
            lower_bound, int(valid and moved == lower_bound))


def _cpu_candidates(totals: np.ndarray, ranks: int, rules: tuple[bool, ...]) -> np.ndarray:
    """在已回传的专家小表上生成快速候选；例如 R8/E128 输出两张 [128,8] 表。

    稳定 FP32 排序和并列 rank 的选择保持确定性；大 token 索引仍在 NPU。
    """
    experts = totals.size
    home_count, batches = experts // ranks, len(rules)
    rank_ids = np.arange(ranks, dtype=np.int64)
    total = int(totals.sum())
    targets = total // ranks + (rank_ids < total % ranks).astype(np.int64)
    home_totals = totals.reshape(ranks, home_count)
    balance = np.broadcast_to(home_totals.sum(axis=1) - targets, (batches, ranks)).copy()
    quotas = np.zeros((batches, ranks, ranks), dtype=np.int64)
    batch_ids = np.arange(batches)
    for _ in range(ranks - 1):
        source = balance.argmax(axis=1)
        target = balance.argmin(axis=1)
        surplus = balance[batch_ids, source]
        deficit = -balance[batch_ids, target]
        amount = np.where(rules, deficit, np.minimum(surplus, deficit))
        np.add.at(quotas, (batch_ids, source, target), amount)
        np.add.at(balance, (batch_ids, source), -amount)
        np.add.at(balance, (batch_ids, target), amount)

    expert_order = np.argsort(-home_totals.astype(np.float32), axis=1, kind='stable')
    target_order = np.argsort(-quotas.astype(np.float32), axis=2, kind='stable')
    ordered_experts = np.take_along_axis(home_totals, expert_order, axis=1)
    ordered_quotas = np.take_along_axis(quotas, target_order, axis=2)
    expert_ends = ordered_experts.cumsum(axis=1)[None, :, :, None]
    target_ends = ordered_quotas.cumsum(axis=2)[:, :, None, :]
    moved_ordered = np.maximum(
        np.minimum(expert_ends, target_ends)
        - np.maximum(expert_ends - ordered_experts[None, :, :, None],
                     target_ends - ordered_quotas[:, :, None, :]), 0)
    by_expert = np.empty_like(moved_ordered)
    np.put_along_axis(by_expert, target_order[:, :, None, :], moved_ordered, axis=3)
    moved = np.empty_like(by_expert)
    np.put_along_axis(moved, expert_order[None, :, :, None], by_expert, axis=2)
    remaining = home_totals[None] - moved.sum(axis=3)
    for home in range(ranks):
        moved[:, home, :, home] += remaining[:, home, :]
    return moved.reshape(batches, experts, ranks)


def _cpu_candidate_metrics(allocation: np.ndarray, totals: np.ndarray, capacity: int) -> np.ndarray:
    """批量验证 CPU 候选及最小迁移证书；例如两张表返回 [2,6] 指标。"""
    batches, experts, ranks = allocation.shape
    home_count = experts // ranks
    home_ids = np.arange(experts) // home_count
    remote = home_ids[:, None] != np.arange(ranks)[None, :]
    enabled = (allocation > 0) & remote
    incoming = enabled.sum(axis=1)
    outgoing = enabled.reshape(batches, ranks, home_count, ranks).sum(axis=(2, 3))
    total = int(totals.sum())
    targets = total // ranks + (np.arange(ranks) < total % ranks)
    valid = ((allocation >= 0).all(axis=(1, 2)) & (totals >= 0).all() & (capacity >= 0)
             & (allocation.sum(axis=2) == totals).all(axis=1)
             & (allocation.sum(axis=1) == targets).all(axis=1)
             & (incoming <= capacity).all(axis=1))
    moved = (allocation * remote).sum(axis=(1, 2))
    peak = np.maximum(incoming.max(axis=1), outgoing.max(axis=1))
    copies = enabled.sum(axis=(1, 2))
    lower_bound = np.maximum(totals.reshape(ranks, home_count).sum(axis=1) - targets, 0).sum()
    certificate = valid & (moved == lower_bound)
    return np.stack((valid, moved, peak, copies, np.broadcast_to(lower_bound, (batches,)),
                     certificate), axis=1).astype(np.int64)


def select_allocation(counts: torch.Tensor, capacity: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """在 CPU 小计数表上选可行规划，优先采用达到迁移下界的候选。

    Args:
        counts: int64[R,E] 全局源计数；每个 EP rank 从同一次 all-gather 获得。
        capacity: 每卡最多接收的不同远程专家数，默认由调用方设为 E/R。

    Returns:
        CPU 计数表、CPU 分配表、是否使用填满接收卡兜底；逐条路由仍留在设备。

    Raises:
        ValueError: 计数形状、数值或容量不合法。
        RuntimeError: 两个候选均不能同时满足均衡与副本容量。

    Example:
        R=4、E=8、B=2、专家总量为 [0,0,4,4,4,4,4,4] 时，
        部分填充超出副本容量，采用填满接收卡候选，迁移 12 条。
    """
    if counts.ndim != 2 or counts.dtype != torch.int64:
        raise ValueError("MoonEP计数必须是int64[R,E]")
    ranks, experts = counts.shape
    if ranks < 1 or experts < 1 or experts % ranks:
        raise ValueError("MoonEP计数的rank数和专家数必须为正，且专家数能被rank数整除")
    if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 0:
        raise ValueError("MoonEP副本容量必须是非负整数")

    # 动态 HCCL split 需要主机整数；只回传一次小表，候选和证书都在 CPU 计算。
    cpu = counts.cpu()
    _checked_count_rows(cpu)
    totals = cpu.numpy().sum(axis=0)
    if ranks == 1:
        return cpu, torch.from_numpy(totals.reshape(experts, 1).copy()), False

    # 两卡时两条规则等价；其余情况先检查部分填充能否证明迁移最优。
    rules = (False,) if ranks == 2 else (False, True)
    candidates = _cpu_candidates(totals, ranks, rules)
    scores = _cpu_candidate_metrics(candidates, totals, capacity)
    certified = [index for index, score in enumerate(scores) if score[5]]
    if certified:
        selected = min(certified, key=lambda index: (scores[index, 2], scores[index, 3], index))
        return cpu, torch.from_numpy(candidates[selected].copy()), False

    fallback = len(rules) - 1
    if not scores[fallback, 0]:
        raise RuntimeError(f"MoonEP快速规划无法满足均衡及副本容量 B={capacity}；候选指标={scores.tolist()}")
    return cpu, torch.from_numpy(candidates[fallback].copy()), True


def _receive_order_metadata(receive_counts: torch.Tensor | np.ndarray) -> tuple[list[int], list[int]]:
    """把CPU接收小表转换成分段平移量，设备无需再排序每一条token选择。

    Args:
        receive_counts: CPU整数表[R,E]，每个源rank按专家升序发送的行数。

    Returns:
        按“专家、源rank”排列的各段平移量与行数，长度均为R×E。
        输出位置j所属段的平移量加上j，即为该行在接收包中的位置。

    Example:
        输入[[2,1],[1,2]]时，接收包的专家号是[0,0,1,0,1,1]。
        四个目标段行数为[2,1,1,2]，平移量为[0,1,-1,0]。
        展开平移量并加[0,1,2,3,4,5]后得到[0,1,3,2,4,5]，
        恰好保持相同专家内原有的源rank及行顺序。
    """
    # 仅访问已经完成D2H的小表；批量整数前缀避免每个专家/源rank进入Python循环。
    counts = receive_counts.numpy() if isinstance(receive_counts, torch.Tensor) else receive_counts
    source_starts = (counts.reshape(-1).cumsum() - counts.reshape(-1)).reshape(counts.shape)
    lengths = counts.T.reshape(-1)
    target_starts = lengths.cumsum() - lengths
    return (source_starts.T.reshape(-1) - target_starts).tolist(), lengths.tolist()


def compile_plan(counts: torch.Tensor, allocation: torch.Tensor, ids: torch.Tensor,
                 rank: int, used_fallback: bool = False) -> ExecutionPlan:
    """将小规划编译成设备索引和HCCL split，反向/重算直接复用。

    Example:
        rank1的active=[e0,e2,e3]时第一段从remote取e0，第二段从home取e2/e3。
    """
    ranks, experts = counts.shape
    home_count = experts // ranks
    device = ids.device
    # CPU小表用共享内存视图批量编译，避免Torch逐操作分发；大token索引仍在设备生成。
    counts_array, allocation_array = counts.numpy(), allocation.numpy()
    source_ends = counts_array.cumsum(axis=0)
    source_starts = source_ends - counts_array
    target_ends = allocation_array.cumsum(axis=1)
    target_starts = target_ends - allocation_array
    overlap = np.maximum(np.minimum(source_ends[:, :, None], target_ends[None])
                         - np.maximum(source_starts[:, :, None], target_starts[None]), 0)
    token_send = overlap[rank].sum(axis=0).tolist()
    token_recv = overlap[:, :, rank].sum(axis=1).tolist()
    allocation_rows = allocation.tolist()
    remote = [[e for e in range(experts) if e // home_count != target and allocation_rows[e][target] > 0]
              for target in range(ranks)]
    send_ids = [e - rank * home_count for target in range(ranks) for e in remote[target]
                if e // home_count == rank]
    weight_send = [sum(e // home_count == rank for e in remote[target]) for target in range(ranks)]
    weight_recv = [sum(e // home_count == source for e in remote[rank]) for source in range(ranks)]
    active = [e for e in range(experts) if allocation_rows[e][rank] > 0]
    remote_row = {e: index for index, e in enumerate(remote[rank])}
    local_positions = [index for index, e in enumerate(active) if e // home_count == rank]
    home_dense = (len(local_positions) == home_count and
                  local_positions == list(range(local_positions[0], local_positions[0] + home_count)))
    remote_positions = [index for index, e in enumerate(active) if e // home_count != rank]
    group_counts = tuple(allocation_rows[e][rank] for e in active)
    # 计数与分配已经在CPU；批量 NumPy 指标代替逐专家 Python 循环。
    # 例如 E128/R8 的指标计算约由 0.24 ms 降为 0.08 ms，不再逐项转换计数列表。
    metrics = tuple(_cpu_candidate_metrics(allocation_array[None], counts_array.sum(0), experts)[0].tolist())

    # 所有短索引和路由前缀只传一次；这些 CPU 元数据本来就用于 HCCL 的动态 split。
    metadata_rows = [list(accumulate(group_counts)), send_ids, local_positions,
                     [active[i] % home_count for i in local_positions], remote_positions,
                     [remote_row[active[i]] for i in remote_positions]]
    spans, ranges = [], []
    begin = 0
    while begin < len(active):
        is_home = active[begin] // home_count == rank
        end = begin + 1
        while end < len(active) and (active[end] // home_count == rank) == is_home:
            end += 1
        rows = [e % home_count if is_home else remote_row[e] for e in active[begin:end]]
        spans.append((is_home, begin, end))
        ranges.append((rows[0], rows[-1] + 1) if rows == list(range(rows[0], rows[-1] + 1)) else None)
        metadata_rows.append(rows)
        begin = end
    offset = source_starts[rank] - (counts_array[rank].cumsum() - counts_array[rank])
    receive_offsets, receive_counts = _receive_order_metadata(overlap[:, :, rank])
    metadata_rows.extend((np.concatenate((offset, target_ends.reshape(-1))).tolist(),
                          receive_offsets, receive_counts))
    lengths = [len(row) for row in metadata_rows]
    packed = torch.tensor([value for row in metadata_rows for value in row], dtype=torch.long).to(device)
    pieces = packed.split(lengths)
    boundaries, sent_ids, home_positions, home_rows, remote_positions_device, remote_rows = pieces[:6]
    weight_spans = tuple((*span, rows) for span, rows in zip(spans, pieces[6:-3]))
    route_metadata, receive_offsets_device, repeat_counts = pieces[-3:]
    # 已知每段源/目标起点，只展开平移量；output_size避免动态尺寸读取设备。
    received = sum(token_recv)
    sort_order = torch.arange(received, device=device) + torch.repeat_interleave(
        receive_offsets_device, repeat_counts, output_size=received)
    return ExecutionPlan(
        allocation, _entry_order(counts, allocation, rank, ids, route_metadata), sort_order,
        boundaries, group_counts, token_send, token_recv, weight_send, weight_recv, sent_ids, weight_spans,
        home_positions, home_rows, remote_positions_device, remote_rows,
        home_count, any(bool(entries) for entries in remote), tuple(metrics), used_fallback, tuple(ranges),
        home_dense,
    )
