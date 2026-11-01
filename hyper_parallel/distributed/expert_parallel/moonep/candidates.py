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
"""MoonEP 设备候选的离线参考；训练规划只使用 planning.py 的 CPU 小表路径。"""

import torch


def device_candidate(totals: torch.Tensor, ranks: int, fill_receiver: bool) -> torch.Tensor:
    """用固定次数设备运算生成分配候选，不读取设备标量。

    Args:
        totals: 同设备 int64 的 [E] 非负专家总计数，E 可被 ranks 整除。
        ranks: EP 大小；构图前已知的主机整数。
        fill_receiver: True 使用单远程 home 规则，False 使用部分填充。

    Returns:
        同设备 int64 的 [E,R] 分配表；调用方仍需检查副本容量。

    Example:
        totals=[0,0,4,4,4,4,4,4]，ranks=4 时，True 得到 M=12，
        False 得到 M=6；后者在 B=2 时因接收三位专家而不合法。
    """
    return device_candidates(totals, ranks, (fill_receiver,))[0]


def device_candidates(totals: torch.Tensor, ranks: int,
                      rules: tuple[bool, ...] = (False, True)) -> torch.Tensor:
    """批量生成迁移候选，共享计数前缀和专家排序，循环中只更新两个余额。

    Args:
        totals: 设备 int64[E] 专家计数。
        ranks: EP 卡数，E 必须能被 R 整除。
        rules: 每个批次的规则；False 部分填充，True 单远程归属。

    Returns:
        int64[B,E,R]；第 b 张表与独立执行 rules[b] 完全一致。

    Example:
        rules=(False,True) 时两条规则同一次 max/min 发射得到各自供需卡；
        rank2 向 rank0 移动3项，只改 balance[2]-=3、balance[0]+=3，
        不再构造 R×R 的 one-hot 外积。
    """
    experts = totals.numel()
    if totals.ndim != 1 or totals.dtype != torch.int64 or ranks < 1 or experts == 0 or experts % ranks:
        raise ValueError("要求 int64 [E] 输入，且 E 能被正整数 R 整除")
    if not rules:
        raise ValueError("迁移规则不能为空")
    local_experts, batches = experts // ranks, len(rules)
    rank_ids = torch.arange(ranks, device=totals.device)
    total = totals.sum()
    targets = total // ranks + (rank_ids < total % ranks).to(torch.int64)
    home_totals = totals.reshape(ranks, local_experts)
    balance = (home_totals.sum(1) - targets).expand(batches, -1).clone()
    fill = torch.tensor(rules, device=totals.device, dtype=torch.bool)[:, None]
    homes, receivers, amounts = [], [], []
    for _ in range(ranks - 1):
        surplus, home = balance.max(1, keepdim=True)
        negative_deficit, target = balance.min(1, keepdim=True)
        deficit = -negative_deficit
        amount = torch.where(fill, deficit, torch.minimum(surplus, deficit))
        # balance总和恒为零，最大值非负、最小值非正；全部平衡时amount自然为零。
        # 后续供需选择只依赖balance，配额表可在循环结束后一次写入。
        homes.append(home)
        receivers.append(target)
        amounts.append(amount)
        balance.scatter_add_(1, home, -amount)
        balance.scatter_add_(1, target, amount)
    quotas = totals.new_zeros((batches, ranks * ranks))
    if amounts:
        pairs = torch.cat(homes, dim=1) * ranks + torch.cat(receivers, dim=1)
        quotas.scatter_add_(1, pairs, torch.cat(amounts, dim=1))
    quotas = quotas.reshape(batches, ranks, ranks)

    # FP32 只决定启发式排序；计数、配额和证书始终保持 int64。
    expert_order = torch.argsort(home_totals.float(), dim=1, descending=True, stable=True)
    target_order = torch.argsort(quotas.float(), dim=2, descending=True, stable=True)
    ordered_experts = home_totals.gather(1, expert_order)
    ordered_quotas = quotas.gather(2, target_order)
    expert_ends = ordered_experts.cumsum(1)[None, :, :, None]
    target_ends = ordered_quotas.cumsum(2)[:, :, None, :]
    expert_starts = expert_ends - ordered_experts[None, :, :, None]
    target_starts = target_ends - ordered_quotas[:, :, None, :]
    ordered_moved = (torch.minimum(expert_ends, target_ends)
                     - torch.maximum(expert_starts, target_starts)).clamp_min_(0)
    # 两次完整置换覆盖每个元素，empty 足够；无须先清零相同大小的输出。
    expert_index = expert_order[None, :, :, None].expand(batches, -1, -1, ranks)
    target_index = target_order[:, :, None, :].expand(-1, -1, local_experts, -1)
    moved = torch.empty_like(ordered_moved).scatter_(2, expert_index, ordered_moved)
    moved = torch.empty_like(moved).scatter_(3, target_index, moved)
    remaining = home_totals[None] - moved.sum(3)
    home_index = rank_ids[None, :, None, None].expand(batches, -1, local_experts, 1)
    moved.scatter_add_(3, home_index, remaining[..., None])
    return moved.reshape(batches, experts, ranks)


def device_plan_metrics(
    allocation: torch.Tensor, totals: torch.Tensor, capacity: torch.Tensor,
) -> torch.Tensor:
    """在设备上校验候选并返回打包指标，不触发主机标量读取。

    Args:
        allocation: int64 [E,R] 候选配额，形状和dtype由调用边界校验。
        totals: int64 [E] 全局专家计数。
        capacity: int64 [R] 各卡远程专家容量；三者处于同一设备。

    Returns:
        int64 [6]，依次为合法标志、M、z、总副本数、M0、M最优证书标志。
        合法性与证书都是设备标志；非法候选的M/z不可用于选解。

    Example:
        第三个四卡例的部分填充返回 [0,6,3,3,6,0]；单home返回
        [1,12,2,4,6,0]。两者都没有 M 最优证书，应进入严格求解。
    """
    return device_plan_metrics_batch(allocation.unsqueeze(0), totals, capacity)[0]


def device_plan_metrics_batch(allocation: torch.Tensor, totals: torch.Tensor,
                              capacity: torch.Tensor) -> torch.Tensor:
    """一次校验 B 张候选，共享远程掩码和迁移下界。

    输入 int64[B,E,R]、[E]、[R]，输出 int64[B,6]，字段与单表接口相同。
    例如两条规则各生成一张表时，容量、总负载和 M0 只需计算一次。
    """
    batches, experts, ranks = allocation.shape
    local_experts = experts // ranks
    rank_ids = torch.arange(ranks, device=totals.device)
    homes = torch.arange(experts, device=totals.device) // local_experts
    remote = homes[:, None] != rank_ids[None, :]
    enabled = (allocation > 0) & remote
    incoming = enabled.sum(1)
    outgoing = enabled.reshape(batches, ranks, local_experts, ranks).sum((2, 3))
    total = totals.sum()
    targets = total // ranks + (rank_ids < total % ranks).to(torch.int64)
    valid = ((allocation >= 0).all((1, 2)) & (totals >= 0).all() & (capacity >= 0).all()
             & (allocation.sum(2) == totals).all(1)
             & (allocation.sum(1) == targets).all(1) & (incoming <= capacity).all(1))
    moved = (allocation * remote).sum((1, 2))
    peak = torch.maximum(incoming.amax(1), outgoing.amax(1))
    copies = enabled.sum((1, 2))
    lower_bound = (totals.reshape(ranks, local_experts).sum(1) - targets).clamp_min(0).sum()
    certificate = valid & (moved == lower_bound)
    return torch.stack((valid.long(), moved, peak, copies,
                        lower_bound.expand(batches), certificate.long()), dim=1)
