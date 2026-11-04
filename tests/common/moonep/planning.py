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
"""Independent scalar metrics for planner validation; no training imports."""


def cpu_plan_metrics(allocation: list[list[int]], totals: list[int],
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
