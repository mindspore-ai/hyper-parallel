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
"""容量约束下最小迁移、最小单卡副本瓶颈的精确整数规划。"""

from dataclasses import dataclass
import math
from numbers import Integral
import time
from typing import Callable, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class CapacityPlan:
    """记录分配和三个按优先级依次优化的指标。

    Example:
        ``allocation[e, d] == 4`` 表示专家 e 的四条路由条目由 d 计算；
        ``moved == 8`` 表示八条条目的计算位置不是它的专家归属 rank。
    """

    allocation: np.ndarray
    targets: np.ndarray
    remote_mask: np.ndarray
    moved: int
    peak_copies: int
    total_copies: int
    unconstrained_lower_bound: int


def solve_capacity_exact(
    totals: Sequence[int],
    ep_size: int,
    replica_capacity: Optional[Union[int, Sequence[int]]] = None,
    progress: Optional[Callable[[str, dict], None]] = None,
    time_limit: Optional[float] = None,
) -> CapacityPlan:
    """精确求解先最小迁移，再最小瓶颈副本数，最后最小总副本数的规划。

    Args:
        totals: 每个专家的全局路由条目数，长度 E；不包含源 rank 维度。
        ep_size: EP 域的 rank 数 R，要求 E 能被 R 整除。
        replica_capacity: 每个目标 rank 可接收的远程专家种数；None 为 E/R。
            可以给统一整数，也可以给长度 R 的整数数组。
        progress: 可选诊断回调，在每一阶段整数规划前后记录状态。
        time_limit: 三阶段整数规划的总秒数；超时即报错，不返回未经证明的近似解。

    Returns:
        x[e,d] 分配表、每 rank 目标条目数、远程副本掩码以及三个最优指标。
        peak_copies 是任一 rank 的最大单向发送/接收副本数，作为等大小
        专家权重的通信瓶颈代理；它不等于真实总延迟。

    Raises:
        ValueError: 输入不合法，或容量约束下无法达到固定均衡目标。
        RuntimeError: 求解器没有证明最优，或返回结果未通过整数约束校验。

    Example:
        ``solve_capacity_exact([0, 0, 4, 4, 4, 4, 4, 4], 4)``
        返回 moved=8、peak_copies=2、total_copies=3。
        ``solve_capacity_exact([4, 0], 2, 0)`` 因禁止副本且负载不均衡而报错。
    """
    # scipy仅用于无法获得设备最优证书时的精确求解；普通导入不加载该可选依赖。
    from scipy.optimize import Bounds, LinearConstraint, milp  # pylint: disable=import-outside-toplevel
    from scipy.sparse import coo_matrix  # pylint: disable=import-outside-toplevel

    if isinstance(ep_size, bool) or not isinstance(ep_size, Integral) or ep_size < 1:
        raise ValueError("ep_size 必须是正整数")
    if time_limit is not None and (isinstance(time_limit, bool) or not math.isfinite(time_limit)
                                   or time_limit <= 0):
        raise ValueError("time_limit 必须是有限正数")
    ranks = int(ep_size)
    raw = list(totals)
    if not raw or len(raw) % ranks:
        raise ValueError("totals 必须非空，专家数必须能被 ep_size 整除")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in raw):
        raise ValueError("totals 必须全部是非负整数")
    # 转成 Python 整数后求和，避免输入为 numpy 整数时累加溢出。
    raw = [int(v) for v in raw]
    # MILP 后端采用双精度，参考器主动限制整数规模，避免大整数失真。
    if sum(raw) > 2**40:
        raise ValueError("参考求解器要求总条目数不超过 2**40")
    experts = len(raw)
    local_experts = experts // ranks
    if replica_capacity is None:
        raw_capacity = [local_experts] * ranks
    elif isinstance(replica_capacity, Integral) and not isinstance(replica_capacity, bool):
        raw_capacity = [int(replica_capacity)] * ranks
    elif isinstance(replica_capacity, bool):
        raise ValueError("replica_capacity 不能是布尔值")
    else:
        try:
            raw_capacity = list(replica_capacity)
        except TypeError as error:
            raise ValueError("replica_capacity 必须是整数或整数序列") from error
    if len(raw_capacity) != ranks or any(
        isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in raw_capacity
    ):
        raise ValueError("replica_capacity 必须包含 R 个非负整数")
    # 超过所有远程专家数的容量与恰好容纳全部远程专家等价。
    capacity = np.array([min(int(v), experts - local_experts) for v in raw_capacity])
    counts = np.asarray(raw, dtype=np.int64)
    total = int(counts.sum())
    targets = np.array([total // ranks + int(d < total % ranks) for d in range(ranks)])
    homes = np.arange(experts) // local_experts
    remote = homes[:, None] != np.arange(ranks)[None, :]

    # 变量依次为 x[E,R]、y[E,R]、z；home 位置的 y 固定为零。
    # x 是计算条目数；y 表示远程副本是否启用；z 是单向副本数最大值。
    cells = experts * ranks
    variables = 2 * cells + 1
    z_index = variables - 1
    lower = np.zeros(variables)
    upper = np.concatenate((np.repeat(counts, ranks), remote.reshape(-1).astype(int),
                            [experts * max(0, ranks - 1)])).astype(float)
    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(terms: list[tuple[int, float]], lo: float, hi: float) -> None:
        """把一个稀疏线性约束追加到参考器的 CPU 矩阵。

        Example:
            ``add_constraint([(0, 1), (1, 1)], 4, 4)`` 添加 x0+x1=4。
        """
        row = len(row_lower)
        for column, coefficient in terms:
            row_ids.append(row)
            col_ids.append(column)
            values.append(coefficient)
        row_lower.append(lo)
        row_upper.append(hi)

    # 每个专家的条目必须守恒；每张卡必须达到固定目标条目数。
    for expert in range(experts):
        add_constraint([(expert * ranks + d, 1) for d in range(ranks)],
                       int(counts[expert]), int(counts[expert]))
    for target in range(ranks):
        add_constraint([(e * ranks + target, 1) for e in range(experts)],
                       int(targets[target]), int(targets[target]))

    for expert in range(experts):
        for target in range(ranks):
            if not remote[expert, target]:
                continue
            x_index = expert * ranks + target
            y_index = cells + x_index
            # y <= x <= n_e*y：整数条目数使 y 精确等价于 x>0。
            add_constraint([(x_index, 1), (y_index, -int(counts[expert]))], -np.inf, 0)
            add_constraint([(y_index, 1), (x_index, -1)], -np.inf, 0)

    for rank in range(ranks):
        incoming = [(cells + e * ranks + rank, 1) for e in range(experts)
                    if homes[e] != rank]
        outgoing = [(cells + e * ranks + d, 1) for e in range(experts)
                    if homes[e] == rank for d in range(ranks) if d != rank]
        add_constraint(incoming, -np.inf, int(capacity[rank]))
        add_constraint(incoming + [(z_index, -1)], -np.inf, 0)
        add_constraint(outgoing + [(z_index, -1)], -np.inf, 0)

    matrix = coo_matrix((values, (row_ids, col_ids)),
                        shape=(len(row_lower), variables)).tocsc()
    base = LinearConstraint(matrix, np.asarray(row_lower), np.asarray(row_upper))
    constraints = [base]
    migration_cost = np.zeros(variables)
    migration_cost[:cells] = remote.reshape(-1)
    peak_cost = np.zeros(variables)
    peak_cost[z_index] = 1
    copies_cost = np.zeros(variables)
    copies_cost[cells:2 * cells] = remote.reshape(-1)
    solution = np.zeros(variables, dtype=np.int64)

    # 分三次固定前一阶段的最优值，避免“大权重加权求和”产生尺度错误。
    deadline = time.monotonic() + time_limit if time_limit is not None else None
    for phase, objective in zip(("migration", "peak_copies", "total_copies"),
                                (migration_cost, peak_cost, copies_cost)):
        started = time.monotonic()
        remaining = max(0.0, deadline - started) if deadline is not None else None
        if remaining == 0.0:
            raise RuntimeError(f"整数规划总时限 {time_limit} 秒已耗尽，阶段 {phase} 未证明最优")
        if progress is not None:
            progress("milp_start", {"phase": phase, "variables": variables,
                                    "constraints": len(row_lower) + len(constraints) - 1,
                                    "mip_rel_gap": 0.0, "time_limit_s": remaining})
        options = {"mip_rel_gap": 0.0}
        if remaining is not None:
            options["time_limit"] = remaining
        result = milp(c=objective, integrality=np.ones(variables),
                      bounds=Bounds(lower, upper), constraints=constraints,
                      options=options)
        if progress is not None:
            gap = getattr(result, "mip_gap", None)
            node_count = getattr(result, "mip_node_count", None)
            dual_bound = getattr(result, "mip_dual_bound", None)
            progress("milp_end", {"phase": phase, "status": int(result.status),
                                  "message": str(result.message), "elapsed_s": time.monotonic() - started,
                                  "mip_gap": float(gap) if gap is not None else None,
                                  "mip_node_count": int(node_count) if node_count is not None else None,
                                  "mip_dual_bound": float(dual_bound) if dual_bound is not None else None})
        if result.status == 2:
            raise ValueError("容量约束下不存在达到固定均衡目标的分配")
        if not result.success or result.x is None or getattr(result, "mip_gap", 1.0) != 0.0:
            raise RuntimeError(f"整数规划阶段 {phase} 未证明最优：{result.message}")
        solution = np.rint(result.x).astype(np.int64)
        if np.max(np.abs(result.x - solution)) > 1e-5:
            raise RuntimeError("求解结果不是可靠的整数解")
        for constraint in constraints:
            actual = constraint.A @ solution
            if np.any(actual < constraint.lb - 1e-6) or np.any(actual > constraint.ub + 1e-6):
                raise RuntimeError("整数化后的结果违反线性约束")
        if np.any(solution < lower) or np.any(solution > upper):
            raise RuntimeError("整数化后的结果违反变量上下界")
        optimum = int(objective @ solution)
        constraints.append(LinearConstraint(objective.reshape(1, -1), optimum, optimum))

    allocation = solution[:cells].reshape(experts, ranks)
    mask = allocation.astype(bool) & remote
    home_load = counts.reshape(ranks, local_experts).sum(axis=1)
    return CapacityPlan(
        allocation=allocation,
        targets=targets,
        remote_mask=mask,
        moved=int((allocation * remote).sum()),
        peak_copies=int(solution[z_index]),
        total_copies=int(mask.sum()),
        unconstrained_lower_bound=int(np.maximum(home_load - targets, 0).sum()),
    )
