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
"""MoonEP 独立脚本实现：动态迁移热专家 token 与权重。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all


@dataclass(frozen=True)
class MoonEPPlan:
    """保存一轮路由的全局分配和本 rank 的收发次序。"""

    allocation: torch.Tensor
    dispatch_order: torch.Tensor
    send_counts: list[int]
    recv_counts: list[int]
    received_experts: torch.Tensor
    weight_send: list[int]
    weight_recv: list[int]
    remote_experts: list[int]
    move_weights: bool


class _MoonEPExchange(torch.autograd.Function):
    """把 token 分发与权重预取合为一个反向同步点，固定集体通信顺序。"""

    @staticmethod
    def forward(ctx: Any, payload: torch.Tensor, gate: torch.Tensor, down: torch.Tensor,
                send_ids: torch.Tensor, token_send: list[int], token_recv: list[int],
                weight_send: list[int], weight_recv: list[int], group: Any,
                move_weights: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """分发 token，并把两组同类型权重打包成一次交换。

        Example:
            两卡热专家迁移时，三个返回张量分别是本 rank 的 token 和两组远程权重。
        """
        ctx.save_for_backward(send_ids)
        ctx.token_send, ctx.token_recv = token_send, token_recv
        ctx.weight_send, ctx.weight_recv = weight_send, weight_recv
        ctx.group, ctx.move_weights = group, move_weights
        ctx.gate_shape, ctx.down_shape = gate.shape, down.shape
        ctx.gate_dtype, ctx.down_dtype = gate.dtype, down.dtype
        ctx.gate_device = gate.device
        ctx.pack_weights = move_weights and gate.dtype == down.dtype
        received = ep_all_to_all(payload, token_send, token_recv, group)
        if not move_weights:
            return (received, gate.new_empty((0,) + tuple(gate.shape[1:])),
                    down.new_empty((0,) + tuple(down.shape[1:])))
        if ctx.pack_weights:
            # 两组权重的专家顺序相同，沿特征维打包后分段仍以专家数计。
            gate_width = gate[0].numel()
            down_width = down[0].numel()
            selected_gate = gate.index_select(0, send_ids).reshape(len(send_ids), gate_width)
            selected_down = down.index_select(0, send_ids).reshape(len(send_ids), down_width)
            packed = torch.cat((selected_gate, selected_down), dim=1)
            remote = ep_all_to_all(packed, weight_send, weight_recv, group)
            remote_gate = remote[:, :gate_width].reshape((sum(weight_recv),) + tuple(gate.shape[1:]))
            remote_down = remote[:, gate_width:].reshape((sum(weight_recv),) + tuple(down.shape[1:]))
            return received, remote_gate, remote_down
        remote_gate = ep_all_to_all(gate.index_select(0, send_ids).contiguous(),
                                   weight_send, weight_recv, group)
        remote_down = ep_all_to_all(down.index_select(0, send_ids).contiguous(),
                                   weight_send, weight_recv, group)
        return received, remote_gate, remote_down

    @staticmethod
    def backward(ctx: Any, grad_received: torch.Tensor, grad_gate: torch.Tensor | None,
                 grad_down: torch.Tensor | None) -> tuple[Any, ...]:
        """先归还副本梯度，再撤销 token 分发，所有 rank 顺序完全一致。

        Example:
            只发送权重而不消费副本的 rank 仍按相同顺序参与打包梯度交换。
        """
        (send_ids,) = ctx.saved_tensors
        local_gate = torch.zeros(ctx.gate_shape, device=ctx.gate_device, dtype=ctx.gate_dtype)
        local_down = torch.zeros(ctx.down_shape, device=ctx.gate_device, dtype=ctx.down_dtype)
        if ctx.move_weights:
            if grad_down is None:
                grad_down = local_down.new_zeros((sum(ctx.weight_recv),) + tuple(ctx.down_shape[1:]))
            if grad_gate is None:
                grad_gate = local_gate.new_zeros((sum(ctx.weight_recv),) + tuple(ctx.gate_shape[1:]))
            if ctx.pack_weights:
                gate_width = ctx.gate_shape[1] * ctx.gate_shape[2]
                down_width = ctx.down_shape[1] * ctx.down_shape[2]
                packed_grads = torch.cat((grad_gate.reshape(grad_gate.shape[0], gate_width),
                                          grad_down.reshape(grad_down.shape[0], down_width)), dim=1)
                returned = ep_all_to_all(packed_grads, ctx.weight_recv, ctx.weight_send, ctx.group)
                returned_gate = returned[:, :gate_width].reshape(
                    (sum(ctx.weight_send),) + tuple(ctx.gate_shape[1:]))
                returned_down = returned[:, gate_width:].reshape(
                    (sum(ctx.weight_send),) + tuple(ctx.down_shape[1:]))
            else:
                returned_down = ep_all_to_all(grad_down.contiguous(), ctx.weight_recv,
                                              ctx.weight_send, ctx.group)
                returned_gate = ep_all_to_all(grad_gate.contiguous(), ctx.weight_recv,
                                              ctx.weight_send, ctx.group)
            local_down.index_add_(0, send_ids, returned_down)
            local_gate.index_add_(0, send_ids, returned_gate)
        grad_payload = ep_all_to_all(grad_received.contiguous(), ctx.token_recv, ctx.token_send, ctx.group)
        return grad_payload, local_gate, local_down, None, None, None, None, None, None, None


def _local_route_order(counts: torch.Tensor, allocation: torch.Tensor, rank: int,
                       local_experts: torch.Tensor, home_only: bool) -> torch.Tensor:
    """按全局专家序号求本 rank 的目标 rank 和发送次序。

    在昇腾上对可被 FP32 精确表示的整数复合键排序，避免 int64 ArgSort 落到 AICPU；
    大形状退回 CPU 规划，保持次序精确。

    Example:
        对 ``[0, 0, 1]``，结果先按目标 rank、再按专家、最后按原槽位排序。
    """
    ranks, experts = counts.shape
    capacity = local_experts.numel()
    device = local_experts.device
    # 小批量时 CPU 排序启动更省；大批量才用昇腾 FP32 ArgSort。
    use_device_sort = device.type == "npu" and capacity >= 2048 and experts * capacity < 2 ** 24
    ids = local_experts.to(torch.long) if use_device_sort else local_experts.cpu().to(torch.long)
    work_device = ids.device
    slots = torch.arange(capacity, device=work_device)
    # 没有迁移时，专家编号的顺序已经决定目标 rank，只需按专家和原槽位排序一次。
    if home_only:
        if use_device_sort:
            return torch.argsort((ids * capacity + slots).to(torch.float32))
        return torch.argsort(ids, stable=True)

    if use_device_sort:
        expert_key = (ids * capacity + slots).to(torch.float32)
        expert_order = torch.argsort(expert_key)
    else:
        expert_order = torch.argsort(ids, stable=True)
    sorted_experts = ids[expert_order]
    starts_cpu = counts[rank].cumsum(0) - counts[rank]
    source_prefix_cpu = counts[:rank].sum(0)
    boundaries_cpu = allocation.cumsum(1)
    # 小规模规划表合成一次 H2D 传输，避免三次独立拷贝和设备同步。
    metadata = torch.cat((starts_cpu, source_prefix_cpu, boundaries_cpu.reshape(-1))).to(work_device)
    starts = metadata[:experts]
    source_prefix = metadata[experts:2 * experts]
    ordinal = slots - starts[sorted_experts] + source_prefix[sorted_experts]
    boundaries = metadata[2 * experts:].reshape(experts, ranks)[sorted_experts]
    sorted_targets = (ordinal.unsqueeze(1) >= boundaries).sum(1)
    # 第一轮已把条目排成 (专家, 原槽位)；第二轮只按目标 rank 排即可。
    # 用排序位置构造唯一 FP32 键，避免再把目标 rank 散写回原槽位。
    if use_device_sort:
        target_key = (sorted_targets * capacity + slots).to(torch.float32)
        return expert_order[torch.argsort(target_key)]
    return expert_order[torch.argsort(sorted_targets, stable=True)]


def plan_moonep(counts: torch.Tensor, rank: int, local_experts: torch.Tensor) -> MoonEPPlan:
    """按 MoonEP 的盈亏规划，使目标 rank 的真实条目数最多相差一。

    Args:
        counts: CPU 上的每源 rank 每专家计数，形状为 ``[R, E]``。
        rank: 本 rank 在 EP group 中的编号。
        local_experts: 本 rank 的扁平 top-k 专家编号，可位于 CPU 或昇腾设备。

    Returns:
        包含 ``allocation[e, d]``、token 发送次序及远程权重交换表的规划。

    Example:
        ``plan_moonep(torch.tensor([[4, 0], [4, 0]]), 0, torch.zeros(4, dtype=torch.long))``
        会把热专家 0 的四个条目移到 rank 1，使两个 rank 各计算四个条目。
    """
    if counts.ndim != 2 or counts.device.type != "cpu":
        raise ValueError("counts 必须是 CPU 上的 [R, E] 张量")
    ranks, experts = counts.shape
    if ranks == 0 or experts == 0 or experts % ranks or not 0 <= rank < ranks:
        raise ValueError("专家数必须能被 EP rank 数整除，且 rank 必须有效")
    source_capacity = int(counts[rank].sum())
    if local_experts.numel() != source_capacity:
        raise ValueError("本地 top-k 数量与 counts 不一致")
    total_entries = int(counts.sum())
    target_capacities = [total_entries // ranks + int(index < total_entries % ranks)
                         for index in range(ranks)]
    per_rank = experts // ranks
    totals = counts.sum(0).tolist()
    allocation = [[0] * ranks for _ in range(experts)]
    balance = [-target for target in target_capacities]
    for expert, total in enumerate(totals):
        home = expert // per_rank
        allocation[expert][home] = int(total)
        balance[home] += int(total)

    # 一次填满接收方的缺口，使其以后不再接收第二个远程归属 rank。
    # 发送方可以暂时低于目标容量，随后作为接收方补足；这与原仓规划一致。
    quotas = [[0] * ranks for _ in range(ranks)]
    while max(balance) > 0:
        home = max(range(ranks), key=lambda index: balance[index])
        target = min(range(ranks), key=lambda index: balance[index])
        amount = -balance[target]
        quotas[home][target] += amount
        balance[home] -= amount
        balance[target] = 0

    # 在 home group 内，优先迁移 token 最多的专家，以减少权重副本数。
    for home in range(ranks):
        remaining = [int(totals[home * per_rank + offset]) for offset in range(per_rank)]
        while max(quotas[home]) > 0:
            target = max(range(ranks), key=lambda index: quotas[home][index])
            offset = max(range(per_rank), key=lambda index: remaining[index])
            amount = min(quotas[home][target], remaining[offset])
            expert = home * per_rank + offset
            allocation[expert][home] -= amount
            allocation[expert][target] += amount
            quotas[home][target] -= amount
            remaining[offset] -= amount

    if any(sum(allocation[expert]) != totals[expert] for expert in range(experts)):
        raise RuntimeError("MoonEP 专家 token 守恒失败")
    if any(sum(allocation[expert][target] for expert in range(experts)) != target_capacities[target]
           for target in range(ranks)):
        raise RuntimeError("MoonEP 目标 rank 容量规划失败")

    # 源 rank 顺序加稳定专家排序定义全局条目次序；向量化定位避免逐 token 的 Python 循环。
    allocation_tensor = torch.tensor(allocation, dtype=torch.int64)
    expert_rows = torch.arange(experts)
    home_only = int(allocation_tensor[expert_rows, expert_rows // per_rank].sum()) == total_entries
    dispatch_order = _local_route_order(counts, allocation_tensor, rank, local_experts, home_only)

    if home_only:
        # 所有专家都留在归属 rank，直接按专家归属求分段；省掉通用区间求交。
        local_begin, local_end = rank * per_rank, (rank + 1) * per_rank
        send_counts = counts[rank].reshape(ranks, per_rank).sum(1).tolist()
        recv_counts = counts[:, local_begin:local_end].sum(1).tolist()
        local_ids = torch.arange(local_begin, local_end)
        received_experts = torch.cat([
            torch.repeat_interleave(local_ids, counts[source, local_begin:local_end])
            for source in range(ranks)
        ])
        return MoonEPPlan(
            allocation=allocation_tensor,
            dispatch_order=dispatch_order,
            send_counts=send_counts,
            recv_counts=recv_counts,
            received_experts=received_experts,
            weight_send=[0] * ranks,
            weight_recv=[0] * ranks,
            remote_experts=[],
            move_weights=False,
        )

    # 源 rank 按 (目标 rank, 专家, 原始槽位) 发送；接收端可直接由全局规划重建专家标签。
    received_parts = []
    source_prefix = counts.cumsum(0)
    allocation_prefix = allocation_tensor.cumsum(1)
    source_begin = torch.zeros(experts, dtype=torch.int64) if rank == 0 else source_prefix[rank - 1]
    source_end = source_prefix[rank]
    target_starts = torch.cat((torch.zeros(experts, 1, dtype=torch.int64), allocation_prefix[:, :-1]), dim=1)
    local_overlaps = (torch.minimum(source_end[:, None], allocation_prefix)
                      - torch.maximum(source_begin[:, None], target_starts)).clamp_min(0)
    # 动态 all-to-all 仍需 CPU 分段列表，直接从 CPU 上已有的全局规划导出。
    # 这一步与 NPU 上异步执行的 dispatch_order 排序可以重叠。
    send_counts = local_overlaps.sum(0).tolist()
    recv_counts = []
    target_begin = torch.zeros(experts, dtype=torch.int64) if rank == 0 else allocation_prefix[:, rank - 1]
    target_end = allocation_prefix[:, rank]
    expert_ids = torch.arange(experts)
    for source in range(ranks):
        source_begin = torch.zeros(experts, dtype=torch.int64) if source == 0 else source_prefix[source - 1]
        source_end = source_prefix[source]
        overlap = (torch.minimum(source_end, target_end) - torch.maximum(source_begin, target_begin)).clamp_min(0)
        received_parts.append(torch.repeat_interleave(expert_ids, overlap))
        recv_counts.append(int(overlap.sum()))

    remote_by_rank = [
        [expert for expert in range(experts)
         if expert // per_rank != target and allocation[expert][target] > 0]
        for target in range(ranks)
    ]
    if any(len(requests) > per_rank for requests in remote_by_rank):
        raise RuntimeError("远程专家副本数量超过 MoonEP 权重槽上界")
    weight_send = [sum(expert // per_rank == rank for expert in remote_by_rank[target])
                   for target in range(ranks)]
    weight_recv = [sum(expert // per_rank == source for expert in remote_by_rank[rank])
                   for source in range(ranks)]
    return MoonEPPlan(
        allocation=allocation_tensor,
        dispatch_order=dispatch_order,
        send_counts=send_counts,
        recv_counts=recv_counts,
        received_experts=torch.cat(received_parts),
        weight_send=weight_send,
        weight_recv=weight_recv,
        remote_experts=remote_by_rank[rank],
        move_weights=not home_only,
    )


class MoonEPExecutor:
    """使用 Torch 集体通信和昇腾 grouped GEMM 执行 MoonEP。"""

    def __init__(self, *, ep_group: Any, ep_size: int, num_experts: int, top_k: int,
                 max_local_num_tokens: int | None, swiglu_limit: float) -> None:
        """记录固定模型尺寸；动态路由计划在每次 forward 构造。

        Example:
            ``MoonEPExecutor(ep_group=None, ep_size=1, num_experts=2, top_k=1,``
            ``max_local_num_tokens=8, swiglu_limit=10.0)`` 可在单卡运行。
        """
        if ep_size <= 0 or num_experts % ep_size or top_k <= 0:
            raise ValueError("MoonEP 的 EP、专家和 top-k 尺寸不合法")
        self.group = ep_group
        self.ranks = ep_size
        self.experts = num_experts
        self.top_k = top_k
        self.max_tokens = max_local_num_tokens
        self.limit = swiglu_limit

    def _exchange_weights(self, weight: torch.Tensor, plan: MoonEPPlan, rank: int) -> torch.Tensor:
        """只预取本步被迁移的专家权重，反向由可微 all-to-all 归还梯度。

        Example:
            两卡中专家 0 被迁到 rank 1 时，rank 0 发送权重 0；反向的权重梯度回到 rank 0。
        """
        if not self._move_weights(plan):
            return weight.new_empty((0,) + tuple(weight.shape[1:]))
        send_ids = self._weight_send_ids(plan, rank, weight.device)
        selected = weight.index_select(0, send_ids)
        if self.ranks == 1:
            return selected
        return ep_all_to_all(selected.contiguous(), plan.weight_send, plan.weight_recv, self.group)

    def _weight_send_ids(self, plan: MoonEPPlan, rank: int, device: torch.device) -> torch.Tensor:
        """按目标 rank、专家编号编排本地待发送的权重行。

        Example:
            本 rank 的专家 0 发往 rank 1 时，结果包含本地编号 0。
        """
        per_rank = self.experts // self.ranks
        ids = [expert - rank * per_rank for target in range(self.ranks)
               for expert in range(rank * per_rank, (rank + 1) * per_rank)
               if target != rank and plan.allocation[expert, target] > 0]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _move_weights(self, plan: MoonEPPlan) -> bool:
        """判断全 EP group 本步是否有远程权重需要交换。

        Example:
            路由完全按 home rank 均衡时返回 ``False``。
        """
        return plan.move_weights

    def _make_plan(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, rank: int) -> MoonEPPlan:
        """同步各 rank 的专家负载并生成当前 microbatch 的迁移计划。

        Example:
            ``executor._make_plan(hidden, ids, 0)`` 给 rank 0 返回 token 和权重的收发次序。
        """
        local_ids = top_k_index.reshape(-1).to(torch.int64)
        local_counts = torch.zeros(self.experts, dtype=torch.int64, device=hidden_states.device)
        local_counts.scatter_add_(0, local_ids, torch.ones_like(local_ids))
        if self.ranks > 1:
            # Gloo 要求输出为一维缓冲区；HCCL 也接受，收集后再恢复矩阵形状。
            gathered = torch.empty(self.ranks * self.experts, dtype=local_counts.dtype,
                                   device=hidden_states.device)
            dist.all_gather_into_tensor(gathered, local_counts, group=self.group)
            counts = gathered.reshape(self.ranks, self.experts).cpu()
        else:
            counts = local_counts.unsqueeze(0).cpu()
        return plan_moonep(counts, rank, local_ids)

    def _dispatch(self, hidden_states: torch.Tensor, plan: MoonEPPlan) -> torch.Tensor:
        """按计划把 routed entry 直接发到承担其专家计算的 rank。

        Example:
            ``executor._dispatch(hidden, plan)`` 返回本 rank 需计算的等容量条目。
        """
        tokens = hidden_states.numel() // hidden_states.shape[-1]
        order = plan.dispatch_order.to(hidden_states.device)
        source_rows = torch.div(order, self.top_k, rounding_mode="floor")
        payload = hidden_states.reshape(tokens, hidden_states.shape[-1]).index_select(0, source_rows)
        return ep_all_to_all(payload, plan.send_counts, plan.recv_counts, self.group) if self.ranks > 1 else payload

    def _prefetch_weights(self, weights: tuple[torch.Tensor, torch.Tensor], plan: MoonEPPlan,
                          rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """预取当前 rank 上被迁移专家的 gate/up 和 down 权重。

        Example:
            ``executor._prefetch_weights((gate_up, down), plan, rank)`` 返回两组远程副本。
        """
        gate, down = weights
        if self.ranks > 1 and plan.move_weights and gate.dtype == down.dtype:
            send_ids = self._weight_send_ids(plan, rank, gate.device)
            gate_width, down_width = gate[0].numel(), down[0].numel()
            selected_gate = gate.index_select(0, send_ids).reshape(len(send_ids), gate_width)
            selected_down = down.index_select(0, send_ids).reshape(len(send_ids), down_width)
            packed = torch.cat((selected_gate, selected_down), dim=1)
            remote = ep_all_to_all(packed, plan.weight_send, plan.weight_recv, self.group)
            return (remote[:, :gate_width].reshape((sum(plan.weight_recv),) + tuple(gate.shape[1:])),
                    remote[:, gate_width:].reshape((sum(plan.weight_recv),) + tuple(down.shape[1:])))
        return tuple(self._exchange_weights(weight, plan, rank) for weight in weights)

    def _expert_forward(self, states: torch.Tensor, weights: tuple[torch.Tensor, torch.Tensor],
                        remote_weights: tuple[torch.Tensor, torch.Tensor],
                        plan: MoonEPPlan, rank: int) -> torch.Tensor:
        """按专家整理接收条目，再用两次 GMM 和 FP32 clipped SwiGLU 计算。

        Example:
            专家标签 ``[1, 0]`` 会先整理为 ``[0, 1]``，计算后恢复原接收顺序。
        """
        labels = plan.received_experts
        if not labels.numel():
            return states
        per_rank = self.experts // self.ranks
        if states.device.type == "npu" and self.experts * labels.numel() < 2 ** 24:
            # FP32 整数键同时编码专家和原顺序，在 NPU 上得到稳定整理次序。
            device_labels = labels.to(states.device)
            positions = torch.arange(labels.numel(), device=states.device)
            sort_order = torch.argsort((device_labels * labels.numel() + positions).float())
        else:
            sort_order = torch.argsort(labels, stable=True).to(states.device)
        sorted_states = states.index_select(0, sort_order)
        active = [expert for expert in range(self.experts) if plan.allocation[expert, rank] > 0]
        remote_index = {expert: index for index, expert in enumerate(plan.remote_experts)}
        gate_up, down = weights
        remote_gate, remote_down = remote_weights
        if not plan.move_weights and active == list(range(rank * per_rank, (rank + 1) * per_rank)):
            # 均衡且本地专家都有 token 时，权重顺序已匹配 GMM，直接复用参数视图。
            grouped_gate, grouped_down = gate_up, down
        else:
            # 一次批量索引替代逐专家切片和 stack，减少小算子的设备启动次数。
            group_ids = [
                expert - rank * per_rank if expert // per_rank == rank else per_rank + remote_index[expert]
                for expert in active
            ]
            indices = torch.tensor(group_ids, dtype=torch.long, device=states.device)
            gate_table = torch.cat((gate_up, remote_gate), dim=0) if plan.remote_experts else gate_up
            down_table = torch.cat((down, remote_down), dim=0) if plan.remote_experts else down
            grouped_gate = gate_table.index_select(0, indices)
            grouped_down = down_table.index_select(0, indices)
        counts = [int(plan.allocation[expert, rank]) for expert in active]
        if states.device.type == "npu":
            # GMM 首次导入会加载可选 torch_npu；CPU 路径不依赖它。
            # 项目已有的 GMM autograd 桥接会为输入与权重生成梯度。
            from hyper_parallel.components.functional import grouped_matmul  # pylint: disable=C0415
            # 专家数量很少，在 CPU 小表上求前缀和，省去一次 NPU cumsum 启动。
            boundaries = torch.tensor(list(accumulate(counts)), device=states.device, dtype=torch.int64)
            gate_up_values = grouped_matmul(  # pylint: disable=not-callable
                sorted_states, grouped_gate, group_list=boundaries,
                group_type=0, group_list_type=0,
            )
        else:
            gate_up_values = torch.cat([
                sorted_states.narrow(0, sum(counts[:index]), count) @ grouped_gate[index]
                for index, count in enumerate(counts)
            ])
        gate, up = gate_up_values.float().chunk(2, dim=-1)
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
        intermediate = functional.silu(gate) * up
        intermediate = intermediate.to(states.dtype)
        if states.device.type == "npu":
            output = grouped_matmul(  # pylint: disable=not-callable
                intermediate, grouped_down, group_list=boundaries,
                group_type=0, group_list_type=0,
            )
        else:
            output = torch.cat([
                intermediate.narrow(0, sum(counts[:index]), count) @ grouped_down[index]
                for index, count in enumerate(counts)
            ])
        restored = torch.empty_like(output)
        restored = restored.index_copy(0, sort_order, output)
        return restored

    def _combine(self, computed: torch.Tensor, top_k_weights: torch.Tensor,
                 plan: MoonEPPlan, output_shape: torch.Size) -> torch.Tensor:
        """把专家结果发回源 rank，再以 FP32 累加每个 token 的 top-k 分支。

        Example:
            ``executor._combine(expert_rows, scores, plan, hidden.shape)`` 恢复原 token 形状。
        """
        returned = (ep_all_to_all(computed, plan.recv_counts, plan.send_counts, self.group)
                    if self.ranks > 1 else computed)
        order = plan.dispatch_order.to(returned.device)
        # 直接按发送次序取权重并累加，省去整块专家输出的逆置换及其反向梯度拷贝。
        weights = top_k_weights.reshape(-1).index_select(0, order).float()
        weighted = returned.float() * weights.unsqueeze(1)
        tokens = top_k_weights.numel() // self.top_k
        result = torch.zeros((tokens, output_shape[-1]), dtype=torch.float32, device=returned.device)
        indices = torch.div(order, self.top_k, rounding_mode="floor")
        result = result.index_add(0, indices, weighted)
        return result.to(returned.dtype).reshape(output_shape)

    def __call__(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                 top_k_weights: torch.Tensor, *, expert_weights: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """规划、分发、预取、专家计算和合并；保留 router/专家梯度。

        Example:
            ``executor(hidden, ids, scores, expert_weights=(gate_up, down))``
            返回与 ``hidden`` 同形状的 routed expert 分支。
        """
        tokens = hidden_states.numel() // hidden_states.shape[-1]
        if (self.max_tokens is not None and tokens > self.max_tokens) or top_k_index.numel() != tokens * self.top_k:
            raise ValueError("MoonEP 的 token 数超过容量或 top-k 形状不匹配")
        rank = dist.get_rank(group=self.group) if self.ranks > 1 else 0
        plan = self._make_plan(hidden_states, top_k_index, rank)
        # 整个 EP group 都没有 routed entry 时，所有 rank 一起跳过零长度交换。
        if not bool(plan.allocation.any()):
            return hidden_states * 0 + sum(weight.sum() * 0 for weight in expert_weights)
        order = plan.dispatch_order.to(hidden_states.device)
        source_rows = torch.div(order, self.top_k, rounding_mode="floor")
        payload = hidden_states.reshape(tokens, hidden_states.shape[-1]).index_select(0, source_rows)
        if self.ranks > 1:
            if self._move_weights(plan):
                received, remote_gate, remote_down = _MoonEPExchange.apply(
                    payload, *expert_weights,
                    self._weight_send_ids(plan, rank, hidden_states.device),
                    plan.send_counts, plan.recv_counts, plan.weight_send, plan.weight_recv,
                    self.group, True,
                )
                remote_weights = (remote_gate, remote_down)
            else:
                # 没有远程权重时只交换 token，省去空权重张量和无用零梯度。
                received = ep_all_to_all(payload, plan.send_counts, plan.recv_counts, self.group)
                remote_weights = tuple(weight.new_empty((0,) + tuple(weight.shape[1:]))
                                       for weight in expert_weights)
        else:
            received = payload
            remote_weights = self._prefetch_weights(expert_weights, plan, rank)
        computed = self._expert_forward(received, expert_weights, remote_weights, plan, rank)
        return self._combine(computed, top_k_weights, plan, hidden_states.shape)


__all__ = ["MoonEPExecutor", "MoonEPPlan", "plan_moonep"]
