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
"""跨层共享权重槽、原生 EFS AG→EP 权重交换及显式专家反向。"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import bind_execution, is_replaying
from hyper_parallel.distributed.expert_parallel.moonep.fsdp_hooks import (
    MoonEPAllGatherExtension, MoonEPGradientExtension,
)
from hyper_parallel.distributed.expert_parallel.moonep.numerics import (
    ExpertSaved, expert_backward_values, expert_forward_values, grouped_values, grouped_weight_grad,
)
from hyper_parallel.distributed.expert_parallel.moonep.planning import ExecutionPlan, compile_plan, select_allocation


@dataclass(frozen=True)
class ExecutionKey:
    """原始专家调用的唯一键；例如(layer7,调用42,版本3)不会与另一微批混用。"""

    layer: int
    invocation: int
    version: int


@dataclass
class _Slot:
    """两组物理权重表与最后读事件；例如layer7退役后layer5才能覆盖槽0。"""

    gate: torch.Tensor
    down: torch.Tensor
    key: ExecutionKey | None = None
    done: Any = None


@dataclass
class _Ticket:
    """当前执行键的槽租约与物化完成事件；例如K6被K7预取后直接命中。"""

    slot: _Slot
    gate: torch.Tensor
    down: torch.Tensor
    ready: Any
    keepalive: list
    consumed: bool = False
    allocation_stream: Any = None
    reclaimable: tuple = ()


@dataclass
class _GradientPacket:
    """在梯度通信域等待归并的 BF16 副本梯度；例如 B7 的 e0 贡献。"""

    returned: torch.Tensor
    packed: torch.Tensor
    plan: ExecutionPlan
    gate_shape: torch.Size
    down_shape: torch.Size
    work: Any
    ready: Any = None


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """取布局张量的本地可微视图；例如EP的全局[E,H,I]变为本rank[P,H,I]。"""
    return value.to_local() if isinstance(value, DTensor) else value


def _communication_domain_stream(group: Any, device: torch.device) -> Any:
    """包装既有通信域的 collective 流，不另向流池申请应用侧流。

    例如 EP 梯度域原本就在流 E 上执行 all-to-all-v，本函数取得 E 的编号，
    返回指向同一条 E 的 Python 包装对象；后续归并仍通过事件保证 C/E 顺序。
    后端需要提供 get_coll_stream_id，以免隐式创建额外的应用流。
    """
    backend = group._get_backend(device)
    stream_id = backend.get_coll_stream_id(device)
    current = torch.npu.current_stream(device)
    stream = torch.npu.Stream(stream_id=stream_id, device_index=device.index,
                              device_type=current.device_type)
    if stream.stream_id != stream_id:
        raise RuntimeError("通信域流包装未指向同一底层流")
    return stream


class MoonEPRuntime:
    """整个顺序模型共享按实际并发请求增长的最终 GMM 表。"""

    def __init__(self, executors: list, *, standalone_prefetch: bool = False) -> None:
        """绑定同形状专家单元；例如两个专家请求同时在途时才分配第二槽。"""
        if not executors:
            raise ValueError("MoonEP要求非空专家列表")
        self.executors = executors
        # 仅无 FSDP 的独立脚本路径用此开关；FSDP 预取由其专家 AG hook 自然触发。
        self.standalone_prefetch = standalone_prefetch
        self.version = 0
        self.sequence = 0
        self.executions: dict[ExecutionKey, tuple[Any, ExecutionPlan]] = {}
        self.trace: list[ExecutionKey] = []
        self.tickets: dict[ExecutionKey, _Ticket] = {}
        self.slots: list[_Slot] = []
        self.weight_workspace: dict[str, torch.Tensor] = {}
        self.weight_workspace_demand: dict[str, int] = {}
        self.weight_workspace_busy = False
        self.stream = None
        self.statistics = Counter()
        self.active_calls = Counter()
        self.shared_executors = set()
        self.capacity = max(executor.home_count + executor.replica_capacity for executor in executors)
        self.gradient_compact_home = executors[0].gradient_compact_home
        if any(executor.gradient_compact_home != self.gradient_compact_home for executor in executors):
            raise ValueError("共享 MoonEP 运行时要求所有专家层使用相同的本地梯度整理策略")
        self.retain_weight_workspace = executors[0].retain_weight_workspace
        if any(executor.retain_weight_workspace != self.retain_weight_workspace for executor in executors):
            raise ValueError("共享 MoonEP 运行时要求所有专家层使用相同的权重工作区生命周期")
        self.weight_send_capacity = max(executor.home_count for executor in executors)
        self.weight_recv_capacity = max(executor.replica_capacity for executor in executors)
        for index, executor in enumerate(executors):
            executor.runtime, executor.layer = self, index
        self._validate_planner_capacity()
        self._configure_groups()

    def _validate_planner_capacity(self) -> None:
        """建模时检查各 EP rank 的容量一致，避免后续独立规划得到不同通信 split。"""
        first = self.executors[0]
        if first.ranks == 1 or not dist.is_initialized():
            return
        local = tuple((executor.experts, executor.replica_capacity) for executor in self.executors)
        gathered = [None] * first.ranks
        dist.all_gather_object(gathered, local, group=first.group)
        if any(other != local for other in gathered):
            raise ValueError("同一 EP 组的 MoonEP 专家数和副本容量必须在所有 rank 上一致")

    def _record_all_gather_workspace_demand(self) -> None:
        """从分片参数元数据记录发送区的AG最低需求，只保存元素数。

        Example:
            home权重144MiB、EFS=2时，发送池首次建立即容纳72MiB转换输入，
            即使前向只发送一个9MiB专家，也不会等首次反向预取才扩到72MiB。
            不申请Tensor，不改变FSDP参数、归约输出或梯度的分配方式。
        """
        self.weight_workspace_demand["send"] = max(
            ((executor.all_gather_extension or MoonEPAllGatherExtension(executor, self)).workspace_demand()
             for executor in self.executors),
            default=0,
        )

    def _configure_groups(self) -> None:
        """复用 DeviceMesh 的 EP/EFS 域；例如 EP4×EFS2 不再创建相同成员的新组。"""
        for executor in self.executors:
            # token、权重和梯度按各自调用次序提交到同一个 EP mesh 组。
            executor.weight_group = executor.group
            executor.gradient_group = executor.group
            scheduler = getattr(executor.owner, "hsdp_scheduler", None)
            if scheduler is not None:
                if len(scheduler.modules) != 1 or scheduler.cell is not executor.owner:
                    raise ValueError("MoonEP要求专家模块是独立FSDP单元")
                scheduler.enable_param_grad_gate()
                state = scheduler.hsdp_state
                groups = {param.mesh_info.shard_process_group
                          for param in state.hsdp_params if param.shard_world_size > 1}
                if len(groups) > 1:
                    raise ValueError("一个专家单元的参数必须使用同一EFS分片组")
                if any(param.all_gather_process_group is not param.mesh_info.shard_process_group
                       for param in state.hsdp_params if param.shard_world_size > 1):
                    raise ValueError("专家AG必须复用DeviceMesh的EFS分片组")
                extension = MoonEPAllGatherExtension(executor, self)
                scheduler.register_all_gather_extension(extension)
                executor.all_gather_extension = extension
                gradient_extension = MoonEPGradientExtension(executor, self)
                scheduler.register_gradient_extension(gradient_extension)
                executor.gradient_extension = gradient_extension

    def new_key(self, executor: Any) -> ExecutionKey:
        """分配原始调用编号；例如同一layer两个微批拥有不同invocation。"""
        self.sequence += 1
        key = ExecutionKey(executor.layer, self.sequence, self.version)
        self.trace.append(key)
        self.active_calls[executor.layer] += 1
        if self.active_calls[executor.layer] > 1:
            self.shared_executors.add(executor.layer)
        return key

    def gradient_stream(self, executor: Any, device: torch.device) -> Any:
        """复用 EP mesh 的唯一通信流；例如 B7 的 dW 接在其权重与 token 通信之后。"""
        if device.type != "npu":
            return None
        if executor.gradient_group is not executor.weight_group or self.stream is None:
            raise RuntimeError("专家梯度必须复用已初始化的 EP mesh 通信流")
        return self.stream

    def start_gradient_return(self, executor: Any, gate: torch.Tensor, down: torch.Tensor,
                              plan: ExecutionPlan) -> tuple[torch.Tensor, torch.Tensor, _GradientPacket | None]:
        """借用本梯度域流提交副本回传；例如 B7 的 dW 在最后一次 dX GMM 前发出。"""
        stream = self.gradient_stream(executor, gate.device) if plan.any_migration else None
        return launch_home_gradients(gate, down, plan, executor.gradient_group, stream,
                                     compact_home=self.gradient_compact_home)

    @staticmethod
    def wait_gradient_return(packet: _GradientPacket) -> tuple[torch.Tensor, torch.Tensor]:
        """在 FSDP 参数梯度就绪后等通信包；例如 B7 的远程 dW 到达再执行 RS。"""
        return wait_home_gradients(packet)

    def _ensure_slots(self, gate: torch.Tensor, down: torch.Tensor) -> None:
        """首次计算只分配一张最终表；例如 BF16 计算不会按 FSDP 深度预留空槽。"""
        if not self.slots:
            # FSDP在首次前向才完成混合精度初始化，此时才能读取真实AG转换输入大小。
            self._record_all_gather_workspace_demand()
            if len({id(executor.weight_group) for executor in self.executors}) != 1:
                raise ValueError("共享权重表要求所有专家层共用同一 EP 权重通信域")
            self.stream = (_communication_domain_stream(self.executors[0].weight_group, gate.device)
                           if gate.device.type == "npu" else None)
            self.slots = [self._new_slot(gate, down)]
        slot = self.slots[0]
        if (slot.gate.shape[1:] != gate.shape[1:] or slot.down.shape[1:] != down.shape[1:]
                or slot.gate.dtype != gate.dtype or slot.gate.device != gate.device):
            raise ValueError("共享专家表槽要求所有专家层的H/I、计算dtype和设备相同")

    def _new_slot(self, gate: torch.Tensor, down: torch.Tensor) -> _Slot:
        """按实际在途请求增加一张专家表；例如 B7 占用首槽时 B6 的 hook 获得第二槽。"""
        return _Slot(gate.new_empty((self.capacity, *gate.shape[1:])),
                     down.new_empty((self.capacity, *down.shape[1:])))

    @torch.no_grad()
    def prepare(self, key: ExecutionKey) -> _Ticket:
        """在预取流完整提交AG→EP→组表，消费时才等待ready；例如B7预取K6。"""
        if key.version != self.version:
            raise RuntimeError("旧执行键不能读取更新后的权重")
        if key in self.tickets:
            self.statistics["ticket_hits"] += 1
            return self.tickets[key]
        executor, plan = self.executions[key]
        # 首次调用已经位于专家FSDP pre-hook之后；后续物化从背景流恢复owner。
        if not self.slots:
            gate, down = executor.read_weights()
            self._ensure_slots(gate, down)
        if self.weight_workspace_busy:
            raise RuntimeError("权重工作区仍在提交或上次物化失败，不能覆盖可能在途的AG/EP输入")
        slot = next((candidate for candidate in self.slots if candidate.key is None), None)
        if slot is None:
            # 预取 hook 与真实消费共用请求表；只在原有槽全部在途时增加物理槽。
            # 容量取决于实际并发租约，不取决于含 dense 单元的 FSDP 预取列表长度。
            template = self.slots[0]
            slot = self._new_slot(template.gate, template.down)
            self.slots.append(slot)
        self.weight_workspace_busy = self.stream is not None
        slot.key = key
        buffers, reclaimable, allocation_stream = None, [], None
        if self.stream is not None:
            allocation_stream = torch.npu.current_stream()
            buffers = self._reserve_weight_buffers(executor, plan, slot, reclaimable)
            self.stream.wait_stream(allocation_stream)
        with torch.npu.stream(self.stream) if self.stream is not None else nullcontext():
            if slot.done is not None:
                self.stream.wait_event(slot.done)
            scheduler = getattr(executor.owner, "hsdp_scheduler", None)
            if scheduler is not None:
                if scheduler.hsdp_state.is_shard:
                    self.statistics["owner_restores"] += 1
                with torch.profiler.record_function("moonep::efs_restore"):
                    gate, down = scheduler.restore_with_all_gather_extension(slot)
            else:
                gate, down = executor.read_weights()
            self._ensure_slots(gate, down)
            with torch.profiler.record_function("moonep::ep_weight_exchange"):
                remote_gate, remote_down, keepalive = self._exchange_weights(executor, plan, gate, down, buffers)
            active = len(plan.group_counts)
            target_gate, target_down = slot.gate[:active], slot.down[:active]
            for (is_home, begin, end, rows), row_range in zip(plan.weight_spans, plan.weight_ranges):
                source_gate = gate if is_home else remote_gate
                source_down = down if is_home else remote_down
                if row_range is not None:
                    first, last = row_range
                    target_gate[begin:end].copy_(source_gate[first:last])
                    target_down[begin:end].copy_(source_down[first:last])
                else:
                    torch.index_select(source_gate, 0, rows, out=target_gate[begin:end])
                    torch.index_select(source_down, 0, rows, out=target_down[begin:end])
            ready = None
            if self.stream is not None:
                for tensor in (gate, down, *keepalive):
                    tensor.record_stream(self.stream)
                ready = torch.npu.Event()
                ready.record(self.stream)
        ticket = _Ticket(slot, target_gate, target_down, ready, keepalive,
                         allocation_stream=allocation_stream, reclaimable=tuple(reclaimable))
        self.weight_workspace_busy = False
        self.tickets[key] = ticket
        self.statistics["materializations"] += 1
        self.statistics["max_live_tickets"] = max(self.statistics["max_live_tickets"], len(self.tickets))
        return ticket

    def _reserve_weight_buffers(self, executor: Any, plan: ExecutionPlan, slot: _Slot,
                                reclaimable: list | None = None) -> tuple | None:
        """在计算流恢复完整权重存储，超过池预算的EP发送包单次申请。

        Returns:
            可选的独立发送包和接收占位；池内包在W借用，完整权重仍由FSDP管理。

        Example:
            B7预取B6时，在C恢复B6的home参数storage，W借3行发送和2行接收切片。
            若需发送17行且预算为16行，才在C单次申请17行；消费端仍等待ready。
            发送量按所有目标的split求和，不能用单卡副本容量限制它。
            reclaimable只登记本次在C新分配的storage，已有完整权重不能推断分配流。
        """
        if executor.all_gather_extension is not None and reclaimable is not None:
            executor.all_gather_extension.reserve_unsharded_storage(reclaimable)
        if not plan.any_migration:
            return None
        width = slot.gate.shape[1] * slot.gate.shape[2] + slot.down.shape[1] * slot.down.shape[2]
        # 常规包在W复用有界池；仅超过发送预算的热点包继续在C单次申请。
        send_limit = self.weight_send_capacity
        buffers = (slot.gate.new_empty((sum(plan.weight_send), width))
                   if sum(plan.weight_send) > send_limit else None, None)
        if reclaimable is not None:
            reclaimable.extend(buffer for buffer in buffers if buffer is not None)
        return buffers

    def _borrow_weight_workspace(self, role: str, template: torch.Tensor,
                                 elements: int, limit: int) -> torch.Tensor:
        """按需建立有界工作区，在唯一预取流复用，超限使用独立临时存储。

        Args:
            role: 发送或接收工作区；发送区同时供AG转换输入使用。
            template: 决定dtype和device的已有权重。
            elements: 当前规划实际需要的元素数。
            limit: 由专家数与副本容量配置推导的工作区预算，不代表立即申请量。

        Returns:
            足够容纳当前请求的连续视图；超预算请求不改变已有工作区。

        Example:
            预算16行，首次接收1行只申请1行；下次接收3行增长为3行，
            再接收4行增长为6行，后续不超过6行均复用。17行走独立申请。
            步末释放Tensor但记住实际需求4行，下一步首次接收1行直接申请4行。
            所有读写在W按AG wait→打包→A2Av wait→填表顺序提交；
            增长创建新storage，已有ticket保留的旧视图不会被resize破坏。
        """
        if elements == 0 or elements > limit:
            if elements > limit:
                self.statistics["weight_workspace_oversized"] += 1
            return template.new_empty(elements)
        demand = max(elements, min(limit, self.weight_workspace_demand.get(role, 0)))
        self.weight_workspace_demand[role] = demand
        buffer = self.weight_workspace.get(role)
        if buffer is not None and (buffer.dtype != template.dtype or buffer.device != template.device
                                   or buffer.numel() > limit):
            raise ValueError("共享权重工作区要求相同的dtype、device且不超过配置预算")
        if buffer is None or buffer.numel() < elements:
            capacity = demand if buffer is None else min(limit, max(demand, 2 * buffer.numel()))
            buffer = template.new_empty(capacity)
            self.weight_workspace[role] = buffer
            self.statistics["weight_workspace_allocations"] += 1
            self.statistics["weight_workspace_peak_bytes"] = max(
                self.statistics["weight_workspace_peak_bytes"],
                sum(value.numel() * value.element_size() for value in self.weight_workspace.values()),
            )
        return buffer[:elements]

    def _exchange_weights(self, executor: Any, plan: ExecutionPlan, gate: torch.Tensor,
                          down: torch.Tensor, buffers: tuple | None = None
                          ) -> tuple[torch.Tensor | None, torch.Tensor | None, list]:
        """打包两矩阵交换，临时缓冲保留到ready；例如e0的两组权重只发一次A2Av。"""
        if not plan.any_migration:
            return None, None, []
        gate_width, down_width = gate.shape[1] * gate.shape[2], down.shape[1] * down.shape[2]
        sent, received = sum(plan.weight_send), sum(plan.weight_recv)
        width = gate_width + down_width
        packed = buffers[0] if buffers is not None else None
        if self.stream is not None:
            if packed is None:
                limit = self.weight_send_capacity * width
                packed = self._borrow_weight_workspace("send", gate, sent * width, limit).view(sent, width)
            limit = self.weight_recv_capacity * width
            remote = self._borrow_weight_workspace("recv", gate, received * width, limit).view(received, width)
        else:
            remote = gate.new_empty((received, width))
        packed = pack_expert_rows(gate, down, plan.send_ids, out=packed)
        # 精确包住 collective 提交；在 ASCEND_PROFILER_OUTPUT 的 CPU 轨道中，
        # HcclAlltoAllV 会嵌套于此标记，并通过 HostToDevice 箭头对应设备通信。
        with torch.profiler.record_function("moonep::all_to_all_v::weight_replica"):
            work = dist.all_to_all_single(remote, packed, plan.weight_recv, plan.weight_send,
                                         group=executor.weight_group, async_op=True)
        # 后续组表马上要在 W 流读取 remote，因此此处只建立 W 对 HCCL 的依赖。
        # 主计算流不在这里等待；它直到 consume() 紧邻专家 GMM 时才等待 ready。
        work.wait()
        self.statistics["weight_exchanges"] += 1
        remote_gate = remote[:, :gate_width].reshape(received, *gate.shape[1:])
        remote_down = remote[:, gate_width:].reshape(received, *down.shape[1:])
        return remote_gate, remote_down, [packed, remote]

    def consume(self, key: ExecutionKey) -> tuple[torch.Tensor, torch.Tensor]:
        """在真正消费流等待当前ready；例如已预取K6无需再执行AG或EP。"""
        ticket = self.prepare(key)
        if self.stream is not None:
            current = torch.npu.current_stream()
            current.wait_event(ticket.ready)
            if current == ticket.allocation_stream:
                # C已排队等待W的最后读取，后续C复用必在该等待之后，无需再等host轮询。
                # 仅擦除确认在本次C分配的storage；其它来源仍由record_stream保护。
                from torch_npu import erase_stream  # pylint: disable=import-outside-toplevel
                for tensor in ticket.reclaimable:
                    if tensor.untyped_storage().nbytes():
                        erase_stream(tensor, self.stream)
        ticket.reclaimable = ()
        ticket.consumed = True
        ticket.keepalive.clear()
        return ticket.gate, ticket.down

    def release(self, key: ExecutionKey) -> None:
        """最后一次GMM读取排队后退役，下一写者等事件；例如K7槽供K5复用。"""
        ticket = self.tickets[key]
        if not ticket.consumed:
            raise RuntimeError("不能按正常路径归还尚未消费的预取")
        if self.stream is not None:
            ticket.slot.done = torch.npu.Event()
            ticket.slot.done.record(torch.npu.current_stream())
        ticket.slot.key = None
        del self.tickets[key]

    def prefetch_previous(self, key: ExecutionKey) -> None:
        """无 FSDP 的脚本路径预取前一层；例如独立三层模型的 K7 预取 K6。"""
        if not self.trace or self.trace[-1] != key:
            raise RuntimeError("首版MoonEP预取要求按原前向调用的逆序执行backward")
        if self.standalone_prefetch and len(self.trace) > 1:
            with torch.profiler.record_function("moonep::prefetch_previous"):
                self.prepare(self.trace[-2])

    def prefetch_fsdp_expert(self, executor: Any) -> None:
        """按原生 FSDP 目标选择专家调用，只扩展 EP all-to-all-v。

        Args:
            executor: 原生 FSDP 刚预取 AG 的目标专家单元。

        Example:
            B7 预取列表含 B6 时，FSDP 先提交 B6 的 EFS AG；本函数找到
            原前向记录的 B6 计划，接续 EP 权重交换并记录 ready。若同层有
            多个未完成微批，则留待真实消费时按调用键选择，避免配错副本。
        """
        if executor.layer in self.shared_executors:
            self.statistics["prefetch_ambiguous"] += 1
            return
        key = next((item for item in reversed(self.trace)
                    if item.layer == executor.layer and self.executions[item][0] is executor), None)
        if key is None or key in self.tickets:
            return
        with torch.profiler.record_function("moonep::fsdp_ep_prefetch"):
            self.prepare(key)
        self.statistics["fsdp_ep_prefetches"] += 1

    def complete(self, key: ExecutionKey) -> None:
        """真实反向已提交后删除小计划；例如R7退出不会调用此函数，B7才会。"""
        if self.trace[-1] != key:
            raise RuntimeError("反向完成次序与执行轨迹不符")
        self.trace.pop()
        del self.executions[key]
        self.active_calls[key.layer] -= 1
        if not self.active_calls[key.layer]:
            del self.active_calls[key.layer]
            self.shared_executors.discard(key.layer)

    def finish_step(self) -> None:
        """优化器更新前检查租约全部归还；例如两个微批均反向后才推进版本。"""
        if self.trace or self.executions or self.tickets or self.active_calls or self.shared_executors:
            raise RuntimeError("尚有MoonEP前向/预取未完成，拒绝进入下一参数版本")
        if any(executor.gradient_extension and executor.gradient_extension.pending for executor in self.executors):
            raise RuntimeError("专家副本梯度尚未在 FSDP 归约前完成归并")
        # 池只在W使用，通信Work与record_stream保护最后读取；不resize单个借用切片。
        if not self.retain_weight_workspace:
            self.weight_workspace.clear()
        self.version += 1


def _selected_expert_rows(value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """仅物化被选中的转置专家；例如从16个转置专家中选1个，避免先复制全部16个。

    输入三维GMM权重与专家行号，输出保留GMM逻辑维度的选中行。
    发送行数达到源专家数时使用原gather，避免重复专家的转置工作随发送量放大。
    """
    if rows.numel() < value.shape[0] and not value.is_contiguous():
        original = value.transpose(1, 2)
        if original.is_contiguous():
            return original.index_select(0, rows).transpose(1, 2)
    return value.index_select(0, rows)


@torch.no_grad()
def pack_expert_rows(gate: torch.Tensor, down: torch.Tensor, rows: torch.Tensor,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """按专家行打包两矩阵，依据源布局避免额外的全专家转置。

    输入 gate[A,H,2I]、down[A,I,H] 与 rows[S]；输出 [S,3HI]。
    例如 rows=[2,0,2] 时每行依次放该专家的 gate/up 和 down，允许重复发送。
    单行连续源直接写入连续out；多行采用gather+cat，避免NPU非连续out的内部临时量。
    可选out为连续[S,3HI]存储；例如C预分配、W填充，避免在W另建最终打包缓存。
    """
    count = rows.numel()
    gate_width, down_width = gate.shape[1] * gate.shape[2], down.shape[1] * down.shape[2]
    if out is not None and (out.shape != (count, gate_width + down_width) or not out.is_contiguous()
                            or out.dtype != gate.dtype or out.device != gate.device):
        raise ValueError("专家打包out必须具有正确shape、dtype、device并且连续")
    if count == 1 and gate.is_contiguous() and down.is_contiguous():
        packed = gate.new_empty((count, gate_width + down_width)) if out is None else out
        gate_view = packed[:, :gate_width].view(count, *gate.shape[1:])
        down_view = packed[:, gate_width:].view(count, *down.shape[1:])
        torch.index_select(gate, 0, rows, out=gate_view)
        torch.index_select(down, 0, rows, out=down_view)
        return packed
    return torch.cat((_selected_expert_rows(gate, rows).reshape(count, gate_width),
                      _selected_expert_rows(down, rows).reshape(count, down_width)), dim=1, out=out)


@torch.no_grad()
def local_home_gradients(gate: torch.Tensor, down: torch.Tensor,
                         plan: ExecutionPlan, *, compact_home: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """抽取当前计算卡产生的 home 梯度；例如 active=[e0,e2] 中只保留本卡 e2。

    返回 [P,H,2I]、[P,I,H]，未在本卡计算的 home 专家写零。
    远程贡献在副本梯度回传后执行相同的 BF16 index_add。
    """
    if compact_home and plan.home_dense:
        # 仅保留紧凑的 home 梯度存储，避免视图持有包含远程专家的整张 GMM 输出。
        start = next((begin for is_home, begin, end, _ in plan.weight_spans
                      if is_home and end - begin == plan.home_count), None)
        if start is None:
            raise RuntimeError("完整 home 专家在物理表中必须占一段连续位置")
        return gate[start:start + plan.home_count].clone(), down[start:start + plan.home_count].clone()
    home_gate = gate.new_zeros((plan.home_count, *gate.shape[1:]))
    home_down = down.new_zeros((plan.home_count, *down.shape[1:]))
    home_gate.index_add_(0, plan.home_rows, gate.index_select(0, plan.home_positions))
    home_down.index_add_(0, plan.home_rows, down.index_select(0, plan.home_positions))
    return home_gate, home_down


@torch.no_grad()
def return_home_gradients(gate: torch.Tensor, down: torch.Tensor, plan: ExecutionPlan,
                          group: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """EP副本梯度求和回归属，不改变FSDP缩放；例如rank1的e0 dW返回rank0。"""
    home_gate, home_down = local_home_gradients(gate, down, plan)
    if not plan.any_migration:
        return home_gate, home_down
    sent = sum(plan.weight_send)
    gate_width, down_width = gate.shape[1] * gate.shape[2], down.shape[1] * down.shape[2]
    # remote_positions 按全局专家号递增，与接收权重行一一对应，没有重复或空洞。
    packed = pack_expert_rows(gate, down, plan.remote_positions)
    returned = packed.new_empty((sent, gate_width + down_width))
    with torch.profiler.record_function("moonep::all_to_all_v::gradient_replica"):
        work = dist.all_to_all_single(returned, packed, plan.weight_send, plan.weight_recv,
                                     group=group, async_op=True)
    work.wait()
    home_gate.index_add_(0, plan.send_ids, returned[:, :gate_width].reshape(sent, *gate.shape[1:]))
    home_down.index_add_(0, plan.send_ids, returned[:, gate_width:].reshape(sent, *down.shape[1:]))
    return home_gate, home_down


@torch.no_grad()
def launch_home_gradients(gate: torch.Tensor, down: torch.Tensor, plan: ExecutionPlan,
                          group: Any, stream: Any, *, compact_home: bool = False
                          ) -> tuple[torch.Tensor, torch.Tensor, _GradientPacket | None]:
    """返回本地梯度并在现有梯度域流提交 EP 回传；例如 B7 先发出 e0 的 BF16 dW。

    Args:
        gate: GMM 产生的 gate/up 专家梯度。
        down: GMM 产生的 down 专家梯度。
        plan: 原前向冻结的专家副本与归属计划。
        group: 专家梯度回传的 EP 通信域。
        stream: 该通信域已有的 collective 流；CPU 测试时为 None。

    Returns:
        可以先交给 autograd 的两份本地 home 梯度，以及等待 FSDP 归并的通信包。
    """
    home_gate, home_down = local_home_gradients(gate, down, plan, compact_home=compact_home)
    if not plan.any_migration:
        return home_gate, home_down, None
    sent = sum(plan.weight_send)
    gate_width = gate.shape[1] * gate.shape[2]
    down_width = down.shape[1] * down.shape[2]
    packed = pack_expert_rows(gate, down, plan.remote_positions)
    returned = packed.new_empty((sent, gate_width + down_width))
    ready = None
    if stream is not None:
        # G 必须在 C 完成打包后读取；通信使用梯度域既有流，避免新增流割裂分配池。
        stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream) if stream is not None else nullcontext():
        with torch.profiler.record_function("moonep::all_to_all_v::gradient_replica"):
            work = dist.all_to_all_single(returned, packed, plan.weight_send, plan.weight_recv,
                                         group=group, async_op=True)
        if stream is not None:
            packed.record_stream(stream)
            returned.record_stream(stream)
            ready = torch.npu.Event()
            ready.record(stream)
    packet = _GradientPacket(returned, packed, plan, gate.shape, down.shape, work, ready)
    return home_gate, home_down, packet


@torch.no_grad()
def wait_home_gradients(packet: _GradientPacket) -> tuple[torch.Tensor, torch.Tensor]:
    """等待梯度域结果并拆出两张专家表；例如 B7 的远程 dW 在 RS 前变为可读。

    Args:
        packet: 前置 hook 保活的 BF16 通信包。

    Returns:
        按原 GMM 布局排列的 gate/up 与 down 远程贡献。
    """
    packet.work.wait()
    if packet.ready is not None:
        torch.npu.current_stream().wait_event(packet.ready)
    sent = sum(packet.plan.weight_send)
    width = packet.gate_shape[1] * packet.gate_shape[2]
    gate = packet.returned[:, :width].reshape(sent, *packet.gate_shape[1:])
    down = packet.returned[:, width:].reshape(sent, *packet.down_shape[1:])
    return gate, down


class _ExpertFunction(torch.autograd.Function):
    """只保存激活，不将临时完整专家表挂到原图；例如F7后可立即复用其槽。"""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, home_gate: torch.Tensor, home_down: torch.Tensor,
                key: ExecutionKey, runtime: MoonEPRuntime) -> torch.Tensor:
        """输出专家主序结果；例如输入[C,H]经过两次GMM仍为[C,H]。"""
        executor, plan = runtime.executions[key]
        gate, down = runtime.consume(key)
        with torch.profiler.record_function("moonep::expert_forward"):
            output, saved = expert_forward_values(x, gate, down, plan.boundaries,
                                                 executor.limit, grouped_values, executor.activation)
        ctx.save_for_backward(saved.x, saved.preact, saved.intermediate, saved.boundaries)
        ctx.key, ctx.runtime = key, runtime
        ctx.shapes = home_gate.shape, home_down.shape
        if not is_replaying():
            runtime.release(key)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """当前权重→必要重放→后继预取→四GMM→归属；例如R7完成后才预取K6。"""
        runtime, key = ctx.runtime, ctx.key
        executor, plan = runtime.executions[key]
        runtime.prepare(key)
        x, preact, intermediate, boundaries = ctx.saved_tensors
        gate, down = runtime.consume(key)
        scheduler = getattr(executor.owner, "hsdp_scheduler", None)
        if scheduler is None:
            runtime.prefetch_previous(key)
        saved = ExpertSaved(x, preact, intermediate, boundaries, executor.limit, executor.activation)
        home_gradients = None

        def launch_replica_gradients(gate_gradient: torch.Tensor, down_gradient: torch.Tensor) -> None:
            """在最后一次 dX GMM 前提交 EP 回传；例如 B7 用 dX 计算掩盖 e0 梯度通信。"""
            nonlocal home_gradients
            with torch.profiler.record_function("moonep::replica_grad_launch"):
                home_gradients = scheduler.launch_gradient_extension((gate_gradient, down_gradient, plan))

        with torch.profiler.record_function("moonep::expert_backward"):
            dx, dg, dd = expert_backward_values(
                grad_output, saved, gate, down, grouped_values, grouped_weight_grad,
                on_weight_gradients=launch_replica_gradients if scheduler is not None else None,
            )
        runtime.release(key)
        if scheduler is None:
            with torch.profiler.record_function("moonep::return_gradients"):
                hg, hd = return_home_gradients(dg, dd, plan, executor.gradient_group)
        else:
            hg, hd = home_gradients
        if (hg.shape, hd.shape) != ctx.shapes:
            raise RuntimeError("MoonEP完整归属梯度与真实参数输入形状不一致")
        runtime.complete(key)
        return dx, hg, hd, None, None


class MoonEPExecutor:
    """接收原始router结果，在实际experts模块边界内执行脚本MoonEP。"""

    def __init__(self, owner: torch.nn.Module, read_weights: Callable, *, ep_group: Any,
                 ep_size: int, num_experts: int, top_k: int, replica_capacity: int | None = None,
                 activation: str = "clipped", swiglu_limit: float = 0.0,
                 prefetch_depth: int = 1,
                 gradient_compact_home: bool = False,
                 retain_weight_workspace: bool = False) -> None:
        """绑定布局读取器；例如 EP=4 时按规划同步分发并回传 token。"""
        if ep_size < 1 or num_experts < 1 or num_experts % ep_size or not 0 < top_k <= num_experts:
            raise ValueError("非法EP、专家数或top-k")
        self.owner, self.read_weights = owner, read_weights
        self.group, self.ranks, self.experts, self.top_k = ep_group, ep_size, num_experts, top_k
        self.rank = dist.get_rank(ep_group) if ep_size > 1 else 0
        self.home_count = num_experts // ep_size
        self.replica_capacity = self.home_count if replica_capacity is None else replica_capacity
        if (isinstance(self.replica_capacity, bool) or not isinstance(self.replica_capacity, Integral)
                or self.replica_capacity < 0):
            raise ValueError("副本容量必须是非负整数")
        self.activation, self.limit = activation, swiglu_limit
        self.prefetch_depth = prefetch_depth
        self.gradient_compact_home = gradient_compact_home
        self.retain_weight_workspace = retain_weight_workspace
        self.runtime = None
        self.layer = -1
        self.weight_group, self.gradient_group = ep_group, ep_group
        self.all_gather_extension = None
        self.gradient_extension = None

    def _plan(self, ids: torch.Tensor) -> ExecutionPlan:
        """只为原始前向规划；例如重算命中原计划，免除再次同步小表。"""
        local = torch.zeros(self.experts, device=ids.device, dtype=torch.int64)
        local.scatter_add_(0, ids, torch.ones_like(ids))
        if self.ranks > 1:
            counts = local.new_empty(self.ranks * self.experts)
            dist.all_gather_into_tensor(counts, local, group=self.group)
            counts = counts.reshape(self.ranks, self.experts)
        else:
            counts = local.unsqueeze(0)
        counts_cpu, allocation, used_fallback = select_allocation(counts, self.replica_capacity)
        self.runtime.statistics["plans"] += 1
        self.runtime.statistics["planner_fallbacks"] += int(used_fallback)
        plan = compile_plan(counts_cpu, allocation, ids, self.rank, used_fallback)
        self.runtime.statistics["moved_entries"] += plan.metrics[1]
        self.runtime.statistics["peak_copies"] = max(self.runtime.statistics["peak_copies"], plan.metrics[2])
        self.runtime.statistics["remote_experts_max"] = max(
            self.runtime.statistics["remote_experts_max"], sum(plan.weight_recv))
        return plan

    def __call__(self, hidden: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor,
                 *, expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """按计划发送hidden、计算并加权回源；例如四token/two-top-k产生八条entry。"""
        if self.runtime is None:
            raise RuntimeError("模型物化/FSDP包装后必须调用configure_moonep统一建立共享生命周期")
        if ids.shape != scores.shape or ids.numel() != hidden.numel() // hidden.shape[-1] * self.top_k:
            raise ValueError("MoonEP路由维度与token数不匹配")
        key = bind_execution(self, lambda: self.runtime.new_key(self))
        if key not in self.runtime.executions:
            with torch.profiler.record_function("moonep::plan"):
                self.runtime.executions[key] = self, self._plan(ids.reshape(-1).long())
        _, plan = self.runtime.executions[key]
        weights = expert_weights if expert_weights is not None else self.read_weights()
        # 权重预取可由上一单元的 FSDP hook 提前触发；这里复用已有租约，
        # 未命中时才按需提交，随后使用同一同步 token 通信路径。
        self.runtime.prepare(key)
        source_rows = plan.dispatch_order // self.top_k
        payload = hidden.reshape(-1, hidden.shape[-1]).index_select(0, source_rows)
        if self.ranks > 1:
            with torch.profiler.record_function("moonep::all_to_all_v::token_dispatch"):
                received = ep_all_to_all(payload, plan.token_send, plan.token_recv, self.group)
        else:
            received = payload
        sorted_x = received.index_select(0, plan.sort_order)
        computed = _ExpertFunction.apply(sorted_x, *weights, key, self.runtime)
        # sort_order是无重复的完整排列；索引赋值的反向只保存索引，避免index_copy额外保留[C,H]输出。
        restored = torch.empty_like(computed)
        restored[plan.sort_order] = computed
        if self.ranks > 1:
            with torch.profiler.record_function("moonep::all_to_all_v::token_return"):
                returned = ep_all_to_all(restored, plan.token_recv, plan.token_send, self.group)
        else:
            returned = restored
        # 评估没有专家反向来完成key；例如no_grad验证后继续训练不会残留旧轨迹。
        if not torch.is_grad_enabled() or not computed.requires_grad:
            self.runtime.complete(key)
        # 与普通EP采用相同类型提升规则，避免额外FP32乘加改变Qwen的BF16舍入。
        selected_scores = scores.reshape(-1).index_select(0, plan.dispatch_order).to(returned.dtype)
        weighted = returned * selected_scores[:, None]
        result = torch.zeros_like(hidden.reshape(-1, hidden.shape[-1]), dtype=weighted.dtype)
        return result.index_add(0, source_rows, weighted).reshape_as(hidden)


def configure_moonep(model: torch.nn.Module, *, prefetch_depth: int | None = None) -> MoonEPRuntime:
    """在模型物化与FSDP包装之后建立运行时；例如48层只按实际并发共享表槽。

    Args:
        model: 已绑定专家执行器的完整模型。
        prefetch_depth: 仅无 FSDP 脚本路径的兼容参数；FSDP 模式不接受此参数。

    Returns:
        运行时统计及finish_step接口；训练每步梯度收尾后、optimizer更新前调用finish_step。
    """
    executors = [module._moonep_executor for module in model.modules()  # pylint: disable=protected-access
                 if isinstance(getattr(module, "_moonep_executor", None), MoonEPExecutor)]
    if any(executor.runtime is not None for executor in executors):
        raise ValueError("同一模型不能重复配置MoonEP运行时")
    # FSDP 决定何时对专家发起 AG；扩展 hook 随之触发 EP 通信，不读取预取深度。
    schedulers = [module.hsdp_scheduler for module in model.modules()
                  if getattr(module, "hsdp_scheduler", None) is not None]
    if schedulers:
        if prefetch_depth is not None:
            raise ValueError("FSDP 模式的专家预取由 AG hook 触发，不接受独立的 MoonEP 预取深度")
        standalone_prefetch = False
    else:
        if prefetch_depth is None:
            depths = {executor.prefetch_depth for executor in executors}
            if len(depths) != 1:
                raise ValueError("同一模型所有MoonEP单元必须采用相同预取配置")
            prefetch_depth = depths.pop()
        if (isinstance(prefetch_depth, bool) or not isinstance(prefetch_depth, Integral)
                or prefetch_depth < 0):
            raise ValueError("无 FSDP 路径的预取配置必须是非负整数")
        standalone_prefetch = bool(prefetch_depth)
    runtime = MoonEPRuntime(executors, standalone_prefetch=standalone_prefetch)
    model._moonep_runtime = runtime  # pylint: disable=protected-access
    return runtime


def has_moonep(module: torch.nn.Module) -> bool:
    """检查是否含生命周期执行器；例如普通EP模型返回False，保持已有构建行为。"""
    return any(isinstance(getattr(child, "_moonep_executor", None), MoonEPExecutor) for child in module.modules())
