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
"""MoonEP 对 FSDP 参数恢复和梯度归约前扩展接口的适配。"""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor


class MoonEPAllGatherExtension:
    """在 FSDP 的 AG 前后处理专家输入与完整参数视图。

    Example:
        反向执行第 7 层时，前置 hook 把第 6 层的两个 FP32 分片复制进
        MoonEP 的 BF16 发送工作区；FSDP 执行并等待 EFS all-gather；
        后置 hook 读取完整 home 权重，后续 EP 迁移仍由 MoonEP 调度。
    """

    def __init__(self, executor: Any, runtime: Any) -> None:
        """绑定一个专家单元；例如第 6 层的所有恢复请求复用同一适配对象。"""
        self.executor = executor
        self.runtime = runtime

    def workspace_demand(self) -> int:
        """计算 AG 转换输入的最大元素数；例如两个分片合计 72 MiB 的 BF16 元素。

        Returns:
            发送工作区至少需要容纳的元素数；零表示沿用 FSDP 自有输入。
        """
        scheduler = getattr(self.executor.owner, "hsdp_scheduler", None)
        if scheduler is None or scheduler.hsdp_state.comm_fusion_policy.enable_comm_fusion:
            return 0
        return sum(parameter.padded_sharded_param_size.numel()
                   for parameter in scheduler.hsdp_state.hsdp_params
                   if parameter.shard_world_size > 1 and parameter.hsdp_placement.dim == 0
                   and parameter.all_gather_process_group is not None
                   and not parameter.offload_to_cpu and parameter.param_dtype is not None
                   and parameter.orig_dtype != parameter.param_dtype)

    def reserve_unsharded_storage(self, reclaimable: list) -> None:
        """在原计算流恢复 FSDP 完整参数的存储；例如 B7 提前为 B6 申请输出。

        Args:
            reclaimable: 本次新申请的底层张量列表，由 MoonEP 保活到通信结束。
        """
        scheduler = getattr(self.executor.owner, "hsdp_scheduler", None)
        if scheduler is None:
            return
        state = scheduler.hsdp_state
        if state.is_shard and not state.comm_fusion_policy.enable_comm_fusion:
            for parameter in state.hsdp_params:
                if parameter.unsharded_param_buffers and parameter.allgather_comm_ctx.allgather_handle is None:
                    empty = [buffer for buffer in parameter.unsharded_param_buffers
                             if buffer.untyped_storage().nbytes() == 0]
                    parameter.alloc_unsharded_param_buffers()
                    reclaimable.extend(empty)

    def fsdp_pre_all_gather(self, scheduler: Any, slot: Any) -> dict[int, torch.Tensor]:
        """为 EFS 恢复借用发送工作区；例如 gate/down 使用互不重叠的 BF16 切片。

        Args:
            scheduler: 持有参数和 AG 句柄的原 FSDP 调度器。
            slot: 本次 MoonEP 权重槽，给出计算 dtype 和专家矩阵宽度。

        Returns:
            参数 id 到借用输入的映射；不满足条件时为空并保留 FSDP 默认路径。
        """
        state = scheduler.hsdp_state
        if not state.is_shard or state.comm_fusion_policy.enable_comm_fusion or self.runtime.stream is None:
            return {}
        parameters = [parameter for parameter in state.hsdp_params
                      if parameter.shard_world_size > 1 and parameter.hsdp_placement.dim == 0
                      and parameter.all_gather_process_group is not None
                      and not parameter.offload_to_cpu and parameter.param_dtype == slot.gate.dtype
                      and parameter.orig_dtype != parameter.param_dtype
                      and parameter.allgather_comm_ctx.allgather_handle is None]
        if not parameters:
            return {}
        sizes = [parameter.padded_sharded_param_size.numel() for parameter in parameters]
        width = slot.gate.shape[1] * slot.gate.shape[2] + slot.down.shape[1] * slot.down.shape[2]
        limit = self.runtime.weight_send_capacity * width
        if sum(sizes) > limit:
            return {}
        scratch = self.runtime._borrow_weight_workspace("send", slot.gate, sum(sizes), limit)
        inputs, offset = {}, 0
        for parameter, size in zip(parameters, sizes):
            inputs[id(parameter)] = scratch.narrow(0, offset, size)
            offset += size
        self.runtime.statistics["pooled_all_gather_parameters"] += len(inputs)
        return inputs

    def fsdp_post_all_gather(self, scheduler: Any, slot: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """AG 完成后取完整 home 权重视图；例如 B6 得到可供 EP 迁移的两张矩阵。

        Args:
            scheduler: 已等待全部 AG 并恢复完整参数的 FSDP 调度器。
            slot: 与前置 hook 相同的权重槽，保证本次恢复的上下文一致。

        Returns:
            gate/up 与 down 的 GMM 视图，不新申请完整参数存储。
        """
        del scheduler, slot
        return self.executor.read_weights()

    def fsdp_after_backward_prefetch(self, scheduler: Any) -> None:
        """接在原生 FSDP AG 预取之后提交 EP 迁移；例如 B7 触发 B6 的权重 all-to-all-v。

        Args:
            scheduler: 刚为本专家提交 AG 的原生 FSDP 调度器。

        Example:
            反向预取深度为 1 时，B7 的 FSDP pre-hook 预取 B6 的 EFS 分片，
            随后本钩子在 EP 通信域流等待该 AG 并发出 B6 的副本迁移；
            B6 的计算只在真正读取权重时等待 ready 事件。
        """
        del scheduler
        self.runtime.prefetch_fsdp_expert(self.executor)

class MoonEPGradientExtension:
    """将 BF16 副本梯度 EP 回传插入 FSDP 原生 RS/AR 之前。

    Example:
        B7 的 dW 产生时前置 hook 在梯度域流发起 all-to-all-v；autograd
        同时交付本地 home 梯度。参数累积完成后，后置 hook 等待 B7 回传，
        把远程贡献加进真实参数梯度，随后由原 FSDP 发起 RS 和 HSDP AR。
    """

    def __init__(self, executor: Any, runtime: Any) -> None:
        """绑定一个专家 FSDP 单元；例如 B7 的多次微批共用同一个待归并队列。"""
        self.executor = executor
        self.runtime = runtime
        self.pending = []
        self._parameter_layout = None

    @staticmethod
    def _local(value: torch.Tensor) -> torch.Tensor:
        """取 DTensor 的本地张量；例如 EP 专家分片的梯度变为本卡 [P,I,H]。"""
        return value.to_local() if isinstance(value, DTensor) else value

    def _layouts(self, scheduler: Any) -> tuple[tuple[Any, bool], tuple[Any, bool]]:
        """按真实共享存储绑定两组参数和 GMM 视图；例如 [P,2I,H] 需要转置 dW。"""
        if self._parameter_layout is not None:
            return self._parameter_layout
        parameters = scheduler.hsdp_state.hsdp_params
        layouts = []
        for view in self.executor.read_weights():
            view = self._local(view)
            matches = []
            for parameter in parameters:
                local = self._local(parameter.unsharded_param)
                if (view.untyped_storage().data_ptr() != local.untyped_storage().data_ptr()
                        or view.storage_offset() != local.storage_offset()):
                    continue
                if view.shape == local.shape and view.stride() == local.stride():
                    matches.append((parameter, False))
                elif (local.ndim == 3 and view.shape == local.transpose(1, 2).shape
                      and view.stride() == local.transpose(1, 2).stride()):
                    matches.append((parameter, True))
            if len(matches) != 1:
                raise RuntimeError("专家 GMM 视图必须唯一对应真实 FSDP 参数或其二维转置")
            layouts.append(matches[0])
        if len(layouts) != 2 or layouts[0][0] is layouts[1][0]:
            raise RuntimeError("专家梯度扩展要求 gate/up 和 down 分别对应独立参数")
        self._parameter_layout = tuple(layouts)
        return self._parameter_layout

    @torch.no_grad()
    def fsdp_pre_gradient_reduce(self, scheduler: Any, context: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """计算本地贡献并发起 EP 回传；例如 B7 的 BF16 e0 副本 dW 先上 G 流。

        Args:
            scheduler: 持有本专家真实参数的 FSDP 调度器。
            context: GMM 的 (gate, down, plan) 梯度与前向冻结的迁移计划。

        Returns:
            可先交给 autograd 累积的两份本地 home 梯度。
        """
        del scheduler
        gate, down, plan = context
        home_gate, home_down, packet = self.runtime.start_gradient_return(self.executor, gate, down, plan)
        if packet is not None:
            self.pending.append(packet)
            self.runtime.statistics["gradient_exchanges"] += 1
        return home_gate, home_down

    @torch.no_grad()
    def fsdp_post_gradient_reduce(self, scheduler: Any) -> None:
        """等待 EP 并归并到真实参数梯度，再允许原生 RS；例如 B7 的 e0 dW 回 rank 0。

        Args:
            scheduler: 即将执行 post_backward 的 FSDP 调度器。
        """
        if not self.pending:
            return
        layouts = self._layouts(scheduler)
        for packet in self.pending:
            remote_gate, remote_down = self.runtime.wait_gradient_return(packet)
            for (parameter, transposed), contribution in zip(layouts, (remote_gate, remote_down)):
                if parameter.unsharded_param.grad is None:
                    raise RuntimeError("FSDP 参数梯度尚未累积，不能归并专家副本梯度")
                target = self._local(parameter.unsharded_param.grad)
                if transposed:
                    contribution = contribution.transpose(1, 2).contiguous()
                target.index_add_(0, packet.plan.send_ids, contribution)
            self.runtime.statistics["gradient_waits"] += 1
        self.pending.clear()
