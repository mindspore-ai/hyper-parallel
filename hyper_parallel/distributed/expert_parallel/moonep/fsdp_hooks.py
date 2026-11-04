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
    """Continue replica preparation from FSDP-owned parameter restoration hooks."""

    def __init__(self, executor: Any, runtime: Any) -> None:
        """Bind one expert unit to its shared weight runtime."""
        self.executor = executor
        self.runtime = runtime

    def fsdp_pre_all_gather(self, _state: Any) -> dict[int, torch.Tensor]:
        """Keep FSDP's per-restore inputs independent of the shared EP send pool.

        Native prefetch can submit several AGs before any wait. Sharing their
        input with the EP packing pool would permit an in-flight overwrite.
        """
        return {}

    def fsdp_post_all_gather(self, _state: Any) -> None:
        """Prepare a retained execution after FSDP has restored its home weights."""
        self.runtime.prefetch_fsdp_expert(self.executor)

    def fsdp_post_prefetch(self, _state: Any) -> None:
        """Prepare replicas even when FSDP retained complete home parameters."""
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
