# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""dispatch_probe: region_dispatch 填 True 还是 False 的开发期判定工具。

公理要求"声明注入时必填 region_dispatch"，正确回答需要知道注入函数能否被
DTensor dispatch 完整穿透。本工具把试错从 apply 期提前到开发期：用 DTensor
试跑注入函数，记录 dispatch 轨迹，报告首个失败的算子并给出填写建议。

用法::

    from hyper_models.components.distributed.dispatch_probe import (
        check_dispatchable)

    report = check_dispatchable(my_compute_fn, example_inputs, mesh)
    print(report)   # dispatchable=True/False + 建议 + 失败算子

判定口径（与公理一致）：
- 试跑全程无异常 → 注入物是纯标准算子，可填 ``region_dispatch=True``；
- 任一算子在 DTensor dispatch/传播中报错（不支持的算子、通信原语作用于
  DTensor、数据依赖分支等）→ 填 ``False``（黑盒托管）。

注意：工具只验证"能否 dispatch"，不验证数值/布局正确性——True 之后仍需
validate 模式的 out_src 真校验兜底。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate

logger = logging.getLogger(__name__)


@dataclass
class DispatchProbeReport:
    """check_dispatchable 的判定报告（print 即可读）。"""

    dispatchable: bool
    """True = 试跑全程 DTensor dispatch 无异常（可填 region_dispatch=True）。"""

    ops: List[str] = field(default_factory=list)
    """dispatch 轨迹（aten opt 全量，含 DTensor 内部重分发）。"""

    failed_op: Optional[str] = None
    """首个失败算子（dispatchable=False 时给出）。"""

    error: Optional[str] = None
    """失败异常摘要（dispatchable=False 时给出）。"""

    recommendation: str = ""
    """region_dispatch 填写建议。"""

    def __str__(self) -> str:
        lines = [
            "=== check_dispatchable 报告 ===",
            f"dispatchable: {self.dispatchable}",
            f"dispatch 轨迹: {len(self.ops)} 个算子"
            + (f"（尾部: {', '.join(self.ops[-3:])}）" if self.ops else ""),
        ]
        if not self.dispatchable:
            lines.append(f"首个失败算子: {self.failed_op}")
            lines.append(f"异常: {self.error}")
        lines.append(f"建议: {self.recommendation}")
        return "\n".join(lines)


class _OpRecorder(TorchDispatchMode):
    """记录 dispatch 轨迹后原样放行（DTensor 子类传播逻辑照常执行）。

    in_flight 只进不出地保留"进入但未正常返回"的算子链——异常时其末尾即
    真正失败的算子（ops 的末尾可能只是失败前最后一个已完成的算子）。
    """

    def __init__(self) -> None:
        self.ops: List[str] = []
        self.in_flight: List[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func).replace("aten.", "")
        self.ops.append(name)
        self.in_flight.append(name)
        result = func(*args, **(kwargs or {}))   # 异常时 in_flight 保留现场
        self.in_flight.pop()
        return result


def check_dispatchable(
    fn,
    example_inputs: Sequence[Any],
    mesh,
    *,
    placements: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
) -> DispatchProbeReport:
    """用 DTensor 试跑注入函数，判定 region_dispatch 应填 True 还是 False。

    Args:
        fn: 待判定的注入物——区域 compute fn ``fn(module, *args)`` 的纯函数
            形态、普通 callable，或 nn.Module（调其 forward）。
        example_inputs: 本 rank 的 local 示例输入（tensor 逐项被
            ``DTensor.from_local`` 包装；非 tensor 项原样透传）。
        mesh: 目标 DeviceMesh（与 plan 坐标系一致的单 dp 切片）。
        placements: 包装用的 placements，缺省全 Replicate（最宽松入口；
            需要模拟真实入口布局时显式给，如 ``(Shard(1),)``）。
        kwargs: 透传给 fn 的关键字参数（同样逐值包装 tensor）。

    Returns:
        DispatchProbeReport —— ``dispatchable=True`` 建议
        ``region_dispatch=True``；``False`` 附首个失败算子与异常摘要，
        建议 ``region_dispatch=False``。
    """
    if placements is None:
        placements = tuple(Replicate() for _ in range(mesh.ndim))

    def wrap(value):
        if isinstance(value, torch.Tensor) and not isinstance(value, DTensor):
            return DTensor.from_local(value, mesh, tuple(placements))
        return value

    dt_args = [wrap(v) for v in example_inputs]
    dt_kwargs = {k: wrap(v) for k, v in (kwargs or {}).items()}
    target = fn
    swapped_params = []
    if isinstance(fn, torch.nn.Module):
        # 模块形态：参数也是 dispatch 参与者——临时包装为 DTensor（试跑后
        # 恢复），否则"DTensor 输入 × plain 参数"在第一个算子就混合报错，
        # 并非注入物本身不可 dispatch
        target = fn.forward
        for submodule in fn.modules():
            for key, param in list(submodule._parameters.items()):
                if param is None or isinstance(param, DTensor):
                    continue
                swapped_params.append((submodule, key, param))
                submodule._parameters[key] = DTensor.from_local(
                    param.detach(), mesh, tuple(placements))

    recorder = _OpRecorder()
    try:
        with recorder:
            target(*dt_args, **dt_kwargs)
    except Exception as exc:  # noqa: BLE001 —— 探针职责：把一切失败归为不可 dispatch
        if recorder.in_flight:
            failed = recorder.in_flight[-1]
        else:
            # c10d 通信等走 __torch_function__ 通道的算子不经过本 mode，
            # 从异常文本反提取算子名（"Operator all_gather does not ..."）
            m = re.search(r"Operator '?([A-Za-z0-9_.]+)'?", str(exc))
            failed = (m.group(1) if m
                      else (recorder.ops[-1] if recorder.ops
                            else "(无算子轨迹——调用前即失败)"))
        return DispatchProbeReport(
            dispatchable=False,
            ops=recorder.ops,
            failed_op=failed,
            error=f"{type(exc).__name__}: {exc}"[:500],
            recommendation=(
                "region_dispatch=False —— 注入物含不可 dispatch 的算子/通信/"
                "数据依赖逻辑（骨架黑盒托管：to_local → local 执行 → 声明式重包）"),
        )
    finally:
        for submodule, key, param in swapped_params:
            submodule._parameters[key] = param   # 恢复 plain 参数
    return DispatchProbeReport(
        dispatchable=True,
        ops=recorder.ops,
        recommendation=(
            "可填 region_dispatch=True —— 试跑全程 DTensor dispatch 穿透无异常"
            "（纯标准算子；仍须 validate 模式 out_src 真校验兜底布局正确性）"),
    )


__all__ = ["DispatchProbeReport", "check_dispatchable"]
