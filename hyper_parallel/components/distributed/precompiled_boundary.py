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
"""precompiled_boundary: 编译期通信规划（05 §4.3）。

RedistOp / PrecompiledBoundary 把 in_src→in_dst、out_src→out_dst 的 placement
差异编译为 RedistOp 序列；运行时零判断直接执行。所有非 identity 通信统一走
DTensor.redistribute()（自研 DTensor 内部按 (src,dst) 自动选最优 collective）。

API 适配（与 05 文档伪代码的差异，自研 DTensor 实际签名）：
- ``DTensor.from_local(local, mesh, placements)``：无 run_check 参数；
- ``dt.redistribute(mesh, placements)``：mesh 为第一个参数，无 async_op。
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Placement, Shard
from hyper_parallel.components.distributed.sharding_config import resolve_placements

logger = logging.getLogger(__name__)


def _classify_collective(src, dst) -> str:
    """从 placement 推导通信类型（调试/profiling 标签，非通信路径选择）。

    只比较有差异的维度——identity 维（如 attention 的 CP 维 Shard(1)→Shard(1)）
    不参与分类，使 TP 维 Shard→Replicate 正确归类为 all_gather。
    """
    if tuple(src) == tuple(dst):
        return "identity"
    diff_src = tuple(s for s, d in zip(src, dst) if s != d)
    diff_dst = tuple(d for s, d in zip(src, dst) if s != d)

    has_shard_src = any(isinstance(p, Shard) for p in diff_src)
    has_partial_src = any(isinstance(p, Partial) for p in diff_src)
    has_shard_dst = any(isinstance(p, Shard) for p in diff_dst)
    all_replicate_dst = all(
        not isinstance(p, (Shard, Partial)) for p in diff_dst
    )

    if has_partial_src and has_shard_dst:
        return "reduce_scatter"
    if has_partial_src and all_replicate_dst:
        return "all_reduce"
    if has_shard_src and all_replicate_dst:
        return "all_gather"
    return "redistribute"


def _get_arg(args, kwargs, name, idx, default=None):
    if name in kwargs:
        return kwargs[name]
    if idx is not None and idx < len(args):
        return args[idx]
    return default


def _set_arg(args, kwargs, name, idx, value):
    if name in kwargs:
        kwargs[name] = value
        return args, kwargs
    if idx is not None and idx < len(args):
        args = list(args)
        args[idx] = value
        return tuple(args), kwargs
    kwargs[name] = value
    return args, kwargs


@dataclass
class RedistOp:
    """一个预编译的 redistribute 操作（05 §4.3.1）。

    collective_type 为调试/profiling 标签；通信统一走 DTensor.redistribute()。
    """
    arg_name: str
    arg_index: Optional[int]
    mesh: object  # DeviceMesh
    src_placements: Tuple[Placement, ...]
    dst_placements: Tuple[Placement, ...]
    collective_type: str

    def execute(self, tensor: torch.Tensor, *, as_dtensor: bool = False):
        """执行通信。

        Args:
            tensor: 输入 local tensor（或 DTensor）。
            as_dtensor: True → 返回 DTensor（校验模式），False → 返回 local tensor。
        """
        if self.collective_type == "identity":
            if isinstance(tensor, DTensor):
                # validate（as_dtensor=True）保持 DTensor；production 返回
                # local——identity op 的输入可能来自 local region 的
                # from_local 重包装（MoE/CP wrapper），boundary 出口必须解包。
                return tensor if as_dtensor else tensor.to_local()
            if as_dtensor:
                return DTensor.from_local(
                    tensor, self.mesh, tuple(self.src_placements))
            return tensor

        # 统一路径：零拷贝包装 → redistribute → 可选 to_local
        if isinstance(tensor, DTensor):
            dt = tensor
        else:
            dt = DTensor.from_local(tensor, self.mesh, tuple(self.src_placements))
        dt = dt.redistribute(self.mesh, tuple(self.dst_placements))
        return dt if as_dtensor else dt.to_local()


class PrecompiledBoundary:
    """编译期通信计划（05 §4.3.3）：in_plan/out_plan 两个 RedistOp 序列。"""

    def __init__(self, spec, mesh, mesh_dim_names):
        self.spec = spec
        self.mesh = mesh
        self.mesh_dim_names = tuple(mesh_dim_names)
        self.in_plan = self._compile_input_plan(spec, mesh, self.mesh_dim_names)
        self.out_plan = self._compile_output_plan(spec, mesh, self.mesh_dim_names)

    # ── 编译 ────────────────────────────────────────────────────────────

    def _compile_input_plan(self, spec, mesh, mesh_dim_names):
        """从 in_src → in_dst 编译输入通信计划（identity 维度自然编译为直通 op）。"""
        plan = []
        all_names = set(spec.in_src.keys()) | set(spec.in_dst.keys())
        for name in sorted(all_names):
            src_p = tuple(resolve_placements(
                spec.in_src.get(name, {}), mesh_dim_names))
            dst_p = tuple(resolve_placements(
                spec.in_dst.get(name, {}), mesh_dim_names))
            plan.append(RedistOp(
                arg_name=name,
                arg_index=None,
                mesh=mesh,
                src_placements=src_p,
                dst_placements=dst_p,
                collective_type=_classify_collective(src_p, dst_p),
            ))
        return plan

    def _compile_output_plan(self, spec, mesh, mesh_dim_names):
        """从 out_src → out_dst 编译输出通信计划（identity 跳过，支持多输出）。

        arg_index 来源优先级：(1) spec.out_names 显式顺序；(2) out_src key 顺序。
        out_src=None 或 out_dst=None → 不编译。
        """
        if spec.out_src is None or spec.out_dst is None:
            return []

        out_names = getattr(spec, "out_names", None) or list(spec.out_src.keys())
        name_to_idx = {name: i for i, name in enumerate(out_names)}

        plan = []
        all_names = set(spec.out_src.keys()) | set(spec.out_dst.keys())
        for name in sorted(all_names):
            src_p = tuple(resolve_placements(
                spec.out_src.get(name, {}), mesh_dim_names))
            dst_p = tuple(resolve_placements(
                spec.out_dst.get(name, {}), mesh_dim_names))
            if src_p == dst_p:
                continue  # identity，不需要通信
            plan.append(RedistOp(
                arg_name=name,
                arg_index=name_to_idx.get(name, 0),
                mesh=mesh,
                src_placements=src_p,
                dst_placements=dst_p,
                collective_type=_classify_collective(src_p, dst_p),
            ))
        return plan

    # ── 运行时执行 ──────────────────────────────────────────────────────

    def redistribute_inputs(self, args, kwargs, *, as_dtensor=False):
        """执行输入重分布。as_dtensor=True → 返回 DTensor（校验模式）。

        arg 未在 args/kwargs 中找到（None）时跳过该 op——如 embed 的
        in_src key "input" 与实际 kwargs 名 "input_ids" 不同名且 identity。
        """
        for op in self.in_plan:
            arg = _get_arg(args, kwargs, op.arg_name, op.arg_index, default=None)
            if arg is None:
                continue
            result = op.execute(arg, as_dtensor=as_dtensor)
            args, kwargs = _set_arg(args, kwargs, op.arg_name, op.arg_index, result)
        return args, kwargs

    def redistribute_outputs(self, outputs, *, as_dtensor_input=False):
        """执行输出重分布（单输出 Tensor / 多输出 tuple，保序，返回同构）。

        as_dtensor_input=True → 输入已是 DTensor（校验模式）。
        """
        is_tuple = isinstance(outputs, (tuple, list))
        outputs_list = list(outputs) if is_tuple else [outputs]
        for op in self.out_plan:
            idx = op.arg_index if op.arg_index is not None else 0
            if idx >= len(outputs_list):
                logger.warning(
                    "PrecompiledBoundary: out_plan expects output '%s' at index %d, "
                    "but module returned only %d outputs. Skipping.",
                    op.arg_name, idx, len(outputs_list),
                )
                continue
            tensor = outputs_list[idx]
            if tensor is None:
                continue
            # as_dtensor_input=True（validate）→ 保持 DTensor 供 out_dst 校验；
            # 否则返回 local（production / 边界最终出口）。
            outputs_list[idx] = op.execute(tensor, as_dtensor=as_dtensor_input)
        return tuple(outputs_list) if is_tuple else outputs_list[0]
