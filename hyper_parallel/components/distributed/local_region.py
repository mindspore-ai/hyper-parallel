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
"""local_region: DTensor -> local -> DTensor 的局部计算区域包装。

基于 ``core.shard.custom_shard`` 的骨架，面向 05 双模式 DTensor 设计（validate 模式
的 MoE local_map / CP attention 内部区域）做了三点增强：

1. **命名参数绑定**：``in_placements`` 为 ``dict[str, placements]``，与
   ``ModuleShardingSpec.in_dst`` 的 dict 契约对齐；positional args 通过
   ``inspect.signature`` 映射到参数名，kwargs 原生支持（HF forward 以 kwargs 为主）。
2. **容错透传**：输入不是 DTensor 时原样透传（production 路径参数已解包的场景）；
   输出已是 DTensor 时不重复包装；全部输入均非 DTensor 时不包装输出。

**无反向缝合的说明（重要）**：hyper_parallel 的 DTensor 是自研的**前向-only**
placement/dispatch 系统，反向不经过 DTensor（不存在 DTensor autograd）。因此本
函数只做前向的 unwrap/wrap，不包含也不需要 autograd.Function 缝合与梯度
placement 声明（区别于 PyTorch ``local_map`` / Titan ``LocalMapConfig.
in_grad_placements``——那些是 torch DTensor 有反向语义的产物）。区域内部的
反向就是 local tensor 上的普通 autograd，梯度直接落在 local 参数分片上，
与 production 模式一致。

与 production 模式的关系：production 的 forward 包装（``_wrap_moe_forward``）在
build 期已把参数永久解包为 plain tensor，边界通信由 ``PrecompiledBoundary`` 执行，
不使用本函数。本函数服务于 **validate 模式**（参数保持 DTensor，区域边界需要
DTensor 契约缝合）与独立使用场景。
"""

import functools
import inspect
from typing import Callable, Dict, Optional, Sequence

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Placement

# 单个 tensor 的 placements：tuple[Placement, ...]，与 mesh 维度对齐
Placements = Sequence[Placement]


def _bind_arg_names(func: Callable) -> Dict[str, int]:
    """把 positional 参数位置映射到参数名（用于 dict 契约的按名查找）。

    签名不可内省（C 扩展等）时返回空 dict——此时仅 kwargs 传参能被
    in_placements 命中，positional 参数全部透传。
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    return {
        name: idx
        for idx, (name, p) in enumerate(sig.parameters.items())
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def _normalize_out_placements(out_placements, num_outputs: int):
    """把 out_placements 归一化为逐输出的 tuple[tuple[Placement, ...] | None, ...]。

    接受的写法：
      - 单输出扁平写法 ``(Partial(), Replicate())``（元素全是 Placement）；
      - 逐输出写法 ``((Partial(), Replicate()), None, (Shard(1), Replicate()))``
        （任一元素是 tuple 或 None，长度须等于输出数）。
    """
    if len(out_placements) == 0:
        raise ValueError("out_placements must not be empty")
    per_output = any(p is None or isinstance(p, tuple) for p in out_placements)
    if per_output:
        if len(out_placements) != num_outputs:
            raise ValueError(
                f"out_placements count {len(out_placements)} does not match "
                f"output count {num_outputs}!"
            )
        return tuple(out_placements)
    if num_outputs != 1:
        raise ValueError(
            f"flat out_placements only valid for single-output functions, "
            f"got {num_outputs} outputs!"
        )
    return (tuple(out_placements),)


def local_region(
    func: Optional[Callable] = None,
    *,
    device_mesh: DeviceMesh,
    in_placements: Optional[Dict[str, Optional[Placements]]] = None,
    out_placements: Optional[Sequence[Optional[Placements]]] = None,
    redistribute_inputs: bool = False,
) -> Callable:
    """把 func 包装为一个 DTensor -> local -> DTensor 的局部计算区域（前向）。

    Args:
        func: 被包装函数（forward 或任意 callable）。也可作装饰器工厂使用。
        device_mesh: DTensor 构造 / redistribute 使用的 mesh。
        in_placements: ``{arg_name: placements}`` —— 区域入口各 DTensor 输入
            期望的 placement。缺省（None 值或未列出）的输入不做 redistribute；
            非 DTensor 输入一律透传。
        out_placements: 区域出口的输出 placement 声明。单输出可扁平写
            ``(Partial(), Replicate())``；多输出逐位置写，非 tensor 输出用
            None 占位。为 None 时不包装输出（原样返回）。
        redistribute_inputs: 入口是否先把输入 redistribute 到 in_placements
            声明的 placement。双模式场景中边界通信已由 PrecompiledBoundary
            完成，传 False（默认）；独立使用时传 True。

    Returns:
        包装后的函数。签名与 func 一致。

    Examples:
        >>> # validate 模式 MoE 模块：边界 DTensor 契约保持，内部 local all-to-all
        >>> wrapped = local_region(
        ...     moe.forward, device_mesh=mesh,
        ...     in_placements={"hidden_states": (Replicate(), Replicate())},
        ...     out_placements=(Partial(), Replicate()),
        ... )

        >>> # 装饰器写法（独立使用，入口自行 redistribute）
        >>> @local_region(device_mesh=mesh,
        ...               in_placements={"x": (Shard(0),)},
        ...               out_placements=((Shard(0),),),
        ...               redistribute_inputs=True)
        ... def my_fn(x, bias=None):
        ...     return x + bias
    """
    def decorator(fn: Callable) -> Callable:
        name_to_idx = None  # 惰性缓存签名映射

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            nonlocal name_to_idx
            if name_to_idx is None:
                name_to_idx = _bind_arg_names(fn)

            args = list(args)
            saw_dtensor = False

            if in_placements:
                for name, placements in in_placements.items():
                    if name in kwargs:
                        from_kwargs, idx = True, None
                        value = kwargs[name]
                    else:
                        from_kwargs = False
                        idx = name_to_idx.get(name)
                        if idx is None or idx >= len(args):
                            continue
                        value = args[idx]

                    if not isinstance(value, DTensor):
                        # 非 DTensor 输入（production 已解包 / 非 tensor 参数）→ 透传
                        continue
                    saw_dtensor = True

                    dt = value
                    if (redistribute_inputs and placements is not None
                            and tuple(dt.placements) != tuple(placements)):
                        dt = dt.redistribute(device_mesh, placements)

                    local_value = dt.to_local()
                    if from_kwargs:
                        kwargs[name] = local_value
                    else:
                        args[idx] = local_value

            out = fn(*args, **kwargs)

            if not saw_dtensor or out_placements is None:
                return out

            single = not isinstance(out, tuple)
            out_items = (out,) if single else out
            placements_items = _normalize_out_placements(out_placements, len(out_items))

            wrapped_out = []
            for item, placements in zip(out_items, placements_items):
                if isinstance(item, DTensor):
                    # 区域内部已自行包装 → 不重复包装
                    wrapped_out.append(item)
                elif isinstance(item, torch.Tensor):
                    if placements is None:
                        raise TypeError(
                            "Tensor output requires non-None out_placements entry!"
                        )
                    wrapped_out.append(
                        DTensor.from_local(item, device_mesh, tuple(placements))
                    )
                else:
                    wrapped_out.append(item)

            return wrapped_out[0] if single else tuple(wrapped_out)

        return wrapped

    if func is not None:
        return decorator(func)
    return decorator
