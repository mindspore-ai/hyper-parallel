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
"""非重入checkpoint的MoonEP执行键与租约上下文。"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from hyper_parallel.core.activation_memory.api import checkpoint_wrapper
from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state


@dataclass
class _Tape:
    """一次checkpoint调用的小型记录；例如原F7和R7共享同一K7和router ID。"""

    executions: list = field(default_factory=list)
    routes: dict = field(default_factory=dict)
    identity: Any = None


@dataclass
class _Scope:
    """当前动态阶段；例如replay=True时读取旧键，不创建新的专家计划。"""

    tape: _Tape
    replay: bool
    cursor: int = 0


_SCOPE: ContextVar[_Scope | None] = ContextVar("moonep_checkpoint_scope", default=None)


def bind_execution(executor: Any, create_key: Any) -> Any:
    """原始前向分配key，重放取回旧key；例如两次layer7调用仍指向同一次原图。"""
    scope = _SCOPE.get()
    if scope is None:
        return create_key()
    if not scope.replay:
        key = create_key()
        scope.tape.executions.append((executor, key))
        if len(scope.tape.executions) > 1:
            raise ValueError("MoonEP重计算键映射要求一个checkpoint区域仅执行一次专家模块")
        return key
    if scope.cursor >= len(scope.tape.executions):
        raise RuntimeError("重放出现原始前向不存在的专家调用")
    expected, key = scope.tape.executions[scope.cursor]
    scope.cursor += 1
    if expected is not executor:
        raise RuntimeError("重计算专家调用顺序发生变化")
    return key


def is_replaying() -> bool:
    """读取动态重算状态；例如R7返回True，随后真实B7返回False。"""
    scope = _SCOPE.get()
    return scope is not None and scope.replay


def checkpoint_route_ids(router: Any, probabilities: torch.Tensor, top_k: int,
                         *, sorted: bool = True) -> torch.Tensor:
    """只复用离散top-k ID；例如R7仍从新logits可微gather分数以保留router梯度。"""
    scope = _SCOPE.get()
    if scope is not None and scope.replay:
        return scope.tape.routes[id(router)]
    with torch.no_grad():
        ids = probabilities.topk(top_k, dim=-1, sorted=sorted).indices
    if scope is not None:
        scope.tape.routes[id(router)] = ids
    return ids


@contextmanager
def _suspend_forward_prefetch(scheduler: Any) -> Iterator[None]:
    """重放时屏蔽普通前向预取并完整恢复；例如R7不错误预取layer8。"""
    # 现有scheduler暂未提供公开作用域API；同时保护backup以覆盖嵌套native hook。
    cells = scheduler.forward_prefetch_cells
    backup = scheduler._backup_forward_fetch  # pylint: disable=protected-access
    scheduler.forward_prefetch_cells = []
    scheduler._backup_forward_fetch = None  # pylint: disable=protected-access
    try:
        yield
    finally:
        scheduler.forward_prefetch_cells = cells
        scheduler._backup_forward_fetch = backup  # pylint: disable=protected-access


@contextmanager
def _phase(block: torch.nn.Module, tape: _Tape, replay: bool) -> Iterator[None]:
    """进入Hyper内置context之后绑定MoonEP状态；例如dense先unpack也会准备K7。"""
    state = get_recompute_state()
    if state is None or state.is_recomputing != replay:
        raise RuntimeError("MoonEP要求Hyper eager非重入checkpoint入口")
    if replay:
        if tape.identity is not state.invocation_id or len(tape.executions) != 1:
            raise RuntimeError("MoonEP重放必须关联原始单专家区域调用")
    else:
        tape.identity = state.invocation_id
    scope = _Scope(tape, replay)
    token = _SCOPE.set(scope)
    try:
        with ExitStack() as stack:
            if replay:
                schedulers = {id(module.hsdp_scheduler): module.hsdp_scheduler for module in block.modules()
                              if getattr(module, "hsdp_scheduler", None) is not None}
                for scheduler in schedulers.values():
                    stack.enter_context(_suspend_forward_prefetch(scheduler))
                executor, key = tape.executions[0]
                executor.runtime.prepare(key)
                executor.runtime.statistics["recomputations"] += 1
            yield
            if replay and scope.cursor != 1:
                raise RuntimeError("重放没有执行原始专家调用")
    finally:
        _SCOPE.reset(token)


def checkpoint_context_factory(block: torch.nn.Module) -> Any:
    """生成与某一区域绑定的context工厂；例如保留DeepSeek已有排除区域后注入此工厂。"""
    def context_factory() -> tuple[Any, Any]:
        """每次区域调用生成独立tape；例如两个微批不共享执行键。"""
        tape = _Tape()
        return _phase(block, tape, False), _phase(block, tape, True)

    return context_factory


def wrap_moonep_checkpoint(block: torch.nn.Module) -> torch.nn.Module:
    """为单MoE区域安装Hyper非重入重计算，应在FSDP包装前调用。

    Args:
        block: 包含一次专家执行的decoder区域，不包含checkpoint参数输入。

    Returns:
        checkpoint包装后的同语义模块；early_stop关闭以保证collective完整重放。

    Example:
        model.layers[7] = wrap_moonep_checkpoint(model.layers[7])；随后再安装FSDP。
    """
    return checkpoint_wrapper(block, use_reentrant=False, early_stop=False,
                              context_fn=checkpoint_context_factory(block))
