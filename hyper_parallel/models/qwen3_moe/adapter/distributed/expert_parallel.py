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
"""expert_parallel: the Qwen3-MoE EP compute archetype factory.

Migrated from ``distributed/expert_parallel/recipes.py`` in M3 (adjust doc
§5.1/§5.4): only the Qwen-specific router contract (``TopKRouter`` module
returning ``(logits, scores, indices)``, no shared expert) sinks into the
model adapter. The combine skeleton, the router-adapter registry and the
dispatch/collective primitives stay in the generic
``auto_models/distributed/expert_parallel`` layer; this factory composes
them through the now-public ``build_ep_compute``.
"""

from functools import partial
from typing import Any, Callable

import torch

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.distributed._builder.forward_rewriter import _install_bound_forward
from hyper_parallel.distributed.expert_parallel.moonep import MoonEPExecutor
from hyper_parallel.distributed.expert_parallel.moonep.fused.executor import SparseMoonEPExecutor
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import checkpoint_route_ids
from hyper_parallel.distributed.expert_parallel.moonep.runtime import local_tensor

from hyper_parallel.distributed.expert_parallel.experts import (
    SHARED_MERGE_MODES,
)
from hyper_parallel.distributed.expert_parallel.recipes import (
    build_ep_compute,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
)
from hyper_parallel.distributed.recipe_spec import (
    local_compute,
)


@local_compute
def qwen3moe_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    use_grouped_gemm: bool = False,
    fix_router: bool = False,
    use_moonep: bool = False,
    use_megamoe: bool = False,
    megamoe_fused: bool = False,
    moonep_replica_capacity: int | None = None,
    moonep_prefetch_depth: int = 1,
    moonep_token_dedup: bool = False,
    moonep_fwd_ratio: float | None = None,
    moonep_bwd_ratio: float | None = None,
) -> Callable:
    """绑定Qwen3-MoE的普通EP或共享权重生命周期MoonEP。

    Args:
        module: 含gate和experts的真实MoE模块，router输出logits、scores和indices。
        mesh: 通用注入的整网拓扑，本函数无需直接使用。
        tp_mesh: 通用TP拓扑，由外层参数布局管理。
        cp_mesh: 通用CP拓扑，由外层序列布局管理。
        ep_mesh: 实际专家分发域。
        use_grouped_gemm: 普通EP基线是否采用grouped GEMM。
        fix_router: 普通EP基线是否采用仅供性能测试的均衡路由。
        use_moonep: 是否启用MoonEP的动态专家放置。
        use_megamoe: 单独启用MegaMoE；与MoonEP组合时选择独立稀疏执行器。
        megamoe_fused: 兼容旧配置的组合开关；组合路径使用独立稀疏执行器。
        moonep_replica_capacity: 不同远程专家数上限；None取E/EP。
        moonep_prefetch_depth: 仅旧 MoonEP 无 FSDP 路径使用；0按需恢复，1提前恢复。稀疏路径由 FSDP 通知预取。
        moonep_token_dedup: 是否在 MoonEP token 传输中启用按目标 rank 去重。
        moonep_fwd_ratio: 前向向量压缩的最大唯一率；None 使用默认值 0.75。
        moonep_bwd_ratio: 反向 dX 压缩的唯一率界限；None 使用默认值 1.0。

    Returns:
        供模型构建器绑定的本地计算函数。

    Example:
        use_moonep=True、E=128、EP=4时，每卡默认至多32份远程专家，整网共享两个表槽。
    """
    del mesh, tp_mesh, cp_mesh
    if not isinstance(moonep_token_dedup, bool):
        raise ValueError("moonep_token_dedup must be a boolean")
    if moonep_token_dedup and not use_moonep:
        raise ValueError("moonep_token_dedup requires use_moonep")
    if megamoe_fused and not (use_moonep and use_megamoe):
        raise ValueError("Qwen MegaMoE外部路由需要同时启用MoonEP和MegaMoE")
    if use_moonep:
        if fix_router:
            raise ValueError("Qwen MoonEP不支持仅供性能测试的均衡路由，请使用自然路由")
        if ep_mesh is None:
            raise ValueError("Qwen MoonEP需要显式EP mesh")
        experts = module.experts
        executor_type = SparseMoonEPExecutor if use_megamoe else MoonEPExecutor
        options = {"token_dedup": moonep_token_dedup,
                   "forward_ratio": moonep_fwd_ratio, "backward_ratio": moonep_bwd_ratio}
        if not use_megamoe:
            options.update(activation="swiglu", prefetch_depth=moonep_prefetch_depth)
        experts._moonep_executor = executor_type(  # pylint: disable=protected-access
            experts, partial(_qwen_expert_weights, experts),
            ep_group=ep_mesh.get_group("ep"), ep_size=ep_mesh["ep"].size(),
            num_experts=module.gate.num_experts, top_k=module.gate.top_k,
            replica_capacity=moonep_replica_capacity,
            **options,
        )
        _install_bound_forward(experts, _qwen_moonep_experts)
        _install_bound_forward(module.gate, _qwen_moonep_router)
        return _qwen_moonep_compute
    if use_megamoe:
        if fix_router:
            raise ValueError("Qwen MegaMoE不支持仅供性能测试的均衡路由")
        if ep_mesh is None:
            raise ValueError("Qwen MegaMoE需要显式EP mesh")
        experts = module.experts
        # Native executor has no parameters; the original expert module remains the FSDP owner.
        object.__setattr__(experts, "_qwen_mega_executor", MegaMoeExperts(
            hidden_size=experts.hidden_size, intermediate_size=experts.intermediate_size,
            num_experts=module.gate.num_experts, top_k=module.gate.top_k,
            ep_group=ep_mesh.get_group("ep"), create_parameters=False, swiglu_limit=None,
        ))
        _install_bound_forward(experts, _qwen_megamoe_experts)
        _install_bound_forward(module.gate, _qwen_moonep_router)
        return _qwen_moonep_compute
    return build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
        archetype_key="qwen3moe_topk_router",
        expected_attrs=["gate", "experts"],
        combine=SHARED_MERGE_MODES["none"],
        use_grouped_gemm=use_grouped_gemm,
        fix_router=fix_router,
    )


def _qwen_expert_weights(experts: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """读取真实参数的GMM视图；例如已转置的GroupedExperts直接返回[P,H,2I]。"""
    gate, down = local_tensor(experts.gate_up_proj), local_tensor(experts.down_proj)
    if not getattr(experts, "is_transposed", False):
        gate, down = gate.transpose(1, 2), down.transpose(1, 2)
    return gate, down


def _qwen_moonep_experts(experts: Any, hidden: torch.Tensor, ids: torch.Tensor,
                         scores: torch.Tensor) -> torch.Tensor:
    """在experts模块/FSDP边界内计算；例如EP4只持有32个home专家的参数边。"""
    return experts._moonep_executor(  # pylint: disable=protected-access
        hidden, ids, scores, expert_weights=_qwen_expert_weights(experts),
    )


def _qwen_megamoe_experts(experts: Any, hidden: torch.Tensor, ids: torch.Tensor,
                          scores: torch.Tensor) -> torch.Tensor:
    """保留Qwen专家参数的FSDP归属，使用完整MegaMoE路由和计算。"""
    return experts._qwen_mega_executor(  # pylint: disable=protected-access
        hidden, ids, scores, expert_weights=_qwen_expert_weights(experts),
    )


def _qwen_moonep_router(router: Any, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """保留Qwen5.x概率公式，重放只复用离散ID；例如top-k=8仍对8个可微score归一化。"""
    logits = torch.nn.functional.linear(hidden.reshape(-1, router.hidden_dim), router.weight)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    indices = checkpoint_route_ids(router, probabilities, router.top_k)
    scores = probabilities.gather(-1, indices)
    if router.norm_topk_prob:
        scores = scores / scores.sum(-1, keepdim=True)
    return logits, scores.to(logits.dtype), indices


def _qwen_moonep_compute(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """路由后调用实际experts子模块；例如[1,S,H]输入最终恢复为同一形状。"""
    indices, scores = MOE_ROUTER_ADAPTERS["qwen3moe"](module, hidden_states)
    return module.experts(hidden_states, indices, scores)


__all__ = [
    "qwen3moe_ep_compute_fn",
]
