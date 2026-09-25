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
"""Kimi-K2.6 EP archetype factory: sigmoid group routing + shared experts.

The family composes its routed branch through the public ``build_ep_compute``
skeleton (same pattern as ``models/qwen3_moe``), which is what keeps the router
choice inside the model adapter instead of the generic EP layer.

``fix_router=True`` selects the family's balanced router: a **benchmark-only**
variant that hands every destination rank the same slot count, so a skewed real
router can neither dominate step time nor grow the busiest rank's expert buffers
during a like-for-like comparison. It ignores the gate scores, so loss /
grad_norm / expert-load statistics from that mode are meaningless and must not be
reported as an efficiency result; correctness runs keep it off.
"""

import logging
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
from hyper_parallel.distributed.recipe_spec import local_compute

logger = logging.getLogger(__name__)


def _global_expert_count(module: Any) -> int:
    """Return the model-level routed expert count.

    Mirrors ``expert_parallel.experts``' own resolution so the family keeps its
    router self-contained: the module (or its ``experts`` child) may carry
    ``num_experts``, otherwise the config declares ``num_experts`` /
    ``n_routed_experts``.

    Raises:
        ValueError: If no owner exposes a routed expert count.
    """
    for owner in (getattr(module, "experts", None), module):
        count = getattr(owner, "num_experts", None)
        if count is not None:
            return int(count)
    config = getattr(module, "config", None)
    for name in ("num_experts", "n_routed_experts"):
        count = getattr(config, name, None)
        if count is not None:
            return int(count)
    raise ValueError(
        f"{type(module).__name__}: cannot determine the global routed expert count"
    )


def _balanced_expert_indices(
        token_count: int,
        experts_per_token: int,
        expert_count: int,
        local_count: int,
        *,
        device: Any,
) -> torch.Tensor:
    """Round-robin ``[token_count, experts_per_token]`` slots over destination ranks.

    Local slot ``i`` is sent to destination ``i % Q`` (``Q = expert_count /
    local_count``), landing on that rank's local expert ``(i // Q) % local_count``.
    Spreading by *rank* first is what makes the load exact: a per-expert round
    robin (``i % expert_count``) replays the same residue pattern on every rank,
    so the leftover slots pile onto a fixed group of ranks.

    Args:
        token_count: Local tokens ``T``.
        experts_per_token: Routed experts per token ``K``.
        expert_count: Global routed experts ``E``.
        local_count: Experts this rank owns ``L``.
        device: Device for the returned index tensor.

    Returns:
        ``[T, K]`` int64 indices naming the expert each slot is routed to.

    Raises:
        ValueError: If ``E`` is not divisible by ``L``.
    """
    if expert_count % local_count != 0:
        raise ValueError(
            f"num_experts ({expert_count}) must be divisible by the local "
            f"expert count ({local_count})"
        )
    destinations = expert_count // local_count
    slots = torch.arange(token_count * experts_per_token, device=device)
    indices = (slots % destinations) * local_count + (slots // destinations) % local_count
    return indices.view(token_count, experts_per_token)


def _balanced_router(module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``deepseekv3`` router variant with a deliberately uniform expert load.

    Runs the real sigmoid-group router to keep the gate in the compute profile
    and its weights differentiable, then *replaces the destination* of every
    slot with a round-robin assignment: every rank receives exactly ``T*K/EP``
    tokens instead of a data-dependent count. Measured on the 293B/18-layer
    config, the real router put 6.3x the mean token count on the busiest rank
    (leaving some experts empty) while this one stays at 1.00x.

    Benchmark-only, see the module docstring: the assignment ignores the gate
    scores, so ``loss`` / ``grad_norm`` and expert-hit distributions from this
    mode are meaningless.
    """
    topk_idx, topk_w = MOE_ROUTER_ADAPTERS["deepseekv3"](module, hidden_states)
    token_count, experts_per_token = topk_idx.shape
    local_count = int(
        getattr(getattr(module, "experts", None), "local_expert_count", None) or 1
    )
    balanced_idx = _balanced_expert_indices(
        int(token_count),
        int(experts_per_token),
        _global_expert_count(module),
        local_count,
        device=topk_idx.device,
    )
    return balanced_idx.to(topk_idx.dtype), topk_w


@local_compute
def kimi_k26_ep_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
        fix_router: bool = False,
        use_grouped_gemm: bool = False,
) -> Callable:
    """Archetype ``deepseekv3_sigmoid_group_shared`` for the Kimi-K2.6 text tower.

    Expected module interface: ``gate``, ``experts``, ``shared_experts``.

    Args:
        module: MoE block exposing the attributes above.
        mesh: Unused (kept for the factory signature).
        tp_mesh: Unused.
        cp_mesh: Unused.
        ep_mesh: Expert-parallel mesh whose ``ep`` axis owns the experts.
        fix_router: Select the benchmark-only balanced router (see
            :func:`_balanced_router`) instead of the real sigmoid-group router.
            Correctness runs must keep this ``False``.
        use_grouped_gemm: Run the local experts through the packed grouped-GEMM
            path instead of the eager per-expert loop.

    Returns:
        A callable ``(module, hidden_states) -> Tensor`` running the block.
    """
    del mesh, tp_mesh, cp_mesh
    logger.info(
        "kimi_k26 EP archetype: fix_router=%s use_grouped_gemm=%s (ep_size=%s)",
        fix_router, use_grouped_gemm,
        getattr(ep_mesh, 'size', lambda *a, **k: None)(),  # pylint: disable=not-callable
    )

    def combine(module: Any, hidden_states: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        """Merge the routed branch with the shared-experts branch."""
        return routed + module.shared_experts(hidden_states)    # nested boundary

    return build_ep_compute(
        module,
        ep_mesh,
        router_fn=_balanced_router if fix_router else MOE_ROUTER_ADAPTERS["deepseekv3"],
        archetype_key="deepseekv3_sigmoid_group_shared",
        expected_attrs=["gate", "experts", "shared_experts"],
        combine=combine,
        use_grouped_gemm=use_grouped_gemm,
    )


__all__ = ["kimi_k26_ep_compute_fn"]
