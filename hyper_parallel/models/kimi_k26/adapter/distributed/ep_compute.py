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

``fix_router=True`` selects the framework's balanced router
(``routing.make_balanced_router``, opt-in through the public ``build_ep_compute``
skeleton): a **benchmark-only** variant that hands every destination rank the same
slot count, so a skewed real router can neither dominate step time nor grow the
busiest rank's expert buffers during a like-for-like comparison. It ignores the gate
scores, so loss / grad_norm / expert-load statistics from that mode are meaningless
and must not be reported as an efficiency result; correctness runs keep it off.
"""

import logging
from typing import Any, Callable, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute

logger = logging.getLogger(__name__)


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
        overlap_shared_expert: bool = False,
        capacity_factor: Optional[float] = None,
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
        overlap_shared_expert: Evaluate ``shared_experts`` while the routed token
            exchange is in flight instead of serializing behind it.  The shared
            branch reads only ``hidden_states``, so the math is unchanged and the
            exchange gets hidden.  The 2-SN reference numbers were measured with
            this on: without it the routed branch exposes the whole exchange
            (measured at GBS 256: +2.28 GB peak, +22% step time).
        capacity_factor: Bound each expert at ``factor x (T*K / E)`` slots per
            sender (``None`` = off).  Read back per layer by
            ``experts._resolve_capacity_factor``, so ``HP_EP_CAPACITY_FACTOR``
            still overrides it.

    Returns:
        A callable ``(module, hidden_states) -> Tensor`` running the block.
    """
    del mesh, tp_mesh, cp_mesh
    logger.info(
        "kimi_k26 EP archetype: fix_router=%s use_grouped_gemm=%s overlap_shared_expert=%s "
        "capacity_factor=%s (ep_size=%s)",
        fix_router, use_grouped_gemm, overlap_shared_expert, capacity_factor,
        getattr(ep_mesh, 'size', lambda *a, **k: None)(),  # pylint: disable=not-callable
    )

    def combine(
        module: Any,
        hidden_states: torch.Tensor,
        routed: torch.Tensor,
        overlapped: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Merge the routed branch with the shared-experts branch."""
        shared = module.shared_experts(hidden_states) if overlapped is None else overlapped
        return routed + shared    # nested boundary

    def overlap_shared(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        """Shared-experts branch, evaluated while the dispatch exchange is in flight."""
        return module.shared_experts(hidden_states)    # nested boundary

    return build_ep_compute(
        module,
        ep_mesh,
        router_fn="deepseekv3",
        fix_router=fix_router,
        archetype_key="deepseekv3_sigmoid_group_shared",
        expected_attrs=["gate", "experts", "shared_experts"],
        combine=combine,
        use_grouped_gemm=use_grouped_gemm,
        capacity_factor=capacity_factor,
        overlap_fn=overlap_shared if overlap_shared_expert else None,
    )


__all__ = ["kimi_k26_ep_compute_fn"]
