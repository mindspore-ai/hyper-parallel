# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Expert compute FLOPs estimation"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context


class EvalExpertCompute:
    """Expert compute (FLOPs) estimation formulas.

    All methods return FLOPs (not bytes), so they must NOT go through
    __wrap_mem_counter — they are not memory quantities.
    """

    @staticmethod
    def router_compute_cost(ccfg: CostModelConfig, ctx: Context) -> float:  # pylint: disable=unused-argument
        """Router (gate) FLOPs — replicated, not scaled by EP.

        Router computes a score vector of length n_exp for each token,
        then selects top-K. The cost is dominated by the linear projection:
          FLOPs = 2 * s * b * h * n_exp  (per microbatch, per layer)

        Router is replicated across EP ranks because topK selection needs
        the full expert score vector.
        """
        return 2 * ccfg.s * ccfg.b * ccfg.h * ccfg.n_exp / ccfg.cp

    @staticmethod
    def expert_compute_cost_balanced(
        ccfg: CostModelConfig, ctx: Context  # pylint: disable=unused-argument
    ) -> float:
        """Routed expert compute FLOPs for balanced token distribution.

        Per-rank FLOPs for routed experts only.
        Each rank holds n_exp/ep experts and processes 1/ep of the tokens
        (balanced assumption):
          FLOPs = 2 * n_ffMM * s * b * h * hff_exp * n_chosen_exp
                  / (ep * cp * t_exp)

        t_exp = etp if etp > 1 else tp (alternative, not multiplicative).
        Factor 2 accounts for multiply-add (MAC = 2 FLOPs).
        n_ffMM is the number of feedforward linear layers per expert
        (SwiGLU: gate+up+down = 3; standard MLP: 2), set in arch_hooks.
        """
        ep = max(ccfg.ep, 1)
        t_exp = ccfg.etp if ccfg.etp > 1 else ccfg.t
        n_ff = max(getattr(ccfg, "n_ffMM", 1), 1)
        return (
            2 * n_ff * ccfg.s * ccfg.b * ccfg.h * ccfg.hff_exp
            * ccfg.n_chosen_exp / (ep * max(t_exp, 1) * ccfg.cp)
        )

    @staticmethod
    def expert_compute_cost_imbalanced(
        ccfg: CostModelConfig, ctx: Context
    ) -> float:
        """Routed expert compute FLOPs for imbalanced token distribution.

        Uses max(tokens_per_rank) to bound per-rank compute (bucket effect).
        Falls back to balanced when tokens_per_expert is None or
        n_exp not divisible by ep.

        t_exp = etp if etp > 1 else tp (alternative, not multiplicative).
        n_ffMM is the number of feedforward linear layers per expert
        (SwiGLU: 3, standard MLP: 2), set in arch_hooks.

        tokens_per_expert: global per-expert token count per microbatch
            (all EP ranks combined, before all-to-all dispatch; None = balanced).
        """
        ep = max(ccfg.ep, 1)
        tokens = ccfg.tokens_per_expert
        if not tokens:
            return EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx)
        if ccfg.n_exp % ep != 0:
            return EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx)
        if len(tokens) < ccfg.n_exp:
            return EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx)
        t_exp = ccfg.etp if ccfg.etp > 1 else ccfg.t
        n_ff = max(getattr(ccfg, "n_ffMM", 1), 1)
        experts_per_rank = ccfg.n_exp // ep
        rank_tokens = []
        for r in range(ep):
            rank_sum = sum(
                tokens[r * experts_per_rank + i] for i in range(experts_per_rank)
            )
            rank_tokens.append(rank_sum)
        max_etp = max(*rank_tokens, 1)
        return (
            2 * n_ff * max_etp * ccfg.h * ccfg.hff_exp
            / (max(t_exp, 1) * ccfg.cp)
        )

    @staticmethod
    def shared_expert_compute_cost(
        ccfg: CostModelConfig, ctx: Context  # pylint: disable=unused-argument
    ) -> float:
        """Shared expert compute FLOPs — replicated, not scaled by EP.

        Shared experts process ALL tokens (not dispatched via EP),
        so their compute is the same regardless of EP degree.
        Shared experts use hff (not hff_exp) for their hidden dimension.
          FLOPs = 2 * n_ffMM * s * b * h * hff * n_shared_exp / (t_exp * cp)

        Factor t_exp because shared expert is TP/ETP-sharded (not EP-sharded).
        t_exp = etp if etp > 1 else tp (alternative, not multiplicative).
        n_ffMM is the number of feedforward linear layers per expert
        (SwiGLU: 3, standard MLP: 2), set in arch_hooks.
        """
        t_exp = ccfg.etp if ccfg.etp > 1 else ccfg.t
        n_ff = max(getattr(ccfg, "n_ffMM", 1), 1)
        return (
            2 * n_ff * ccfg.s * ccfg.b * ccfg.h * ccfg.hff
            * ccfg.n_shared_exp / (max(t_exp, 1) * ccfg.cp)
        )

    @staticmethod
    def expert_layer_compute(ccfg: CostModelConfig, ctx: Context) -> float:
        """Dispatcher: routes to balanced or imbalanced compute based on tokens_per_expert."""
        if ccfg.n_exp <= 1:
            return 0
        router = EvalExpertCompute.router_compute_cost(ccfg, ctx)
        if ccfg.tokens_per_expert is not None:
            expert = EvalExpertCompute.expert_compute_cost_imbalanced(ccfg, ctx)
        else:
            expert = EvalExpertCompute.expert_compute_cost_balanced(ccfg, ctx)
        shared = EvalExpertCompute.shared_expert_compute_cost(ccfg, ctx)
        return router + expert + shared
