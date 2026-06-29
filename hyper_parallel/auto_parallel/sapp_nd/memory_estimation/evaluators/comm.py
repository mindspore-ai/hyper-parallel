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
"""Communication volume submodule"""
from __future__ import annotations
from typing import TYPE_CHECKING
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context


class EvalLayerComm:
    """Communication volume formulas class"""

    @staticmethod
    def dp_comm_non_exp(ccfg: CostModelConfig, ctx: Context) -> float:
        """DP/OP comm for non-expert parameters"""
        non_exp, _, _ = ctx.eval.num_p(ccfg, ctx)
        dp_comm_non_exp = 0
        # Non expert ZeRO LvL 2
        if ccfg.comm_d_non_exp == 2:
            dp_comm_non_exp += non_exp / (ccfg.cp * ccfg.t)
            dp_comm_non_exp += non_exp / ccfg.t
        # Non expert ZeRO LvL 3
        if ccfg.comm_d_non_exp == 3:
            dp_comm_non_exp += non_exp / ccfg.t
        return dp_comm_non_exp

    @staticmethod
    def dp_comm_exp(ccfg: CostModelConfig, ctx: Context) -> float:
        """DP/OP comm for expert parameters"""
        _, routed, shared = ctx.eval.num_p(ccfg, ctx)
        exp_param_size = routed + shared
        if exp_param_size == 0:
            return 0
        dp_comm_exp = 0
        # Expert ZeRO LvL 2
        if ccfg.comm_d_exp == 2:
            dp_comm_exp += exp_param_size / (ccfg.cp * ccfg.t_exp * ccfg.ep)
            dp_comm_exp += exp_param_size / max(ccfg.ep, ccfg.t_exp)
        # Expert ZeRO LvL 3
        if ccfg.comm_d_exp == 3:
            dp_comm_exp += exp_param_size / (ccfg.cp * ccfg.t_exp * ccfg.ep)
        return dp_comm_exp

    @staticmethod
    def dp_comm_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """DP/OP comm sum"""
        non_exp = EvalLayerComm.dp_comm_non_exp(ccfg, ctx)
        exp = EvalLayerComm.dp_comm_exp(ccfg, ctx)
        return non_exp + exp

    @staticmethod
    def tp_comm_non_exp(ccfg: CostModelConfig, ctx: Context, mb: int) -> float:
        """TP comm for non-expert parameters"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        tp_comm_non_exp = 0.25 * ccfg.n_gather
        tp_comm_non_exp *= ccfg.s * ccfg.b * ccfg.h * mb
        if ccfg.n_exp > 1:
            tp_comm_non_exp = (
                0.25
                * ccfg.n_gather
                * ccfg.h
                * ccfg.h
                * ccfg.bytes_compute
                * ccfg.n_attMM
            )
        res = (
            EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.gather)
            * ccfg.comm_t
            * tp_comm_non_exp
            / ccfg.cp
        )
        return res

    @staticmethod
    def tp_comm_exp(ccfg: CostModelConfig, ctx: Context, mb: int) -> float:
        """TP comm for expert parameters"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        tp_comm_exp = 0.25 * ccfg.n_gather
        tp_comm_exp *= ccfg.s * ccfg.b * ccfg.hff * mb
        if ccfg.n_exp > 1:
            # Routed experts use hff_exp, shared experts use hff
            routed_comm = ccfg.n_exp / ccfg.ep * ccfg.hff_exp
            shared_comm = ccfg.n_shared_exp * ccfg.hff
            tp_comm_exp = (
                0.25
                * ccfg.n_gather
                * ccfg.h
                * ccfg.bytes_compute
                * ccfg.n_ffMM
                * (routed_comm + shared_comm)
            )
        res = (
            EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.gather)
            * ccfg.comm_t
            * tp_comm_exp
            / ccfg.cp
        )
        return res

    @staticmethod
    def tp_comm_layer(ccfg: CostModelConfig, ctx: Context, mb: int) -> float:
        """TP comm sum"""
        non_exp = EvalLayerComm.tp_comm_non_exp(ccfg, ctx, mb)
        exp = EvalLayerComm.tp_comm_exp(ccfg, ctx, mb)
        return non_exp + exp

    @staticmethod
    def cp_comm_non_exp(ccfg: CostModelConfig, ctx: Context) -> float:
        """CP comm for non-expert parameters"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        rec_factor = EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.gather) * int(
            ccfg.p == 1
        )
        if ccfg.cp_algo in ["colossalai_cp", "hybird_cp"]:
            return (
                ccfg.comm_cp
                * 2
                * ccfg.s
                * ccfg.b
                * ((2 * 0.5 * rec_factor + 0.5) * ccfg.n_attMM * ccfg.h)
                / (ccfg.t)
            )
        if ccfg.cp_algo == "ulysses_cp":
            return (
                ccfg.comm_cp
                * 2
                * ccfg.s
                * ccfg.b
                * ((0.5 * rec_factor + 0.5) * ccfg.n_attMM * ccfg.h)
                / (ccfg.t)
            )
        return 0

    @staticmethod
    def cp_comm_exp(ccfg: CostModelConfig, _) -> float:
        """CP comm for expert parameters"""
        if ccfg.cp_algo in ["colossalai_cp", "hybird_cp", "ulysses_cp"]:
            res = ccfg.comm_cp * 2 * ccfg.s * ccfg.b * ccfg.n_ffMM * ccfg.hff
            return res / ccfg.t
        return 0

    @staticmethod
    def cp_comm_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """CP comm sum"""
        non_exp = EvalLayerComm.cp_comm_non_exp(ccfg, ctx)
        exp = EvalLayerComm.cp_comm_exp(ccfg, ctx)
        return non_exp + exp

    @staticmethod
    def ep_comm_layer_balanced(
        ccfg: CostModelConfig, ctx: Context, mb: int  # pylint: disable=unused-argument
    ) -> float:
        """EP comm for balanced token distribution (byte volume).

        Uses (ep-1)/ep correction: only (ep-1)/ep fraction of local tokens
        actually cross rank boundaries in an all-to-all dispatch/combine pair.
        Result is in bytes (like TP activation comm), unlike CP/DP which are
        in element counts (parameter comm).
        """
        if ccfg.ep <= 1 or ccfg.comm_ep == 0:
            return 0
        t_local = mb * ccfg.n_chosen_exp * ccfg.s * ccfg.b / ccfg.cp
        t_cross = t_local * (ccfg.ep - 1) / ccfg.ep
        return t_cross * ccfg.h * ccfg.bytes_compute * 2 * ccfg.comm_ep

    @staticmethod
    def ep_comm_layer_imbalanced(
        ccfg: CostModelConfig, ctx: Context, mb: int
    ) -> float:
        """EP comm for imbalanced (skewed) token distribution (byte volume).

        Uses max(rank_tokens) to bound communication volume.
        Normalized with (ep-1)/ep cross-rank factor and mb scaling,
        so it reduces to balanced when token distribution is uniform.
        Falls back to balanced when tokens_per_expert is empty
        or n_exp not divisible by ep.

        tokens_per_expert: global per-expert token count per microbatch
            (all EP ranks combined, before all-to-all dispatch; None = balanced).
            Under uniform distribution, each rank's share equals
            n_chosen_exp * s * b / (cp * t), matching t_local in the balanced formula.

        Result is in bytes (like TP activation comm), unlike CP/DP which are
        in element counts (parameter comm).
        """
        if ccfg.ep <= 1 or ccfg.comm_ep == 0:
            return 0
        tokens = ccfg.tokens_per_expert
        if not tokens:
            return EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        if ccfg.n_exp % ccfg.ep != 0:
            logger.warning(
                "n_exp=%d not divisible by ep=%d, falling back to balanced",
                ccfg.n_exp,
                ccfg.ep,
            )
            return EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
        experts_per_rank = ccfg.n_exp // ccfg.ep
        rank_tokens = []
        for r in range(ccfg.ep):
            rank_sum = sum(
                tokens[r * experts_per_rank + i] for i in range(experts_per_rank)
            )
            rank_tokens.append(rank_sum)
        max_inbound = max(rank_tokens)
        # max_inbound: per-rank inbound tokens for one microbatch
        # multiply by mb for the full pipeline stage, by (ep-1)/ep for cross-rank fraction
        t_cross = max_inbound * mb * (ccfg.ep - 1) / ccfg.ep
        return t_cross * ccfg.h * ccfg.bytes_compute * 2 * ccfg.comm_ep

    @staticmethod
    def ep_comm_layer(ccfg: CostModelConfig, ctx: Context, mb: int) -> float:
        """EP comm dispatcher: balanced or imbalanced based on tokens_per_expert."""
        if ccfg.ep <= 1 or ccfg.comm_ep == 0:
            return 0
        if ccfg.tokens_per_expert is not None:
            return EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, mb)
        return EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)
