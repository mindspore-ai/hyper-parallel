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
"""Communication volume submodule"""
from __future__ import annotations
from typing import TYPE_CHECKING
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import (
    CPAlgo,
    _resolve_cp_algo,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    compute_kv_dim,
)

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
        # Level 0-1-2 :
        # Either GradAR
        # Or GradRS + ParamAG
        # Level 3
        # GradRS + FWD ParamAG + BWD ParamAG
        # When FSDP is active, all-gather is accounted for by fsdp_comm_layer;
        # dp_comm only counts the gradient reduce-scatter.

        # Non expert ZeRO LvL 2
        if ccfg.comm_d_non_exp == 2:
            dp_comm_non_exp += non_exp / (ccfg.cp * ccfg.t)
            dp_comm_non_exp += non_exp / ccfg.t
        # Non expert ZeRO LvL 3
        # Non expert ZeRO LvL 3 / FSDP
        if ccfg.comm_d_non_exp == 3:
            dp_comm_non_exp += non_exp / (ccfg.cp * ccfg.t)
            if not ccfg.fsdp:
                dp_comm_non_exp += non_exp / ccfg.t
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
        # Level 0-1-2 :
        # Either GradAR
        # Or GradRS + ParamAG
        # Level 3
        # GradRS + FWD ParamAG + BWD ParamAG
        # When FSDP is active, all-gather is accounted for by fsdp_comm_layer;
        # dp_comm only counts the gradient reduce-scatter.

        # Expert ZeRO LvL 2
        if ccfg.comm_d_exp == 2:
            dp_comm_exp += exp_param_size / (ccfg.cp * ccfg.t_exp * ccfg.ep)
            dp_comm_exp += exp_param_size / max(ccfg.ep, ccfg.t_exp)
        # Expert ZeRO LvL 3
        # Expert ZeRO LvL 3 / FSDP
        if ccfg.comm_d_exp == 3:
            dp_comm_exp += exp_param_size / (ccfg.cp * ccfg.t_exp * ccfg.ep)
            if not ccfg.fsdp:
                dp_comm_exp += exp_param_size / max(ccfg.ep, ccfg.t_exp)
                dp_comm_exp += exp_param_size / max(ccfg.ep, ccfg.t_exp)
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
        )  # [HYPOTHESIS]
        # hybird_cp is a known typo for hybrid_cp kept for backward compat
        if ccfg.cp_algo in ["colossalai_cp", "hybrid_cp", "hybird_cp"]:
            # FW Ring P2P + BW Ring P2P
            # KV transfers, can be recomputed
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
    def cp_comm_exp(ccfg: CostModelConfig, _: Context) -> float:
        """CP comm for expert parameters"""
        # hybird_cp is a known typo for hybrid_cp kept for backward compat
        if ccfg.cp_algo in ["colossalai_cp", "hybrid_cp", "hybird_cp", "ulysses_cp"]:
            # FW Ring P2P + BW Ring P2P
            # or FW + BW All2Alls
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
        When EP>1: uses (ep-1)/ep correction for cross-rank all-to-all.
        When EP=1 & n_exp>1: expert dispatch/combine still occurs (within
        the TP group for MoE routing). Volume = 2 * n_chosen_exp * b * s * h
        per layer, routed through TP bandwidth.
        Result is in bytes (like TP activation comm), unlike CP/DP which are
        in element counts (parameter comm).
        """
        del ctx
        if ccfg.ep <= 1 or ccfg.comm_ep == 0:
            return 0
        if ccfg.comm_ep == 0:
            return 0
        if ccfg.n_exp <= 1:
            return 0
        if ccfg.ep <= 1:
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

        When EP=1 & n_exp>1: falls back to balanced dispatch/combine volume.

        tokens_per_expert: global per-expert token count per microbatch
            (all EP ranks combined, before all-to-all dispatch; None = balanced).
            Under uniform distribution, each rank's share equals
            n_chosen_exp * s * b / (cp * t), matching t_local in the balanced formula.

        Result is in bytes (like TP activation comm), unlike CP/DP which are
        in element counts (parameter comm).
        """
        if ccfg.ep <= 1 or ccfg.comm_ep == 0:
            return 0
        if ccfg.comm_ep == 0:
            return 0
        if ccfg.n_exp <= 1:
            return 0
        if ccfg.ep <= 1:
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
        if ccfg.comm_ep == 0:
            return 0
        if ccfg.n_exp <= 1:
            return 0
        if ccfg.ep <= 1:
            return 0
        if ccfg.tokens_per_expert is not None:
            return EvalLayerComm.ep_comm_layer_imbalanced(ccfg, ctx, mb)
        return EvalLayerComm.ep_comm_layer_balanced(ccfg, ctx, mb)

    @staticmethod
    def cp_comm_buffer(ccfg: CostModelConfig, ctx: Context) -> float:
        """Estimate CP communication buffer memory per layer.

        Ring CP (colossalai_cp / hybrid_cp):
            Intra-node: all-gather among device_per_node ranks → buffer for
            (intra_ranks - 1) extra KV chunks.
            Cross-node: intra-node all-gather result stays resident, plus 1
            full-node KV chunk as ring receive buffer → peak is
            (2 * intra_ranks - 1) extra chunks.

        Ulysses CP:
            All2All rearranges heads across CP ranks.  Peak buffer is
            1 send chunk + 1 receive chunk = 2 chunks, where each chunk
            is the per-rank activation slice being exchanged.
        """
        del ctx
        if ccfg.cp <= 1:
            return 0.0

        s, b, cp = ccfg.s, ccfg.b, ccfg.cp
        fp16_bytes = 2
        kv_bytes = fp16_bytes * 2

        cp_algo = _resolve_cp_algo(ccfg)

        if cp_algo == CPAlgo.ULYSSES_CP:
            kv_dim = compute_kv_dim(ccfg)
            chunk = s * b * (kv_dim / cp) * kv_bytes
            intra_ranks = min(int(cp), int(ccfg.device_per_node))
            if cp <= ccfg.device_per_node:
                extra_chunks = intra_ranks - 1
            else:
                extra_chunks = 2 * intra_ranks - 1
            return extra_chunks * chunk

        kv_dim = compute_kv_dim(ccfg)
        chunk = (s / cp) * b * kv_dim * kv_bytes

        intra_ranks = min(int(cp), int(ccfg.device_per_node))

        if cp <= ccfg.device_per_node:
            extra_chunks = intra_ranks - 1
        else:
            extra_chunks = 2 * intra_ranks - 1

        return extra_chunks * chunk

    @staticmethod
    def fsdp_comm_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """FSDP/HSDP communication volume estimation."""
        non_exp, routed, shared = ctx.eval.num_p(ccfg, ctx)
        exp = routed + shared
        d_shard = ccfg.d_shard_or_d

        non_exp_comm = (
            ccfg.comm_fsdp
            * non_exp / (d_shard * ccfg.cp * ccfg.t)
            * ccfg.bytes_compute
            * 2
        )
        exp_comm = (
            ccfg.comm_fsdp
            * exp / (d_shard * ccfg.ep * ccfg.cp * ccfg.t_exp)
            * ccfg.bytes_compute
            * 2
            if ccfg.n_exp > 1
            else 0
        )

        if getattr(ccfg, "comm_hsdp", 0) > 0:
            d_replicate = ccfg.d // d_shard
            sharded_non_exp = non_exp / (d_shard * ccfg.cp * ccfg.t)
            sharded_exp = exp / (d_shard * ccfg.cp * ccfg.t_exp) if ccfg.n_exp > 1 else 0
            hsdp_comm = ccfg.comm_hsdp / d_replicate * (sharded_non_exp + sharded_exp) * ccfg.bytes_compute
            non_exp_comm += hsdp_comm

        return non_exp_comm + exp_comm

    @staticmethod
    def fsdp_buffer_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """FSDP/HSDP all-gather buffer memory (bytes)."""
        non_exp, routed, shared = ctx.eval.num_p(ccfg, ctx)
        non_exp_buf = (
            ccfg.comm_fsdp * ccfg.fsdp_all_gather_buffer
            * non_exp * ccfg.bytes_p / (ccfg.cp * ccfg.t)
        )
        exp_buf = (
            ccfg.comm_fsdp * ccfg.fsdp_all_gather_buffer
            * (routed + shared) * ccfg.bytes_p
            / (ccfg.ep * ccfg.cp * ccfg.t_exp)
            if ccfg.n_exp > 1
            else 0
        )
        return non_exp_buf + exp_buf

    @staticmethod
    def fsdp_buffer_comm(ccfg: CostModelConfig, ctx: Context) -> float:
        """FSDP/HSDP all-gather buffer size (bytes)."""
        non_exp, routed, shared = ctx.eval.num_p(ccfg, ctx)
        non_exp_buf = (
            ccfg.comm_fsdp * ccfg.fsdp_all_gather_buffer
            * non_exp * ccfg.bytes_compute / (ccfg.cp * ccfg.t)
        )
        exp_buf = (
            ccfg.comm_fsdp * ccfg.fsdp_all_gather_buffer
            * (routed + shared) * ccfg.bytes_compute
            / (ccfg.ep * ccfg.cp * ccfg.t_exp)
            if ccfg.n_exp > 1
            else 0
        )
        return non_exp_buf + exp_buf

    @staticmethod
    def fsdp_grad_buffer_comm(ccfg: CostModelConfig, ctx: Context) -> float:
        """FSDP/HSDP gradient reduce-scatter buffer size (bytes)."""
        non_exp, routed, shared = ctx.eval.num_p(ccfg, ctx)
        non_exp_buf = (
            ccfg.comm_fsdp * ccfg.fsdp_all_gather_buffer
            * non_exp * ccfg.bytes_grad / (ccfg.cp * ccfg.t)
        )
        exp_buf = (
            ccfg.comm_fsdp * ccfg.fsdp_all_gather_buffer
            * (routed + shared) * ccfg.bytes_grad
            / (ccfg.ep * ccfg.cp * ccfg.t_exp)
            if ccfg.n_exp > 1
            else 0
        )
        return non_exp_buf + exp_buf

    @staticmethod
    def hsdp_inter_buffer_comm(ccfg: CostModelConfig, ctx: Context) -> float:
        """HSDP inter-node reduce-scatter buffer size (bytes)."""
        if getattr(ccfg, "comm_hsdp", 0) <= 0:
            return 0.0
        non_exp, routed, shared = ctx.eval.num_p(ccfg, ctx)
        exp = routed + shared
        d_shard = ccfg.d_shard_or_d
        sharded_non_exp = non_exp / (d_shard * ccfg.cp * ccfg.t)
        sharded_exp = (
            exp / (d_shard * ccfg.cp * ccfg.t_exp)
            if ccfg.n_exp > 1
            else 0
        )
        return (
            ccfg.comm_hsdp
            * (sharded_non_exp + sharded_exp)
            * ccfg.bytes_compute
        )

    @staticmethod
    def dp_buffer_comm(ccfg: CostModelConfig, ctx: Context) -> float:
        del ccfg, ctx
        return 0.0

    @staticmethod
    def tp_buffer_comm(ccfg: CostModelConfig, ctx: Context, mb: int) -> float:
        """Peak HBM buffer for TP all-gather (bytes)."""
        if ccfg.t <= 1 or ccfg.comm_t == 0:
            return 0
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        rec_coeff = EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.gather)

        attn_buf = ccfg.s * ccfg.b * ccfg.h * ccfg.bytes_compute
        ffn_buf = ccfg.s * ccfg.b * 2 * ccfg.hff * ccfg.bytes_compute

        peak_buf = max(attn_buf, ffn_buf)
        return rec_coeff * ccfg.comm_t * mb * peak_buf / ccfg.cp
