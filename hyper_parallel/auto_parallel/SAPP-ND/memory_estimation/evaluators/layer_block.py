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
"""Layer's blocks submodule"""
from __future__ import annotations
from typing import TYPE_CHECKING
from paradise.common.layer_type import LayerType
from memory_estimation.evaluators.utils import EvalUtils

if TYPE_CHECKING:
    from toolkits.paradise.common.cost_model_preprocess import CostModelConfig
    from toolkits.memory_estimation._context import Context

mb = EvalUtils.mb


class EvalAttn:
    """Attention formulas class"""

    @staticmethod
    def num_params_mla(ccfg: CostModelConfig, _) -> float:
        """Parameters count for Multi-Head Latent Attention"""
        # W_up_q = ccfg.dc_q * ccfg.dh * ccfg.a
        # W_up_k = ccfg.dc_kv * ccfg.dh * ccfg.n_kv
        # W_up_v = ccfg.dc_kv * ccfg.dh * ccfg.n_kv
        # W_down_q = ccfg.dc_q * ccfg.h
        # W_down_kv = ccfg.dc_kv * ccfg.h
        # W_q_rope = ccfg.a * ccfg.dhr * ccfg.dc_q
        # W_k_rope = ccfg.dhr * ccfg.h
        # Wo = ccfg.h * ccfg.a * ccfg.dh

        c_kv_fact = ccfg.dc_kv * (ccfg.n_kv * ccfg.dh + ccfg.h)
        c_q_fact = ccfg.dc_q * (ccfg.a * ccfg.dh + ccfg.h + ccfg.a * ccfg.dhr)
        rest_fact = (ccfg.h * ccfg.a * ccfg.dh) + (ccfg.h * ccfg.dhr)
        res = (
            0.5 * ccfg.n_attMM * c_kv_fact
            + 0.25 * ccfg.n_attMM * c_q_fact
            + 0.25 * ccfg.n_attMM * rest_fact
        )
        return res

    @staticmethod
    def num_params_attn(ccfg: CostModelConfig, ctx: Context) -> float:
        """Parameters count for Multi-Head/Grouped-Q./Multi-Q. Attention"""
        if ccfg.dc_kv == 0:
            # Q,O and K,V have distinct shapes
            return 0.5 * ccfg.n_attMM * (
                ccfg.h * ccfg.h + ccfg.h
            ) + 0.5 * ccfg.n_attMM * (ccfg.h * ccfg.n_kv * ccfg.dh + ccfg.h)
        return EvalAttn.num_params_mla(ccfg, ctx)

    @staticmethod
    def attn_qkv_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """QKV linear Activations"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        att_qkv_size = 0
        if ccfg.dc_kv == 0:
            n_op = ccfg.n_attMM + ccfg.n_attParamCast
            att_qkv_size = (
                ccfg.s
                * ccfg.b
                * ccfg.bytes_compute
                * (
                    0.25 * n_op * ccfg.h
                    + 0.5 * n_op * ccfg.dh * ccfg.n_kv
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.attBMM)
                    * ccfg.n_attBMM
                    * ccfg.dh
                )
            )
        else:
            q_size = (
                0.25
                * (ccfg.n_attMM + ccfg.n_attParamCast)
                * (ccfg.dc_q + 2 * ccfg.a * (ccfg.dh + ccfg.dhr))
            )
            k_size = (
                0.25
                * (ccfg.n_attMM + ccfg.n_attParamCast)
                * (ccfg.dhr + ccfg.n_kv * (2 * ccfg.dh + ccfg.dhr))
            )
            v_size = (
                0.25
                * (ccfg.n_attMM + ccfg.n_attParamCast)
                * (ccfg.n_kv * ccfg.dh + ccfg.dc_kv)
            )
            att_qkv_size = (
                ccfg.s
                * ccfg.b
                * ccfg.bytes_compute
                * (
                    q_size
                    + k_size
                    + v_size
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.attBMM)
                    * ccfg.n_attBMM
                    * ccfg.dh
                )
            )
        micro_factor = ctx.micro_factor
        return micro_factor * att_qkv_size / (ccfg.t * ccfg.cp)

    @staticmethod
    def attn_score_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Score/Softmax Activations"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        att_score = (
            ccfg.s_fa
            * ccfg.b
            * ccfg.a
            * ccfg.s
            * (
                ccfg.n_softmax
                * (
                    ccfg.rec_op.softmax * ccfg.bytes_softmax
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.dropout)
                    * ccfg.bytes_dropout
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.headCast)
                    * ccfg.bytes_compute
                )
            )
        )
        micro_factor = ctx.micro_factor
        return micro_factor * att_score / (ccfg.t * ccfg.cp)

    @staticmethod
    def attn_proj_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Output projection Activations"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        att_proj = (
            ccfg.s
            * ccfg.b
            * ccfg.h
            * ccfg.bytes_compute
            * (
                0.25 * (ccfg.n_attMM + ccfg.n_attParamCast)
                + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.dropout)
                * ccfg.n_dropout
                * ccfg.bytes_dropout
            )
        )
        micro_factor = ctx.micro_factor
        return micro_factor * att_proj / max(ccfg.sp, ccfg.cp)


class EvalFFn:
    """Feed-forward formulas class"""

    @staticmethod
    def num_params_ffn(ccfg: CostModelConfig, _) -> float:
        """Parameters count"""
        experts_param_size = (
            (ccfg.n_exp + ccfg.n_shared_exp)
            * max(ccfg.n_ffMM, ccfg.n_ffBMM)
            * (ccfg.hff * ccfg.h + ccfg.hff)
        )
        return experts_param_size

    @staticmethod
    def ffn_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """ "Activations count"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        tok_size = ccfg.s * ccfg.b
        n_mm = max(ccfg.n_ffMM, ccfg.n_ffBMM)
        if n_mm % 2 == 0:
            matmul = 0.5 * ccfg.h + 0.5 * ccfg.hff
        else:
            matmul = 1 / 3 * ccfg.h + 2 / 3 * ccfg.hff
        matmul *= ccfg.bytes_compute * n_mm
        activ_fun = ccfg.bytes_compute * ccfg.hff
        activ_fun *= EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.ffAct)
        pcast = ccfg.bytes_compute * ccfg.hff * ccfg.n_ffParamCast
        activ_size = matmul + pcast + activ_fun
        micro_factor = ctx.micro_factor
        return micro_factor * tok_size * activ_size / (ccfg.t * ccfg.cp)

    @staticmethod
    def ffn_router_and_concat_activations(
        ccfg: CostModelConfig, ctx: Context
    ) -> float:
        """MoE router and output activations"""
        # Router activations (logits, probs, mask)
        r = ccfg.s * ccfg.b * ccfg.bytes_compute
        r *= 2 * ccfg.n_exp + ccfg.n_chosen_exp
        # Concat all exp output
        c = ccfg.s * ccfg.b * ccfg.bytes_compute * ccfg.h
        micro_factor = ctx.micro_factor
        return micro_factor * (r + c) / (ccfg.t * ccfg.cp)

    @staticmethod
    def shared_exp_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Shared expert activations"""
        return ccfg.n_shared_exp * EvalFFn.ffn_activations(ccfg, ctx)

    @staticmethod
    def routed_exp_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """MoE topK activations"""
        tok_size = ccfg.s * ccfg.b
        activ_size = EvalFFn.ffn_activations(ccfg, ctx) / tok_size
        avg_num_toks = tok_size * ccfg.n_chosen_exp / ccfg.n_exp
        if not ccfg.gmm:  # Capacity mode
            expert_capacity = avg_num_toks * ccfg.cap_fact * ccfg.n_exp
            routed_activ = activ_size * expert_capacity
        else:  # Dropless mode
            load = avg_num_toks * ccfg.n_exp * ctx.dropless_tok_factor
            routed_activ = load * activ_size
        return routed_activ

    @staticmethod
    def ffn_moe_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Sum of Activations"""
        return (
            EvalFFn.routed_exp_activations(ccfg, ctx)
            + EvalFFn.shared_exp_activations(ccfg, ctx)
            + EvalFFn.ffn_router_and_concat_activations(ccfg, ctx)
        )


class EvalNorm:
    """Normalization formulas class"""

    @staticmethod
    def num_params_norm(ccfg: CostModelConfig, _) -> float:
        """Parameters count"""
        return ccfg.n_normOp * 2 * ccfg.h

    @staticmethod
    def norm_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Activations"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        norm = (
            ccfg.s
            * ccfg.b
            * ccfg.bytes_norm
            * ccfg.h
            * ccfg.n_normOp
            * EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.normOp)
        )
        micro_factor = ctx.micro_factor
        return micro_factor * norm / (ccfg.t * ccfg.cp)
