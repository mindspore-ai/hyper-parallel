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
from paradise.common.layer_type import LayerType
from memory_estimation.evaluators.utils import EvalUtils

if TYPE_CHECKING:
    from toolkits.paradise.common.cost_model_preprocess import CostModelConfig
    from toolkits.memory_estimation._context import Context


class EvalLayerComm:
    """Communication volume formulas class"""

    @staticmethod
    def dp_comm_non_exp(ccfg: CostModelConfig, ctx: Context) -> float:
        """DP/OP comm for non-expert parameters"""
        non_exp, _ = ctx.eval.num_p(ccfg, ctx)
        dp_comm_non_exp = 0
        # Level 0-1-2 :
        # Either GradAR
        # Or GradRS + ParamAG
        # Level 3
        # GradRS + FWD ParamAG + BWD ParamAG

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
        _, exp_param_size = ctx.eval.num_p(ccfg, ctx)
        dp_comm_exp = 0
        # Level 0-1-2 :
        # Either GradAR
        # Or GradRS + ParamAG
        # Level 3
        # GradRS + FWD ParamAG + BWD ParamAG

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
        # Log purpose
        return non_exp + exp

    @staticmethod
    def tp_comm_non_exp(ccfg: CostModelConfig, ctx: Context, mb: int) -> float:
        """TP comm for non-expert parameters"""
        # 0.5 gather for Attn/FFn
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        tp_comm_non_exp = 0.25 * ccfg.n_gather
        tp_comm_non_exp *= ccfg.s * ccfg.b * ccfg.h * mb
        if ccfg.n_exp > 1:
            # Analysis from IRs and peak CSVs:
            # Parameters quantity estimation approach
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
        # 0.5 for FWD gather only
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        tp_comm_exp = 0.25 * ccfg.n_gather
        tp_comm_exp *= ccfg.s * ccfg.b * ccfg.hff * mb
        if ccfg.n_exp > 1:
            # Analysis from IRs and peak CSVs:
            # Parameters quantity estimation approach
            tp_comm_exp = (
                0.25
                * ccfg.n_gather
                * ccfg.h
                * ccfg.hff
                * ccfg.bytes_compute
                * ccfg.n_ffMM
                * (ccfg.n_exp / ccfg.ep + ccfg.n_shared_exp)
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
        # Log purpose
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
        if ccfg.cp_algo in ["colossalai_cp", "hybird_cp"]:
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
            # FW + BW All2Alls
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
    def ep_comm_layer(ccfg: CostModelConfig, _, mb: int) -> float:
        """EP comm"""
        # [HYPOTHESIS]
        a2a = (
            ccfg.comm_ep
            * (mb * (ccfg.n_chosen_exp) * ccfg.s * ccfg.b * ccfg.h)
            / (ccfg.cp * ccfg.t)
        )
        # print("a2a",a2a,ccfg.comm_ep)
        res = a2a * 2
        return res
