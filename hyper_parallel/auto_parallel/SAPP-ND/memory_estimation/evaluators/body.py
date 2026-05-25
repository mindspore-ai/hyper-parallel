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
"""Body module"""
from __future__ import annotations
from typing import TYPE_CHECKING
from memory_estimation.logger import logger
from memory_estimation.evaluators.utils import EvalUtils

if TYPE_CHECKING:
    from toolkits.paradise.common.cost_model_preprocess import CostModelConfig
    from toolkits.memory_estimation._context import Context
    from typing import Tuple


class EvalBody:
    """Body layer formulas class"""

    @staticmethod
    def num_params_layer(
        ccfg: CostModelConfig, ctx: Context
    ) -> Tuple[float, float]:
        """Parameters count"""
        return (
            ctx.attn_num_p(ccfg, ctx) + ctx.norm_num_p(ccfg, ctx),
            ctx.ffn_num_p(ccfg, ctx),
        )

    @staticmethod
    def stat_p_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """model param"""
        non_exp_p, exp_p = ctx.eval.num_p(ccfg, ctx)
        # Expert
        # Partial DP for shared exp, Full DP for routed exp
        b_p_exp = (
            ccfg.n_exp
            / (ccfg.n_exp + ccfg.n_shared_exp)
            * ccfg.bytes_p
            / ccfg.shard_p_os_exp
            + ccfg.n_shared_exp
            / (ccfg.n_exp + ccfg.n_shared_exp)
            * ccfg.bytes_p
            / ccfg.shard_p_os_exp_partial
        )
        # Non expert
        b_p_non_exp = ccfg.bytes_p / ccfg.shard_p_os_non_exp_partial
        return exp_p * b_p_exp + non_exp_p * b_p_non_exp

    @staticmethod
    def stat_os_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """optim state"""
        if ctx.swap_os:
            return 0
        non_exp_p, exp_p = ctx.eval.num_p(ccfg, ctx)
        # Expert
        # Partial DP for shared exp, Full DP for routed exp
        b_os_exp = (
            ccfg.n_exp
            / (ccfg.n_exp + ccfg.n_shared_exp)
            * 2
            * ccfg.bytes_os
            / ccfg.shard_p_os_exp
            + ccfg.n_shared_exp
            / (ccfg.n_exp + ccfg.n_shared_exp)
            * 2
            * ccfg.bytes_os
            / ccfg.shard_p_os_exp_partial
        )
        # Non expert
        b_os_non_exp = 2 * ccfg.bytes_os / ccfg.shard_p_os_non_exp_partial
        return exp_p * b_os_exp + non_exp_p * b_os_non_exp

    @staticmethod
    def stat_grad_layer(ccfg: CostModelConfig, ctx: Context) -> float:
        """gradients"""
        non_exp_p, exp_p = ctx.eval.num_p(ccfg, ctx)
        # Expert
        # Partial DP for shared exp, Full DP for routed exp
        b_grad_exp = (
            ccfg.n_exp
            / (ccfg.n_exp + ccfg.n_shared_exp)
            * ccfg.bytes_grad
            / ccfg.shard_grad_exp
            + ccfg.n_shared_exp
            / (ccfg.n_exp + ccfg.n_shared_exp)
            * ccfg.bytes_grad
        )
        # Non expert
        b_grad_non_exp = ccfg.bytes_grad / ccfg.shard_grad_non_exp
        return exp_p * b_grad_exp + non_exp_p * b_grad_non_exp

    # No recompute and select recompute

    @staticmethod
    def layer_activ(ccfg: CostModelConfig, ctx: Context) -> float:
        """activations"""
        attn_size = sum(
            [
                ctx.attn_qkv_activ(ccfg, ctx),
                ctx.attn_score_activ(ccfg, ctx),
                ctx.attn_proj_activ(ccfg, ctx),
            ]
        )
        if ccfg.n_exp == 1:
            ffn_size = ctx.ffn_activ(ccfg, ctx)
        else:
            ffn_size = ctx.ffn_moe_activ(ccfg, ctx)
        norm_size = ctx.norm_activ(ccfg, ctx)
        return attn_size + ffn_size + norm_size

    # Full recompute

    @staticmethod
    def fullrec_layer_activ(ccfg: CostModelConfig, ctx: Context) -> float:
        """activations"""
        micro_factor = ctx.micro_factor
        forward_activation = (
            micro_factor * ccfg.bytes_compute * ccfg.s * ccfg.b * ccfg.h
        )
        forward_activation /= ccfg.shard_recompute_input
        return forward_activation

    @staticmethod
    def fullrec_layer_activ_gradclip(
        ccfg: CostModelConfig, ctx: Context
    ) -> float:
        """special case with gradient clipping"""
        non_exp_p, exp_p = ctx.eval.num_p(ccfg, ctx)
        grad_clip_mem = (
            exp_p
            * (
                ccfg.n_exp
                / (ccfg.n_exp + ccfg.n_shared_exp)
                * ccfg.bytes_os
                / ccfg.shard_p_os_exp
                + ccfg.n_shared_exp
                / (ccfg.n_exp + ccfg.n_shared_exp)
                * ccfg.bytes_os
                / ccfg.shard_p_os_exp_partial
            )
            + non_exp_p
        )
        grad_clip_mem *= ccfg.bytes_os / ccfg.shard_p_os_non_exp_partial
        grad_clip_mem *= int(ccfg.has_clip)
        forward_activation = EvalBody.fullrec_layer_activ(ccfg, ctx)
        dp_comm_size = ctx.eval.dyn.comm.dp(ccfg, ctx)
        if forward_activation + dp_comm_size > grad_clip_mem:
            return forward_activation
        logger.debug(
            "gradient clipping %s > %s",
            EvalUtils.mb(grad_clip_mem),
            EvalUtils.mb(forward_activation + dp_comm_size),
        )
        return grad_clip_mem

    @staticmethod
    def fullrec_layer_comm_gradclip(
        ccfg: CostModelConfig, ctx: Context
    ) -> float:
        """special case with gradient clipping"""
        if EvalBody.fullrec_layer_activ_gradclip(ccfg, ctx) > 0:
            return ctx.eval.dyn.comm.dp(ccfg, ctx)
        return 0
