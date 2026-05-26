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
"""Tail submodule"""
from __future__ import annotations
from typing import TYPE_CHECKING
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context


class EvalMTP:
    """MTP"""

    @staticmethod
    def num_params_mtp(ccfg: CostModelConfig, _) -> float:
        """Param count (MTP)"""
        # Linear + Norm
        return 2 * ccfg.h * ccfg.h + 4 * ccfg.h

    @staticmethod
    def stat_mtp_p(ccfg: CostModelConfig, ctx: Context) -> float:
        """static mem for model param (MTP)"""
        if not ccfg.n_mtp:
            return 0
        extra_param_size = EvalMTP.num_params_mtp(ccfg, ctx)
        b_p = ccfg.bytes_p
        if ccfg.is_shard_mtp_param:
            b_p /= ccfg.shard_p_os_non_exp_partial
        extra = ccfg.n_mtp * extra_param_size * b_p
        # Shared Head
        ctx.current_node = LayerType.EMBEDDING_LAYER
        head = ccfg.n_mtp * ctx.eval.stat.p(ccfg, ctx)
        # Shared Tail
        ctx.current_node = LayerType.OUTPUT_LAYER
        tail = ccfg.n_mtp * EvalTailSingle.stat_output_single_p(ccfg, ctx)
        return head + extra + tail

    @staticmethod
    def stat_mtp_os(ccfg: CostModelConfig, ctx: Context) -> float:
        """static mem for optimizer states (MTP)"""
        if not ccfg.n_mtp or ctx.swap_os:
            return 0
        extra_param_size = EvalMTP.num_params_mtp(ccfg, ctx)
        b_os = 2 * ccfg.bytes_os
        if ccfg.is_shard_mtp_param:
            b_os /= ccfg.shard_p_os_non_exp_partial
        extra = ccfg.n_mtp * extra_param_size * b_os
        # Shared Head
        ctx.current_node = LayerType.EMBEDDING_LAYER
        head = ccfg.n_mtp * ctx.eval.stat.os(ccfg, ctx)
        # Shared Tail
        ctx.current_node = LayerType.OUTPUT_LAYER
        tail = ccfg.n_mtp * EvalTailSingle.stat_output_single_os(ccfg, ctx)
        return head + extra + tail

    @staticmethod
    def stat_mtp_grad(ccfg: CostModelConfig, ctx: Context) -> float:
        """static mem for gradients (MTP)"""
        if not ccfg.n_mtp:
            return 0
        extra_param_size = EvalMTP.num_params_mtp(ccfg, ctx)
        b_grad = ccfg.bytes_grad
        if ccfg.is_shard_mtp_param:
            b_grad /= ccfg.shard_grad_non_exp
        extra = ccfg.n_mtp * extra_param_size * b_grad
        # Shared Head
        ctx.current_node = LayerType.EMBEDDING_LAYER
        head = ccfg.n_mtp * ctx.eval.stat.grad(ccfg, ctx)
        # Shared Tail
        ctx.current_node = LayerType.OUTPUT_LAYER
        tail = ccfg.n_mtp * EvalTailSingle.stat_output_single_grad(ccfg, ctx)
        return head + extra + tail

    @staticmethod
    def activ_mtp(ccfg: CostModelConfig, ctx: Context) -> float:
        """activation mem (MTP)"""
        if not ccfg.n_mtp:
            return 0
        micro_factor = ctx.micro_factor
        res = micro_factor * ccfg.n_mtp * ccfg.bytes_compute
        res *= ccfg.s * ccfg.b * 3 * ccfg.h
        # Shared Head
        ctx.current_node = LayerType.EMBEDDING_LAYER
        res += ccfg.n_mtp * ctx.eval.dyn.activation(ccfg, ctx)
        # Shared Tail
        ctx.current_node = LayerType.OUTPUT_LAYER
        res += ccfg.n_mtp * EvalTailSingle.activ_out_single(ccfg, ctx)
        return res

    @staticmethod
    def comm_mtp(ccfg: CostModelConfig, ctx: Context) -> float:
        """communication mem (MTP)"""
        if not ccfg.n_mtp:
            return 0
        mtp_dp_comm_size = 0
        param_size = EvalMTP.num_params_mtp(ccfg, ctx)
        mtp_dp_comm_size += (
            ccfg.comm_d_non_exp * ccfg.n_mtp * param_size / (ccfg.t * ccfg.cp)
        )
        # Shared Head
        ctx.current_node = LayerType.EMBEDDING_LAYER
        param_size = ctx.eval.num_p(ccfg, ctx)
        mtp_dp_comm_size += (
            ccfg.comm_d_non_exp * ccfg.n_mtp * param_size / (ccfg.t * ccfg.cp)
        )
        mtp_dp_comm_size += ccfg.n_mtp * ctx.eval.dyn.comm.dp(ccfg, ctx)
        # Shared Tail
        ctx.current_node = LayerType.OUTPUT_LAYER
        param_size = ctx.eval.num_p(ccfg, ctx)
        mtp_dp_comm_size += (
            ccfg.comm_d_non_exp * ccfg.n_mtp * param_size / (ccfg.t * ccfg.cp)
        )
        mtp_dp_comm_size += ccfg.n_mtp * EvalTailSingle.comm_out_single(
            ccfg, ctx
        )
        return mtp_dp_comm_size


class EvalTailSingle:
    """Single tail layer formulas class"""

    @staticmethod
    def stat_output_single_p(ccfg: CostModelConfig, ctx: Context) -> float:
        """static mem for model param (lmhead)"""
        param_size = ctx.eval.num_p(ccfg, ctx)
        b_p = ccfg.bytes_p
        b_p /= ccfg.shard_p_os_non_exp_partial
        return param_size * b_p

    @staticmethod
    def stat_output_single_os(ccfg: CostModelConfig, ctx: Context) -> float:
        """static mem for optim state (lmhead)"""
        if ctx.swap_os:
            return 0
        param_size = ctx.eval.num_p(ccfg, ctx)
        b_os = 2 * ccfg.bytes_os
        b_os /= ccfg.shard_p_os_non_exp_partial
        return param_size * b_os

    @staticmethod
    def stat_output_single_grad(ccfg: CostModelConfig, ctx: Context) -> float:
        """static mem for gradient (lmhead)"""
        param_size = ctx.eval.num_p(ccfg, ctx)
        b_grad = ccfg.bytes_grad
        b_grad /= ccfg.shard_grad_non_exp
        return param_size * b_grad

    @staticmethod
    def activ_out_single(ccfg: CostModelConfig, ctx: Context) -> float:
        """activation mem (lmhead)"""
        micro_factor = ctx.micro_factor
        last_norm = ccfg.s * ccfg.b * ccfg.bytes_norm * ccfg.h
        lm_head = ccfg.s * ccfg.b * ccfg.bytes_compute * ccfg.v
        activ_size = last_norm + lm_head
        activ_size /= ccfg.shard_output_activ
        return micro_factor * activ_size

    @staticmethod
    def comm_out_single(ccfg: CostModelConfig, ctx: Context) -> float:
        """communicaiton mem (lmhead)"""
        return (
            ccfg.comm_d_non_exp
            * ctx.eval.num_p(ccfg, ctx)
            / (ccfg.t * ccfg.cp)
        )


class EvalTail:
    """Single tail layer formulas class"""

    @staticmethod
    def num_params_output(ccfg: CostModelConfig, _) -> float:
        """Parameters count (lmhead)"""
        return ccfg.h * ccfg.v + ccfg.v

    @staticmethod
    def stat_output_p(ccfg: CostModelConfig, ctx: Context) -> float:
        """total model param"""
        return sum(
            [
                EvalTailSingle.stat_output_single_p(ccfg, ctx),
                EvalMTP.stat_mtp_p(ccfg, ctx),
            ]
        )

    @staticmethod
    def stat_output_os(ccfg: CostModelConfig, ctx: Context) -> float:
        """total optim state"""
        return sum(
            [
                EvalTailSingle.stat_output_single_os(ccfg, ctx),
                EvalMTP.stat_mtp_os(ccfg, ctx),
            ]
        )

    @staticmethod
    def stat_output_grad(ccfg: CostModelConfig, ctx: Context) -> float:
        """total gradients"""
        return sum(
            [
                EvalTailSingle.stat_output_single_grad(ccfg, ctx),
                EvalMTP.stat_mtp_grad(ccfg, ctx),
            ]
        )

    @staticmethod
    def activ_output(ccfg: CostModelConfig, ctx: Context) -> float:
        """total activations"""
        return sum(
            [
                EvalTailSingle.activ_out_single(ccfg, ctx),
                EvalMTP.activ_mtp(ccfg, ctx),
            ]
        )

    @staticmethod
    def comm_output(ccfg: CostModelConfig, ctx: Context) -> float:
        """total communications"""
        return sum(
            [
                EvalTailSingle.comm_out_single(ccfg, ctx),
                EvalMTP.comm_mtp(ccfg, ctx),
            ]
        )
