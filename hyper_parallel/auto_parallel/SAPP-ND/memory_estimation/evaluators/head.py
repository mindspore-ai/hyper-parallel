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
"""Head submodule"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolkits.paradise.common.cost_model_preprocess import CostModelConfig
    from toolkits.memory_estimation._context import Context


class EvalHead:
    """Head layer formulas class"""

    @staticmethod
    def num_params_embed(ccfg: CostModelConfig, _) -> float:
        """Parameter count"""
        return ccfg.h * ccfg.v

    @staticmethod
    def stat_embed_p(ccfg: CostModelConfig, ctx: Context) -> float:
        """model param"""
        if ccfg.tie_emb_out:
            return 0
        param_size = ctx.eval.num_p(ccfg, ctx)
        param_size /= ccfg.shard_embed
        b_p = ccfg.bytes_p
        b_p /= ccfg.cp
        return param_size * b_p

    @staticmethod
    def stat_embed_os(ccfg: CostModelConfig, ctx: Context) -> float:
        """optim state"""
        if ctx.swap_os or ccfg.tie_emb_out:
            return 0
        param_size = ctx.eval.num_p(ccfg, ctx)
        param_size /= ccfg.shard_embed
        b_os = 2 * ccfg.bytes_os
        b_os /= ccfg.cp
        return param_size * b_os

    @staticmethod
    def stat_embed_grad(ccfg: CostModelConfig, ctx: Context) -> float:
        """gradient"""
        if ccfg.tie_emb_out:
            return 0
        param_size = ctx.eval.num_p(ccfg, ctx)
        param_size /= ccfg.shard_embed
        b_grad = ccfg.bytes_grad
        b_grad /= ccfg.cp
        return param_size * b_grad

    @staticmethod
    def dp_comm_embed(ccfg: CostModelConfig, ctx: Context) -> float:
        """DP Communication size"""
        return (
            ccfg.comm_d_non_exp
            * ctx.eval.num_p(ccfg, ctx)
            / (ccfg.shard_embed * ccfg.cp)
        )

    @staticmethod
    def tp_comm_embed(ccfg: CostModelConfig, _) -> float:
        """TP Communication size"""
        return (
            ccfg.rec_op.gather
            * ccfg.comm_t
            * ccfg.s
            * ccfg.h
            * ccfg.b
            * (ccfg.t - 1)
            / (ccfg.t * ccfg.cp)
        )

    @staticmethod
    def activ_embed(ccfg: CostModelConfig, ctx: Context) -> float:
        """activations"""
        micro_factor = ctx.micro_factor
        activ_size = micro_factor * ccfg.bytes_compute / (ccfg.t * ccfg.cp)
        activ_size *= ccfg.s * ccfg.b * ccfg.h
        return activ_size
