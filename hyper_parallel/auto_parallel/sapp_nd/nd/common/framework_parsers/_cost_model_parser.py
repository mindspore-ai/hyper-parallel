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
"""cost model parser module"""
from __future__ import annotations
from typing import TYPE_CHECKING

import math
from abc import ABC
from abc import abstractmethod

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import _CostModVar


class _CostModelParser(ABC):
    """abstract parser class"""

    def __init__(self, ccfg: _CostModVar):
        self.ccfg = ccfg
        self.config = ccfg.config

    @abstractmethod
    def parse(self):
        """Parse the cost model configuration and populate the cost model variables.

        Subclasses must implement this method to read framework-specific
        configuration values into the shared _CostModVar instance.
        """

    def config_optimizer_shard(self, ccfg):
        """OP related variables"""
        # Non expert params
        ccfg.shard_p_os_non_exp_partial = (
            ccfg.os_max_shard if ccfg.has_op else ccfg.t
        ) * ccfg.cp
        ccfg.shard_p_os_non_exp = (
            (ccfg.d if ccfg.has_op else 1) * ccfg.cp * ccfg.t
        )
        ccfg.shard_grad_non_exp = (
            ccfg.shard_p_os_non_exp if ccfg.has_grad_shard else ccfg.t
        )

        # Expert params
        ccfg.shard_p_os_exp_partial = math.gcd(
            ccfg.n_exp,
            (ccfg.os_max_shard if ccfg.has_op else 1) * ccfg.t_exp,
        )
        ccfg.shard_p_os_exp = (
            (ccfg.d_exp if ccfg.has_op else 1) * ccfg.cp * ccfg.t_exp
        )
        ccfg.shard_grad_exp = (
            ccfg.shard_p_os_exp
            if ccfg.has_grad_shard
            else ccfg.t_exp
        )
        ccfg.shard_grad_exp_partial = (
            ccfg.shard_p_os_exp_partial
            if ccfg.has_grad_shard
            else ccfg.t_exp
        )

    # def config_op_level(self, ccfg, strategy):
    #     def full_partial():
    #         return Config({"full":0, "partial":0})
    #     def exp_or_not():
    #         return Config({
    #             "non_exp":full_partial(),
    #             "exp":full_partial()
    #         })
    #     ccfg.op = Config({
    #         "p":exp_or_not(),
    #         "os":exp_or_not(),
    #         "grad"exp_or_not()
    #     })
    #     shard_strat = {
    #         "grad":0, #zero 1
    #         "os+grad":0, #zero 2
    #         "p+os+grad":0, # zero 3
    #         "p+os":0 # zero2 mindspore
    #     }
    #     shard_strat[strategy]

    def init_hff(self):
        """MindFormers format for FFn hidden size"""
        # Assuming following 3 variables are already parsed
        hidden_size = self.ccfg.h
        ffn_dim_multiplier = self.ccfg.fdm
        multiple_of = self.ccfg.multiple_of
        hff = 4 * hidden_size
        if ffn_dim_multiplier:
            hff = int((ffn_dim_multiplier + 0.01) * hff)
        hff = int(2 * hff / 3)
        hff = multiple_of * ((hff + multiple_of - 1) // multiple_of)
        return hff

    def config_comm_flag(self, ccfg):
        """comm flag variables"""
        ccfg.comm_d_non_exp = (
            0
            if ((ccfg.d == 1) or not ccfg.has_op)
            else (2 if not ccfg.has_grad_shard else 3)
        )  # data parallel comm factor
        ccfg.comm_d_exp = (
            0
            if ((ccfg.d_exp == 1) or not ccfg.has_op)
            else (2 if not ccfg.has_grad_shard else 3)
        )  # data parallel comm factor
        ccfg.comm_t = float(ccfg.t > 1)  # tensor parallel comm factor
        ccfg.comm_ep = float(
            ccfg.ep > 1 or ccfg.n_exp > 1
        )  # expert parallel comm factor
        ccfg.comm_cp = float(ccfg.cp > 1)  # context parallel comm factor

    def config_dp_tp_exp(self, ccfg):
        """MoE strategy variables"""
        if ccfg.etp:
            ccfg.t_exp = ccfg.etp
            # d * t = inner dp * outer dp * etp
            # inner dp = EP, outer dp = the rest
            ccfg.d_exp = ccfg.d * ccfg.t * ccfg.cp // ccfg.t_exp // ccfg.ep
        else:
            ccfg.t_exp = ccfg.t
            if ccfg.d >= ccfg.ep:
                ccfg.d_exp = ccfg.d // ccfg.ep
            else:
                ccfg.d_exp = ccfg.d * ccfg.t // ccfg.ep
            if ccfg.t_exp * ccfg.ep > ccfg.d * ccfg.t:
                ccfg.t_exp = 1

        exp_group1_invalid = ccfg.d_exp < 1 or ccfg.t_exp < 1
        exp_group2_invalid = ccfg.hff_exp < 1 or ccfg.n_exp < 1
        if exp_group1_invalid or exp_group2_invalid:
            raise TypeError(
                f"MoE parsing error: d_exp({ccfg.d_exp})/t_exp({ccfg.t_exp})/"
                f"hff_exp({ccfg.hff_exp})/n_exp({ccfg.n_exp})/"
                f"DP = {ccfg.d}, TP = {ccfg.t}, EP = {ccfg.ep}/"
            )
