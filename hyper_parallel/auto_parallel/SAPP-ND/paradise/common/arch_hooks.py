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
"""Custom variables per model (expert knowledge)"""
import math
from paradise.common.config import Config
from paradise.common.cost_model_preprocess import CostModelConfig
from memory_estimation.logger import logger


class CWrap:
    """Temporary evaluator-like instance"""

    def __init__(self, e) -> None:
        self.ccfg = e
        self.set_ccfg = lambda hook: hook(self.ccfg)
        self.get_model_name = lambda: self.ccfg.model_name

    def reset(self, e):
        """Replace the wrapped config object."""
        self.ccfg = e

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return lambda *args, **kwargs: None
        return self.__dict__[attr]

    def set_strategy(self, **kwargs):
        """Forward strategy updates to the wrapped config."""
        self.ccfg.set_strategy(**kwargs)

    def get_strategy(self):
        """Return strategy from the wrapped config."""
        return self.ccfg.get_strategy()


def custom_default_transformer(ccfg):
    """base"""
    ccfg.n_attMM = 4  # num attention matmul
    ccfg.n_attBMM = 2  # num attention batch matmul
    ccfg.n_attParamCast = (
        ccfg.n_attMM if not ccfg.has_op else 0
    )  # num attention parameters cast
    ccfg.n_ffMM = 3  # num feedforward matmul
    ccfg.n_ffBMM = 0  # num feedforward batch matmul
    ccfg.n_ffParamCast = (
        ccfg.n_ffMM if not ccfg.has_op else 0
    )  # num feedforward parameters cast
    ccfg.n_softmax = 1  # num softmax
    ccfg.n_dropout = 0  # num dropout
    ccfg.n_normOp = 2  # num normalization
    ccfg.n_gather = 4  # num gather (TP)
    ccfg.bytes_grad = 4 if ccfg.p > 1 else 0  # gradients
    ccfg.bytes_os = 4  # optimizer states
    ccfg.bytes_dropout = 0  # dropout mask
    ccfg.bytes_norm = 4  # normalization input


def custom_llama2(ccfg):
    """llama2"""
    custom_default_transformer(ccfg)
    ccfg.n_gather = 4  # num gather (TP)
    ccfg.bytes_grad = 2  # gradients


def custom_mixtral(ccfg):
    """mixtral"""
    ccfg.n_attMM = 4  # num attention matmul
    ccfg.n_attBMM = 2  # num attention batch matmul
    ccfg.n_attParamCast = (
        ccfg.n_attMM if not ccfg.has_op else 0
    )  # num attention parameters cast
    ccfg.n_ffMM = 0  # num feedforward matmul
    ccfg.n_ffBMM = 3  # num feedforward batch matmul
    ccfg.n_ffParamCast = (
        ccfg.n_ffMM if not ccfg.has_op else 0
    )  # num feedforward parameters cast
    ccfg.n_softmax = 2  # num softmax
    ccfg.n_dropout = 0  # num dropout
    ccfg.n_normOp = 5  # num normalization
    ccfg.n_gather = 4  # num gather (TP)
    ccfg.bytes_grad = 2 if ccfg.p > 1 else 0  # gradients
    ccfg.bytes_os = 4  # optimizer states
    ccfg.bytes_dropout = 0  # dropout mask
    ccfg.bytes_norm = 4  # normalization input
    ccfg.hff = ccfg.hff_exp


def custom_t5(ccfg):
    """t5"""

    # Encoder + Decoder
    def encode(c):
        c.n_attMM = 4  # num attention matmul
        c.n_attBMM = 1  # num attention batch matmul
        c.n_attParamCast = (
            c.n_attMM if not c.has_op else 0
        )  # num attention parameters cast
        c.n_ffMM = 2  # num feedforward matmul
        c.n_ffBMM = 0  # num feedforward batch matmul
        c.n_ffParamCast = (
            c.n_ffMM if not c.has_op else 0
        )  # num feedforward parameters cast
        c.n_softmax = 2  # num softmax
        c.n_dropout = 5  # num dropout
        c.n_normOp = 2  # num normalization
        c.n_gather = 4  # num gather (TP)
        c.bytes_grad = 4 if c.p > 1 else 0  # gradients
        c.bytes_os = 4  # optimizer states
        c.bytes_dropout = 1  # dropout mask
        c.bytes_norm = 4  # normalization input

    def decode(c):
        c.n_attMM = 8  # num attention matmul
        c.n_attBMM = 2  # num attention batch matmul
        c.n_attParamCast = (
            c.n_attMM if not c.has_op else 0
        )  # num attention parameters cast
        c.n_ffMM = 2  # num feedforward matmul
        c.n_ffBMM = 0  # num feedforward batch matmul
        c.n_ffParamCast = (
            c.n_ffMM if not c.has_op else 0
        )  # num feedforward parameters cast
        c.n_softmax = 4  # num softmax
        c.n_dropout = 7  # num dropout
        c.n_normOp = 3  # num normalization
        c.n_gather = 6  # num gather (TP)
        c.bytes_grad = 4 if c.p > 1 else 0  # gradients
        c.bytes_os = 4  # optimizer states
        c.bytes_dropout = 1  # dropout mask
        c.bytes_norm = 4  # normalization input

    def hook_encode(e):
        if isinstance(e, CostModelConfig):
            e = CWrap(e)
        e.set_ccfg(encode)

    def hook_decode(e):
        if isinstance(e, CostModelConfig):
            e = CWrap(e)
        e.set_ccfg(decode)

    ccfg.layer_custom_config = [
        (ccfg.n_lay // 2, hook_encode),
        (ccfg.n_lay // 2, hook_decode),
    ]


def custom_pangualpha(ccfg):
    """pangualpha"""
    ccfg.n_attMM = 4  # num attention matmul
    ccfg.n_attBMM = 1  # num attention batch matmul
    ccfg.n_attParamCast = (
        ccfg.n_attMM if not ccfg.has_op else 0
    )  # num attention parameters cast
    ccfg.n_ffMM = 2  # num feedforward matmul
    ccfg.n_ffBMM = 0  # num feedforward batch matmul
    ccfg.n_ffParamCast = (
        ccfg.n_ffMM if not ccfg.has_op else 0
    )  # num feedforward parameters cast
    ccfg.n_softmax = 2  # num softmax
    ccfg.n_dropout = 5  # num dropout
    ccfg.n_normOp = 4  # num normalization
    ccfg.n_gather = 4  # num gather (TP)
    ccfg.bytes_grad = 4 if ccfg.p > 1 else 0  # gradients
    ccfg.bytes_os = 4  # optimizer states
    ccfg.bytes_dropout = 1  # dropout mask
    ccfg.bytes_norm = 4  # normalization input


def custom_deepseek3(ccfg):
    """deepseekv3"""
    saved = Config({})
    if ccfg.config_format == "yaml":
        saved.hff = (
            ccfg.config.model.model_config.intermediate_size
            if ccfg.config.model.model_config.intermediate_size
            else ccfg.parser.init_hff()
        )
    elif ccfg.config_format == "json":
        saved.hff = ccfg.ffn_hidden_size
    else:
        saved.hff = ccfg.specs.inter_dim
        if not saved.hff:
            saved.hff = ccfg.specs.hidden_dim
        if not saved.hff:
            saved.hff = ccfg.h
    saved.n_chosen_exp = ccfg.n_chosen_exp
    saved.n_exp = ccfg.n_exp
    saved.n_shared_exp = ccfg.n_shared_exp
    saved.ep = ccfg.ep
    custom_default_transformer(ccfg)
    ccfg.dh = 128

    def dense(c):
        c.hff = saved.hff
        c.n_chosen_exp = 1
        c.n_exp = 1
        c.n_shared_exp = 0

    def moe(c):
        c.hff = c.hff_exp
        c.n_chosen_exp = saved.n_chosen_exp
        c.n_exp = saved.n_exp
        c.n_shared_exp = saved.n_shared_exp

    def hook_dense(e):
        if isinstance(e, CostModelConfig):
            e = CWrap(e)
        # e.ccfg.ep = 1
        e.set_ccfg(dense)
        e.ccfg.ep = 1
        # e.set_strategy(ep=1)

    def hook_moe(e):
        if isinstance(e, CostModelConfig):
            e = CWrap(e)
        # e.ccfg.ep = saved.ep
        e.set_ccfg(moe)
        e.ccfg.ep = saved.ep
        # e.set_strategy(ep=saved.ep)

    n_moe = ccfg.n_lay - ccfg.k_1st_dense
    ccfg.layer_custom_config = [
        (ccfg.k_1st_dense, hook_dense),
        (n_moe, hook_moe),
        (ccfg.n_mtp, hook_moe if n_moe > 0 else hook_dense),
    ]


def custom_qwen(ccfg):
    """qwen2"""
    custom_default_transformer(ccfg)
    # if "72b" in ccfg.model_name :
    #     ccfg.s = ccfg.s * 3/4
    ccfg.shard_recompute_input = ccfg.t
    ccfg.shard_output_activ = ccfg.t
    # ccfg.bytes_grad = 4


def custom_cm(ccfg):
    """llama moe"""
    shard_p_os_exp = ccfg.shard_p_os_exp_partial
    shard_p_os_non_exp_partial = math.gcd(ccfg.n_exp, ccfg.shard_p_os_non_exp)
    shard_embed = ccfg.t
    custom_deepseek3(ccfg)

    def custom_shard(c):
        c.shard_p_os_exp = shard_p_os_exp
        c.shard_p_os_non_exp_partial = shard_p_os_non_exp_partial
        c.shard_embed = shard_embed

    for idx, f in enumerate(ccfg.layer_custom_config):

        def wrap_hook(e, f=f):
            if isinstance(e, CostModelConfig):
                e = CWrap(e)
            f[1](e)
            e.set_ccfg(custom_shard)

        ccfg.layer_custom_config[idx] = (f[0], wrap_hook)

    def num_params_norm_cm(c, _):
        return c.n_normOp * 2 * c.h + 0.5 * c.n_attMM * c.dh

    ccfg.overwrite_eval_functions["num_params_norm"] = num_params_norm_cm


def check_and_apply_custom_hook(e):
    """routing hooks"""
    if isinstance(e, CostModelConfig):
        e = CWrap(e)
    map_modelname_custom = {
        "llama2": custom_llama2,
        "mixtral": custom_mixtral,
        "t5": custom_t5,
        "pangualpha": custom_pangualpha,
        "deepseek": custom_deepseek3,
        "qwen": custom_qwen,
        "cm": custom_cm,
    }
    for k, v in map_modelname_custom.items():
        if k in e.get_model_name().lower():
            e.set_ccfg(v)
            return
    logger.warning(
        "Hook not defined for: %s. Default one is chosen", e.get_model_name()
    )
    e.set_ccfg(custom_default_transformer)
