# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Skeleton of a memory estimation hook class.

Copy the class, give @hook_runner the model name to hook, and replace the
placeholder formula and cost model override.
"""
from typing import Any

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hook_base import MemEvalHook, hook_runner


class Template(MemEvalHook):
    """Registers a placeholder for every hookable formula."""

    @staticmethod
    def f(ccfg: Any, ctx: Any) -> None:
        """Formula to get hooked. The placeholder returns None."""
        del ccfg, ctx

    @staticmethod
    def custom_ccfg(ccfg: Any) -> None:
        """Cost model variables to overwrite. The placeholder changes nothing."""
        del ccfg

    @staticmethod
    @hook_runner("model name")
    def run_hooks(e: Any) -> None:
        """Register the placeholders on the evaluator e."""
        c = Template
        e.set_ccfg(c.custom_ccfg)
        e.set_passes(vpp_less_mem=False, swap_os=False, dropless_tok_factor=1)
        e.set_head_eval_fun(num_p=c.f, stat=c.f, dyn=c.f)
        e.set_tail_eval_fun(num_p=c.f, stat=c.f, dyn=c.f)
        e.set_body_eval_fun(
            "NOT_REC_LAYER",
            num_p=c.f,
            stat_p=c.f,
            stat_os=c.f,
            stat_grad=c.f,
            dyn_activ=c.f,
            dyn_dp_comm=c.f,
            dyn_tp_comm=c.f,
            dyn_cp_comm=c.f,
            dyn_ep_comm=c.f,
        )
        e.set_attn_eval_fun(
            num_p=c.f,
            qkv=c.f,
            score=c.f,
            proj=c.f,
        )
        e.set_ffn_eval_fun(
            num_p=c.f,
            activation=c.f,
            moe_activ=c.f
        )
        e.set_norm_eval_fun(num_p=c.f, activation=c.f)
        e.set_pp_micro_factor_eval_fun("1f1b", c.f)
