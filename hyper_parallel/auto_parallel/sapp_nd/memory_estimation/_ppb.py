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
"""PPB input module"""
from __future__ import annotations

from typing import Callable, Tuple, TYPE_CHECKING

from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig

# The balancer's recompute options.
_OPTIONS = ("NONE", "SLCT", "FULL")
# Where the balancer reads each option's memory per micro-batch.
_MEMORY_KEY = {
    "NONE": "memory_activation",
    "SLCT": "memory_select_rec",
    "FULL": "memory_recompute",
}


class _PPB:
    """Pipeline balance payload builder."""

    def __init__(self, eval_cfg: Config, inner_dyn_fun: Callable) -> None:
        """Initialize _PPB with evaluation config and dynamic memory function.

        Args:
            eval_cfg: Evaluation configuration object.
            inner_dyn_fun: Function to compute inner dynamic memory.
        """
        self.eval_cfg = eval_cfg
        self._inner_dynamic_mem = inner_dyn_fun
        self.mb = EvalUtils.mb

    @staticmethod
    def add_to_ppb_list(ppb_lay_desc: list, desc: dict) -> None:
        """layer description list preparation"""
        if desc:
            already_comp = False
            body_idx = 0
            for d in ppb_lay_desc:
                if all(v == d[k] for k, v in desc.items()):
                    # already exist desc
                    d["nb_layer"] += 1
                    already_comp = True
                if d["type"] == "BODY":
                    body_idx += 1
            if desc and not already_comp:
                desc["nb_layer"] = 1
                if desc["type"] == "BODY":
                    desc["name"] = f"BODY_{body_idx}"
                else:
                    desc["name"] = desc["type"]
                ppb_lay_desc += [desc]

    def lay_ppb(self, ccfg: CostModelConfig, ctx: Context, res_stat: float) -> dict:
        """layer description preparation"""
        desc = {"model_name": ccfg.model_name}
        original_enable_node_log = ctx.enable_node_log
        ctx.enable_node_log = False
        try:
            if ctx.current_node == ctx.head_node:
                d_emb = self.mb(sum(self._inner_dynamic_mem(ppb=True)))
                desc["type"] = "HEAD"
                desc["memory_parameter"] = self.mb(res_stat) + d_emb
                desc["time"] = 1
            elif ctx.current_node == ctx.tail_node:
                d_out = self.mb(sum(self._inner_dynamic_mem(ppb=True)))
                desc["type"] = "TAIL"
                desc["memory_parameter"] = self.mb(res_stat) + d_out
                desc["time"] = 1
            else:
                self._body_memory(desc, ccfg, ctx, res_stat)
        finally:
            ctx.enable_node_log = original_enable_node_log
        return desc

    def _body_memory(self, desc: dict, ccfg: CostModelConfig, ctx: Context, res_stat: float) -> None:
        """Describe a body layer's memory under each option.

        The balancer charges an option's memory once per micro-batch in
        flight, and the layer's parameter memory once. Activations are kept
        per micro-batch. Of the communication buffers, what grows with the
        micro-batches in flight is charged per micro-batch, and the rest,
        the largest any option keeps, once.
        """
        many = self._many(ccfg)
        ctx.current_node = LayerType.NOT_REC_LAYER
        dyn = {"NONE": self._dynamic_mem(many)}
        ctx.current_node = LayerType.SEL_REC_LAYER
        dyn["SLCT"] = self._dynamic_mem(many)
        ctx.current_node = LayerType.FULL_REC_LAYER
        dyn["FULL"] = self._dynamic_mem(many)
        desc["type"] = "BODY"
        desc["memory_parameter"] = self.mb(res_stat) + self.mb(max(once for _, _, once in dyn.values()))
        for name in _OPTIONS:
            activation, per_micro_batch, _ = dyn[name]
            desc[_MEMORY_KEY[name]] = self.mb(activation) + self.mb(per_micro_batch)
        desc["time"] = 1

    @staticmethod
    def _many(ccfg: CostModelConfig) -> int:
        """The most micro-batches a stage keeps in flight under 1F1B, and at least 2."""
        return max(2, min(getattr(ccfg, "p", 1), getattr(ccfg, "m", 1)))

    def _dynamic_mem(self, many: int) -> Tuple[float, float, float]:
        """``(activation, buffers per micro-batch, buffers once)`` of the current layer.

        The buffers are split by how they grow from one micro-batch in
        flight to *many*.
        """
        _, more = self._inner_dynamic_mem(default_micro_factor=many)
        activation, one = self._inner_dynamic_mem(ppb=True)
        per_micro_batch = (more - one) / (many - 1)
        return activation, per_micro_batch, one - per_micro_batch

    def ppb_combine_bodies(self, ppb_lay_desc: list) -> None:
        """combine descriptions into a new body"""
        if not self.eval_cfg.ppb_combined:
            return
        for new_body in self.eval_cfg.ppb_combined:
            desc = {
                "model_name": "combined",
                "type": "BODY",
                "memory_parameter": 0,
                "memory_activation": 0,
                "memory_recompute": 0,
                "time": 1,
                "nb_layer": 1,
                "name": "COMBINED",
            }
            idx = -1
            for mod, t in new_body:
                target = next(
                    (
                        d
                        for d in ppb_lay_desc
                        if d["model_name"] == mod and d["type"] == t.upper()
                    ),
                    None,
                )
                if target:
                    desc["model_name"] += "_" + mod
                    desc["name"] += "_" + target["name"]
                    for key, value in target.items():
                        if key.startswith("memory"):
                            desc[key] = desc.get(key, 0) + value
                    target_idx = ppb_lay_desc.index(target)
                    idx = target_idx if idx < 0 else min(idx, target_idx)
                    del ppb_lay_desc[target_idx]
            idx = max(idx, 0)
            ppb_lay_desc.insert(idx, desc)
