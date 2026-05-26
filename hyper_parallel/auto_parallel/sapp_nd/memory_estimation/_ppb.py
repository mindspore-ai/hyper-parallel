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
"""PPB input module"""

from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils


class _PPB:
    """Pipeline balance payload builder."""

    def __init__(self, eval_cfg, inner_dyn_fun):
        self.eval_cfg = eval_cfg
        self._inner_dynamic_mem = inner_dyn_fun
        self.mb = EvalUtils.mb

    def add_to_ppb_list(self, ppb_lay_desc, desc):
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

    def lay_ppb(self, ccfg, ctx, res_stat):
        """layer description preparation"""
        ctx.enable_node_log = False
        desc = {}
        desc["model_name"] = ccfg.model_name
        if ctx.current_node == ctx.head_node:
            d_emb = self.mb(sum(self._inner_dynamic_mem(ppb=True)))
            desc["type"] = "HEAD"
            desc["memory_parameter"] = self.mb(res_stat) + d_emb
            # desc["time"] = perfs[0]
            desc["time"] = 1
        elif ctx.current_node == ctx.tail_node:
            d_out = self.mb(sum(self._inner_dynamic_mem(ppb=True)))
            desc["type"] = "TAIL"
            desc["memory_parameter"] = self.mb(res_stat) + d_out
            # desc["time"] = perfs[-1]
            desc["time"] = 1
        else:
            ctx.current_node = LayerType.NOT_REC_LAYER
            dyn_nrec = self._inner_dynamic_mem(ppb=True)
            ctx.current_node = LayerType.SEL_REC_LAYER
            dyn_srec = self._inner_dynamic_mem(ppb=True)
            ctx.current_node = LayerType.FULL_REC_LAYER
            dyn_frec = self._inner_dynamic_mem(ppb=True)
            c = max(dyn_nrec[1], dyn_srec[1], dyn_frec[1])  # Temporary
            desc["type"] = "BODY"
            desc["memory_parameter"] = self.mb(res_stat)
            desc["memory_parameter"] += self.mb(c)
            desc["memory_activation"] = self.mb(dyn_nrec[0])
            desc["memory_select_rec"] = self.mb(dyn_srec[0])
            desc["memory_recompute"] = self.mb(dyn_frec[0])
            desc["time"] = 1
        ctx.enable_node_log = True
        # if any(
        #     k in desc and desc[k] > 0
        #     for k in [
        #         "memory_parameter",
        #         "memory_activation",
        #         "memory_recompute",
        #         "memory_select_rec",
        #     ]
        # ):
        #     # Ignore values 0
        #     return desc
        return desc

    def ppb_combine_bodies(self, ppb_lay_desc):
        """combine descriptions into a new body"""
        if not self.eval_cfg.ppb_combined:
            return
        for new_body in self.eval_cfg.ppb_combined:
            desc = {
                "model_name": "combined",
                "type": "BODY",
                "memory_parameter": 0,
                "memory_activation": 0,
                "memory_select_rec": 0,
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
                    for m in desc:
                        if m.startswith("memory") and m in target:
                            desc[m] += target[m]
                    target_idx = ppb_lay_desc.index(target)
                    idx = target_idx if idx < 0 else min(idx, target_idx)
                    del ppb_lay_desc[target_idx]
            idx = max(idx, 0)
            ppb_lay_desc.insert(idx, desc)

    def lay_ppb_new(self, ccfg, ctx, res_stat):
        """layer description preparation"""
        ctx.enable_node_log = False
        desc = {}
        desc["model_name"] = ccfg.model_name
        desc["memory_activation"] = {"NONE": 0, "FULL": 0}
        if ctx.current_node == ctx.head_node:
            d_emb = self.mb(sum(self._inner_dynamic_mem(ppb=True)))
            desc["memory_parameter"] = self.mb(res_stat) + d_emb
            desc["type"] = "HEAD"
        elif ctx.current_node == ctx.tail_node:
            d_out = self.mb(sum(self._inner_dynamic_mem(ppb=True)))
            desc["memory_parameter"] = self.mb(res_stat) + d_out
            desc["type"] = "TAIL"
        else:
            ctx.current_node = LayerType.NOT_REC_LAYER
            dyn_nrec = self._inner_dynamic_mem(ppb=True)
            ctx.current_node = LayerType.FULL_REC_LAYER
            dyn_frec = self._inner_dynamic_mem(ppb=True)
            c = max(dyn_nrec[1], dyn_frec[1])  # Temporary
            desc["memory_parameter"] = self.mb(res_stat)
            desc["memory_parameter"] += self.mb(c)
            desc["memory_activation"]["NONE"] = self.mb(dyn_nrec[0])
            desc["memory_activation"]["FULL"] = self.mb(dyn_frec[0])
            desc["type"] = "BODY"
        desc["options"] = ["NONE", "FULL"]
        desc["forward_time"] = {"NONE": 1, "FULL": 1}
        desc["backward_time"] = {"NONE": 1, "FULL": 1}
        desc["time"] = 1
        ctx.enable_node_log = True
        # if any(
        #     k > 0
        #     for k in [
        #         desc["memory_parameter"],
        #         desc["memory_activation"]["NONE"],
        #         desc["memory_activation"]["FULL"],
        #     ]
        # ):
        #     # Ignore values 0
        #     return desc
        # return {}
        return desc

    def ppb_combine_bodies_new(self, ppb_lay_desc):
        """combine descriptions into a new body"""
        if not self.eval_cfg.ppb_combined:
            return
        for new_body in self.eval_cfg.ppb_combined:
            desc = {
                "model_name": "combined",
                "type": "BODY",
                "memory_parameter": 0,
                "memory_activation": {"NONE": 0, "FULL": 0},
                "options": ["NONE", "FULL"],
                "forward_time": {"NONE": 1, "FULL": 1},
                "backward_time": {"NONE": 1, "FULL": 1},
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
                    desc["memory_parameter"] += target["memory_parameter"]
                    desc["memory_activation"]["NONE"] += target[
                        "memory_activation"
                    ]["NONE"]
                    desc["memory_activation"]["FULL"] += target[
                        "memory_activation"
                    ]["FULL"]
                    target_idx = ppb_lay_desc.index(target)
                    idx = target_idx if idx < 0 else min(idx, target_idx)
                    del ppb_lay_desc[target_idx]
            idx = max(idx, 0)
            ppb_lay_desc.insert(idx, desc)
