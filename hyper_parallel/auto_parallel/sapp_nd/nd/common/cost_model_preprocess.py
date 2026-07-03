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
"""parse config for cost model"""
import re
import inspect
from copy import deepcopy
from pprint import pformat

# from hyper_parallel.auto_parallel.sapp_nd.nd.config import Config, YamlObject
from hyper_parallel.auto_parallel.sapp_nd.nd.common.generate_partitions import PartitionGenerator
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger


# class CostModelConfig(Config) :
class CostModelConfig(PartitionGenerator):
    """cost model variables class"""

    def __init__(
        self,
        input_config=None,
        hook_cls=None,
        framework=None,
        source_code=None,
    ):
        super().__init__(input_config, hook_cls, framework, source_code)
        logger.debug(
            "parser = %s for %s", str(self.parser), str(self.model_name)
        )

    def __str__(self):
        return "CostModelConfig attributes:\n" + pformat(
            {
                k: v
                for k, v in vars(self).items()
                if isinstance(v, (int, float, str, bool))
            }
        )

    def __getattr__(self, attr):
        call_source = inspect.currentframe().f_back.f_code.co_name
        if attr not in self.__dict__:
            logger.warning(
                "[%s] Attribute %s does not exist. "
                "Value '0' will be assigned.",
                call_source,
                attr,
            )
            return 0
        return self.__dict__[attr]

    def __copy__(self):
        res = object.__new__(type(self))
        res.__dict__.update(self.__dict__)
        return res

    def __deepcopy__(self, memo):
        res = object.__new__(type(self))
        for k, v in self.__dict__.items():
            setattr(res, k, deepcopy(v, memo))
        return res

    def fp_bytes(self, precision):
        """Return bytes size for datatype"""
        if precision and isinstance(precision, str):
            res = re.match(r"[^0-9]*([0-9]+)[^0-9]*", precision)
            if res:
                return int(res.group(1)) // 8
        logger.warning("No bytes detected from FP Precision: %s", precision)
        return 0

    def print_stages_i(self, stage_id, stage):
        """for print_stages"""
        stage_layers = []
        for chunk in stage:
            chunk_lay_occ = []
            if chunk:
                layer, count = chunk[0], 1
                for lay_id in range(1, len(chunk)):
                    if chunk[lay_id] == layer:
                        count += 1
                    else:
                        chunk_lay_occ += [f"{count}{layer.name[0]}"]
                        layer, count = chunk[lay_id], 1
                chunk_lay_occ += [f"{count}{layer.name[0]}"]
            stage_layers += [chunk_lay_occ]
        logger.info("stage _%s : %s", stage_id, stage_layers)

    def print_stages(self, stages, spec_stage_id=-1):
        """Call after generate_partitions"""
        if spec_stage_id == -1:
            for stage_id, stage in enumerate(stages):
                self.print_stages_i(stage_id, stage)
        elif 0 <= spec_stage_id < len(stages):
            self.print_stages_i(spec_stage_id, stages[spec_stage_id])
        else:
            logger.warning("Incorrect spec_stage_id")

    def count_layers(self, stages):
        """Count non-embedding and non-output layers in generated stages."""
        return sum(sum(len(layer) for layer in chunk) for chunk in stages) - 2

    def print_parallelism(self):
        """strategy pretty printer"""
        if not self.multimodal:
            logger.info("%s Parallelism used :", self.model_name)
            logger.info(
                "DP %s, TP %s, PP %s, EP %s, CP %s, VPP %s",
                self.d,
                self.t,
                self.p,
                self.ep,
                self.cp,
                self.vp,
            )
            logger.info(
                "d_exp %s, t_exp %s, os_max_shard %s, etp %s",
                self.d_exp,
                self.t_exp,
                self.os_max_shard,
                self.etp,
            )
            logger.info(
                "shard_grad_exp %s, shard_grad_non_exp %s",
                self.shard_grad_exp,
                self.shard_grad_non_exp,
            )
            logger.info(
                "shard_p_os_exp %s, shard_p_os_non_exp %s",
                self.shard_p_os_exp,
                self.shard_p_os_non_exp,
            )
            logger.info(
                "shard_embed %s, shard_output_activ %s, shard_rec_input %s",
                self.shard_embed,
                self.shard_output_activ,
                self.shard_recompute_input,
            )
        else:
            for m in self.mm_ccfgs:
                self.mm_ccfgs[m].print_parallelism()

    def strategy_num_devices(self):
        """total num devices"""
        return self.d * self.t * self.cp * self.p

    def is_consistent_pp_config(self):
        """check if pp/offset/recomputation consistency"""

        def is_valid_cfg(cfg):
            if cfg is None or isinstance(cfg, (int, bool)):
                return True
            if not isinstance(cfg, list) or not cfg:
                return False
            if isinstance(cfg[0], int):
                return len(cfg) == self.p
            if isinstance(cfg[0], list):
                return len(cfg) == self.vp and all(
                    isinstance(c, list) and len(c) == self.p for c in cfg
                )
            return False

        return (
            is_valid_cfg(self.offset)
            and is_valid_cfg(self.full_rec)
            and is_valid_cfg(self.sel_rec)
        )

    @staticmethod
    def __maybe_set_int(target, attr, value):
        """Set an integer strategy attribute when an override is supplied."""
        if isinstance(value, int):
            setattr(target, attr, value)

    def __strategy_target(self, model_name):
        """Get the config object targeted by a strategy update."""
        if not self.multimodal:
            return self
        if model_name in self.mm_ccfgs:
            return self.mm_ccfgs[model_name]
        raise TypeError(
            f"{self.model_name}:  model_name is required (multimodal)"
        )

    def set_strategy(self, **kwargs):
        """overwrite parallelism"""
        model_name = kwargs.get("model_name", None)
        dp = kwargs.get("dp", None)
        tp = kwargs.get("mp", None)
        cp = kwargs.get("cp", None)
        ep = kwargs.get("ep", None)
        op = kwargs.get("op", None)
        etp = kwargs.get("etp", None)
        pp = kwargs.get("pp", None)
        vpp = kwargs.get("vpp", None)
        off = kwargs.get("offset", None)
        fr = kwargs.get("full_rec", None)
        sr = kwargs.get("sel_rec", None)
        m = kwargs.get("mb", None)
        b = kwargs.get("mbs", None)
        target_ccfg = self.__strategy_target(model_name)

        for attr, value in (
            ("d", dp),
            ("t", tp),
            ("ep", ep),
            ("etp", etp),
            ("cp", cp),
            ("vp", vpp),
            ("p", pp),
            ("m", m),
            ("b", b),
        ):
            self.__maybe_set_int(target_ccfg, attr, value)
        target_ccfg.sp = target_ccfg.t
        if op is not None and isinstance(op, int):
            target_ccfg.os_max_shard = op
            # Sync has_op with os_max_shard: op<=1 means no optimizer sharding
            target_ccfg.has_op = op > 1
        target_ccfg.gbs = target_ccfg.b * target_ccfg.d * target_ccfg.m
        logger.debug(
            "in ccfg: DP = %d, TP = %d, EP = %d, CP = %d, "
            "PP = %d, MB = %d, MBS = %d, VPP = %d",
            target_ccfg.d,
            target_ccfg.t,
            target_ccfg.ep,
            target_ccfg.cp,
            target_ccfg.p,
            target_ccfg.m,
            target_ccfg.b,
            target_ccfg.vp,
        )
        if hasattr(target_ccfg.parser, "config_shard_emb"):
            target_ccfg.parser.config_shard_emb()
        target_ccfg.parser.config_dp_tp_exp(target_ccfg)
        target_ccfg.parser.config_optimizer_shard(target_ccfg)
        target_ccfg.parser.config_comm_flag(target_ccfg)
        if fr is not None:
            target_ccfg.full_rec = fr
        if sr is not None:
            target_ccfg.sel_rec = sr
        if isinstance(off, (int, list)):
            target_ccfg.offset = off
        if not target_ccfg.is_consistent_pp_config():
            raise AttributeError(
                f"{target_ccfg.model_name}: "
                "Inconsistent pipeline parallel variables "
                f"pp {target_ccfg.p} vpp {target_ccfg.vp} "
                f"offset {target_ccfg.offset} "
                f"full_rec {target_ccfg.full_rec} "
                f"sel_rec {target_ccfg.sel_rec}"
            )
        self.__maybe_set_int(target_ccfg, "cp", cp)

    def get_strategy(self):
        """return parallelism/recompute strategies"""

        def strategy(mm):
            return {
                "dp": mm.d,
                "tp": mm.t,
                "pp": mm.p,
                "ep": mm.ep,
                "cp": mm.cp,
                "vpp": mm.vp,
                "op": mm.os_max_shard,
                "gbs": mm.b * mm.m * mm.d,
                "sched": mm.pp_sched,
                "offset": mm.offset,
                "full_rec": mm.full_rec,
                "sel_rec": mm.sel_rec,
            }

        # logger.output("get_strat ccfg")
        if self.multimodal:
            return {mm.model_name: strategy(mm) for mm in self.mm_ccfgs.values()}
        return strategy(self)

    def layer_custom_config_callback(self, fun):
        """
        Use input fun as callback for layer_custom_config
        Only for overwriting cost model variables
        """
        for idx, f in enumerate(self.layer_custom_config):

            def wrap(e, hook=f[1]):
                hook(e)
                if isinstance(e, CostModelConfig):
                    fun(self)
                else:
                    e.set_ccfg(fun)

            wrap.__name__ = f"{f[1].__name__}_{fun.__name__}"
            self.layer_custom_config[idx] = (f[0], wrap)
