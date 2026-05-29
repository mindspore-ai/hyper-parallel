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
"""backbone memory estimation module"""
from __future__ import annotations
from typing import TYPE_CHECKING
import os
import ast
import math
import logging
import inspect
import pprint
import importlib
import matplotlib.pyplot as plt
from PIL import Image
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger as nd_logger
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context, MemType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._bwd_overhead import _BackwardOverhead
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._ppb import _PPB
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hook_base import MemEvalHook

if TYPE_CHECKING:
    from typing import Any, Dict, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
EVAL_YML = os.path.join(current_dir, "configs_eval/default.yaml")


class _Backbone:
    """backbone class"""

    def __init__(self, config: Any, **kwargs):
        self._child_cls = self
        self.mb = EvalUtils.mb
        self.eval_cfg = Config(kwargs.get("eval_yml", EVAL_YML))
        self._ctx = kwargs.get("ctx", Context())
        self._ccfg = kwargs.get("ccfg", None)
        self.framework = kwargs.get("framework", None)
        self.source_code = kwargs.get("source_code", None)
        self.hook_cls, self.config_path = None, None
        if not self._ccfg:
            self.hook_cls = kwargs.get("hook_cls", self.eval_cfg.hook_class)
            if isinstance(self.hook_cls, str):
                self._load_eval_yaml_hook_cls(self.hook_cls)
            if self.hook_cls and not isinstance(self.hook_cls, MemEvalHook):
                raise AttributeError(
                    f"'{self.hook_cls}' is not a MemEvalHook instance"
                )
            if config:
                if kwargs.get("log_level", 1) == 0:
                    # logger.setLevel(logging.CRITICAL)
                    nd_logger.setLevel(logging.CRITICAL)
                self.update_config(config)
            else:
                raise AttributeError("missing config")
        self.evaluator_instances = None
        self.ppb = None
        self._overhead_obj = _BackwardOverhead(
            self, self._ccfg, self._ctx, self._inner_dynamic_mem
        )
        self._ppb_obj = _PPB(self.eval_cfg, self._inner_dynamic_mem)

    @property
    def ccfg(self) -> CostModelConfig:
        """read-only"""
        return self._ccfg

    @property
    def ctx(self) -> Context:
        """read-only"""
        return self._ctx

    def _load_eval_yaml_hook_cls(self, hook_cls):
        """hook_class in eval yaml"""
        target_mod_path = None
        try:
            # search in folder 'hooks'
            hooks_dir = os.path.join(current_dir, "hooks")
            for f in os.listdir(hooks_dir):
                if f.endswith(".py"):
                    mod_path = f"hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hooks.{f.split('.')[0]}"
                    spec = importlib.util.find_spec(mod_path)
                    if spec is None or spec.origin is None:
                        continue
                    with open(spec.origin, "r", encoding="utf-8") as mf:
                        source = mf.read()
                        tree = ast.parse(source)
                        mod_cls = next(
                            (
                                node
                                for node in ast.walk(tree)
                                if isinstance(node, ast.ClassDef)
                                and node.name == hook_cls
                            ),
                            None,
                        )
                        if mod_cls:
                            target_mod_path = mod_path
                            break
            if target_mod_path:
                module = importlib.import_module(target_mod_path)
                self.hook_cls = getattr(module, hook_cls)()
        except (ModuleNotFoundError, ImportError) as e:
            print(e)

    # Peak memory estimation

    def update_config(self, new_config: Any) -> None:
        """processing input config"""
        if self._ccfg is not None:
            self._ccfg.update_config(new_config, self.hook_cls, self.framework, self.source_code)
        else:
            self._ccfg = CostModelConfig(new_config, self.hook_cls, self.framework, self.source_code)
        if isinstance(new_config, str):
            if not self.config_path:
                logger.info(
                    "%s Process config file: %s",
                    "=" * 30,
                    new_config.split("/")[-1],
                )
            self.config_path = new_config
        self.evaluator_instances = None
        self.ppb = None

    def _inner_static_mem(self) -> float:
        """static memory evaluation for backbone estimation"""
        if self._ctx.current_node in self._ctx.node_eval:
            p = self._ctx.eval.stat.p(self._ccfg, self._ctx)
            ost = self._ctx.eval.stat.os(self._ccfg, self._ctx)
            grad = self._ctx.eval.stat.grad(self._ccfg, self._ctx)
            res = p + ost + grad
            self._ctx.save2log("_param", res)
            return p + ost + grad
        return 0

    def _inner_dynamic_mem(
        self, ppb=False, default_micro_factor=None
    ) -> tuple[float, float]:
        """dynamic memory evaluation for backbone estimation"""
        if self._ctx.current_node in self._ctx.node_eval:
            if ppb:
                micro_factor = 1
            elif default_micro_factor:
                micro_factor = default_micro_factor
            else:
                sched = self._ccfg.pp_sched
                micro_factor = self._ctx.pp_micro_eval[sched](
                    self._ccfg, self._ctx
                )
            self._ctx.micro_factor = max(1, micro_factor)
            activation = self._ctx.eval.dyn.activation(self._ccfg, self._ctx)
            self._ctx.save2log("_activ", activation)
            comm = self._inner_comm_mem(micro_factor)
            return activation, comm
        return 0, 0

    def _inner_comm_mem(self, micro_factor) -> float:
        """dynamic memory evaluation for backbone estimation"""
        if self._ctx.current_node in self._ctx.node_eval:
            comm_eval_field = self._ctx.eval.dyn.comm
            comm_cat = {
                "dp": MemType.AG_COMM,
                "tp": MemType.AG_COMM,
                "cp": MemType.AG_COMM,
                "ep": MemType.A2A_COMM,
            }
            comm_mem = {}
            for k, fun in vars(comm_eval_field).items():
                sig_param = inspect.signature(fun).parameters.values()
                if (
                    # any(
                    #     p.kind == inspect.Parameter.VAR_KEYWORD
                    #     for p in sig_param
                    # )
                    # and len(sig_param) > 2
                    len(sig_param) > 2
                ):
                    comm = fun(self._ccfg, self._ctx, micro_factor)
                else:
                    comm = fun(self._ccfg, self._ctx)
                comm_mem[k] = comm
            res = EvalUtils.eval_expr_insight(
                expr=self._ctx.comm_expr,
                ctx=self._ctx,
                mem_val=comm_mem,
                mem_cat=comm_cat,
            )
            self._ctx.save2log("_comm", res)
            return res
        return 0

    def __subplot_stages(self, stage_i, stat_mem_i, dyn_mem_i, i) -> None:
        """Plot static and dynamic memory for one pipeline stage."""
        _, ax = plt.subplots(figsize=(5, 8))
        bottoms = {"stat": 0, "dyn": 0}
        color_stat = plt.get_cmap("cividis")
        color_dyn = plt.get_cmap("plasma")
        total_lay = sum(len(chunk) for chunk in stage_i)
        color_denominator = max(1, total_lay - 1)
        for chunk_id, chunk in enumerate(stage_i):
            for lay_id, lay_type in enumerate(chunk):
                real_id = self._ctx.real_lay_ids[chunk_id][i][lay_id]
                name = f"Lay_{real_id}_{lay_type.name[0]}"
                idx = chunk_id * len(dyn_mem_i) + lay_id
                stat = self.mb(stat_mem_i[chunk_id][lay_id])
                dyn = self.mb(dyn_mem_i[chunk_id][lay_id])
                ax.bar(
                    0.1,
                    [dyn],
                    bottom=bottoms["dyn"],
                    label=name,
                    color=color_dyn(idx / color_denominator),
                    width=0.15,
                    linewidth=0.5,
                    edgecolor="black",
                )
                ax.text(
                    0.2,
                    bottoms["dyn"] + dyn / 2,
                    f"DYN_{name}",
                    va="center",
                    fontsize=5,
                )
                ax.bar(
                    0.5,
                    [stat],
                    bottom=bottoms["stat"],
                    label=name,
                    color=color_stat(idx / color_denominator),
                    width=0.15,
                    linewidth=0.5,
                    edgecolor="black",
                )
                ax.text(
                    0.6,
                    bottoms["stat"] + stat / 2,
                    f"STAT_{name}",
                    va="center",
                    fontsize=5,
                )
                bottoms["dyn"] += dyn
                bottoms["stat"] += stat
        ax.axhline(
            self.get_max_device_memory(),
            color="red",
            linewidth=1,
            ls="dotted",
        )
        ax.axhline(
            bottoms["dyn"] + bottoms["stat"],
            color="blue",
            linewidth=1,
            ls="dashed",
        )
        ax.text(
            0,
            self.get_max_device_memory(),
            "Device Memory",
            color="red",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            0.62,
            bottoms["dyn"] + bottoms["stat"],
            "Prediction total",
            color="blue",
            va="bottom",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_ylabel("Size (MB)")
        ax.set_xlim([0, 0.8])
        ax.set_xlabel(f"Stage_{i}")

    def __plot_stages(self, stages, stat_mems, dyn_mems) -> None:
        """plot bars for estimations"""

        logger.info("Plotting predictions in plots/")
        if not os.path.exists("plots"):
            os.makedirs("plots")
        imgs = []
        for stage_id in range(self._ccfg.p):
            self.__subplot_stages(
                stages[stage_id],
                stat_mems[stage_id],
                dyn_mems[stage_id],
                stage_id,
            )
            img = f"plots/MemPlot_stage_{stage_id}.png"
            imgs += [(stage_id, img)]
            logger.info("save plot: %s", img)
            plt.savefig(img, dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()
        # Concatenate every stage plots
        if imgs:
            imgs = sorted(imgs)
            canvas = [Image.open(i) for _, i in imgs]
            stage_canva = Image.new(
                "RGB",
                (
                    canvas[0].size[0]
                    * 2 ** math.ceil(math.log2(len(canvas)) / 2),
                    canvas[0].size[1]
                    * 2 ** math.floor(math.log2(len(canvas)) / 2),
                ),
            )
            offset_x, offset_y = 0, 0
            for i in canvas:
                stage_canva.paste(i, (offset_x, offset_y))
                offset_x += i.size[0]
                if offset_x >= stage_canva.size[0]:
                    offset_x = 0
                    offset_y += i.size[1]
            stage_canva.save("plots/MemPlot_all_stages.png")
            logger.info("save stage plot: plots/MemPlot_all_stages.png")

    def __update_stage_logs(self, stage_logs: list, stage_id: int) -> None:
        """update stage's temporary buffers from ctx's buffers"""
        if not stage_logs[stage_id].node_compute_log:
            stage_logs[stage_id].node_compute_log = {}
        stage_logs[stage_id].node_compute_log.update(
            self._ctx.node_compute_log
        )
        if not stage_logs[stage_id].accu_mem_type:
            stage_logs[stage_id].accu_mem_type = {
                mt: 0 for mt in list(MemType)
            }
        for mem_type in list(MemType):
            val = self._ctx.accu_mem_type[mem_type]
            stage_logs[stage_id].accu_mem_type[mem_type] += val

    def __preprocess_layer_custom_config_list(self, stages: list) -> list:
        """flatten layer_custom_config for backbone estimation"""
        flatten = sum(
            [[f[1]] * f[0] for f in self._ccfg.layer_custom_config], []
        )
        total_n_lay = self._ccfg.n_lay + self._ccfg.n_mtp
        total_n_lay_stages = self._ccfg.count_layers(stages)
        if not self._ccfg.multimodal and total_n_lay != total_n_lay_stages:
            raise(AttributeError(
                f"Mismatch of num_layers between parsed value ({total_n_lay})"
                f" and generated partitions ({total_n_lay_stages})"
                f" => offset may be incorrect ({self._ccfg.offset})"
            ))
        if self._ccfg.n_lay > 0 and len(flatten) != total_n_lay:
            raise AttributeError(
                f"layer_custom_config occurrences ({len(flatten)})"
                f" != num_layers ({total_n_lay})"
            )
        if self._ccfg.pp_sched == "zero_bubble_v":
            n_layer_first_chunk = (
                sum(len(s[0]) for s in stages) - 1
            )  # Except embedding layer
            flatten = (
                flatten[:n_layer_first_chunk]
                + flatten[n_layer_first_chunk:][::-1]
            )
        return flatten

    def _estimate_backbone(self, *args) -> Tuple[list, Dict]:
        """Evaluator's main function for estimation"""
        stages = args[0]
        spec_stage_id = args[3]

        if spec_stage_id >= self._ccfg.p or spec_stage_id < 0:
            spec_stage_id = -1
        # Process partition generation
        if not stages:
            stages = self._ccfg.generate_partitions_vpp()
            #     multimodal=self._ccfg.multimodal
            # )
        if not self._ccfg.multimodal:
            return self.__estimate_stages_backbone(
                stages, args[1], args[2], spec_stage_id, args[4]
            )
        res = []
        original_ccfg = self._ccfg
        common_lc = []
        self.evaluator_instances = []
        # Build common layer_custom_config + Build temporary evaluators
        for m in self._ccfg.mm_order:
            self._ccfg.mm_ccfgs[m].config = original_ccfg.config
            if not self._child_cls:
                raise AttributeError("expected non null _child_cls")
            tmp_evaluator = type(self._child_cls)(
                None,
                ccfg=self._ccfg.mm_ccfgs[m],
                trace_fun=self.toggle_func_trace
            )
            tmp_evaluator.import_eval_yaml()
            num_layer = tmp_evaluator.get_num_layers()
            # assert not isinstance(num_layer, tuple)
            strategy = tmp_evaluator.get_strategy()
            full_rec = strategy["full_rec"]
            offset = strategy["offset"]
            self._ccfg.hooks_dict[m](tmp_evaluator)
            strategy = tmp_evaluator.get_strategy()
            if (
                tmp_evaluator.get_num_layers() != num_layer
                or strategy["full_rec"] != full_rec
                or strategy["offset"] != offset
            ):
                # Reverify num layers, recomp, offset after hook
                stages[m] = (
                    tmp_evaluator.ccfg.generate_partitions_vpp_unimodal()
                )
                tmp_evaluator.set_layer_custom(None)
            common_lc += self._ccfg.mm_ccfgs[m].layer_custom_config
            self.evaluator_instances += [tmp_evaluator]
            if args[1]:
                logger.info("Submodule %s", self._ccfg.mm_ccfgs[m].model_name)
                tmp_evaluator.print_ctx()
                self.print_stages(stages[m])
        self.set_layer_custom(common_lc)
        if args[1]:
            logger.info(
                "Combined layer_custom_config for %s\n%s",
                self._ccfg.model_name,
                pprint.pformat(self._ccfg.layer_custom_config, compact=True),
            )
            logger.info(
                "Sub evaluator instances for  %s\n%s",
                self._ccfg.model_name,
                pprint.pformat(self.evaluator_instances, compact=True),
            )

        res = self.__estimate_stages_backbone(
            self._ccfg.combine_partition_multimodal(stages),
            args[1],
            args[2],
            spec_stage_id,
            args[4],
        )
        self._ccfg = original_ccfg
        return res

    def __estimate_stages_backbone(self, *args) -> Tuple[list, Dict]:
        """Evaluator's main function for stage estimation"""
        stages = args[0]
        verbose = args[1]
        compute_ppb = args[2]
        spec_stage_id = args[3]

        if verbose:
            logger.info("Partition of layers :")
            self._ccfg.print_stages(stages, spec_stage_id)
        insights = []
        # Compute peak memory
        flatten = self.__preprocess_layer_custom_config_list(stages)
        if verbose:
            logger.info(
                "Flatten layer_custom_config\n%s",
                pprint.pformat(
                    [f if not f else f.__name__ for f in flatten], compact=True
                ),
            )

        stage_misc = {
            "stat": [[[0 for _ in c] for c in s] for s in stages],
            "dyn": [[[0 for _ in c] for c in s] for s in stages],
            "logs": [Config({}) for _ in range(self._ccfg.p)],
        }
        # tmp_ppb_lay_desc = []  # PPB purpose
        ppb_lay_desc = []
        record_lay_types = {}
        self.__chunk_stage_lay_loops(
            flatten,
            stages,
            record_lay_types,
            stage_misc,
            verbose,
            compute_ppb,
            ppb_lay_desc,  # tmp_ppb_lay_desc,
        )
        self.__postprocess_stages(
            stages,
            record_lay_types,
            stage_misc,
            verbose,
            spec_stage_id,
            insights,
        )
        # PPB Input
        ppb_input = None
        if compute_ppb == 1:
            self._ppb_obj.ppb_combine_bodies(ppb_lay_desc)
            ppb_input = {"layers_description": ppb_lay_desc}
        elif compute_ppb == 2:
            self._ppb_obj.ppb_combine_bodies_new(ppb_lay_desc)
            ppb_input = {"layers_description_new": ppb_lay_desc}
        if args[4]:  # Plot
            self.__plot_stages(stages, stage_misc["stat"], stage_misc["dyn"])
        return insights, ppb_input

    def __update_evaluator(self, node, verbose):
        if self.evaluator_instances and node == self._ctx.head_node:
            tmp_eval = self.evaluator_instances.pop(0)
            self._ccfg = tmp_eval.ccfg
            self._ctx.copy_tmp_buff(tmp_eval.ctx)
            self._ctx = tmp_eval.ctx
            if verbose:
                logger.info(
                    "Update ccfg and ctx, module : %s",
                    self._ccfg.model_name,
                )

    def __update_next_layer_custom_function(self, *args):
        """Apply the next layer custom hook before evaluating a layer."""
        flatten, verbose = args[0], args[1]
        record_lay_types = args[2]
        stage_id, chunk_id, lay_id = args[3], args[4], args[5]
        node = args[6]
        if self.is_regular_layer(node) and flatten:
            hook = flatten.pop(0)
            if hook:
                if verbose:
                    logger.info("Apply hook %s", hook.__name__)
                record_lay_types[(stage_id, chunk_id, lay_id)] = (
                    self._ccfg,
                    self._ctx,
                    hook,
                )
                hook(self)
            else:
                record_lay_types[(stage_id, chunk_id, lay_id)] = (
                    self._ccfg,
                    self._ctx,
                    lambda _: None,
                )
        else:
            record_lay_types[(stage_id, chunk_id, lay_id)] = (
                self._ccfg,
                self._ctx,
                lambda _: None,
            )
        if verbose:
            logger.info(
                "stage_id=%s, chunk_id=%s, lay_id=%s, node=%s",
                stage_id,
                chunk_id,
                lay_id,
                node,
            )
            self._ccfg.print_parallelism()

    def __chunk_stage_lay_loops(self, *args):
        """Evaluate every layer in every stage and chunk."""
        flatten, stages, record_lay_types = args[0], args[1], args[2]
        sm = args[3]
        verbose, compute_ppb, ppb_lay_desc = args[4], args[5], args[6]
        self._ctx.real_lay_ids = []
        count = 0
        for chunk_id in range(self._ccfg.vp):
            self._ctx.real_lay_ids += [[]]
            for stage_id in range(self._ccfg.p):
                self._ctx.real_lay_ids[chunk_id] += [[]]
                self._ctx.init_tmp_buff()
                for lay_id in range(len(stages[stage_id][chunk_id])):
                    node = stages[stage_id][chunk_id][lay_id]
                    if self.is_regular_layer(node):
                        self._ctx.real_lay_ids[chunk_id][stage_id] += [count]
                        count += 1
                    else:
                        self._ctx.real_lay_ids[chunk_id][stage_id] += [""]
                    # Update evaluator (multimodal)
                    self.__update_evaluator(node, verbose)
                    # Update next layer custom function
                    self.__update_next_layer_custom_function(
                        flatten,
                        verbose,
                        record_lay_types,
                        stage_id,
                        chunk_id,
                        lay_id,
                        node,
                    )
                    self._ctx.current_stage_id = stage_id
                    self._ctx.current_chunk_id = chunk_id
                    self._ctx.current_lay_id = lay_id
                    self._ctx.current_node = node
                    sm["stat"][stage_id][chunk_id][
                        lay_id
                    ] = self._inner_static_mem()
                    sm["dyn"][stage_id][chunk_id][lay_id] = sum(
                        self._inner_dynamic_mem()
                    )
                    if verbose:
                        logger.info("pp micro factor for dynamic: %s",self._ctx.micro_factor)
                    # PPB Purpose
                    if compute_ppb == 1:
                        desc = self._ppb_obj.lay_ppb(
                            self._ccfg,
                            self._ctx,
                            sm["stat"][stage_id][chunk_id][lay_id],
                        )
                        self._ppb_obj.add_to_ppb_list(ppb_lay_desc, desc)
                    elif compute_ppb == 2:
                        desc = self._ppb_obj.lay_ppb_new(
                            self._ccfg,
                            self._ctx,
                            sm["stat"][stage_id][chunk_id][lay_id],
                        )
                        self._ppb_obj.add_to_ppb_list(ppb_lay_desc, desc)
                self.__update_stage_logs(sm["logs"], stage_id)

    def __postprocess_stages(self, *args):
        """Build memory insights from raw stage evaluation buffers."""
        stages, record_lay_types = args[0], args[1]
        sm = args[2]
        verbose, spec_stage_id = args[3], args[4]
        insights = args[5]
        for stage_id in range(self._ccfg.p):
            ins = {}  # Mem Insights purpose
            ins["Static"] = sum(
                sum(mem for mem in c) for c in sm["stat"][stage_id]
            )
            ins["Dynamic"] = sum(
                sum(mem for mem in c) for c in sm["dyn"][stage_id]
            )
            self._ctx.init_tmp_buff()
            if not self._ccfg.freeze:
                ins["Dynamic"] += self._overhead_obj.estimate(
                    stages, stage_id, record_lay_types
                )
                self.__update_stage_logs(sm["logs"], stage_id)
            safety_buffer = 1024 * 1024 * 1024  # 1 GB
            if ins["Dynamic"] > 0:
                ins["Dynamic"] += safety_buffer
            stage_accu = sm["logs"][stage_id].accu_mem_type
            ins["ModelParameters"] = self.mb(stage_accu[MemType.MODEL_PARAM])
            ins["OptimizerStates"] = self.mb(stage_accu[MemType.OPTIM_STATE])
            ins["AccumulGradients"] = self.mb(stage_accu[MemType.ACCU_GRAD])
            ins["Attn"] = self.mb(stage_accu[MemType.ATTN_ACTIV])
            ins["FFn"] = self.mb(stage_accu[MemType.FFN_ACTIV])
            ins["Norm"] = self.mb(stage_accu[MemType.NORM_ACTIV])
            ins["AllGather Comm"] = self.mb(stage_accu[MemType.AG_COMM])
            ins["All2All Comm"] = self.mb(stage_accu[MemType.A2A_COMM])
            ins["Node Log"] = sm["logs"][stage_id].node_compute_log
            # VERBOSE
            if verbose and spec_stage_id in (-1, stage_id):
                self.__verbose_insights(sm, stage_id, ins)
            ins["Static"] = self.mb(ins["Static"])
            ins["Dynamic"] = self.mb(ins["Dynamic"])
            insights += [ins]

    def __verbose_insights(self, *args):
        """logging purpose"""
        sm = args[0]
        stage_id = args[1]
        ins = args[2]
        stat_i = max(1, ins["Static"])
        dyn_i = max(1, ins["Dynamic"])
        # logs_i = sm["logs"][stage_id]
        accu_i = sm["logs"][stage_id].accu_mem_type
        logger.info(
            "stage _%s : %s MB",
            stage_id,
            self.mb(ins["Static"] + ins["Dynamic"]),
        )
        logger.info(
            "\tStatic\t%s\t"
            "ModelParam %s (%s%%), "
            "OptimStates %s (%s%%), "
            "Gradients %s (%s%%)",
            self.mb(ins["Static"]),
            ins["ModelParameters"],
            round(accu_i[MemType.MODEL_PARAM] / stat_i * 100),
            ins["OptimizerStates"],
            round(accu_i[MemType.OPTIM_STATE] / stat_i * 100),
            ins["AccumulGradients"],
            round(accu_i[MemType.ACCU_GRAD] / stat_i * 100),
        )
        logger.info(
            "\tDynamic\t%s\t"
            "Attn %s (%d%%), "
            "FFn %s (%d%%), "
            "Norm %s (%d%%), "
            "AllGather Comm %s (%d%%), "
            "All2All Comm %s (%d%%), ",
            self.mb(ins["Dynamic"]),
            ins["Attn"],
            round(accu_i[MemType.ATTN_ACTIV] / dyn_i * 100),
            ins["FFn"],
            round(accu_i[MemType.FFN_ACTIV] / dyn_i * 100),
            ins["Norm"],
            round(accu_i[MemType.NORM_ACTIV] / dyn_i * 100),
            ins["AllGather Comm"],
            round(accu_i[MemType.AG_COMM] / dyn_i * 100),
            ins["All2All Comm"],
            round(accu_i[MemType.A2A_COMM] / dyn_i * 100),
        )
        logger.info(
            "\tNode eval log : \n %s \n %s",
            "> Foreach: (stage_id,chunk_id,lay_id,name) -> (mem type,value)",
            pprint.pformat(ins["Node Log"], width=300),
        )

    def apply_hook(self, hook, ccfg=None, ctx=None):
        """apply hook on evaluator"""
        self._ccfg = ccfg if ccfg else self._ccfg
        self._ctx = ctx if ctx else self._ctx
        hook(self)

    def set_layer_custom(self, _):
        """child implement"""
        pass  # pylint: disable=unnecessary-pass

    def is_regular_layer(self, _):
        """child implement"""
        return False

    def import_eval_yaml(self):
        """child implement"""
        pass  # pylint: disable=unnecessary-pass

    def get_num_layers(self):
        """child implement"""
        return 0

    def get_strategy(self):
        """child implement"""
        return {}

    def get_max_device_memory(self):
        """child implement"""
        return 0

    def print_stages(self, _):
        """child implement"""
        pass  # pylint: disable=unnecessary-pass

    def print_ctx(self):
        """child implement"""
        pass  # pylint: disable=unnecessary-pass
