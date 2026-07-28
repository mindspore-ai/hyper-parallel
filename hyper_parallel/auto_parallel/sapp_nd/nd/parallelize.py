# Copyright 2024-2026 Huawei Technologies Co., Ltd
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
"""find parallelization"""

import time
import copy
import multiprocessing as proc
import json
import os
import logging

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_performance

from hyper_parallel.auto_parallel.sapp_nd.nd.global_config import GlobalConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import validate_cp_constraints
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import detect_attention_type

# logger = proc.log_to_stderr()
# logger.setLevel(proc.SUBDEBUG)


class ParallelizeLayer:
    """Parallelize one layer type"""

    def __init__(
        self,
        evaluator,
        machine,
        global_batch_size=None,
        dimensions=None,
        **extra_config,
    ):

        self.enable_debug = logger.level < logging.CRITICAL
        self.machine = machine
        if "mppb" in extra_config:
            manual_ppb = extra_config.pop("mppb")
        else:
            manual_ppb = False

        self.mem_eval = evaluator

        self.model_name = self.mem_eval._ccfg.model_name
        logger.debug("model is %s", self.model_name)

        if "mem_for_ppb" in extra_config:
            reserve_mem = extra_config.pop("mem_for_ppb")
            self.mem_eval._ccfg.device_capacity.decrease(reserve_mem)

        if "max_mem" in extra_config:
            max_mem = extra_config.pop("max_mem")
            if max_mem is not None:
                self.mem_eval._ccfg.device_capacity.set(max_mem)

        logger.debug("before global config init")

        if "sub_model" in extra_config:
            sub_model = extra_config.pop("sub_model")
            if sub_model is not None:
                self.config = GlobalConfig(
                    self.mem_eval._ccfg.mm_ccfgs[sub_model],
                    dimensions,
                    mppb=manual_ppb,
                )
            else:
                self.config = GlobalConfig(
                    self.mem_eval._ccfg, dimensions, mppb=manual_ppb
                )
        else:
            self.config = GlobalConfig(
                self.mem_eval._ccfg, dimensions, mppb=manual_ppb
            )

        self.mem_eval.set_passes(**extra_config)

        self.machine.update_num_if_none(
            self.config.ccfg.strategy_num_devices()
        )

        if global_batch_size:
            self.global_batch_size = global_batch_size
        else:
            self.global_batch_size = self.config.ccfg.gbs

        self.bound_space()

    def bound_space(self):
        """Set bounds for parallel dimensions"""
        vpp = (
            1
            if Dim.VPP in self.config.dimensions
            else Dim.VPP.from_config(self.config.ccfg)
        )
        pp_bound = min(
            self.machine.pipeline_bound(),
            self.config.total_layer_num() // vpp,
            self.global_batch_size,
        )
        Dim.PP.set_bound(pp_bound)
        logger.info(
            "PP bound is %d, machine bound = %d, L = %d, VPP = %d, B = %d",
            pp_bound,
            self.machine.pipeline_bound(),
            self.config.total_layer_num(),
            vpp,
            self.global_batch_size,
        )
        Dim.EP.set_bound(self.config.ccfg.n_exp)
        # if (
        #     self.config.dimensions.count(Dim.EP) > 0
        #     and Dim.EP.from_config(self.config.ccfg) <= 1
        # ):
        #     Dim.EP.set_bound(1)
        #     self.config.dimensions.remove(Dim.EP)
        kv_heads = self.config.ccfg.n_kv
        if kv_heads:
            Dim.TP.set_bound(kv_heads)
            logger.warning(
                "Because of n_kv_heads, MP will be limited to %s",
                str(kv_heads),
            )
        else:
            # num_head % (TP * UP) == 0. Add UP later
            Dim.TP.set_bound(
                Hard.highest_power_of_2_divisor(self.config.ccfg.a)
            )

    def filtered_out(self, _):
        """Manual conditions to remove config patterns"""
        # if parallel_config.has_dim(Dim.EP):
        #     if self.config.dim_val(Dim.EP, parallel_config) < 8:
        #         return True
        return False

    def is_valid(self, parallel_config):
        """Check configuration validity"""
        if not parallel_config.is_valid():
            logger.warning("configuration %s not valid", str(parallel_config))
            return False
        if not self.config.moe_valid(parallel_config):
            logger.warning("expert parallel is higher than expert number")
            return False
        if hasattr(self.config, 'ep_constraints_valid') and not self.config.ep_constraints_valid(parallel_config):
            logger.warning("EP divisibility constraints not satisfied")
            return False
        if self.filtered_out(parallel_config):
            logger.warning("Config manually filtered out")
            return False

        if hasattr(parallel_config, 'dims_val') and Dim.CP in parallel_config.dims_val:
            cp_degree = parallel_config.dims_val[Dim.CP]
            if cp_degree > 1:
                seq_len = self.config.ccfg.s
                tp_degree = parallel_config.dims_val.get(Dim.TP, 1)
                pp_degree = parallel_config.dims_val.get(Dim.PP, 1)
                device_per_node = self.machine.device.intra_node_num()
                total_devices = self.machine.number

                attention_type = detect_attention_type(self.config.ccfg).name.lower()

                bw_intra = self.config.ccfg.bw_intra
                bw_inter = self.config.ccfg.bw_inter

                sp_enabled = bool(parallel_config.dims_val.get(Dim.SP, False))

                cp_result = validate_cp_constraints(
                    seq_len=seq_len,
                    cp_degree=cp_degree,
                    tp_degree=tp_degree,
                    pp_degree=pp_degree,
                    device_per_node=device_per_node,
                    attention_type_str=attention_type,
                    bw_intra=bw_intra,
                    bw_inter=bw_inter,
                    total_devices=total_devices,
                    sp_enabled=sp_enabled,
                    cp_algo=getattr(self.config.ccfg, 'cp_algo', 'colossalai_cp'),
                    attention_heads=self.config.ccfg.a,
                    num_kv_heads=getattr(self.config.ccfg, 'n_kv', 0),
                )

                if not cp_result.is_valid:
                    logger.warning("CP constraints violated: %s", cp_result.error_message)
                    return False

                if cp_result.warning_message:
                    logger.info("CP warning: %s", cp_result.warning_message)

        gbs = self.config.global_batch_size(parallel_config)
        if not gbs == self.global_batch_size:
            logger.error(
                "wrong global batch size: ccfg is %d, instead of %d",
                gbs,
                self.global_batch_size,
            )
            return False
        return True

    def memory_estim(self, debugger=None):
        """Whether the config fits memory"""
        logger.debug("estimate_peak")
        verbose = logger.level < logging.INFO
        self.mem_eval.set_config(self.config.ccfg)  # = self.config.ccfg
        # self.mem_eval = EvaluatorV2(self.config)
        logger.debug("ccfg = %s", str(self.config.ccfg))
        peak = self.mem_eval.estimate_peak(
            verbose=verbose
        )  # (logger.level>2))
        logger.debug("peak memory = %d", peak)
        if debugger and debugger.is_enabled():
            debugger.info[Debug.MemParts.TOTAL] = peak
        return peak

    def generate_search_space(self, folder, threads_num):
        """Return a search space computed with memory estimation"""
        space = ({}, 0)
        configs = []
        results = {}
        if threads_num:
            with proc.Pool(processes=threads_num) as pool:
                logger.debug("before loops")
                results, size = self.device_loops(space, pool)
                logger.debug("%d results", len(results))
                for config, result in results.items():
                    logger.debug("result = %s", str(result))
                    logger.debug(
                        "before get: is ready ? %s", str(result.ready())
                    )
                    logger.debug(
                        "before get: is successful ? %s",
                        str(result.successful()),
                    )
                    # if result.successful():
                    peak_mem = result.get(1)
                    logger.debug(
                        "after get: is ready ? %s", str(result.ready())
                    )
                    logger.debug(
                        "after get: is successful ? %s",
                        str(result.successful()),
                    )
                    logger.debug("peak_mem = %s", str(peak_mem))
                    if self.mem_eval.mem_fit(peak_mem):
                        configs.append((config, peak_mem))
                pool.close()
                pool.join()
        else:
            results, size = self.device_loops(space, None)
            for config, peak_mem in results.items():
                if self.mem_eval.mem_fit(peak_mem):
                    configs.append((config, peak_mem))
                    if folder:
                        self.config.write(folder, config)
        logger.output("%d valid configurations generated", size)
        logger.output("%d configuration fitting memory to order", len(configs))

        return configs

    def device_loops(self, space, pool):
        """Exploration loop nest level 0: parallel dimensions dividing devices"""
        for tp in self.config.space(Dim.TP, self.machine.number):
            for pp in self.config.space(Dim.PP, self.machine.number // tp):
                for cp in self.config.space(
                    Dim.CP, self.machine.number // tp // pp
                ):
                    logger.debug(
                        "dp = %d / %d / %d / %d",
                        self.machine.number,
                        tp,
                        cp,
                        pp,
                    )
                    dp = self.machine.number // tp // cp // pp
                    if dp < 1:
                        break
                    space = self.batch_loops(space, pool, (dp, tp, pp, cp))
        return space

    def batch_loops(self, space, pool, dtpc_p):
        """Exploration loop nest level 1: dimensions dividing batch (except already processed DP)"""
        dp, _, pp, _ = dtpc_p
        # if pp > 1:
        for mbs in self.config.space(
            Dim.MBS, self.global_batch_size // pp // dp
        ):
            logger.debug("mbn= %d / %d / %d", self.global_batch_size, dp, mbs)
            mbn = self.global_batch_size // dp // mbs
            space = self.parallel_loops(space, pool, (dtpc_p, (mbs, mbn)))
        # else:
        #     logger.debug("no pipeline so mbn = 1")
        #     mbs = self.global_batch_size // dp
        #     space = self.parallel_loops(space, pool, (dtpc_p, (mbs, 1)))
        return space

    def parallel_loops(self, space, pool, dims):
        """Exploration loop nest level 2: dimensions dependent on others"""
        dtpc_p, mbsn = dims
        dp, tp, pp, _ = dtpc_p
        for ep in self.config.space(Dim.EP, dp * tp):
            for vpp in self.config.range_space(
                Dim.VPP, min(4, pp, self.config.total_layer_num() // pp)
            ):
                for op in self.config.space(
                    Dim.OP, self.config.max_op(dp, tp, ep)
                ):
                    for sp in self.config.bool_space(Dim.SP):
                        space = self.inside_loop_nest(
                            space,
                            pool,
                            (dtpc_p, mbsn, (ep, vpp, op, sp)),
                        )
        return space

    def inside_loop_nest(self, space, pool, dims):
        """Exploration loop nest statements"""
        dtpc_p, mbsn, evos_p = dims
        configs, size = space
        parallel_config = self.config.make_parallel_config(
            dtpc_p, mbsn, evos_p
        )
        logger.info("test config %d : %s", size, str(parallel_config))
        size += 1

        if self.is_valid(parallel_config) and self.config.set_parallel_config(
            parallel_config
        ):
            if pool is None:
                if self.enable_debug:
                    mem_debugger = Debug.Debug(
                        parallel_config,
                        info_type=Debug.MemParts,
                        enable=self.enable_debug,
                        output_file="debug_mem.csv",
                    )
                    # try:
                    peak = self.memory_estim(mem_debugger)
                    mem_debugger.write()
                else:
                    peak = self.memory_estim()
                # except:
                # logger.error()
                # return (configs, size)
            else:
                # logger.debug("before evaluator copy")
                # evaluator = copy.deepcopy(self.mem_eval)
                logger.debug("before apply_async")
                peak = pool.apply_async(
                    pool_estimate_memory,
                    args=(copy.deepcopy(self.config.ccfg),),
                    # args=(evaluator,),
                    # self.memory_estim,
                )
                logger.debug("after apply_async")
            configs[parallel_config] = peak

        return (configs, size)

    def order_search_space(self, space, threads_num, cache_file):
        """Sort the search space computed with performance estimation"""
        if not space:
            return ([], [])
        multiproc = False
        if threads_num and threads_num > 5 * len(space):
            multiproc = True
        scored_space = []
        debug_parts = []
        for config, mem in space:
            self.config.set_parallel_config(config)
            values = []
            if multiproc:
                with proc.Pool(processes=threads_num) as pool:
                    score = pool.apply_async(
                        pool_estimate_performance,
                        args=(
                            copy.deepcopy(self.config),
                            self.machine.device,
                            cache_file,
                        ),
                    )
            else:
                if self.enable_debug:
                    debugger = Debug.Debug(
                        config,
                        info_type=Debug.PerfParts,
                        enable=self.enable_debug,
                    )
                    score = estimate_performance(
                        self.config.ccfg,
                        debugger=debugger,
                        device_type=self.machine.device,
                        memory=mem,
                        cache_file=cache_file,
                    )
                    debugger.write()
                    debug_parts = list(debugger.info.keys())
                    values = list(debugger.info.values())
                    del values[-2:]
                    del debug_parts[-2:]
                else:
                    score = estimate_performance(
                        self.config.ccfg,
                        device_type=self.machine.device,
                        memory=mem,
                    )
            scored_space.append((config, mem, score, values))

            logger.info("config %s has score %f", str(config), score)

        if multiproc:
            new_scored_space = []
            pool.close()
            pool.join()
            for config, mem, score, values in scored_space:
                new_scored_space.append((config, mem, score.get(), values))
        else:
            new_scored_space = scored_space
        return (sorted(new_scored_space, key=lambda x: x[2]), debug_parts)

    def order_space_test_comm_classified(self, space, order_by=2):
        """Order the given space with performance estimation"""
        scored_space = []
        debug_parts = []
        for config, real_time, real_comm_wait in space:
            debugger = Debug.Debug(
                config, info_type=Debug.PerfParts, enable=self.enable_debug
            )
            self.config.set_parallel_config(config)
            peak_mem = self.memory_estim()
            score = estimate_performance(
                self.config.ccfg,
                debugger=debugger,
                device_type=self.machine.device,
                stage_focused=0,
            )  # , memory = mem)
            debugger.write()
            debug_parts = list(debugger.info.keys())
            values = list(debugger.info.values())
            del values[-2:]
            scored_space.append(
                (config, peak_mem, real_time, score, values, real_comm_wait)
            )

            logger.info("config %s has score %f", str(config), score)
        del debug_parts[-2:]
        return (sorted(scored_space, key=lambda x: x[order_by]), debug_parts)

    def order_space_test(self, space, order_by=2):
        """Order the given space with performance estimation"""
        scored_space = []
        debug_parts = []
        for config, real_time in space:
            debugger = Debug.Debug(
                config, info_type=Debug.PerfParts, enable=self.enable_debug
            )
            logger.info("Test config %s", str(config))
            self.config.set_parallel_config(config)
            logger.debug(self.mem_eval.get_strategy())
            peak_mem = self.memory_estim()
            score = estimate_performance(
                self.config.ccfg,
                debugger=debugger,
                device_type=self.machine.device,
            )  # , memory = mem)
            debugger.write()
            debug_parts = list(debugger.info.keys())
            values = list(debugger.info.values())
            del values[-2:]
            scored_space.append((config, peak_mem, real_time, score, values))

            logger.info("config %s has score %f", str(config), score)
        del debug_parts[-2:]
        return (sorted(scored_space, key=lambda x: x[order_by]), debug_parts)

    def plot_title(self):
        """Generate plot title"""
        return (
            f"{self.model_name} on {self.machine.number}"
            + f" {self.machine.device} with {self.global_batch_size} GBS"
        )

    def run_generation_to_ordering(
        self, yaml_folder, threads_num=None, top_num=None, cache_file=None
    ):
        """Test some functions"""
        start = time.time()
        space = self.generate_search_space(yaml_folder, threads_num)
        generation = time.time()
        scored_space, dbg = self.order_search_space(
            space, threads_num, cache_file=cache_file
        )
        ordering = time.time()
        logger.output(
            space_to_string(scored_space, max_num=top_num, debug_parts=dbg)
        )
        logger.output(
            "Space generation took %.2fs and ordering took %.2fs",
            generation - start,
            ordering - generation,
        )
        is_not = " NOT" if not self.config.balancing.from_config else ""
        logger.output(
            "Offset & Recompute were%s computed from config info", is_not
        )
        logger.output(
            "Device number is %d, global batch size is %d, dimensions are %s",
            self.machine.number,
            self.global_batch_size,
            str(self.config.dimensions),
        )
        if self.enable_debug:
            file_path = os.path.dirname(os.path.realpath(__file__))
            output_path = os.path.join(file_path, "output")
            if scored_space:
                Debug.plot_nd(
                    scored_space,
                    output_path,
                    dbg,
                    title=self.plot_title(),
                    max_num=top_num,
                )
        return scored_space

    def to_ppb(self, scored_space, k, cfg_name):
        """Create an input file for pipeline balancing"""
        parallel_config = scored_space[k][0]
        self.config.set_parallel_config(parallel_config)
        self.mem_eval.update_config(self.config)
        m = cfg_name + "_nd_to_ppb_" + str(k)
        s = self.config.dim_val(Dim.PP, parallel_config)
        mb = self.config.dim_val(Dim.MBN, parallel_config)
        i = self.config.dim_val(Dim.VPP, parallel_config)
        mem = str(self.config.ccfg.device_capacity.to_mb)
        filename = (
            os.path.dirname(os.path.realpath(__file__))
            + "/../pipeline_balance/layers/"
            + m
            + ".json"
        )
        with open(filename, "w+", encoding="utf-8") as fp:
            json.dump(
                self.mem_eval.estimate_layer_memory(
                    device_type=self.machine.device
                ),
                fp,
                indent=4,
            )
        logger.output(
            "To run pipeline balancing on configuration %s:"
            "\npython run_pipeline_balance.py "
            "-m %d -s %d -mb %d -i %d -mem %d",
            parallel_config,
            m,
            s,
            mb,
            i,
            mem,
        )
        logger.output("Warning: currently select_recompute_memory \
                should be removed & layer time need to be added")

    def test_from_csv(self, csv_f, output_path=None):
        """Run estimation tests against a real run profiling in csv format"""
        configs, row_num = Debug.get_real_data(csv_f)
        configs_estimated, debug_parts = self.order_space_test(
            configs, order_by=2
        )
        if output_path is not None:
            Debug.plot_vs_real(
                configs_estimated,
                csv_f,
                output_path,
                debug_parts,
                title=self.plot_title(),
            )
        correl, topk = Debug.correlation_topk(configs_estimated, csv_f)
        return correl, topk, row_num

    def test_from_csv_comm_classified(
        self, csv_f, output_path=None, plot_idle=False
    ):
        """Run test to compare estimation with detailed profiling"""
        configs = Debug.get_comm_classified_data(csv_f, plot_idle=plot_idle)
        configs_estimated, debug_parts = self.order_space_test_comm_classified(
            configs, order_by=2
        )

        if output_path is not None:
            Debug.plot_vs_real_comm_classified(
                configs_estimated,
                csv_f,
                output_path,
                debug_parts,
                title=self.plot_title(),
                plot_idle=plot_idle,
            )

        return Debug.correlation_with_classified_comms(configs_estimated)


class ParallelizeMultiModal(ParallelizeLayer):
    """Parallelize a MultiModel"""

    def __init__(
        self,
        evaluator,
        machine,
        global_batch_size=None,
        dimensions=None,
        **extra_config,
    ):

        super().__init__(
            evaluator,
            machine,
            global_batch_size=global_batch_size,
            dimensions=dimensions,
            sub_model="deepseekv3",
            **extra_config,
        )


class Parallelize:  # pylint: disable=R0903
    """Main class instantiated by one of the above two"""

    def __init__(
        self,
        framework,
        config,
        machine,
        **extra_config,
    ):
        logger.debug("before evaluator init")
        if "model" in extra_config:
            model_name = extra_config.pop("model")
            mem_eval = EvaluatorV2(
                config, framework=framework, hook_cls=model_name, machine=machine
            )
        else:
            mem_eval = EvaluatorV2(config, framework=framework, machine=machine)

        if "global_batch_size" in extra_config:
            global_batch_size = extra_config.pop("global_batch_size")
        else:
            global_batch_size = None

        if "dimensions" in extra_config:
            dimensions = extra_config.pop("dimensions")
        else:
            dimensions = None

        if mem_eval.ccfg.multimodal:
            logger.debug("MultiModal is triggered")
            self.instance = ParallelizeMultiModal(
                mem_eval,
                machine,
                global_batch_size=global_batch_size,
                dimensions=dimensions,
                **extra_config,
            )
        else:
            self.instance = ParallelizeLayer(
                mem_eval,
                machine,
                global_batch_size=global_batch_size,
                dimensions=dimensions,
                sub_model=None,
                **extra_config,
            )

    def __getattr__(self, name):
        return self.instance.__getattribute__(name)


def space_to_string(space, max_num=None, debug_parts=None):
    """Space printer"""
    i = 0
    s = ""
    if max_num is not None:
        s += "Top " + str(max_num) + " configurations:\n"
    else:
        s += "\n"
    if len(space) == 0:
        return s
    s += "\t"
    for d in space[0][0].all_dims:
        s += str(d) + " " * (6 - len(str(d)))
    s += "Memory    Performance score  "
    if debug_parts is not None:
        for dbg_part in debug_parts:
            s += "\t" + dbg_part.short_name()
    s += "\n"
    for config in space:
        if max_num is not None and max_num == i:
            break
        s += "\t"
        for v in config[0].values():
            s += v + " " * (6 - len(v))
        s += str(config[1]) + " MB  "  # + str(config[2])
        s += f"{(config[2]):16.12e}"
        for v in config[3]:
            s += f"\t{(100*v/config[2]):.2f}%"
        s += "\n"
        i += 1
    return s


def pool_estimate_memory(config):
    """Calls memory estimation for multiprocessing"""
    logger.debug("estimate_peak")
    # print("estimate_peak")
    e = EvaluatorV2(config)
    return e.estimate_peak()


# def pool_estimate_memory(evaluator):
#     """Calls memory estimation for multiprocessing"""
#     logger.debug("estimate_peak")
#     return evaluator.estimate_peak()


def pool_estimate_performance(config, device):
    """Calls performance estimation for multiprocessing"""
    return estimate_performance(config, device_type=device)
