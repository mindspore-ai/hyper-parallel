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
"""find parallelization"""

from collections import Counter
import time
import json
import os
import logging
from typing import Any, Optional, Tuple

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_performance
from hyper_parallel.auto_parallel.sapp_nd.nd.global_config import GlobalConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import validate_cp_constraints
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    arm_strategy_guard,
    detect_attention_type,
    unset_reads_report,
)

# logger = proc.log_to_stderr()
# logger.setLevel(proc.SUBDEBUG)

def state_machine_links(ccfg: Any, device: Any) -> None:
    """Give a config, and each submodule's, its machine's ranks a node and link bandwidths (B5).

    The CP check, its topology penalty and the CP cost read them off the
    config, where nothing assigned them, so they held the config's defaults,
    8 ranks a node, 400 and 25 GB/s, whatever machine was chosen.  The node
    is the device's first level and its links the first two levels'
    bandwidths, the second level's falling back to the first's.

    Args:
        ccfg: The config the search prices; a multimodal one's submodules
            take the same machine.
        device: The machine's ``Hard.Type``; nothing is stated without one.
    """
    levels = list(getattr(device, "level_bandwidth", None) or [])
    if device is None or not levels:
        return
    for cfg in [ccfg, *(getattr(ccfg, "mm_ccfgs", None) or {}).values()]:
        cfg.device_per_node = device.intra_node_num()
        cfg.bw_intra = levels[0]
        cfg.bw_inter = levels[1] if len(levels) > 1 else levels[0]


class ParallelizeLayer:
    """Parallelize one layer type"""

    def __init__(
        self,
        evaluator: Any,
        machine: Any,
        global_batch_size: Any = None,
        dimensions: Any = None,
        **extra_config: Any,
    ) -> None:
        """Build the search driver for one evaluator and machine."""
        self.enable_debug = logger.level < logging.CRITICAL
        self.machine = machine
        if "mppb" in extra_config:
            manual_ppb = extra_config.pop("mppb")
        else:
            manual_ppb = False

        self.mem_eval = evaluator
        state_machine_links(self.mem_eval._ccfg, getattr(machine, "device", None))

        self.model_name = self.mem_eval._ccfg.model_name
        logger.debug("model is %s", self.model_name)

        # The cap replaces the device's capacity, and the reserve comes out
        # of whichever capacity holds.
        if "max_mem" in extra_config:
            max_mem = extra_config.pop("max_mem")
            if max_mem is not None:
                self.mem_eval._ccfg.device_capacity.set(max_mem)

        if "mem_for_ppb" in extra_config:
            reserve_mem = extra_config.pop("mem_for_ppb")
            self.mem_eval._ccfg.device_capacity.decrease(reserve_mem)

        logger.debug("before global config init")

        if "sub_model" in extra_config:
            sub_model = extra_config.pop("sub_model")
            if sub_model is not None:
                self.config = GlobalConfig(
                    self.mem_eval._ccfg.mm_ccfgs[sub_model],
                    dimensions,
                    mppb=manual_ppb,
                    parent=self.mem_eval._ccfg,
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
        # From here on the configs this search owns take a strategy only
        # through set_strategy; an estimator's copy of one starts unarmed.
        arm_strategy_guard(self.mem_eval.ccfg)

    def bound_space(self) -> None:
        """Set bounds for parallel dimensions"""
        # Bounds live on the module-level dimensions: start each search from
        # none, or it keeps the tightest bound any earlier search set.
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
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
        self._report_bounds(vpp)

    def _report_bounds(self, vpp: int) -> None:
        """Name the bound each searched dimension takes, and why.

        A degree over its bound is never generated, so no check refuses it
        and no estimate drops it: this line is the only trace it leaves.
        """
        heads = "KV heads" if self.config.ccfg.n_kv else "attention heads' largest power of two"
        bounds = (
            (Dim.PP, f"the machine's {self.machine.pipeline_bound()}, {self.config.total_layer_num()} layers "
                     f"over VPP {vpp}, a batch of {self.global_batch_size}"),
            (Dim.EP, "the model's experts"),
            (Dim.TP, f"the model's {heads}"),
        )
        named = [f"{dim} up to {dim.get_bound()} ({why})" for dim, why in bounds if dim in self.config.dimensions]
        if named:
            logger.output("The search bounds %s", "; ".join(named))

    @staticmethod
    def filtered_out(_: Any) -> bool:
        """Manual conditions to remove config patterns"""
        # if parallel_config.has_dim(Dim.EP):
        #     if self.config.dim_val(Dim.EP, parallel_config) < 8:
        #         return True
        return False

    def _refuse(self, reason: str) -> bool:
        """Count a candidate refused for *reason*, for the search's summary, and return False."""
        self.__dict__.setdefault("refused", Counter())[reason] += 1
        return False

    def is_valid(self, parallel_config: Any) -> bool:
        """Check configuration validity"""
        if not parallel_config.is_valid():
            logger.warning("configuration %s not valid", str(parallel_config))
            return self._refuse("a degree out of bounds")
        if not self.config.moe_valid(parallel_config):
            logger.warning("expert parallel is higher than expert number")
            return self._refuse("EP over the experts or the ranks they spread over")
        if hasattr(self.config, 'ep_constraints_valid') and not self.config.ep_constraints_valid(parallel_config):
            logger.warning("EP divisibility constraints not satisfied")
            return self._refuse("EP divisibility")
        if self.filtered_out(parallel_config):
            logger.warning("Config manually filtered out")
            return self._refuse("filtered out")

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
                    return self._refuse("CP constraints")

                if cp_result.warning_message:
                    logger.info("CP warning: %s", cp_result.warning_message)

        gbs = self.config.global_batch_size(parallel_config)
        if not gbs == self.global_batch_size:
            logger.error(
                "wrong global batch size: ccfg is %d, instead of %d",
                gbs,
                self.global_batch_size,
            )
            return self._refuse("global batch")
        return True

    def priced(self) -> Any:
        """The config a candidate is priced on: the whole model, every submodule of it.

        The search drives one submodule's strategy (:class:`GlobalConfig`
        gives the others the same degrees), and a candidate is priced on the
        model the run trains: a vision tower's parameters, activations and
        compute count toward the stage that holds them (F2).
        """
        return self.mem_eval.ccfg

    def memory_estim(self, debugger: Any = None) -> Any:
        """Whether the config fits memory"""
        logger.debug("estimate_peak")
        verbose = logger.level < logging.INFO
        self.mem_eval.set_config(self.priced())
        # self.mem_eval = EvaluatorV2(self.config)
        logger.debug("ccfg = %s", self.config.ccfg)
        peak = self.mem_eval.estimate_peak(
            verbose=verbose
        )  # (logger.level>2))
        logger.debug("peak memory = %d", peak)
        if debugger and debugger.is_enabled():
            debugger.info[Debug.MemParts.TOTAL] = peak
        return peak

    def _drop(self, config: Any, peak: float) -> None:
        """Say that the memory budget drops *config*, and by how much its peak misses it."""
        budget = self.mem_eval.get_max_device_memory()
        logger.output(
            "%s dropped: its peak of %.0f MB is over the %.0f MB budget by %.0f MB",
            config,
            peak,
            budget,
            peak - budget,
        )

    def _summarise_search(self, size: int, priced: int, fitting: int) -> None:
        """The search's two counts: what was tested, refused and priced, and what fits."""
        refused = self.__dict__.get("refused", Counter())
        reasons = ", ".join(f"{reason} {count}" for reason, count in refused.most_common())
        logger.output(
            "%d configurations tested: %d refused%s, %d priced for memory",
            size,
            sum(refused.values()),
            f" ({reasons})" if reasons else "",
            priced,
        )
        logger.output(
            "%d configuration fitting memory to order, %d dropped over the memory budget",
            fitting,
            priced - fitting,
        )

    def generate_search_space(self, folder: Any, threads_num: Any) -> Any:  # pylint: disable=unused-argument
        """Return a search space computed with memory estimation.

        A candidate the memory budget drops is named with its peak, and the
        summary counts the candidates the checks refused apart from those the
        budget dropped, each check by name.  *threads_num* is unused: the
        candidates are priced in this process.
        """
        configs = []
        self.refused = Counter()
        with Debug.holding_open():
            results, size = self.device_loops(({}, 0), None)
        for config, peak_mem in results.items():
            if self.mem_eval.mem_fit(peak_mem):
                configs.append((config, peak_mem))
                if folder:
                    self.config.write(folder, config)
            else:
                self._drop(config, peak_mem)
        self._summarise_search(size, len(results), len(configs))

        return configs

    def device_loops(self, space: Any, collect: Optional[list]) -> Tuple[dict, int]:
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
                    space = self.batch_loops(space, collect, (dp, tp, pp, cp))
        return space

    def batch_reachable(self) -> bool:
        """Whether any candidate the loops build makes the global batch; one line says so where none does.

        Each candidate's batch is its DP x MB x MBS, MB the batch over DP and
        MBS, so a batch that is no whole number of micro-batches of any DP x
        MBS the loops reach is refused in every candidate, each with an error.
        The search stops before building one, with this line instead.
        """
        number = self.machine.number
        reached = set()
        for tp in self.config.space(Dim.TP, number):
            for pp in self.config.space(Dim.PP, number // tp):
                for cp in self.config.space(Dim.CP, number // tp // pp):
                    dp = number // tp // cp // pp
                    if dp < 1:
                        break
                    for mbs in self.config.space(Dim.MBS, self.global_batch_size // pp // dp):
                        mbn = self.global_batch_size // dp // mbs
                        candidate = self.config.make_parallel_config((dp, tp, pp, cp), (mbs, mbn), (1, 1, 1, False))
                        if self.config.global_batch_size(candidate) == self.global_batch_size:
                            return True
                        reached.add(self.config.dim_val(Dim.DP, candidate) * mbs)
        logger.error(
            "No candidate makes a global batch of %d (the yaml states %d): the candidates the search builds "
            "have DP x MBS of %s, and %d is no whole number of micro-batches of any. State -b as a multiple "
            "of one of them, or let DP shrink by naming PP, CP or MP in -l.",
            self.global_batch_size,
            self.config.ccfg.gbs,
            ", ".join(str(batch) for batch in sorted(reached)),
            self.global_batch_size,
        )
        return False

    def batch_loops(self, space: Any, collect: Optional[list], dtpc_p: Any) -> Tuple[dict, int]:
        """Exploration loop nest level 1: dimensions dividing batch (except already processed DP)"""
        dp, _, pp, _ = dtpc_p
        # if pp > 1:
        for mbs in self.config.space(
            Dim.MBS, self.global_batch_size // pp // dp
        ):
            logger.debug("mbn= %d / %d / %d", self.global_batch_size, dp, mbs)
            mbn = self.global_batch_size // dp // mbs
            space = self.parallel_loops(space, collect, (dtpc_p, (mbs, mbn)))
        # else:
        #     logger.debug("no pipeline so mbn = 1")
        #     mbs = self.global_batch_size // dp
        #     space = self.parallel_loops(space, pool, (dtpc_p, (mbs, 1)))
        return space

    def parallel_loops(self, space: Any, collect: Optional[list], dims: Any) -> Tuple[dict, int]:
        """Exploration loop nest level 2: dimensions dependent on others"""
        dtpc_p, mbsn = dims
        dp, tp, pp, cp = dtpc_p
        for ep in self.config.space(Dim.EP, self.config.max_ep(dp, tp, cp)):
            for vpp in self.config.range_space(
                Dim.VPP, min(4, pp, self.config.total_layer_num() // pp)
            ):
                for op in self.config.space(
                    Dim.OP, self.config.max_op(dp, tp, ep, cp)
                ):
                    for sp in self.config.bool_space(Dim.SP):
                        space = self.inside_loop_nest(
                            space,
                            collect,
                            (dtpc_p, mbsn, (ep, vpp, op, sp)),
                        )
        return space

    def inside_loop_nest(self, space: Any, collect: Optional[list], dims: Any) -> Tuple[dict, int]:
        """Exploration loop nest statements: price a candidate that passes the checks, or collect it.

        With *collect*, a list, the candidate and its loop degrees are
        appended instead of priced, after the same checks and the same
        setting of the config as here.
        """
        dtpc_p, mbsn, evos_p = dims
        configs, size = space
        parallel_config = self.config.make_parallel_config(
            dtpc_p, mbsn, evos_p
        )
        logger.info("test config %d : %s", size, parallel_config)
        size += 1

        if self.is_valid(parallel_config) and self.config.set_parallel_config(
            parallel_config
        ):
            if collect is not None:
                collect.append((parallel_config, dims))
                return (configs, size)
            if self.enable_debug:
                mem_debugger = Debug.Debug(
                    parallel_config,
                    info_type=Debug.MemParts,
                    enable=self.enable_debug,
                    output_file="debug_mem.csv",
                )
                peak = self.memory_estim(mem_debugger)
                mem_debugger.write()
            else:
                peak = self.memory_estim()
            configs[parallel_config] = peak

        return (configs, size)

    def order_search_space(self, space: Any, threads_num: Any,  # pylint: disable=unused-argument
                           cache_file: Any) -> Any:
        """Sort the search space computed with performance estimation.

        *threads_num* is unused: the candidates are priced in this process.
        """
        if not space:
            return ([], [])
        scored_space = []
        debug_parts = []
        with Debug.holding_open():
            for config, peak in space:
                self.config.set_parallel_config(config)
                values = []
                priced = self.priced()
                if self.enable_debug:
                    debugger = Debug.Debug(
                        config,
                        info_type=Debug.PerfParts,
                        enable=self.enable_debug,
                    )
                    score = estimate_performance(
                        priced,
                        debugger=debugger,
                        device_type=self.machine.device,
                        memory=peak,
                        cache_file=cache_file,
                    )
                    debugger.write()
                    debug_parts = list(debugger.info.keys())
                    values = list(debugger.info.values())
                    del values[-2:]
                    del debug_parts[-2:]
                else:
                    score = estimate_performance(
                        priced,
                        device_type=self.machine.device,
                        memory=peak,
                        cache_file=cache_file,
                    )
                scored_space.append((config, peak, score, values))
                logger.info("config %s has score %f", config, score)
        return (sorted(scored_space, key=lambda x: x[2]), debug_parts)

    def order_space_test_comm_classified(self, space: Any, order_by: Any = 2) -> Any:
        """Order the given space with performance estimation.

        A measured configuration the cost model cannot represent, such as
        expert parallelism wider than its stage, is left out and named, so
        that it does not end the comparison of every other one.
        """
        scored_space = []
        debug_parts = []
        for config, real_time, real_comm_wait in space:
            # The comparison needs the per-part split at any verbosity; debug.csv stays opt-in.
            debugger = Debug.Debug(config, info_type=Debug.PerfParts, enable=True)
            try:
                self.config.set_parallel_config(config)
                peak_mem = self.memory_estim()
                score = estimate_performance(
                    self.priced(),
                    debugger=debugger,
                    device_type=self.machine.device,
                    stage_focused=0,
                )  # , memory = mem)
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                logger.output("ND cannot cost %s, left out of the comparison: %s", config, exc)
                continue
            if self.enable_debug:
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

    def order_space_test(self, space: Any, order_by: Any = 2) -> Any:
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
                self.priced(),
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

    def plot_title(self) -> str:
        """Generate plot title"""
        return (
            f"{self.model_name} on {self.machine.number}"
            + f" {self.machine.device} with {self.global_batch_size} GBS"
        )

    def _search_and_order(self, yaml_folder: Any, threads_num: Any, cache_file: Any) -> Tuple[list, list, float, float]:
        """Generate and order the space.

        Returns:
            The ordered entries, the parts of their scores, and the seconds
            generation and ordering took.
        """
        if not self.batch_reachable():
            return [], [], 0.0, 0.0
        start = time.time()
        space = self.generate_search_space(yaml_folder, threads_num)
        generated = time.time()
        scored_space, dbg = self.order_search_space(space, threads_num, cache_file=cache_file)
        return scored_space, dbg, generated - start, time.time() - generated

    def _ranked_search(self, yaml_folder: Any, threads_num: Any, cache_file: Any,
                       ranking_csv: Optional[str]) -> Tuple[list, list, float, float]:
        """Search and order the space, rank its entries and write the ranking.

        Returns:
            What :meth:`_search_and_order` returns, each entry's
            configuration given its rank.
        """
        scored_space, dbg, generation, ordering = self._search_and_order(yaml_folder, threads_num, cache_file)
        for rank, entry in enumerate(scored_space, start=1):
            entry[0].rank = rank
        if ranking_csv:
            Debug.write_ranking_csv(scored_space, ranking_csv)
            logger.output(
                "ND's order of %d configuration(s) written to %s",
                len(scored_space),
                ranking_csv,
            )
        return scored_space, dbg, generation, ordering

    def _log_recompute_source(self, priced: Any) -> None:
        """Log where the offsets and the recompute came from, and a stated mode the candidates were not priced in."""
        is_not = " NOT" if not self.config.balancing.from_config else ""
        logger.output(
            "Offset & Recompute were%s computed from config info", is_not
        )
        stated_mode = getattr(priced, "stated_ac_mode", None)
        if not self.config.balancing.from_config and stated_mode not in (None, "full"):
            # The balancing gives every candidate full recompute without
            # -mppb, whatever the run's mode, which cost an hour twice
            # because nothing said so (H1).
            logger.output(
                "The run's activation checkpoint mode is %s, and every candidate above is priced fully "
                "recomputed: pass -mppb to price the run's own",
                stated_mode,
            )

    def run_generation_to_ordering(
        self,
        yaml_folder: Any,
        threads_num: Any = None,
        top_num: Any = None,
        cache_file: Any = None,
        ranking_csv: Optional[str] = None,
    ) -> Any:
        """Search, order and print the configurations that fit memory.

        ``ranking_csv``, when given, receives every one of them in ND's order.
        It is written before anything is plotted, so a plot that fails cannot
        take the ranking with it.
        """
        scored_space, dbg, generation, ordering = self._ranked_search(
            yaml_folder, threads_num, cache_file, ranking_csv
        )

        logger.output(
            space_to_string(scored_space, max_num=top_num, debug_parts=dbg)
        )
        priced = self.priced()
        _log_priced_caveats(priced)
        logger.output(
            "Space generation took %.2fs and ordering took %.2fs",
            generation,
            ordering,
        )
        self._log_recompute_source(priced)
        logger.output(
            "Device number is %d, global batch size is %d, dimensions are %s",
            self.machine.number,
            self.global_batch_size,
            str(self.config.dimensions),
        )
        if self.enable_debug:
            output_path = Debug.output_dir()
            if scored_space:
                Debug.plot_nd(
                    scored_space,
                    output_path,
                    dbg,
                    title=self.plot_title(),
                    max_num=top_num,
                )
        return scored_space

    def to_ppb(self, scored_space: Any, k: Any, cfg_name: Any, folder: Optional[str] = None) -> str:
        """Write the pipeline balancer's layer description of the k-th configuration.

        Args:
            scored_space: The ordered search space.
            k: Rank of the configuration to describe.
            cfg_name: Prefix of the model name the balancer is given.
            folder: Where to write; ND's output directory by default.

        Returns:
            The path of the file written.
        """
        parallel_config = scored_space[k][0]
        self.config.set_parallel_config(parallel_config)
        self.mem_eval.set_config(self.config.ccfg)
        name = f"{cfg_name}_nd_to_ppb_{k}"
        folder = os.path.abspath(folder or Debug.output_dir())
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name + ".json")
        description = self.mem_eval.estimate_layer_memory(device_type=self.machine.device)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(description, fp, indent=4)
        logger.output(
            "To run pipeline balancing on configuration %s:"
            "\npython run_pipeline_balance.py -lf %s -m %s -s %s -mb %s -i %s -mem %s",
            parallel_config,
            folder,
            name,
            self.config.dim_val(Dim.PP, parallel_config),
            self.config.dim_val(Dim.MBN, parallel_config),
            self.config.dim_val(Dim.VPP, parallel_config),
            int(self.config.ccfg.device_capacity.to_mb().size),
        )
        return path

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
        evaluator: Any,
        machine: Any,
        global_batch_size: Any = None,
        dimensions: Any = None,
        **extra_config: Any,
    ) -> None:
        """Drive the search from the submodule the parser marked as main."""
        super().__init__(
            evaluator,
            machine,
            global_batch_size=global_batch_size,
            dimensions=dimensions,
            sub_model=getattr(
                evaluator.ccfg, "mm_main", None
            ) or evaluator.ccfg.mm_order[-1],
            **extra_config,
        )


class Parallelize:  # pylint: disable=R0903
    """Main class instantiated by one of the above two"""

    def __init__(self, framework: Any, config: Any, machine: Any, **extra_config: Any) -> None:
        """Dispatch to the unimodal or multimodal search driver."""
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


def _log_priced_caveats(priced: Any) -> None:
    """Log what qualifies every number of the table: the length it prices, MTP layers unpriced, fields never set."""
    if not getattr(priced, "seq_len_stated", True):
        # A warning at parse time scrolls past a search's lines; this one
        # sits under the table it qualifies (I11).
        logger.output(
            "Every number above is priced at %d tokens, the model's context limit: the config states no "
            "training sequence length (dataset.data_transform.max_seq_len or dataset.data_config.seq_length)",
            priced.s,
        )
    if getattr(priced, "mtp_unpriced", 0):
        # A stated MTP depth looked priced and was not (I2).
        logger.output(
            "The model states %d MTP layer(s), which the AutoModels trainer does not build: no number above "
            "prices them",
            priced.mtp_unpriced,
        )
    for line in unset_reads_report():
        logger.output(line)


def space_to_string(space: Any, max_num: Any = None, debug_parts: Any = None) -> str:
    """Format ranked configurations for CLI output.

    Args:
        space: Configurations with memory, performance score and score components.
            TOP uses the rank stored on each configuration, falling back to its
            position in the supplied list when it has not been ranked.
        max_num: Maximum number of configurations to print.
        debug_parts: Performance components to name in the header.
    """
    s = ""
    if max_num is not None:
        s += "Top " + str(max_num) + " configurations:\n"
    else:
        s += "\n"
    if len(space) == 0:
        return s
    s += "\tTOP   "
    for d in space[0][0].all_dims:
        s += str(d) + " " * (6 - len(str(d)))
    s += "Memory    Performance score  "
    if debug_parts is not None:
        for dbg_part in debug_parts:
            s += "\t" + dbg_part.short_name()
    s += "\n"
    for i, config in enumerate(space):
        if max_num is not None and max_num == i:
            break
        s += f"\t{config[0].rank or i + 1:<6}"
        for v in config[0].values():
            s += v + " " * (6 - len(v))
        s += str(config[1]) + " MB  "  # + str(config[2])
        s += f"{(config[2]):16.12e}"
        for v in config[3]:
            s += f"\t{(100*v/config[2]):.2f}%"
        s += "\n"
    return s
